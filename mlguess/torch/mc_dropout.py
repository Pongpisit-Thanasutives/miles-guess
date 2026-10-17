"""Monte Carlo predictive moments with an explicit output distribution."""

from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler, WeightedRandomSampler


_DROPOUT_TYPES = (
    torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
    torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout,
)
_CLASSIFICATION_KINDS = {"probabilities", "logits", "evidence", "dirichlet"}
_OUTPUT_KINDS = _CLASSIFICATION_KINDS | {"regression", "gaussian_std", "gaussian_variance", "nig"}


def enable_dropout(model):
    """Enable only dropout modules, leaving other evaluation states unchanged."""
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()


@contextmanager
def inference_context(model, *, dropout=False):
    """Freeze training statistics and restore every module's prior mode."""
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if dropout:
            enable_dropout(model)
        with torch.no_grad():
            yield
    finally:
        for module, training in states:
            module.training = training


def as_model_input(data, model, *, device=None):
    """Convert NumPy or tensor inputs to the model's device and floating dtype."""
    parameter = next(model.parameters(), None)
    if parameter is None:
        parameter = next((buffer for buffer in model.buffers() if buffer.is_floating_point()), None)
    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data)
    if parameter is None:
        return torch.as_tensor(data, device=device)
    return torch.as_tensor(data, device=parameter.device if device is None else device, dtype=parameter.dtype)


def _matrix(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must have floating shape (samples, outputs)")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value


def _parameters(output, count, output_kind):
    if isinstance(output, (tuple, list)):
        if len(output) != count:
            raise ValueError(f"{output_kind} requires {count} parameter tensors")
        values = tuple(_matrix(value, output_kind) for value in output)
    else:
        output = _matrix(output, output_kind)
        if output.shape[-1] % count:
            raise ValueError(f"{output_kind} output width must be divisible by {count}")
        # Concatenated parameters use blocks: [all means, all scales, ...].
        values = output.chunk(count, dim=-1)
    if any(value.shape != values[0].shape for value in values):
        raise ValueError(f"{output_kind} parameter shapes must match")
    return values


def _entropy(prob):
    # A single column represents Bernoulli P(Y=1); multiple columns categorical.
    full_prob = torch.cat((prob, 1 - prob), dim=-1) if prob.shape[-1] == 1 else prob
    return -torch.special.xlogy(full_prob, full_prob).sum(dim=-1)


def _output_moments(output, output_kind):
    """Return mean, conditional aleatoric, within-model epistemic, expected H."""
    if output_kind == "nig":
        mean, v, alpha, beta = _parameters(output, 4, output_kind)
        if (v <= 0).any() or (alpha <= 1).any() or (beta <= 0).any():
            raise ValueError("Finite NIG predictive variance requires v>0, alpha>1 and beta>0")
        aleatoric = beta / (alpha - 1)
        epistemic = aleatoric / v
        return mean, aleatoric, epistemic, None
    if output_kind in {"gaussian_std", "gaussian_variance"}:
        mean, scale = _parameters(output, 2, output_kind)
        if (scale < 0).any():
            raise ValueError("Gaussian standard deviations/variances must be nonnegative")
        aleatoric = scale.square() if output_kind == "gaussian_std" else scale
        return mean, aleatoric, torch.zeros_like(mean), None
    output = _matrix(output, output_kind)
    if output_kind == "regression":
        return output, torch.zeros_like(output), torch.zeros_like(output), None
    if output_kind in {"evidence", "dirichlet"}:
        alpha = torch.relu(output) + 1 if output_kind == "evidence" else output
        if alpha.shape[-1] < 2 or (alpha <= 0).any():
            raise ValueError("Dirichlet parameters require at least two classes and alpha>0")
        strength = alpha.sum(dim=-1, keepdim=True)
        if not torch.isfinite(strength).all():
            raise ValueError("Dirichlet total concentration must be finite")
        prob = alpha / strength
        epistemic = prob * (1 - prob) / (strength + 1)
        aleatoric = prob * (1 - prob) - epistemic
        # E_{P~Dir(alpha)} H(Categorical(P)); includes within-model uncertainty.
        expected_entropy = (
            torch.digamma(strength + 1)
            - (prob * torch.digamma(alpha + 1)).sum(dim=-1, keepdim=True)
        ).squeeze(-1)
        return prob, aleatoric, epistemic, expected_entropy
    if output_kind == "logits":
        prob = torch.sigmoid(output) if output.shape[-1] == 1 else torch.softmax(output, dim=-1)
    else:
        prob = output
        if (prob < 0).any() or (prob > 1).any():
            raise ValueError("probabilities must lie in [0,1]; specify output_kind='logits' for raw logits")
        if prob.shape[-1] > 1 and not torch.allclose(
            prob.sum(dim=-1), torch.ones_like(prob[:, 0]), rtol=1e-5, atol=1e-7
        ):
            raise ValueError("Categorical probabilities must sum to one for each sample")
    return prob, prob * (1 - prob), torch.zeros_like(prob), _entropy(prob)


def _scale_moments(mean, aleatoric, epistemic, y_scaler):
    """Apply one affine inverse transform to means and squared slope to variance."""
    if y_scaler is None or y_scaler is False:
        return mean, aleatoric, epistemic
    from mlguess.uncertainty import affine_inverse_scale

    slope = affine_inverse_scale(y_scaler, mean.shape[-1])
    scale = torch.as_tensor(slope, dtype=mean.dtype, device=mean.device).square()
    mean = torch.as_tensor(
        y_scaler.inverse_transform(mean.detach().cpu().numpy()),
        dtype=mean.dtype, device=mean.device,
    )
    return mean, aleatoric * scale, epistemic * scale


def monte_carlo_dropout(data_loader,
                        forward_passes,
                        model,
                        n_classes=None,
                        n_samples=None,
                        batch_size=1024,
                        uncertainty=False,
                        *,
                        output_kind=None,
                        y_scaler=None):
    """Return a finite equal-weight mixture's uncertainty decomposition.

    ``output_kind`` explicitly defines model outputs: ``regression`` means
    point predictions; ``gaussian_std``/``gaussian_variance`` mean (mu, scale);
    ``nig`` means (mu, v, alpha, beta); ``logits``/``probabilities`` mean class
    scores; ``evidence`` means raw evidence transformed by ReLU+1; and
    ``dirichlet`` means positive concentration parameters. Distribution
    parameters may be tuples or concatenated parameter blocks. A one-column
    logit/probability output represents binary P(Y=1).

    When omitted, use ``model.mc_output_kind``; the legacy
    ``uncertainty=True`` explicitly selects ``evidence``. Generic models without
    either declaration must specify ``output_kind``. ``n_classes`` optionally
    checks the number of classes or regression targets; ``n_samples`` checks
    the number of samples. ``batch_size`` is retained for compatibility; the
    loader controls batching. Data must have identical order on every pass;
    use ``shuffle=False`` and no stochastic validation transforms.

    Estimated moments are NumPy arrays. ``epistemic`` is the sum of
    ``within_epistemic`` and ``between_epistemic``; ``total`` is aleatoric plus
    epistemic. The legacy ``variance`` key denotes between-member variance.
    Variances use population normalization (ddof=0), so the finite-mixture law
    of total variance holds exactly, including one pass.

    Point regression cannot estimate observational noise: it returns
    ``aleatoric=None``, ``total=None``, and ``aleatoric_available=False``.
    Its epistemic component is the between-pass variance. Classification also
    returns entropy and mutual information, including within-Dirichlet and
    between-pass contributions. Target scaling supports affine sklearn
    scalers only and occurs before aggregation.
    """
    if output_kind is None:
        output_kind = "evidence" if uncertainty else getattr(model, "mc_output_kind", None)
    elif uncertainty and output_kind != "evidence":
        raise ValueError("uncertainty=True conflicts with output_kind; use the explicit output_kind alone")
    if output_kind not in _OUTPUT_KINDS:
        raise ValueError(f"Specify output_kind as one of {sorted(_OUTPUT_KINDS)}")
    if not isinstance(forward_passes, int) or isinstance(forward_passes, bool) or forward_passes < 1:
        raise ValueError("forward_passes must be a positive integer")
    if (n_samples is not None and n_samples < 1) or (n_classes is not None and n_classes < 1):
        raise ValueError("n_samples and n_classes must be positive when specified")
    categorical = output_kind in _CLASSIFICATION_KINDS
    if categorical and y_scaler is not None and y_scaler is not False:
        raise ValueError("y_scaler is only valid for regression outputs")
    if isinstance(data_loader, DataLoader) and isinstance(
        data_loader.sampler, (RandomSampler, SubsetRandomSampler, WeightedRandomSampler)
    ):
        raise ValueError("Monte Carlo passes require consistent sample order; use shuffle=False")
    if isinstance(data_loader, DataLoader) and getattr(data_loader.sampler, "shuffle", False):
        raise ValueError("Monte Carlo passes require a non-shuffled sampler")
    pass_means, pass_aleatoric, pass_epistemic, pass_expected_entropy = [], [], [], []
    with inference_context(model, dropout=True):
        for _ in range(forward_passes):
            means, aleatoric, epistemic, expected_entropies = [], [], [], []
            for batch in data_loader:
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                output = model(as_model_input(inputs, model))
                mean, ale, epi, expected_entropy = _output_moments(output, output_kind)
                mean, ale, epi = _scale_moments(mean, ale, epi, y_scaler)
                if not all(torch.isfinite(value).all() for value in (mean, ale, epi)):
                    raise ValueError("Predictive moments must be finite")
                means.append(mean.detach().cpu())
                aleatoric.append(ale.detach().cpu())
                epistemic.append(epi.detach().cpu())
                if categorical:
                    expected_entropies.append(expected_entropy.detach().cpu())
            if not means:
                raise ValueError("data_loader yielded no batches; it must be reusable for each pass")
            mean = torch.cat(means, dim=0)
            if n_samples is not None and mean.shape[0] != n_samples:
                raise ValueError(f"Expected {n_samples} samples, got {mean.shape[0]}")
            if n_classes is not None and mean.shape[1] != n_classes:
                raise ValueError(f"Expected {n_classes} outputs, got {mean.shape[1]}")
            if pass_means and mean.shape != pass_means[0].shape:
                raise ValueError("Each Monte Carlo pass must contain the same samples and outputs")
            pass_means.append(mean)
            pass_aleatoric.append(torch.cat(aleatoric, dim=0))
            pass_epistemic.append(torch.cat(epistemic, dim=0))
            if categorical:
                pass_expected_entropy.append(torch.cat(expected_entropies, dim=0))
    means = torch.stack(pass_means)
    mean = means.mean(dim=0)
    aleatoric = torch.stack(pass_aleatoric).mean(dim=0)
    within = torch.stack(pass_epistemic).mean(dim=0)
    between = means.var(dim=0, correction=0)
    epistemic = within + between
    result = {
        "mean": mean, "aleatoric": aleatoric, "epistemic": epistemic,
        "total": aleatoric + epistemic, "within_epistemic": within,
        "between_epistemic": between, "variance": between,
    }
    if categorical:
        entropy = _entropy(mean)
        member_entropy = torch.stack([_entropy(member) for member in pass_means]).mean(dim=0)
        expected_entropy = torch.stack(pass_expected_entropy).mean(dim=0)
        result.update(
            entropy=entropy, mutual_info=entropy - expected_entropy,
            mutual_info_within=member_entropy - expected_entropy,
            mutual_info_between=entropy - member_entropy,
        )
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Aggregated uncertainty moments must be finite; check parameter magnitudes and dtype")
    result = {key: value.numpy() for key, value in result.items()}
    result["aleatoric_available"] = output_kind != "regression"
    if output_kind == "regression":
        result["aleatoric"] = None
        result["total"] = None
    return result
