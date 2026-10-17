"""Monte Carlo dropout and finite-mixture uncertainty aggregation for Keras 3."""

from contextlib import contextmanager

import keras
import keras.ops as ops
import numpy as np

from mlguess.uncertainty import ensemble_uncertainties, inverse_scale_moments, dirichlet_uncertainties


def _all_layers(model):
    """Walk public layer collections, including nested models, exactly once."""
    seen = set()
    pending = [model]
    while pending:
        layer = pending.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        yield layer
        pending.extend(getattr(layer, "layers", ()))


@contextmanager
def _dropout_only(model):
    """Enable dropout while all other layers stay in inference mode.

    In particular, BatchNormalization running moments are neither updated nor
    used as batch statistics. GaussianNoise also remains disabled. Calls are
    restored even if inference raises. Do not use the same model concurrently
    from another thread while sampling.
    """
    dropout_types = tuple(getattr(keras.layers, name) for name in (
        "Dropout", "SpatialDropout1D", "SpatialDropout2D", "SpatialDropout3D",
        "AlphaDropout", "GaussianDropout",
    ))
    replacements = []
    try:
        for layer in _all_layers(model):
            if isinstance(layer, dropout_types):
                original = layer.call

                def stochastic_call(inputs, *args, _call=original, **kwargs):
                    kwargs["training"] = True
                    return _call(inputs, *args, **kwargs)

                replacements.append((layer, original))
                layer.call = stochastic_call
            elif isinstance(layer, keras.layers.RNN) and (
                getattr(layer, "dropout", 0) or getattr(layer, "recurrent_dropout", 0)
            ):
                raise NotImplementedError(
                    "Selective MC dropout does not support dropout internal to recurrent layers."
                )
        yield
    finally:
        for layer, original in reversed(replacements):
            layer.call = original


def mc_raw_predictions(model, x, forward_passes, batch_size=None):
    """Return raw network samples with shape ``(passes, samples, parameters)``."""
    if isinstance(forward_passes, bool) or not isinstance(forward_passes, (int, np.integer)) or forward_passes < 1:
        raise ValueError("forward_passes must be a positive integer")
    if len(x) == 0:
        raise ValueError("MC prediction requires at least one sample")
    batch_size = len(x) if batch_size is None else batch_size
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    samples = []
    with _dropout_only(model):
        for _ in range(forward_passes):
            batches = [
                ops.convert_to_numpy(model(x[start:start + batch_size], training=False))
                for start in range(0, len(x), batch_size)
            ]
            samples.append(np.concatenate(batches, axis=0))
    return np.stack(samples, axis=0)


def monte_carlo_ensemble(model, x_test, y_test, forward_passes, y_scaler=None):
    """Aggregate regression MC dropout using the law of total variance.

    Returns ``(mean, aleatoric_variance, epistemic_variance)`` in target units.
    Aleatoric variance is the mean conditional variance. Epistemic variance
    includes both each evidential member's epistemic variance and the population
    variance (``ddof=0``) of the sampled means. Gaussian networks emit standard
    deviations, which their uncertainty helper squares before aggregation.

    For deterministic regressors, aleatoric variance is unknown (``None``),
    while the sampled-mean variance supplies the epistemic component only.
    Both current Keras regressors and the legacy wrappers are supported.
    ``y_test`` is retained for compatibility and is used only to check shapes.
    Target transforms must be supported affine sklearn scalers.
    """
    targets = np.asarray(y_test)
    if targets.ndim not in (1, 2) or len(targets) != len(x_test):
        raise ValueError("y_test must have shape (samples,) or (samples, targets)")
    n_targets = 1 if targets.ndim == 1 else targets.shape[1]
    network = getattr(model, "model", None)
    network = model if network is None else network
    draws = mc_raw_predictions(network, x_test, forward_passes, getattr(model, "batch_size", None))
    means, aleatoric, epistemic = [], [], []
    deterministic = getattr(model, "_uncertainty_kind", None) == "deterministic"
    if hasattr(model, "evidential") and hasattr(model, "uncertainty"):
        deterministic = not (model.evidential or model.uncertainty)
    if deterministic or not hasattr(model, "calc_uncertainties"):
        if draws.shape[1:] != (len(x_test), n_targets):
            raise ValueError("Predicted target shape does not match y_test")
        means, _, _ = inverse_scale_moments(draws, np.zeros_like(draws), np.zeros_like(draws), y_scaler)
        return means.mean(axis=0), None, means.var(axis=0, ddof=0)
    for output in draws:
        components = model.calc_uncertainties(output, y_scaler=y_scaler)
        if isinstance(components, (tuple, list)):
            if len(components) == 2:
                mu, ale = components
                epi = np.zeros_like(ale)
            elif len(components) == 3:
                mu, ale, epi = components
            else:
                raise ValueError("Regression uncertainty helpers must return two or three components")
        else:
            if components.shape[-1] != 3 * n_targets:
                raise ValueError("Expected current Keras uncertainties as concatenated [mean, aleatoric, epistemic]")
            mu, ale, epi = np.split(components, 3, axis=-1)
        if np.shape(mu) != (len(x_test), n_targets):
            raise ValueError("Predicted target shape does not match y_test")
        means.append(mu)
        aleatoric.append(ale)
        epistemic.append(epi)
    result = ensemble_uncertainties(np.stack(means), np.stack(aleatoric), np.stack(epistemic), axis=0)
    return result["mean"], result["aleatoric"], result["epistemic"]



def categorical_mc_moments(raw_samples, *, evidential=False, from_logits=False):
    """Combine categorical MC samples into marginal class-indicator moments.

    ``raw_samples`` has shape (passes, samples, classes). Evidential values are
    decoded as alpha=relu(output)+1. Otherwise values must be probabilities, or
    logits when ``from_logits=True``. The five returned arrays are probability,
    aleatoric variance, epistemic variance, predictive entropy, and mutual
    information. Dirichlet within-pass epistemic variance and entropy are
    retained, as well as variation between dropout passes.
    """
    from scipy.special import digamma, xlogy

    output = np.asarray(raw_samples, dtype=np.float64)
    if (output.ndim != 3 or min(output.shape[:2]) < 1 or output.shape[-1] < 2
            or not np.all(np.isfinite(output))):
        raise ValueError("Categorical samples must be finite with shape (passes, samples, classes>=2)")
    if evidential:
        alpha = np.maximum(output, 0) + 1
        probability, aleatoric, within_epistemic, _ = dirichlet_uncertainties(alpha)
        strength = alpha.sum(axis=-1, keepdims=True)
        expected_entropy = (digamma(strength + 1).squeeze(-1)
                            - np.sum(probability * digamma(alpha + 1), axis=-1))
    else:
        if from_logits:
            exponential = np.exp(output - np.max(output, axis=-1, keepdims=True))
            probability = exponential / exponential.sum(axis=-1, keepdims=True)
        else:
            probability = output
            if (np.any(probability < 0) or np.any(probability > 1)
                    or not np.allclose(probability.sum(axis=-1), 1., rtol=1e-6, atol=1e-7)):
                raise ValueError("Categorical probability samples must lie in [0,1] and sum to one")
        within_epistemic = np.zeros_like(probability)
        aleatoric = probability * (1 - probability)
        expected_entropy = -np.sum(xlogy(probability, probability), axis=-1)
    result = ensemble_uncertainties(probability, aleatoric, within_epistemic, axis=0)
    mean = result["mean"]
    entropy = -np.sum(xlogy(mean, mean), axis=-1)
    mutual_information = entropy - expected_entropy.mean(axis=0)
    return mean, result["aleatoric"], result["epistemic"], entropy, mutual_information


def reset_weights(model):
    """Reset initialized layer variables using the backend-independent Keras API."""
    for layer in _all_layers(model):
        for name in ("kernel", "bias", "recurrent_kernel", "gamma", "beta", "moving_mean", "moving_variance"):
            variable = getattr(layer, name, None)
            initializer = getattr(layer, f"{name}_initializer", None)
            if variable is not None and initializer is not None:
                variable.assign(initializer(variable.shape, dtype=variable.dtype))
