"""Train and evaluate scalar station models on the splits prepared by nn_data.

The reference's point fits use squared error. Evidential fits use the existing
NIG marginal likelihood and evidence regularizer. In both cases, a two-stage
model can regularize the cosine similarity of the *pre-clamp* corrections.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from mlguess.uncertainty import affine_inverse_scale, nig_uncertainties
from .models import CNN1D, DB, TwoStageMLP, TwoStageCNN1D, TwoStageDB, seed_everything
from .nn_data import source_sum_scaler_params
from .regression_losses import EvidentialRegressionLoss


_ARCHITECTURES = {c.__name__.lower(): c for c in (CNN1D, DB, TwoStageMLP, TwoStageCNN1D, TwoStageDB)}


def build_model(name, model_config, prepared):
    """Instantiate the named architecture using training-fitted dimensions and scales.

    The key/control split is contiguous and has no effect on the backbone
    topology; two-stage models always read their source channels first. By
    default their floor is zero in *raw* units. An explicit `threshold` is in
    standardized source coordinates as in the reference implementation.
    """
    key = name.replace("_", "").lower()
    if key not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}. Expected: {list(_ARCHITECTURES)}")
    if prepared.model_name.replace("_", "").lower() != key:
        raise ValueError("Use the same architecture to prepare the data and build the model.")
    kwargs = dict(model_config)
    n_features = int(prepared.feature_scaler.n_features_in_)
    n_sources = len(prepared.source_names)
    if key.endswith("db"):
        derived = {"n_fts": n_features, "n_fstatic": int(prepared.static_scaler.n_features_in_)}
    elif key == "cnn1d":
        derived = {"in_features": n_features}
    else:
        n_key = n_sources + int(tuple(prepared.ts_cols)[n_sources:n_sources + 1] == ("aod",))
        derived = {"n_fkey": n_key, "n_fcontrol": n_features - n_key}
    if key.startswith("twostage"):
        if not n_sources:
            raise ValueError("A two-stage model requires named source contribution columns.")
        derived.update(n_sources=n_sources, **source_sum_scaler_params(
            prepared.feature_scaler, prepared.target_scaler, n_sources))
        kwargs.setdefault("threshold", None)
    collisions = set(derived) & kwargs.keys()
    if collisions:
        raise ValueError(f"These model options are derived from the training data: {sorted(collisions)}")
    return _ARCHITECTURES[key](**kwargs, **derived)


@dataclass
class TrainingHistory:
    """Losses are weighted by the number of observed station-time targets."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0  # one-based; the final epoch for a fixed-length fit

    @property
    def epochs_run(self):
        return len(self.train_loss)


def cosine_diversity_loss(delta):
    """Absolute pairwise cosine similarity of *columns* (source correction vectors)."""
    d = delta.reshape(-1, delta.shape[-1])
    if d.shape[1] < 2:
        return d.sum() * 0
    d = F.normalize(d, dim=0)
    return _off_diagonal_mean(d.T @ d)


def _off_diagonal_mean(matrix):
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask].abs().mean()


def _batches(split, batch_size, rng=None):
    """Keep each station sequence intact; shuffle groups and stations only."""
    keys = list(split.tensors)
    if rng is not None:
        rng.shuffle(keys)
    for key in keys:
        tensors = split.tensors[key]
        n = tensors[0].shape[0]
        if any(x.shape[0] != n for x in tensors):
            raise ValueError("Mismatched station counts in a length group.")
        order = np.arange(n) if rng is None else rng.permutation(n)
        chunk_size = max(n, 1) if batch_size is None else batch_size
        for start in range(0, n, chunk_size):
            ix = order[start:start + chunk_size]
            yield key, ix, tuple(x[ix] for x in tensors)


def _unpack(model, output):
    if model.two_stage:
        prediction, corrected, delta, _ = output
    else:
        prediction, corrected, delta = output, None, None
    return (tuple(prediction) if model.evidential else (prediction,)), corrected, delta


def _epoch(model, split, batch_size, loss_fn, lambda_sim, optimizer=None, rng=None):
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    total_loss, n_total, gram = 0.0, 0, None
    with torch.set_grad_enabled(training):
        for _, _, tensors in _batches(split, batch_size, rng):
            *inputs, target = (x.to(device) for x in tensors)
            observed = torch.isfinite(target)
            n = int(observed.sum().item())
            if not n:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            params, _, delta = _unpack(model, model(*inputs))
            if any(p.shape != target.shape for p in params):
                raise ValueError("Prediction and target shapes must agree; refusing to broadcast labels.")
            pp = [p[observed] for p in params]
            y = target[observed]
            loss = loss_fn(*pp, y) if model.evidential else F.mse_loss(pp[0], y)
            if model.two_stage and lambda_sim:
                # MLP targets [N,1] and sequences [N,T] both index one K-vector per label.
                d = delta.reshape(-1, delta.shape[-1])[observed.reshape(-1)]
                if training:
                    loss = loss + lambda_sim * cosine_diversity_loss(d)
                elif d.shape[-1] > 1:
                    # Validation cosine similarity is computed once across *all*
                    # observed examples, independent of grouping and batch size.
                    dd = d.detach().to(device="cpu", dtype=torch.float64)
                    block = dd.T @ dd
                    gram = block if gram is None else gram + block
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite model loss on observed targets.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * n
            n_total += n
    if not n_total:
        raise ValueError("The training/validation split contains no observed targets.")
    value = total_loss / n_total
    if gram is not None:
        norms = gram.diag().clamp_min(0).sqrt().clamp_min(1e-12)
        value += lambda_sim * _off_diagonal_mean(gram / (norms[:, None] * norms[None, :])).item()
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite aggregate model loss.")
    return value


def fit_model(model, train, validation=None, *, epochs=25, learning_rate=1e-3, batch_size=16,
              patience=None, lambda_sim=0.05, evidential_coef=0.01, seed=88,
              device="cpu", checkpoint_path=None):
    """Fit with Adam. If supplied, validation selects the weights restored at return.

    A batch contains up to `batch_size` complete equal-length station sequences,
    or rows for an MLP. Use `batch_size=None` for one update per length group,
    exactly as in the attached sequence training routines. No sequence is
    padded or split across updates. To reproduce
    initialization, call `seed_everything(seed)` **before** building the model;
    this function controls subsequent dropout and batch ordering. `checkpoint_path`
    optionally saves the selected model's state_dict; reuse its training scalers.
    """
    for key, value in (("epochs", epochs), ("batch_size", batch_size), ("patience", patience)):
        if key in ("batch_size", "patience") and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if patience is not None and validation is None:
        raise ValueError("patience requires a validation split.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    for key, value in (("lambda_sim", lambda_sim), ("evidential_coef", evidential_coef)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be nonnegative and finite.")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = EvidentialRegressionLoss(coef=evidential_coef)
    history, best_value, best_state = TrainingHistory(), float("inf"), None
    for epoch in range(1, epochs + 1):
        history.train_loss.append(_epoch(model, train, batch_size, loss_fn, lambda_sim, optimizer, rng))
        if validation is None:
            history.best_epoch = epoch
            continue
        value = _epoch(model, validation, batch_size, loss_fn, lambda_sim)
        history.val_loss.append(value)
        if value < best_value:
            best_value, history.best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif patience is not None and epoch - history.best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return history


def predict_frame(model, split, prepared, *, batch_size=128, device=None):
    """Predict every input row, including those without labels; use original units.

    The frame follows `split.frame` order. Every two-stage row includes the
    vector `s_corrected` (in `prepared.source_names` order), a named
    `corrected_<source>`, and `delta_<source>` = corrected minus the input
    source actually used after preprocessing. NIG beta and all three
    variances are in squared target units; nu and alpha are dimensionless.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if device is not None:
        model.to(device)
    device = next(model.parameters()).device
    frame = split.frame.copy()
    n = len(frame)
    params = np.empty((n, 4 if model.evidential else 1), dtype=float)
    seen = np.zeros(n, dtype=bool)
    if model.two_stage:
        names = tuple(prepared.source_names)
        stats = source_sum_scaler_params(prepared.feature_scaler, prepared.target_scaler, len(names))
        for key, value in stats.items():
            saved = getattr(model, key).detach().cpu().numpy().ravel()
            expected = np.asarray(value).reshape(-1)
            if saved.shape != expected.shape or not np.allclose(saved, expected, rtol=1e-6, atol=1e-7):
                raise ValueError("Source/target scales differ from the fitted model. "
                                 "Use the preprocessor fitted on its training split.")
        corrected, original = (np.empty((n, len(names))) for _ in range(2))
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for key, ix, tensors in _batches(split, batch_size):
                positions = np.asarray(split.groups[key]["row_positions"])[ix].reshape(-1)
                if (not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0)
                        or np.any(positions >= n) or len(np.unique(positions)) != len(positions)
                        or np.any(seen[positions])):
                    raise ValueError("Invalid or duplicate prediction row positions.")
                seen[positions] = True
                inputs = [x.to(device) for x in tensors[:-1]]
                parts, sc, _ = _unpack(model, model(*inputs))
                params[positions] = np.column_stack([p.cpu().numpy().reshape(-1) for p in parts])
                if model.two_stage:
                    k = len(names)
                    corrected[positions] = sc.cpu().numpy().reshape(-1, k)
                    original[positions] = inputs[0][..., :k].cpu().numpy().reshape(-1, k)
    finally:
        model.train(was_training)
    if not n or not seen.all() or not np.isfinite(params).all():
        raise ValueError("Empty split, incomplete row coverage, or nonfinite predictions.")
    target_scaler = prepared.target_scaler
    frame["pred"] = target_scaler.inverse_transform(params[:, :1]).ravel()
    if model.evidential:
        frame["nu"], frame["alpha"] = params[:, 1], params[:, 2]
        frame["beta"] = params[:, 3] * affine_inverse_scale(target_scaler, 1)[0] ** 2
        _, a, e, total = nig_uncertainties(frame["pred"], frame["nu"], frame["alpha"], frame["beta"])
        frame["aleatoric_variance"], frame["epistemic_variance"], frame["total_variance"] = a, e, total
    if model.two_stage:
        source_scaler = prepared.feature_scaler
        scale, center = source_scaler.scale_[:len(names)], source_scaler.mean_[:len(names)]
        delta = (corrected - original) * scale
        corrected = corrected * scale + center
        if not np.isfinite(corrected).all() or not np.isfinite(delta).all():
            raise ValueError("Nonfinite source contributions.")
        frame["s_corrected"] = list(corrected)
        for k, name in enumerate(names):
            frame[f"corrected_{name}"] = corrected[:, k]
            frame[f"delta_{name}"] = delta[:, k]
    return frame


def regression_metrics(frame, target_col="pm25"):
    """Scalar point metrics in physical units, scored only on observed labels."""
    y = frame[target_col].to_numpy(dtype=float)
    pred = frame["pred"].to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions cannot be scored.")
    valid = np.isfinite(y)
    if not valid.any():
        raise ValueError("No observed test targets are available for scoring.")
    y, residual = y[valid], pred[valid] - y[valid]
    sse, sst = np.sum(residual**2), np.sum((y - np.mean(y))**2)
    return {"n_observed": int(valid.sum()), "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))), "r2": float(1 - sse/sst) if sst > 0 else np.nan}
