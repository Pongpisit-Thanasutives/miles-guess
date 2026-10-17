"""Predictive moments and the law of total variance for finite mixtures.

All uncertainty values are variances, not standard deviations. Ensemble
members must refer to the same inputs, targets/classes, and physical units.
These functions operate on NumPy arrays and require no neural-network backend.
"""
import numpy as np


def _finite_array(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def ensemble_uncertainties(means, aleatoric, epistemic=None, axis=0, weights=None):
    """Combine member moments without averaging their distribution parameters.

    ``means``, ``aleatoric`` and optional ``epistemic`` have identical shapes,
    conventionally (members, samples, targets). The member axis is removed.
    ``aleatoric`` is mandatory: point predictions do not identify observation
    noise. Pass explicit zeros only when a point-mass mixture is intended.
    Omitted within-member epistemic variance is zero (e.g. Gaussian members).

    The result uses finite-mixture population variance (denominator M for
    equal weights), including M=1. Optional weights are nonnegative member
    probabilities; positive sums are normalized. This is not an unbiased
    sample estimator of an unknown population of models.
    """
    means = _finite_array(means, "means")
    if means.ndim < 1 or not isinstance(axis, (int, np.integer)):
        raise ValueError("means needs a member axis and axis must be an integer")
    if not -means.ndim <= axis < means.ndim:
        raise ValueError("axis is outside the array dimensions")
    axis %= means.ndim
    count = means.shape[axis]
    if count == 0:
        raise ValueError("at least one member is required")
    if aleatoric is None:
        raise ValueError("aleatoric variance is unknown; supply explicit member variances")
    aleatoric = _finite_array(aleatoric, "aleatoric")
    epistemic = np.zeros_like(means) if epistemic is None else _finite_array(epistemic, "epistemic")
    for name, variance in (("aleatoric", aleatoric), ("epistemic", epistemic)):
        if variance.shape != means.shape or np.any(variance < 0):
            raise ValueError(f"{name} must match means.shape and be nonnegative")
    if weights is None:
        weights = np.full(count, 1.0 / count)
    else:
        weights = _finite_array(weights, "weights")
        if weights.shape != (count,) or np.any(weights < 0) or weights.max(initial=0) <= 0:
            raise ValueError("weights must be a nonnegative vector with positive sum, one per member")
        # Normalize in two steps to avoid overflow for very large weights.
        weights = weights / weights.max()
        weights = weights / weights.sum()
    weight_shape = [1] * means.ndim
    weight_shape[axis] = count
    weights = weights.reshape(weight_shape)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.sum(weights * means, axis=axis)
        between = np.sum(weights * (means - np.expand_dims(mean, axis)) ** 2, axis=axis)
        within = np.sum(weights * epistemic, axis=axis)
        aleatoric = np.sum(weights * aleatoric, axis=axis)
        epistemic = within + between
        result = dict(mean=mean, aleatoric=aleatoric, epistemic=epistemic,
                      total=aleatoric + epistemic, within_epistemic=within,
                      between_epistemic=between)
    for name, value in result.items():
        _finite_array(value, name)
    return result


def nig_uncertainties(mu, nu, alpha, beta):
    """Return (mean, aleatoric, epistemic, total) for a valid NIG prediction.

    Requires nu > 0, alpha > 1 and beta > 0. The predictive distribution is
    Student-t with df=2*alpha and scale²=beta*(1+nu)/(nu*alpha).
    """
    mu, nu, alpha, beta = np.broadcast_arrays(*[
        _finite_array(value, name) for value, name in
        ((mu, "mu"), (nu, "nu"), (alpha, "alpha"), (beta, "beta"))
    ])
    if np.any(nu <= 0) or np.any(alpha <= 1) or np.any(beta <= 0):
        raise ValueError("NIG finite-variance moments require nu>0, alpha>1 and beta>0")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        aleatoric = beta / (alpha - 1)
        epistemic = aleatoric / nu
        total = aleatoric + epistemic
    for name, value in (("aleatoric", aleatoric), ("epistemic", epistemic), ("total", total)):
        _finite_array(value, name)
    return mu, aleatoric, epistemic, total


def dirichlet_uncertainties(alpha):
    """Return predictive class means and marginal indicator variances.

    The last axis is classes. These are marginal variances, not a full
    class-covariance matrix. Dirichlet concentration must be strictly positive.
    """
    alpha = _finite_array(alpha, "alpha")
    if alpha.ndim < 1 or alpha.shape[-1] < 2 or np.any(alpha <= 0):
        raise ValueError("alpha must contain positive concentrations for at least two classes")
    with np.errstate(over="ignore"):
        strength = alpha.sum(axis=-1, keepdims=True)
    _finite_array(strength, "Dirichlet strength")
    probability = alpha / strength
    total = probability * (1 - probability)
    epistemic = total / (strength + 1)
    aleatoric = total * (strength / (strength + 1))
    return probability, aleatoric, epistemic, total


def affine_inverse_scale(y_scaler, n_targets):
    """Signed slopes of an explicitly supported fitted affine inverse scaler.

    Nonlinear/custom transformations are rejected: a local slope or training
    variance is not generally an exact transformation of predictive variance.
    """
    if not isinstance(n_targets, (int, np.integer)) or n_targets < 1:
        raise ValueError("n_targets must be positive")
    if y_scaler is None or y_scaler is False:
        return np.ones(n_targets)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    if isinstance(y_scaler, (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler)):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(y_scaler)
        if y_scaler.n_features_in_ != n_targets:
            raise ValueError("scaler must have one fitted feature per target")
    if isinstance(y_scaler, MinMaxScaler):
        scale = np.reciprocal(y_scaler.scale_)
    elif isinstance(y_scaler, (StandardScaler, RobustScaler, MaxAbsScaler)):
        scale = np.ones(n_targets) if y_scaler.scale_ is None else y_scaler.scale_
    else:
        raise TypeError("uncertainty scaling requires a fitted StandardScaler, MinMaxScaler, RobustScaler or MaxAbsScaler")
    scale = _finite_array(scale, "inverse scale")
    if scale.shape != (n_targets,) or np.any(scale == 0):
        raise ValueError("scaler must have one nonzero inverse slope per target")
    return scale


def inverse_scale_moments(mean, aleatoric, epistemic, y_scaler=None):
    """Inverse-transform means once and multiply variances by inverse slope².

    Inputs have shape (..., targets); all leading dimensions are preserved.
    Use shape (samples, 1) for a single target.
    """
    mean = _finite_array(mean, "mean")
    aleatoric = _finite_array(aleatoric, "aleatoric")
    epistemic = _finite_array(epistemic, "epistemic")
    if mean.ndim < 1 or aleatoric.shape != mean.shape or epistemic.shape != mean.shape:
        raise ValueError("moments must have matching shapes (..., targets)")
    if np.any(aleatoric < 0) or np.any(epistemic < 0):
        raise ValueError("variances must be nonnegative")
    slope = affine_inverse_scale(y_scaler, mean.shape[-1])
    if y_scaler is not None and y_scaler is not False:
        mean = _finite_array(y_scaler.inverse_transform(mean.reshape(-1, mean.shape[-1])), "inverse mean").reshape(mean.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        aleatoric = aleatoric * slope**2
        epistemic = epistemic * slope**2
    _finite_array(aleatoric, "inverse aleatoric")
    _finite_array(epistemic, "inverse epistemic")
    return mean, aleatoric, epistemic
