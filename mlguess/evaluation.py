"""Held-out evaluation of NIG regression and categorical predictions.

These functions do not fit models, choose hyperparameters, or calibrate outputs.
Call them after training and validation decisions have been frozen. Regression
intervals use the actual NIG predictive Student-t distribution.
"""
import numpy as np
import pandas as pd
from scipy.stats import t

from mlguess.uncertainty import inverse_scale_moments, nig_uncertainties


def evaluate_nig(y_true, parameters, y_scaler=None, interval_levels=(.5, .8, .9, .95)):
    """Evaluate one NIG model, in physical target units.

    ``parameters`` is a four-tuple (mu, nu, alpha, beta) of NumPy arrays with
    shape (samples, targets), in training-target units. ``y_true`` is already
    in physical units. A fitted affine ``y_scaler`` transforms the predictions.
    An ensemble is a mixture, not a single Student-t: do not pass averaged
    NIG parameters to this function.

    Returns per-target errors and Student-t NLL, PIT values, central predictive
    intervals, empirical coverage, and the variance decomposition. NLL is the
    marginal negative log density in the supplied target units. Multiple
    targets are evaluated separately; no cross-target covariance is modeled.
    R² is NaN when a test target is constant.
    """
    if not isinstance(parameters, (tuple, list)) or len(parameters) != 4:
        raise ValueError("parameters must be (mu, nu, alpha, beta)")
    arrays = [np.asarray(value, dtype=float) for value in parameters]
    arrays = [value[:, None] if value.ndim == 1 else value for value in arrays]
    y = np.asarray(y_true, dtype=float)
    y = y[:, None] if y.ndim == 1 else y
    if y.ndim != 2 or min(y.shape) < 1 or not np.all(np.isfinite(y)):
        raise ValueError("y_true must be finite with shape (samples, targets)")
    if any(value.shape != y.shape for value in arrays):
        raise ValueError("all parameter arrays must match y_true.shape")
    levels = np.asarray(interval_levels, dtype=float)
    if (levels.ndim != 1 or levels.size == 0 or not np.all(np.isfinite(levels))
            or np.any((levels <= 0) | (levels >= 1)) or np.unique(levels).size != levels.size):
        raise ValueError("interval_levels must be distinct values strictly between zero and one")
    mu, nu, alpha, beta = arrays
    mu, aleatoric, epistemic, _ = nig_uncertainties(mu, nu, alpha, beta)
    mu, aleatoric, epistemic = inverse_scale_moments(mu, aleatoric, epistemic, y_scaler)
    with np.errstate(over="ignore", invalid="ignore"):
        # Scale² = predictive variance * (df-2)/df; this avoids beta*nu overflow.
        total = aleatoric + epistemic
        df = 2 * alpha
        scale = np.sqrt(total * ((alpha - 1) / alpha))
    if (not np.all(np.isfinite(df)) or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)):
        raise ValueError("predictive Student-t parameters must be finite and nondegenerate")
    distribution = t(df=df, loc=mu, scale=scale)
    log_density = distribution.logpdf(y)
    if not np.all(np.isfinite(log_density)):
        raise ValueError("predictive log densities are not finite")
    residual = y - mu
    ss_residual = np.sum(residual**2, axis=0)
    ss_target = np.sum((y-y.mean(axis=0))**2, axis=0)
    r2 = np.full(y.shape[1], np.nan)
    np.divide(ss_residual, ss_target, out=r2, where=ss_target > 0)
    r2 = 1 - r2
    metrics = pd.DataFrame(dict(mae=np.mean(np.abs(residual), axis=0),
                                rmse=np.sqrt(np.mean(residual**2, axis=0)),
                                r2=r2, nll=-log_density.mean(axis=0)))
    metrics.index.name = "target"
    intervals, records = {}, []
    for level in levels:
        tail = (1-float(level))/2
        lower, upper = distribution.ppf(tail), distribution.isf(tail)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Predictive interval exceeds finite numeric range")
        intervals[float(level)] = (lower, upper)
        coverage = np.mean((y >= lower) & (y <= upper), axis=0)
        width = np.mean(upper-lower, axis=0)
        for target in range(y.shape[1]):
            records.append(dict(target=target, nominal=float(level),
                                observed=float(coverage[target]), mean_width=float(width[target])))
    return dict(mean=mu, aleatoric=aleatoric, epistemic=epistemic, total=total,
                df=df, predictive_scale=scale, pit=distribution.cdf(y),
                intervals=intervals, metrics=metrics, coverage=pd.DataFrame.from_records(records))


def evaluate_classifier(y_true, probabilities, n_bins=10):
    """Evaluate fixed categorical probabilities on held-out integer labels.

    Labels must be integers 0..K-1, aligned with the probability columns.
    Brier score is the mean of the SUM of squared class errors (range 0..2).
    ECE uses equal-width top-label confidence bins, with zero in the first
    bin and one in the last. Empty reliability bins have NaN means. ECE depends
    on binning and sample size; the full reliability table is also returned.
    A zero probability assigned to a true class gives infinite log loss.
    """
    p = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y_true)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if (p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 2 or labels.shape != (len(p),)
            or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-7)):
        raise ValueError("probabilities must be finite normalized (samples, classes>=2) with matching labels")
    if (not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)
            or np.any(labels >= p.shape[1])):
        raise ValueError("labels must be integers from zero to the number of classes minus one")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    predicted = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = predicted == labels
    bin_index = np.minimum((confidence*n_bins).astype(int), n_bins-1)
    rows, ece = [], 0.
    for index in range(n_bins):
        selected = bin_index == index
        count = int(selected.sum())
        avg_conf = float(confidence[selected].mean()) if count else np.nan
        avg_acc = float(correct[selected].mean()) if count else np.nan
        if count:
            ece += count/len(p)*abs(avg_acc-avg_conf)
        rows.append(dict(lower=index/n_bins, upper=(index+1)/n_bins, count=count,
                         confidence=avg_conf, accuracy=avg_acc))
    one_hot = np.eye(p.shape[1])[labels]
    with np.errstate(divide="ignore"):
        log_loss = -np.log(p[np.arange(len(p)), labels]).mean()
    confusion = np.zeros((p.shape[1], p.shape[1]), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    return dict(metrics=dict(accuracy=float(correct.mean()), log_loss=float(log_loss),
                             brier=float(np.mean(np.sum((p-one_hot)**2, axis=1))), ece=float(ece)),
                reliability=pd.DataFrame.from_records(rows), confusion=confusion)
