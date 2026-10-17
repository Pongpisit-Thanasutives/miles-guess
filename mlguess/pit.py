import numpy as np
from scipy.stats import norm, percentileofscore

# path to test /glade/work/schreck/repos/evidential/main/results/evidential/test.csv

def pit_histogram(y_true, y_pred, pred_type="ensemble", bins=10):
    """ 
    Calculate PIT histogram values for different types of ensemble predictions. Predictions
    can take one of three formats:
    * ensemble: array of n deterministic ensemble predictions with shape (n_samples, n_members).
    * gaussian: arary of 1 gaussian distribution prediction (mean and standard deviation) with shape (n_samples, n_params)
    * gaussian_ensemble: array of multiple gaussian distributions for the same prediction with shape (n_samples, n_params, n_members)

    Args:
        y_true: true values with shape (n_samples,)
        y_pred: predictions in format of gaussian, gaussian_ensemble, or ensemble.
        pred_type: 'ensemble', 'gaussian', or 'gaussian_ensemble'
        bins: number of bins or array of specific bin edges. Bins should cover space between 0              and 1.
    
    Returns:
        pit_hist, pit_bins        
    """
    if pred_type == "gaussian":
        pit_quantiles = probability_integral_transform_gaussian(y_true, y_pred)
    elif pred_type == "gaussian_ensemble":
        pit_quantiles = pit_gaussian_ensemble(y_true, y_pred)
    elif pred_type == "ensemble":
        pit_quantiles = probability_integral_transform_ensemble(y_true, y_pred)
    else:
        raise ValueError("pred_type must be ensemble, gaussian, or gaussian_ensemble")
    pit_hist, pit_bins = pit_histogram_values(pit_quantiles, bins=bins) 
    return pit_hist, pit_bins


def pit_deviation(y_true, y_pred, pred_type="ensemble", bins=10):
    """Runs pit_histogram then calculates the pit_deviation. See docstring for pit_histogram.
    
    """
    pit_hist, pit_bins = pit_histogram(y_true, y_pred, pred_type=pred_type, bins=bins)
    n_bins = pit_hist.shape[0]
    if not np.allclose(np.diff(pit_bins), 1 / n_bins) or not np.allclose(pit_bins[[0, -1]], [0, 1]):
        raise ValueError("PIT deviation requires equally spaced bins over [0, 1]")
    n_samples = len(y_true)
    pit_deviation = np.sqrt(1 / n_bins * np.sum((pit_hist / n_samples - 1 / n_bins) ** 2))
    return pit_deviation

def pit_deviation_worst(n_bins):
    """Calculate the worst possible PITD score based on the number of bins. Assumes all the forecasts
    end up in one of the outermost bins.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("At least two bins are required for PIT skill")
    return np.sqrt(1 / n_bins * ((n_bins - 1) * (1 / n_bins) ** 2 + (1 - 1 / n_bins) ** 2))


def pit_deviation_skill_score(y_true, y_pred, pred_type="ensemble", bins=10):
    """Calculate PITD score relative to the worst possible PITD for a given number of bins.
    Ranges from 0 to 1. Explicit bin edges should be equally spaced over [0, 1].
    """
    pitd_score = pit_deviation(y_true, y_pred, pred_type=pred_type, bins=bins)
    n_bins = bins if np.ndim(bins) == 0 else len(bins) - 1
    pitd_worst = pit_deviation_worst(n_bins)
    return 1 - pitd_score / pitd_worst


def probability_integral_transform_ensemble(y_true, y_pred_ens):
    """Calculate the probability integral transform quantiles for an ensemble of predictions

    Args:
        y_true: true values with shape (n_samples,)
        y_pred_ens: predicted ensemble values with shape (n_samples, n_ensemble_members)
    
    Ties retain scipy's average percentage rank convention (``kind="rank"``).
    This finite-ensemble rank diagnostic is not a randomized discrete PIT.

    Returns:
        pit_quantiles: for each sample, the true value's quantile in the predicted distribution.
    """
    y_true, y_pred_ens = _pit_inputs(y_true, y_pred_ens, "ensemble")
    pit_quantiles = np.zeros(y_true.shape)
    for i in range(y_true.shape[0]):
        pit_quantiles[i] = percentileofscore(y_pred_ens[i], y_true[i], kind="rank") / 100.0
    return pit_quantiles


def probability_integral_transform_gaussian(y_true, y_pred_gaussian):
    """Calculate the probability integral transform quantiles for a single Gaussian distribution.

    Args:
        y_true: true values with shape (n_samples,)
        y_pred_gaussian: predicted Gaussian parameters (mean, stand. dev.) with shape (n_samples, n_params)
    
    Returns:
        pit_quantiles: for each sample, the true value's quantile in the predicted distribution.
    """
    y_true, y_pred_gaussian = _pit_inputs(y_true, y_pred_gaussian, "gaussian")
    mean, sigma = y_pred_gaussian.T
    # At zero spread use the right-continuous CDF of a point mass. Unlike
    # continuous predictive distributions, its PIT is not generally uniform.
    pit_quantiles = (y_true >= mean).astype(float)
    positive = sigma > 0
    pit_quantiles[positive] = norm.cdf(y_true[positive], mean[positive], sigma[positive])
    return pit_quantiles



def pit_gaussian_ensemble(y_true, y_pred_gauss_ens):
    """Calculate the probability integral transform quantile for an ensemble of Gaussian parametric models

    Args:
        y_true: true values with shape (n_samples,)
        y_pred_gauss_ens: ensemble of gaussian predictions (mean and standard deviation) with shape (n_samples, n_params, n_members)
 
    Returns:
        pit_quantiles: for each sample, the true value's quantile in the predicted distribution.
    """
    y_true, y_pred_gauss_ens = _pit_inputs(y_true, y_pred_gauss_ens, "gaussian_ensemble")
    pit_quantiles_members = np.zeros((y_true.shape[0], y_pred_gauss_ens.shape[-1]))
    for m in range(y_pred_gauss_ens.shape[-1]):
        pit_quantiles_members[:, m] = probability_integral_transform_gaussian(y_true, y_pred_gauss_ens[:, :, m])
    pit_quantiles = np.mean(pit_quantiles_members, axis=1)
    return pit_quantiles


def pit_histogram_values(pit_quantiles, bins=10):
    # Quantiles have a fixed probability domain. Inferring the range from the
    # observed values can make a narrow, badly calibrated distribution appear
    # uniform. NumPy ignores ``range`` when explicit edges are supplied.
    pit_quantiles = np.asarray(pit_quantiles, dtype=float)
    if pit_quantiles.ndim != 1 or pit_quantiles.size == 0 or not np.all(np.isfinite(pit_quantiles)) or np.any((pit_quantiles < 0) | (pit_quantiles > 1)):
        raise ValueError("PIT quantiles must be a nonempty finite vector in [0, 1]")
    if np.ndim(bins) != 0:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0) or not np.allclose(edges[[0, -1]], [0, 1]):
            raise ValueError("PIT bin edges must increase from 0 to 1")
    pit_hist, pit_bins = np.histogram(pit_quantiles, bins=bins, range=(0.0, 1.0))
    return pit_hist, pit_bins


def _pit_inputs(y_true, prediction, kind):
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    ndim = 3 if kind == "gaussian_ensemble" else 2
    if y_true.ndim != 1 or y_true.size == 0 or prediction.ndim != ndim or prediction.shape[0] != len(y_true) or prediction.shape[-1] == 0:
        raise ValueError("PIT requires nonempty predictions with one row per observation")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(prediction)):
        raise ValueError("PIT observations and predictions must be finite")
    if kind != "ensemble" and (prediction.shape[1] != 2 or np.any(prediction[:, 1] < 0)):
        raise ValueError("Gaussian predictions require mean and nonnegative standard deviation")
    return y_true, prediction
