"""Regression scores and Gaussian spread/skill summaries."""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import properscoring as ps
from mlguess.pit import pit_deviation_skill_score


def _regression_arrays(y_true, y_pred, total=None):
    arrays = [np.asarray(x, dtype=float) for x in (y_true, y_pred)]
    if total is not None:
        arrays.append(np.asarray(total, dtype=float))
    arrays = [x[:, None] if x.ndim == 1 else x for x in arrays]
    if any(x.ndim != 2 or x.shape != arrays[0].shape or x.size == 0 for x in arrays):
        raise ValueError("Targets, predictions and variances must have matching (samples, targets) shapes")
    if any(not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("Targets, predictions and variances must be finite")
    if total is not None and np.any(arrays[2] < 0):
        raise ValueError("Predictive variance must be nonnegative")
    return arrays


def _gaussian_crps(y_true, mu, sigma):
    """Gaussian CRPS including its exact zero-spread (absolute error) limit."""
    y_true, mu, sigma = np.broadcast_arrays(y_true, mu, sigma)
    result = np.abs(y_true - mu).astype(float)
    positive = sigma > 0
    result[positive] = ps.crps_gaussian(y_true[positive], mu=mu[positive], sig=sigma[positive])
    return result


def regression_metrics(y_true, y_pred, total=None, split="val"):
    """Compute scores for 1-D or multioutput targets; ``total`` is variance.

    Undefined R-squared scores (fewer than two samples or occupied spread bins)
    are reported as NaN, rather than a synthetic numerical penalty.
    """
    arrays = _regression_arrays(y_true, y_pred, total)
    y_true, y_pred = arrays[:2]
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        f"{split}_mse": mse,
        f"{split}_rmse": np.sqrt(mse),
        f"{split}_mae": mean_absolute_error(y_true, y_pred),
        f"{split}_r2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        f"{split}_mape": np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100,
    }
    if total is not None:
        total = arrays[2]
        sigma = np.sqrt(total)
        pitd = []
        for col in range(y_true.shape[1]):
            score = pit_deviation_skill_score(
                y_true[:, col], np.stack([y_pred[:, col], sigma[:, col]], -1),
                pred_type="gaussian", bins=10,
            )
            metrics[f"{split}_pitd_{col}"] = score
            pitd.append(score)
        metrics[f"{split}_pitd"] = np.mean(pitd)
        metrics[f"{split}_crps"] = _gaussian_crps(y_true, y_pred, sigma).mean()
        spread_scores = rmse_crps_skill_scores(y_true, y_pred, total)
        metrics[f"{split}_r2_rmse_sigma"] = spread_scores["r2_rmse"]
        metrics[f"{split}_r2_crps_sigma"] = spread_scores["r2_crps"]
    return metrics


def rmse_crps_skill_scores(y, mu, total, filter_top_percentile=0):
    """Average each output's R-squared between binned spread and RMSE/CRPS."""
    y, mu, total = _regression_arrays(y, mu, total)
    scores = {"r2_rmse": [], "r2_crps": []}
    for col in range(y.shape[1]):
        result = calculate_skill_score(y[:, col], mu[:, col], np.sqrt(total[:, col]),
                                      num_bins=100, log=True,
                                      filter_top_percentile=filter_top_percentile)
        for key, metric in (("r2_rmse", "rmse"), ("r2_crps", "crps")):
            scores[key].append(r2_score(result["bin"], result[metric]) if len(result) > 1 else np.nan)
    return {key: np.mean(value) for key, value in scores.items()}


def calculate_skill_score(y_true, y_pred, sigma, num_bins=10, log=False, filter_top_percentile=0):
    """Bin one target's Gaussian standard deviations and summarize RMSE/CRPS.

    Paired nonfinite samples are omitted. A zero standard deviation retains
    its exact deterministic score; a small positive value is used only for
    logarithmic bin assignment. Multioutput data must be handled per target.
    """
    arrays = [np.asarray(x, dtype=float) for x in (y_true, y_pred, sigma)]
    arrays = [x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x for x in arrays]
    if any(x.ndim != 1 or x.shape != arrays[0].shape for x in arrays):
        raise ValueError("Provide matching one-dimensional targets, predictions and standard deviations")
    if not isinstance(num_bins, (int, np.integer)) or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")
    if not 0 <= filter_top_percentile < 100:
        raise ValueError("filter_top_percentile must be in [0, 100)")
    df = pd.DataFrame(dict(zip(("y_true", "y_pred", "sigma"), arrays)))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if np.any(df["sigma"] < 0):
        raise ValueError("Standard deviations must be nonnegative")
    if df.empty:
        raise ValueError("At least one finite paired sample is required")
    spread = np.log(np.maximum(df["sigma"], 1e-10)) if log else df["sigma"]
    df["bin"] = pd.cut(spread, bins=num_bins)
    threshold = np.percentile(df["sigma"], 100 - filter_top_percentile)
    df = df[df["sigma"] <= threshold]
    records = []
    for bin_name, group in df.groupby("bin", observed=True):
        records.append({
            "bin": np.exp(bin_name.left) if log else bin_name.left,
            "rmse": np.sqrt(mean_squared_error(group["y_true"], group["y_pred"])),
            "crps": _gaussian_crps(group["y_true"], group["y_pred"], group["sigma"]).mean(),
            "count": len(group),
        })
    return pd.DataFrame.from_records(records, columns=["bin", "rmse", "crps", "count"])
