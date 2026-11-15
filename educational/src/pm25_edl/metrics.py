"""Scores in physical PM2.5 units. Evaluation uses finite *observed* pm25 labels.

Training scores use pm25_filled only through score_training_predictions.

gamma is the predictive mean, nu the NIG mean precision multiplier, alpha the
inverse-gamma shape and beta its scale. The predictive Student-t scale below
is not the predictive standard deviation. This distinction matters for PIT,
NLL and central prediction intervals.
"""
import numpy as np
import pandas as pd
from scipy.stats import t


def _observed(frame):
    if 'pm25' not in frame:
        raise ValueError("Scoring requires observed 'pm25'; 'pm25_filled' is training-only.")
    y = frame['pm25'].to_numpy(dtype=float)
    mask = np.isfinite(y)
    if not mask.any():
        raise ValueError("No finite observed PM2.5 targets to score.")
    return frame.loc[mask], y[mask]


def _distribution(frame):
    """Return (location, degrees of freedom, scale) in original PM2.5 units."""
    gamma, nu, a, b = (frame[key].to_numpy(dtype=float)
                       for key in ('gamma', 'nu', 'alpha', 'beta'))
    if not (all(np.isfinite(v).all() for v in (gamma, nu, a, b))
            and (nu > 0).all() and (a > 1).all() and (b > 0).all()):
        raise ValueError("Expected finite NIG parameters with nu,beta>0 and alpha>1.")
    return gamma, 2 * a, np.sqrt((b / a) * (1 + 1 / nu))


def _coverage(level):
    if not 0 < level < 1:
        raise ValueError("interval_level must be strictly between 0 and 1.")
    return 1 - level


def predictive_interval(frame, *, interval_level=0.90):
    """Central Student-t prediction interval for every row, labeled or not."""
    tail = _coverage(interval_level)
    loc, df, scale = _distribution(frame)
    half_width = t.isf(tail / 2, df) * scale
    return loc - half_width, loc + half_width


def predictive_pit(frame):
    """Total predictive PIT at each finite observed target, in input order."""
    obs, y = _observed(frame)
    loc, df, scale = _distribution(obs)
    return t.cdf(y, df=df, loc=loc, scale=scale)


def score_predictions(frame, *, interval_level=0.90):
    """R² (undefined: NaN), RMSE, MAE, predictive NLL, and mean Winkler score.

    The latter is an interval score: width plus 2/(1-level) times the distance
    outside the interval. Lower is better for every reported loss/score except R².
    NLL is a log *density* in the reported physical units; it can be negative.
    """
    obs, y = _observed(frame)
    loc, df, scale = _distribution(obs)
    residual = y - loc
    sse = np.dot(residual, residual)
    sst = np.sum((y - y.mean()) ** 2)
    tail = _coverage(interval_level)
    half_width = t.isf(tail / 2, df) * scale
    winkler = 2 * half_width + (2 / tail) * np.maximum(abs(residual) - half_width, 0)
    return dict(n_obs=len(y), r2=float(1 - sse / sst) if sst > 0 else float('nan'),
                rmse=float(np.sqrt(sse / len(y))), mae=float(np.mean(abs(residual))),
                nll=float(-t.logpdf(y, df=df, loc=loc, scale=scale).mean()),
                mean_winkler=float(winkler.mean()))


def score_training_predictions(frame, *, interval_level=0.90):
    """Score finite pm25_filled targets, which may include training imputations.

    Uses the same physical-unit scores as score_predictions without changing
    its observed-only evaluation contract or modifying the input frame.
    Here n_obs counts finite training targets, not exclusively observations.
    """
    if 'pm25_filled' not in frame:
        raise ValueError("Training scoring requires 'pm25_filled'.")
    if not np.isfinite(frame['pm25_filled'].to_numpy(dtype=float)).any():
        raise ValueError("No finite training PM2.5 targets to score.")
    training = frame.drop(columns=['pm25'], errors='ignore').rename(columns={'pm25_filled': 'pm25'})
    return score_predictions(training, interval_level=interval_level)


def discard_curve(frame, *, fractions=None, uncertainty='total_std'):
    """RMSE after removing the most uncertain observed rows.

    If a cutoff splits equally uncertain rows, use their group's average squared
    error (expected retained MSE over random tie breaks). This avoids dependence
    on station/file order. Report the attainable fraction, retaining ≥1 row.
    """
    obs, y = _observed(frame)
    u = obs[uncertainty].to_numpy(dtype=float)
    pred = obs['gamma'].to_numpy(dtype=float)
    if not (np.isfinite(u).all() and np.isfinite(pred).all() and (u >= 0).all()):
        raise ValueError("Discard curves require finite predictions and nonnegative finite uncertainty.")
    fractions = np.linspace(0, .9, 10) if fractions is None else np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or not np.all((fractions >= 0) & (fractions < 1)):
        raise ValueError("fractions must be a vector of fractions in [0, 1).")
    _, group, counts = np.unique(u, return_inverse=True, return_counts=True)
    error_sums = np.bincount(group, weights=(y - pred) ** 2)
    ends = np.cumsum(counts)
    prefix = np.r_[0., np.cumsum(error_sums)]
    keep = len(y) - np.floor(fractions * len(y) + 1e-12).astype(int)
    keep = np.clip(keep, 1, len(y))
    k = np.searchsorted(ends, keep)
    take = keep - (ends[k] - counts[k])
    mse = (prefix[k] + take / counts[k] * error_sums[k]) / keep
    return pd.DataFrame({'discard_fraction': 1 - keep / len(y), 'n_retained': keep,
                         'rmse': np.sqrt(mse)})


def _self_test():
    # nu=1, alpha=2, beta=1 -> Student-t(df=4, scale=1).
    f = pd.DataFrame(dict(pm25=[0., 2., np.nan], gamma=0., nu=1., alpha=2.,
                          beta=1., total_std=np.sqrt(2.), pm25_filled=999.))
    m = score_predictions(f)
    assert m['n_obs'] == 2 and m['mae'] == 1 and m['r2'] == -1
    np.testing.assert_allclose(m['rmse'], np.sqrt(2.))
    np.testing.assert_allclose(m['nll'], -t.logpdf([0., 2.], df=4).mean())
    np.testing.assert_allclose(predictive_pit(f), [0.5, t.cdf(2., 4)])
    lo, hi = predictive_interval(f)
    np.testing.assert_allclose(hi, t.ppf(.95, 4))
    np.testing.assert_allclose(m['mean_winkler'], (hi - lo).mean())
    np.testing.assert_allclose(discard_curve(f, fractions=[0, .5]).rmse, np.sqrt(2.))
    training = score_training_predictions(f)
    assert training['n_obs'] == 3 and training['mae'] == 999.
    assert score_predictions(f) == m  # Training scoring does not mutate the observations.


if __name__ == '__main__':
    _self_test()
    print("Student-t scores, observation mask, interval and discard checks passed.")
