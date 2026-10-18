"""Clock-time summaries and Figure 11 style plots for regression uncertainty.

These are summaries of an already frozen test set. They neither select a model
nor change its uncertainty estimates. All inputs must be in physical target
units, with variances in squared target units.
"""

from numbers import Integral, Number
import warnings

import numpy as np
import pandas as pd


def _clock_times(timestamps, n_samples):
    """Parse a single common clock without silently interpreting row indices."""
    raw = np.asarray(timestamps)
    if raw.ndim != 1 or raw.shape[0] != n_samples:
        raise ValueError("timestamps must be one-dimensional and aligned with the samples")
    if raw.dtype.kind in "biufc" or any(
        isinstance(value, (Number, bool, np.bool_)) for value in raw
    ):
        raise ValueError("timestamps must be datetimes, not numeric row indices")
    try:
        # A single named timezone is preserved, including daylight-saving
        # transitions. Mixed offsets/awareness must first be converted by the
        # caller to an explicitly chosen common timezone.
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            clock = pd.DatetimeIndex(
                pd.to_datetime(timestamps, errors="raise", format="mixed")
            )
    except (TypeError, ValueError, FutureWarning, OverflowError) as exc:
        raise ValueError(
            "timestamps must be valid datetimes in one common timezone; "
            "convert mixed timezones or mixed naive/aware values explicitly first"
        ) from exc
    if clock.hasnans:
        raise ValueError("timestamps must not contain missing values")
    return clock


def _target_names(target_names, n_targets):
    if target_names is None:
        return [f"target_{index}" for index in range(n_targets)]
    if isinstance(target_names, str):
        raise ValueError("target_names must contain one distinct name per target")
    try:
        names = list(target_names)
    except TypeError as exc:
        raise ValueError("target_names must contain one distinct name per target") from exc
    if (
        len(names) != n_targets
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != n_targets
    ):
        raise ValueError("target_names must contain one distinct nonempty name per target")
    return names


def diurnal_uncertainty_summary(
    timestamps,
    observed,
    predicted,
    aleatoric_variance,
    epistemic_variance,
    *,
    target_names=None,
    bin_minutes=10,
):
    """Summarize test observations by actual time of day, in source clock time.

    Arrays must have matching shape ``(samples,)`` or ``(samples, targets)``.
    Each target is summarized independently. Within each clock-time bin, truth
    and predictions are arithmetic means. Uncertainty curves are
    ``sqrt(mean(variance))``: the RMS conditional standard deviation used in
    the authors' Figure 11 plotting code. They are neither mean standard
    deviations, standard errors of the bin mean, nor the total standard
    deviation of a mixture over observations. In particular, variation of
    predicted means within a bin is not added to epistemic uncertainty.

    ``bin_minutes`` is a positive integer divisor of 1440. Bins are left-closed
    and timestamps are floored to the bin start. Every bin is returned for
    every target, ordered by target and minute of day; an unobserved bin has
    count zero and NaN summary values. There is no interpolation or smoothing.
    All observations receive equal weight, including repeated timestamps.

    Naive datetimes retain the source clock; no timezone is inferred. Aware
    datetimes retain their common timezone. Convert mixed timezones or mixed
    naive/aware inputs to a deliberately chosen common timezone beforehand.
    Local times repeated at a daylight-saving transition share a clock bin.
    Numeric row indices and missing/invalid timestamps are rejected.

    Returns a long pandas DataFrame with ``target``, ``minute_of_day``,
    ``count``, ``observed_mean``, ``predicted_mean``, ``aleatoric_sd``, and
    ``epistemic_sd``. ``count`` is the number of test observations in the bin,
    not the number of independent days. The DataFrame attrs record the clock
    and aggregation conventions.
    """
    if (
        isinstance(bin_minutes, (bool, np.bool_))
        or not isinstance(bin_minutes, Integral)
        or bin_minutes <= 0
        or 1440 % bin_minutes != 0
    ):
        raise ValueError("bin_minutes must be a positive integer divisor of 1440")

    arrays = []
    for name, value in (
        ("observed", observed),
        ("predicted", predicted),
        ("aleatoric_variance", aleatoric_variance),
        ("epistemic_variance", epistemic_variance),
    ):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite numeric array") from exc
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape (samples, targets)")
        arrays.append(array)
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("all observation, prediction, and variance shapes must match")
    if any((array < 0).any() for array in arrays[2:]):
        raise ValueError("aleatoric and epistemic variances must be nonnegative")

    n_samples, n_targets = arrays[0].shape
    names = _target_names(target_names, n_targets)
    clock = _clock_times(timestamps, n_samples)
    minutes = (clock.hour * 60 + clock.minute) // bin_minutes * bin_minutes
    all_bins = pd.Index(np.arange(0, 1440, bin_minutes), name="minute_of_day")
    frames = []
    for index, name in enumerate(names):
        values = pd.DataFrame(
            dict(
                minute_of_day=minutes,
                observed_mean=arrays[0][:, index],
                predicted_mean=arrays[1][:, index],
                aleatoric_variance=arrays[2][:, index],
                epistemic_variance=arrays[3][:, index],
            )
        )
        grouped = values.groupby("minute_of_day", sort=True)
        summary = grouped.mean().reindex(all_bins)
        summary["count"] = grouped.size().reindex(all_bins, fill_value=0).astype(int)
        summary["aleatoric_sd"] = np.sqrt(summary.pop("aleatoric_variance"))
        summary["epistemic_sd"] = np.sqrt(summary.pop("epistemic_variance"))
        summary.insert(0, "target", name)
        frames.append(summary.reset_index())

    result = pd.concat(frames, ignore_index=True)[
        ["target", "minute_of_day", "count", "observed_mean", "predicted_mean",
         "aleatoric_sd", "epistemic_sd"]
    ]
    result.attrs.update(
        bin_minutes=int(bin_minutes),
        clock_timezone=str(clock.tz) if clock.tz is not None else None,
        clock_convention="source clock" if clock.tz is None else "supplied timezone",
        uncertainty_aggregation="sqrt(mean(per-observation variance))",
        weighting="equal weight per observation",
    )
    return result


def _plot_text(values, names, argument, *, broadcast=False):
    if values is None:
        return names
    if isinstance(values, str):
        if broadcast:
            return [values] * len(names)
        raise ValueError(f"{argument} must contain one string per target")
    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError(f"{argument} must contain one string per target") from exc
    if len(result) != len(names) or any(not isinstance(value, str) for value in result):
        raise ValueError(f"{argument} must contain one string per target")
    return result


def plot_diurnal_uncertainty(
    timestamps,
    observed,
    predicted,
    aleatoric_variance,
    epistemic_variance,
    *,
    target_names=None,
    target_labels=None,
    units=None,
    bin_minutes=10,
):
    """Plot test-set diurnal means and RMS uncertainty in Figure 11's style.

    See :func:`diurnal_uncertainty_summary` for aggregation and time semantics.
    This mirrors the paper's four curves: blue triangles for truth, orange
    squares for predictions, green circles for aleatoric SD, and red pentagons
    for epistemic SD. All supplied targets get their own panel, wrapping after
    three columns. No absent targets, observations, or uncertainty components
    are manufactured. Empty bins break the lines. No smoothing, clipping, or
    fixed scientific y-axis limits are applied.

    ``target_labels`` supplies one display label per target; ``units`` may be a
    shared string or one string per target. Truth and predictions must already
    use those units, and input variances their squares. Returns
    ``(figure, axes, summary)`` with a one-dimensional array of active axes and
    the complete summary table including per-bin counts.
    """
    import matplotlib.pyplot as plt

    summary = diurnal_uncertainty_summary(
        timestamps, observed, predicted, aleatoric_variance, epistemic_variance,
        target_names=target_names, bin_minutes=bin_minutes,
    )
    names = summary["target"].drop_duplicates().tolist()
    labels = _plot_text(target_labels, names, "target_labels")
    unit_labels = _plot_text(units, [""] * len(names), "units", broadcast=True)
    n_cols = min(3, len(names))
    n_rows = (len(names) + n_cols - 1) // n_cols
    fig, grid = plt.subplots(
        n_rows, n_cols, figsize=(4.3 * n_cols, 3.8 * n_rows),
        squeeze=False, constrained_layout=True,
    )
    axes = grid.ravel()[:len(names)]
    for unused in grid.ravel()[len(names):]:
        fig.delaxes(unused)

    styles = (
        ("observed_mean", "Truth", "#0072BD", "^"),
        ("predicted_mean", "Predicted", "#FF7F00", "s"),
        ("aleatoric_sd", "Aleatoric SD", "#00A000", "o"),
        ("epistemic_sd", "Epistemic SD", "#FF2424", "p"),
    )
    zone = summary.attrs["clock_timezone"]
    clock_label = "source clock; timezone unspecified" if zone is None else zone
    for index, (ax, name, label, unit) in enumerate(zip(axes, names, labels, unit_labels)):
        rows = summary.loc[summary["target"] == name]
        hours = rows["minute_of_day"].to_numpy() / 60
        available = np.flatnonzero(rows["count"].to_numpy() > 0)
        # One marker near each four-hour reference time. Choosing among
        # occupied bins keeps sparse test sets visible without filling gaps.
        references = np.arange(0, 24, 4)
        markers = np.unique([
            available[np.argmin(np.abs(hours[available] - reference))]
            for reference in references
        ]).tolist()
        for column, legend_label, color, marker in styles:
            ax.plot(
                hours, rows[column].to_numpy(), color=color, marker=marker,
                markevery=markers, markersize=5, linewidth=1.2, label=legend_label,
            )
        panel = chr(ord("a") + index) if index < 26 else str(index + 1)
        ax.set_title(f"({panel}) {label}", loc="left")
        ax.set_ylabel("Mean / RMS SD" + (f" ({unit})" if unit else ""))
        ax.set_xlabel(f"Time of day ({clock_label})")
        ticks = np.arange(0, 24, 2)
        ax.set_xticks(ticks, [f"{hour:02d}:00" for hour in ticks])
        ax.tick_params(axis="x", labelrotation=70)
        ax.set_xlim(-0.3, 24)
        ax.grid(False)
        ax.legend(fontsize=8, frameon=True)
    return fig, axes, summary
