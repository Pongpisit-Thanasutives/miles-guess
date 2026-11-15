"""Held-out PM2.5 diagnostics. Accept the physical-unit frame from predict_frame."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import discard_curve, predictive_pit


def plot_pit(predictions, *, bins=10):
    """PIT histogram of finite observations under the *total* Student-t forecast."""
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    pit = predictive_pit(predictions)
    fig, ax = plt.subplots(figsize=(5.4, 3.5), layout="constrained")
    ax.hist(pit, bins=np.linspace(0, 1, bins + 1), density=True, color="#447799",
            edgecolor="white", label=f"Observed days (n={len(pit):,})")
    ax.axhline(1, ls="--", color="black", lw=1.2, label="Uniform density")
    ax.set(xlim=(0, 1), xlabel="Predictive CDF at the observation (PIT)",
           ylabel="Density", title="Calibration of the total predictive distribution")
    ax.legend(frameon=False)
    return fig


def plot_discard(predictions):
    """RMSE on observations retained after successively removing the most uncertain."""
    curve = discard_curve(predictions, uncertainty="total_std")
    fig, ax = plt.subplots(figsize=(5.4, 3.5), layout="constrained")
    ax.plot(curve["discard_fraction"], curve["rmse"], "o-", color="#335F75", ms=4)
    ax.set(xlabel="Fraction discarded (largest predictive SD first)",
           ylabel="Retained RMSE (µg m⁻³)", title="Does uncertainty rank forecast errors?")
    ax.grid(axis="y", alpha=.2)
    return fig


def plot_station(predictions, *, station=None, start=None, end=None):
    """Station timeline: observed data, mean, and separate one-component SD bands.

    The default is the first station alphabetically with a finite observation,
    falling back to the first station when all are unlabeled. Bounds are descriptive SDs,
    not predictive intervals. Actual calendar gaps break the forecast lines.
    """
    station_ids = predictions["stcode"].astype(str)
    if station is None:
        if station_ids.empty:
            raise ValueError("Empty predictions")
        eligible = station_ids[np.isfinite(predictions["pm25"].to_numpy(dtype=float))]
        station = min(eligible) if not eligible.empty else min(station_ids)
    frame = predictions.loc[station_ids.eq(str(station))].copy()
    if frame.empty:
        raise ValueError(f"No predictions for station {station!r}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), errors="raise"))
    if dates.hasnans or not dates.is_unique:
        raise ValueError("Expected one valid date per prediction at the selected station")
    frame.index = dates
    frame = frame.sort_index().loc[start:end]
    if frame.empty:
        raise ValueError("No predictions in the requested station/date window")
    # Reindex only for display; no feature, prediction, or observation is imputed.
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="D"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                             sharey=True, layout="constrained")
    y = frame["pred"].to_numpy(dtype=float)
    observed = frame["pm25"].to_numpy(dtype=float)
    for ax, (name, column, color) in zip(axes, [
        ("Aleatoric", "aleatoric_std", "#00826D"),
        ("Epistemic", "epistemic_std", "#946420"),
    ]):
        sd = frame[column].to_numpy(dtype=float)
        ax.fill_between(frame.index, y - sd, y + sd, color=color, alpha=.25,
                        label=f"Mean ± 1 {name.lower()} SD")
        ax.plot(frame.index, y, color=color, lw=1.3, marker=".", ms=2.5, label="Predictive mean")
        ax.scatter(frame.index, observed, s=10, color="#282C32", alpha=.65, label="Observed", zorder=3)
        ax.set_ylabel("PM2.5 (µg m⁻³)")
        ax.legend(frameon=False, loc="upper left", ncol=3)
    axes[0].set_title(f"Station {station} · daily PM2.5 in the test period")
    axes[-1].set_xlabel("Date")
    return fig


def _self_check():
    # A real observation can be absent even when the station has a prediction.
    frame = pd.DataFrame(dict(stcode=["A"] * 4, date=pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]),
        pm25=[10., np.nan, 12., 9.], pred=[10., 11., 11., 10.]))
    frame = frame.assign(gamma=frame["pred"], nu=2., alpha=3., beta=2.,
                         aleatoric_std=1., epistemic_std=np.sqrt(.5), total_std=np.sqrt(1.5))
    for fn in (plot_pit, plot_discard, plot_station):
        fig = fn(frame)
        fig.canvas.draw()
        if fn is plot_station:
            assert len(fig.axes) == 2
            assert np.isnan(fig.axes[0].lines[0].get_ydata()[2])  # Jan 3 is a calendar gap
        plt.close(fig)


if __name__ == "__main__":
    _self_check()
