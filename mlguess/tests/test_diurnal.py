"""Analytical checks of the authors' Figure 11 aggregation convention."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlguess.diurnal import diurnal_uncertainty_summary, plot_diurnal_uncertainty


def example():
    return dict(
        timestamps=["2024-06-01 12:09:59", "2024-06-02 12:00:00", "2024-06-01 03:10:00"],
        observed=np.array([1., 5., 9.]),
        predicted=np.array([0., 10., 8.]),
        aleatoric_variance=np.array([1., 9., 4.]),
        epistemic_variance=np.array([4., 16., 0.]),
    )


def test_analytic_rms_uncertainty_and_empty_bins():
    result = diurnal_uncertainty_summary(**example(), target_names=["flux"])
    assert list(result.columns) == [
        "target", "minute_of_day", "count", "observed_mean", "predicted_mean",
        "aleatoric_sd", "epistemic_sd",
    ]
    assert len(result) == 144
    assert result.target.unique().tolist() == ["flux"]
    assert result.minute_of_day.tolist() == list(range(0, 1440, 10))
    assert result["count"].sum() == 3
    noon = result.set_index("minute_of_day").loc[720]
    assert noon["count"] == 2
    assert noon.observed_mean == 3
    assert noon.predicted_mean == 5
    assert noon.aleatoric_sd == pytest.approx(np.sqrt(5))
    assert noon.epistemic_sd == pytest.approx(np.sqrt(10))
    assert noon.aleatoric_sd != 2  # Mean SD is a different estimand.
    assert noon.epistemic_sd != pytest.approx(np.sqrt(10 + 25))  # No mixture spread.
    empty = result[result["count"] == 0]
    assert empty.iloc[:, 3:].isna().all().all()
    assert result.attrs["clock_timezone"] is None


def test_multiple_targets_and_observation_order_do_not_change_values():
    data = example()
    for name in ("observed", "predicted", "aleatoric_variance", "epistemic_variance"):
        data[name] = np.column_stack((data[name], data[name] * 4))
    first = diurnal_uncertainty_summary(**data, target_names=["wind", "heat"])
    for name in data:
        data[name] = np.asarray(data[name])[[2, 0, 1]]
    second = diurnal_uncertainty_summary(**data, target_names=["wind", "heat"])
    pd.testing.assert_frame_equal(first, second)
    noon = first[first.minute_of_day == 720].set_index("target")
    assert noon.loc["heat", "observed_mean"] == 4 * noon.loc["wind", "observed_mean"]
    assert noon.loc["heat", "aleatoric_sd"] == 2 * noon.loc["wind", "aleatoric_sd"]
    assert len(first) == 288


def test_equal_weight_per_observation_including_repeated_timestamps():
    result = diurnal_uncertainty_summary(
        ["2024-01-01 06:00"] * 2 + ["2024-01-02 06:00"],
        [0, 0, 9], [1, 1, 10], [1, 1, 16], [0, 0, 0],
    )
    row = result.loc[result.minute_of_day == 360].iloc[0]
    assert row["count"] == 3
    assert row.observed_mean == 3  # Not 4.5 from equal-weight daily means.
    assert row.aleatoric_sd == pytest.approx(np.sqrt(6))


def test_flooring_near_midnight_and_fifteen_minute_bins():
    result = diurnal_uncertainty_summary(
        ["2024-01-01 23:59:59.999999", "2024-01-02 00:14:59", "2024-01-02 00:15"],
        [3, 1, 2], [3, 1, 2], [0, 0, 0], [0, 0, 0], bin_minutes=15,
    )
    occupied = result[result["count"] > 0]
    assert occupied.minute_of_day.tolist() == [0, 15, 1425]
    assert occupied.observed_mean.tolist() == [1, 2, 3]


def test_naive_datetime64_preserves_source_clock():
    data = example()
    data["timestamps"] = np.array(data["timestamps"], dtype="datetime64[ns]")
    result = diurnal_uncertainty_summary(**data)
    assert result[result["count"] > 0].minute_of_day.tolist() == [190, 720]
    assert result.attrs["clock_convention"] == "source clock"


def test_aware_common_timezone_preserved_instead_of_utc_conversion():
    data = example()
    data["timestamps"] = pd.to_datetime(data["timestamps"]).tz_localize("Asia/Tokyo")
    result = diurnal_uncertainty_summary(**data)
    assert result[result["count"] > 0].minute_of_day.tolist() == [190, 720]
    assert result.attrs["clock_timezone"] == "Asia/Tokyo"


def test_repeated_dst_clock_time_is_one_diurnal_bin():
    clock = pd.to_datetime(["2024-11-03 05:30Z", "2024-11-03 06:30Z"]).tz_convert("America/New_York")
    result = diurnal_uncertainty_summary(clock, [2, 4], [1, 3], [1, 1], [0, 0])
    row = result[result["count"] > 0].iloc[0]
    assert row.minute_of_day == 90
    assert row["count"] == 2
    assert row.observed_mean == 3


@pytest.mark.parametrize("timestamps", [
    [0, 1, 2], np.array([0, 1, 2], dtype=object), [False, True, False],
    ["2024-01-01", None, "2024-01-02"],
    ["2024-01-01", "not a timestamp", "2024-01-02"],
    ["2024-01-01", "2024-01-01T00:00:00Z", "2024-01-02"],
    ["2024-01-01T00:00:00+01:00", "2024-01-01T00:00:00+02:00", "2024-01-02T00:00:00+01:00"],
    [["2024-01-01"], ["2024-01-02"], ["2024-01-03"]],
    ["2024-01-01"],
])
def test_rejects_invalid_or_ambiguous_timestamps(timestamps):
    data = example()
    data["timestamps"] = timestamps
    with pytest.raises(ValueError, match="timestamps"):
        diurnal_uncertainty_summary(**data)


@pytest.mark.parametrize("bin_minutes", [True, False, 0, -10, 7, 10.0, "10", 1441])
def test_rejects_invalid_bins(bin_minutes):
    with pytest.raises(ValueError, match="bin_minutes"):
        diurnal_uncertainty_summary(**example(), bin_minutes=bin_minutes)


@pytest.mark.parametrize("name,value", [
    ("observed", []), ("predicted", [1, 2]), ("predicted", [[1, 2], [3, 4], [5, 6]]),
    ("observed", [0, np.nan, 2]), ("aleatoric_variance", [0, np.inf, 2]),
    ("predicted", [0, 1j, 2]),
    ("aleatoric_variance", [-1, 0, 1]), ("epistemic_variance", [0, -1, 2]),
    ("epistemic_variance", np.zeros((3, 1, 1))),
])
def test_rejects_invalid_prediction_arrays(name, value):
    data = example()
    data[name] = value
    with pytest.raises(ValueError):
        diurnal_uncertainty_summary(**data)


@pytest.mark.parametrize("names", [[], [""], [" "] , "wind", [1], ["a", "b"]])
def test_rejects_invalid_target_names(names):
    with pytest.raises(ValueError, match="target_names"):
        diurnal_uncertainty_summary(**example(), target_names=names)


def test_duplicate_target_names_rejected():
    data = example()
    for name in ("observed", "predicted", "aleatoric_variance", "epistemic_variance"):
        data[name] = np.column_stack((data[name], data[name]))
    with pytest.raises(ValueError, match="distinct"):
        diurnal_uncertainty_summary(**data, target_names=["flux", "flux"])


def test_plot_uses_clock_hours_breaks_gaps_and_matches_paper_markers():
    fig, axes, summary = plot_diurnal_uncertainty(
        **example(), target_names=["u_star"], target_labels=["Friction velocity"], units="m/s",
    )
    try:
        assert axes.shape == (1,)
        ax = axes[0]
        assert len(ax.lines) == 4
        assert [line.get_marker() for line in ax.lines] == ["^", "s", "o", "p"]
        assert [line.get_label() for line in ax.lines] == ["Truth", "Predicted", "Aleatoric SD", "Epistemic SD"]
        x = ax.lines[0].get_xdata()
        y = ax.lines[0].get_ydata()
        np.testing.assert_allclose(x, np.arange(0, 1440, 10) / 60)
        np.testing.assert_allclose(x[np.isfinite(y)], [3 + 10 / 60, 12])
        assert np.isnan(y[20:70]).all()  # No connection over absent clock bins.
        assert "source clock" in ax.get_xlabel()
        assert "m/s" in ax.get_ylabel()
        assert "RMS SD" in ax.get_ylabel()
        assert ax.get_title(loc="left") == "(a) Friction velocity"
        assert [label.get_text() for label in ax.get_xticklabels()] == [f"{hour:02d}:00" for hour in range(0, 24, 2)]
        assert summary["count"].sum() == 3
        # Sparse occupied bins retain visible markers.
        assert len(ax.lines[0].get_markevery()) == 2
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_plot_wraps_panels_without_manufacturing_targets_or_clipping():
    data = example()
    for name in ("observed", "predicted", "aleatoric_variance", "epistemic_variance"):
        data[name] = np.repeat(data[name][:, None], 4, axis=1)
    data["predicted"][0, 3] = 10000
    data["timestamps"] = pd.to_datetime(data["timestamps"]).tz_localize("UTC")
    fig, axes, _ = plot_diurnal_uncertainty(
        **data, target_names=["a", "b", "c", "d"], units=["m/s", "K", "W", "Pa"],
    )
    try:
        assert axes.shape == (4,)
        assert len(fig.axes) == 4
        assert axes[3].get_ylim()[1] > 5000
        assert axes[3].get_ylabel().endswith("(Pa)")
        assert axes[0].get_xlabel() == "Time of day (UTC)"
        assert axes[3].get_subplotspec().rowspan.start == 1
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kwargs", [{"units": ["a", "b"]}, {"units": 7}, {"target_labels": "flux"}])
def test_invalid_plot_labels_fail_before_creating_a_figure(kwargs):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_diurnal_uncertainty(**example(), **kwargs)
    assert plt.get_fignums() == before
