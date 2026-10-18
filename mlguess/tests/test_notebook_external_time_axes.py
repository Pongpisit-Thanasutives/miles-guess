"""Focused plotting checks; all tiny arrays here are test fixtures, not research results."""
import ast
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def cell_source(name, index):
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    return "".join(notebook["cells"][index]["source"])


def shap_plotter():
    tree = ast.parse(cell_source("shap.ipynb", 23))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "plot_time_series")
    namespace = {"np": np, "plt": plt, "torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "shap_plot_time_series", "exec"), namespace)
    return namespace["plot_time_series"]


@pytest.fixture(autouse=True)
def close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def test_shap_preserves_observations_alignment_and_unequal_batch_sizes():
    batches = [[torch.tensor([[1.0, -3.0], [3.0, -5.0]])],
               [torch.tensor([[7.0, -9.0]])]]
    times = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-01 06:00"], format="mixed", utc=True)
    fig, axes = shap_plotter()(batches, ["Prediction"], times)
    line = axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(orig=False), mdates.date2num(times[[1, 2, 0]].to_pydatetime()))
    np.testing.assert_allclose(line.get_ydata(), [4.0, 8.0, 2.0])
    assert len(line.get_ydata()) == 3  # All observations, not two batch means.
    assert "UTC" in axes[0].get_xlabel()
    assert isinstance(axes[0].xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    assert fig is axes[0].figure


@pytest.mark.parametrize("times, message", [
    (None, "timestamps"),
    ([0, 1], "indices"),
    (np.array([0, 1], dtype=object), "indices"),
    (["2020-01-01"], "count"),
    (["2020-01-01", None], "missing"),
])
def test_shap_rejects_missing_or_unaligned_time_metadata(times, message):
    with pytest.raises(ValueError, match=message):
        shap_plotter()([[torch.ones((2, 3))]], ["Prediction"], times)


def paper_fixture():
    columns = ["friction_velocity:surface:m_s-1", "kinematic_sensible_heat_flux:surface:K_m_s-1",
               "kinematic_latent_heat_flux:surface:g_kg-1_m_s-1"]
    frame = pd.DataFrame({"Time": ["2003-03-03 23:45:00", "2003-03-03 01:15:00",
                                    "2003-03-03 07:30:00", "2003-03-03 00:00:00"]})
    for i, col in enumerate(columns):
        frame[col] = np.array([0.02, 0.01, 0.03, 0.015]) * (i + 1)
        frame[f"{col}_pred"] = frame[col] + 0.001
        frame[f"{col}_ale"] = [0.0004, 0.0001, 0.0009, 0.000225]
        frame[f"{col}_epi"] = frame[f"{col}_ale"] / 4
    return columns, frame


def test_paper_selected_day_uses_sorted_real_datetimes():
    columns, frame = paper_fixture()
    col = columns[1]
    ns = {"np": np, "pd": pd, "plt": plt, "dq": frame, "condition": np.ones(len(frame), dtype=bool),
          "gamma": frame[col], "gamma_pred": frame[f"{col}_pred"],
          "_ale": np.sqrt(frame[f"{col}_ale"]), "_epi": np.sqrt(frame[f"{col}_epi"])}
    exec(cell_source("paper_figures_regression.ipynb", 49), ns)
    line = ns["ax"].lines[0]
    order = np.argsort(pd.to_datetime(frame.Time).to_numpy())
    np.testing.assert_allclose(line.get_xdata(orig=False), mdates.date2num(pd.to_datetime(frame.Time).to_numpy()[order]))
    np.testing.assert_allclose(line.get_ydata(), frame[col].to_numpy()[order])
    assert "Sensible heat" in ns["ax"].get_ylabel()
    # Nonuniform timestamps must not acquire equally spaced category positions.
    assert len(np.unique(np.round(np.diff(line.get_xdata(orig=False)), 8))) > 1


@pytest.mark.parametrize("index", [51, 57])
def test_paper_daily_summaries_use_physical_hours_for_sparse_times(index, monkeypatch):
    columns, frame = paper_fixture()
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    grouped = frame.copy()
    grouped["time_of_day"] = grouped.Time.str.split().str[1]
    grouped = grouped.drop(columns="Time").groupby("time_of_day").mean()
    ns = {"np": np, "pd": pd, "plt": plt, "output_cols": columns, "dg": grouped}
    exec(cell_source("paper_figures_regression.ipynb", index), ns)
    expected_hours = [0.0, 1.25, 7.5, 23.75]
    for ax in ns["axs"]:
        np.testing.assert_allclose(ax.lines[0].get_xdata(), expected_hours)
        assert ax.get_xlim() == (0.0, 24.0)
        assert "Hour of day" in ax.get_xlabel()
    if index == 51:
        np.testing.assert_allclose(ns["axs"][0].lines[0].get_ydata(),
                                   np.sqrt(grouped[f"{columns[1]}_ale"]))
    else:
        np.testing.assert_allclose(ns["axs"][0].lines[0].get_ydata(), grouped[columns[0]])
