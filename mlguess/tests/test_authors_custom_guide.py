"""Guard the authors-data starting point in the separate custom-data guide."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name,expected", [
    ("sample_cabauw_surface_layer.csv", "fe9ead12574f5041da1fc683ed85bf994aff0cab17227edec5398592b4ce2bde"),
    ("ptype_sample.parquet", "d10c68959901d2ee265c11678a11a48d1bb306304c50e126d7a2ef5d81eaa00b"),
])
def test_bundled_author_data_is_unchanged(name, expected):
    assert hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == expected


def test_custom_guide_default_split_matches_original_notebook(tmp_path):
    guide = json.loads((ROOT / "notebooks/evidential_learning_your_data.ipynb").read_text())
    original = json.loads((ROOT / "notebooks/legacy/regression_example_torch.ipynb").read_text())
    namespace = dict(ROOT=ROOT, Path=Path, np=np, pd=pd,
                     GroupShuffleSplit=GroupShuffleSplit, train_test_split=train_test_split,
                     display=lambda value: None, TASK="regression", DATA_PATH=None,
                     FEATURE_COLUMNS=None, TARGET_COLUMNS=None, OUTPUT_DIR=tmp_path,
                     PLOT_TIME_COLUMN=None, PLOT_TIME_WINDOW=None, TIME_COLUMN=None,
                     author_config=yaml.safe_load((ROOT / "config/evidential_regression_torch.yml").read_text()))
    exec("".join(guide["cells"][6]["source"]), namespace)
    exec("".join(guide["cells"][8]["source"]), namespace)
    expected = dict(data=namespace["frame"], flat_seed=1000, data_seed=0,
                    GroupShuffleSplit=GroupShuffleSplit)
    exec("".join(original["cells"][8]["source"]), expected)
    for current, archived in [("train_idx", "train_data"), ("val_idx", "valid_data"),
                              ("test_idx", "test_data")]:
        np.testing.assert_array_equal(namespace[current], expected[archived].index.to_numpy())
    assert [len(namespace[key]) for key in ("train_idx", "val_idx", "test_idx")] == [55007, 6598, 10592]
    assert namespace["FEATURE_COLUMNS"] == namespace["author_config"]["data"]["input_cols"]
    assert namespace["TARGET_COLUMNS"] == ["friction_velocity:surface:m_s-1"]

    assert namespace["plot_time_column"] == "Time"
    assert "Time" not in namespace["FEATURE_COLUMNS"]
    assert namespace["plot_times"].dt.tz is None
    expected_times = pd.to_datetime(namespace["frame"].iloc[namespace["test_idx"]]["Time"])
    pd.testing.assert_series_equal(namespace["plot_times"].iloc[namespace["test_idx"]], expected_times)


@pytest.fixture(params=["evidential_learning_your_data.ipynb", "dna_example_torch.ipynb"])
def time_helpers(request):
    notebook = json.loads((ROOT / "notebooks" / request.param).read_text())
    namespace = dict(np=np, pd=pd)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name in {"parse_plot_times", "prediction_display"}]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), request.param, "exec"), namespace)
    return namespace


def test_time_plot_preserves_prediction_pairing_and_gaps(time_helpers):
    frame = pd.DataFrame({"collected": ["2024-01-03", "2024-01-01", "2024-05-01", "2024-01-01"]})
    times = time_helpers["parse_plot_times"](frame, "collected")
    test_rows = np.array([2, 0, 3, 1])  # Deliberately not chronological or row sorted.
    means = np.array([20., 30., 40., 50.])
    selected, coordinates = time_helpers["prediction_display"](means, times.iloc[test_rows], "7D")
    np.testing.assert_array_equal(selected, [2, 3, 1])
    np.testing.assert_array_equal(means[selected], [40., 50., 30.])
    np.testing.assert_array_equal(coordinates.to_numpy(), times.iloc[[3, 1, 0]].to_numpy())
    np.testing.assert_array_equal(test_rows, [2, 0, 3, 1])
    all_rows, _ = time_helpers["prediction_display"](means, times.iloc[test_rows])
    np.testing.assert_array_equal(all_rows, [2, 3, 1, 0])


@pytest.mark.parametrize("values", [["2024-01-01", "not-a-date"], ["2024-01-01", None], [1., float("nan")]])
def test_explicit_invalid_plot_times_raise(time_helpers, values):
    with pytest.raises(ValueError, match=r"invalid times at row positions \[1\]"):
        time_helpers["parse_plot_times"](pd.DataFrame({"time": values}), "time")
    with pytest.raises(ValueError, match="absent from the CSV"):
        time_helpers["parse_plot_times"](pd.DataFrame({"time": values}), "missing")


def test_numeric_time_and_mean_fallback_use_actual_coordinates(time_helpers):
    times = time_helpers["parse_plot_times"](pd.DataFrame({"seconds": [50., 5., 10.]}), "seconds")
    means = np.array([4., -2., 9.])
    selected, coordinates = time_helpers["prediction_display"](means, times, 10.)
    np.testing.assert_array_equal(selected, [1, 2])
    np.testing.assert_array_equal(coordinates, [5., 10.])
    selected, coordinates = time_helpers["prediction_display"](means)
    np.testing.assert_array_equal(coordinates, [-2., 4., 9.])
    np.testing.assert_array_equal(selected, [1, 0, 2])
    with pytest.raises(ValueError, match="positive number"):
        time_helpers["prediction_display"](means, times, "7D")


def test_plot_timezone_interpretation_is_explicit(time_helpers):
    naive = time_helpers["parse_plot_times"](pd.DataFrame({"time": ["2024-01-01", "2024-01-02"]}), "time")
    assert naive.dt.tz is None
    aware = time_helpers["parse_plot_times"](pd.DataFrame({"time": ["2024-01-01T09:00:00+09:00", "2024-01-01T00:00:00+00:00"]}), "time")
    assert str(aware.dt.tz) == "UTC"
    assert aware.iloc[0] == aware.iloc[1] == pd.Timestamp("2024-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="mixes timezone-naive and timezone-aware"):
        time_helpers["parse_plot_times"](pd.DataFrame({"time": ["2024-01-01", "2024-01-02T00:00:00Z"]}), "time")
