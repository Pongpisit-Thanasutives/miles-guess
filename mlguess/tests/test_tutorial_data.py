"""Validate leakage boundaries in the editable custom-data notebook."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def load_splitter():
    notebook = Path(__file__).parents[2] / "notebooks/evidential_learning_your_data.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = next("".join(cell["source"]) for cell in cells
                  if cell["cell_type"] == "code" and "def split_rows(" in "".join(cell["source"]))
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    namespace = dict(np=np, pd=pd, GroupShuffleSplit=GroupShuffleSplit,
                     train_test_split=train_test_split)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace["split_rows"]


@pytest.mark.parametrize("strategy", ["random", "group", "time", "datetime"])
def test_custom_tutorial_split_has_no_shared_rows_or_groups(strategy):
    frame = pd.DataFrame({"group": np.repeat(np.arange(30), 4),
                          "time": np.repeat(np.arange(30), 4)})
    labels = np.tile(np.arange(3), 40)
    if strategy == "datetime":
        frame["time"] = pd.to_datetime(frame["time"], unit="D").astype(str)
        strategy = "time"
    splits = load_splitter()(frame, "classification", strategy, labels, .2, .2, 42,
                             group_column="group", time_column="time")
    assert sorted(np.concatenate(splits).tolist()) == list(range(len(frame)))
    if strategy == "group":
        group_sets = [set(frame.iloc[rows]["group"]) for rows in splits]
        assert group_sets[0].isdisjoint(group_sets[1] | group_sets[2])
        assert group_sets[1].isdisjoint(group_sets[2])
    if strategy == "time":
        train, validation, test = [frame.iloc[rows]["time"] for rows in splits]
        assert train.max() < validation.min() < test.min()
        assert validation.max() < test.min()
    if strategy == "random":
        assert all(set(labels[rows]) == {0, 1, 2} for rows in splits)


def test_custom_tutorial_rejects_invalid_split_fractions():
    with pytest.raises(ValueError, match="fractions"):
        load_splitter()(pd.DataFrame({"x": range(30)}), "regression", "random",
                        np.arange(30), .6, .5, 42)
