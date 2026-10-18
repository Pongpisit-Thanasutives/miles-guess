"""Guard the authors-data starting point in the separate custom-data guide."""
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
