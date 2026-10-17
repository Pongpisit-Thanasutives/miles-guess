"""Audit regression tests for splitting, preprocessing, CPU apps and PBS dry runs."""

import copy
import importlib
from pathlib import Path
import shlex
import sys
import types

import numpy as np
import pandas as pd
import pytest
import yaml

from mlguess.preprocessing import load_preprocessor, load_preprocessing
from mlguess.splitting import load_splitter
from mlguess.keras.data import _calendar_day, load_ptype_uq, preprocess_data


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["GroupKFold", "GroupShuffleSplit", "ShuffleSplit", "StratifiedKFold", "StratifiedShuffleSplit", "StratifiedGroupKFold", "TimeSeriesSplit"])
def test_splitter_execution_and_partition_boundaries(name):
    x = np.arange(60).reshape(-1, 1)
    y = np.arange(60) % 3
    groups = np.repeat(np.arange(15), 4)
    splitter = load_splitter(name, n_splits=3, train_size=.7)
    kwargs = {"groups": groups} if "Group" in name else {}
    for train, valid in splitter.split(x, y, **kwargs):
        assert not set(train) & set(valid)
        if "Group" in name:
            assert not set(groups[train]) & set(groups[valid])
        if name == "TimeSeriesSplit":
            assert train.max() < valid.min()


def test_preprocessor_honors_options_without_mutating_config():
    params = {"with_mean": False, "with_std": False}
    transform = load_preprocessor("standard", params)
    x = np.array([[10., 30.], [20., 50.]])
    np.testing.assert_array_equal(transform.fit_transform(x), x)
    assert params == {"with_mean": False, "with_std": False}
    scaled = load_preprocessor("normalize", {"feature_range": [-2, 2]}).fit_transform(x)
    np.testing.assert_allclose(scaled, [[-2., -2.], [2., 2.]])
    quantile = load_preprocessor("quantile", {"n_quantiles": 7, "random_state": 3}, seed=9)
    assert quantile.n_quantiles == 7 and quantile.random_state == 3
    with pytest.raises(TypeError, match="Invalid parameters"):
        load_preprocessor("normalize", {"with_mean": True})
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        load_preprocessor("not-a-scaler", {})


def test_load_preprocessing_optional_options_and_no_fit():
    conf = {"data": {"scaler_x": {"type": "standard"}, "scaler_y": None}}
    before = copy.deepcopy(conf)
    x_scaler, y_scaler = load_preprocessing(conf)
    assert not hasattr(x_scaler, "mean_") and y_scaler is False
    assert conf == before
    assert load_preprocessing({"data": {}}) == (False, False)


def test_shipped_scaler_configuration_parameters_are_valid():
    for path in (ROOT / "config").rglob("*.yml"):
        conf = yaml.safe_load(path.read_text())
        if isinstance(conf, dict) and isinstance(conf.get("data"), dict):
            load_preprocessing(conf)


def test_ptype_holds_out_whole_calendar_days(monkeypatch):
    rows = []
    for day in range(1, 13):
        for hour in [0, 12]:
            rows.append({"datetime": f"2026-01-{day:02d}T{hour:02d}:00:00", "wetbulb0_filter": 0., "usa": 1.,
                         "ra_percent": 1., "sn_percent": 0., "pl_percent": 0., "fzra_percent": 0.})
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame.copy())
    conf = {"data_path": "fixture.parquet", "qc": 0, "case_studies": {"case": [pd.Timestamp("2026-01-02")]},
            "test_cutoff": pd.Timestamp("2026-01-10"), "ensemble": {"n_splits": 2}, "seed": 7, "train_size1": .7}
    result = load_ptype_uq(conf, data_split=1, drop_mixed=True)
    days = {name: set(part.day) for name, part in result.items()}
    assert not days["train"] & days["val"]
    assert not (days["train"] | days["val"]) & days["test"]
    assert days["test"] == {"2026-01-02", "2026-01-10", "2026-01-11", "2026-01-12"}
    assert sum(map(len, result.values())) == len(frame)
    assert all((part.groupby("day").size() == 2).all() for part in result.values())
    with pytest.raises(ValueError, match="data_split"):
        load_ptype_uq(conf, data_split=2)
    with pytest.raises(ValueError, match="missing"):
        _calendar_day(None)


def class_frames():
    return {
        "train": pd.DataFrame({"x": [0., 2.], "a": [1, 0], "b": [0, 0], "c": [0, 1]}, index=[10, 20]),
        "val": pd.DataFrame({"x": [100.], "a": [0], "b": [1], "c": [0]}, index=[99]),
        "test": pd.DataFrame(columns=["x", "a", "b", "c"], dtype=float),
    }


@pytest.mark.parametrize("encoding", ["label", "onehot"])
def test_preprocess_train_only_and_fixed_declared_class_axis(encoding):
    data = class_frames()
    transformed, scalers = preprocess_data(data, ["x"], ["a", "b", "c"], encoder_type=encoding)
    np.testing.assert_allclose(scalers["input"].mean_, [1.])
    np.testing.assert_allclose(transformed["val_x"], [[99.]])
    assert transformed["train_x"].index.tolist() == [10, 20]
    assert transformed["val_x"].index.tolist() == [99]
    if encoding == "onehot":
        np.testing.assert_array_equal(transformed["train_y"], [[1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_array_equal(transformed["val_y"], [[0., 1., 0.]])
        assert transformed["test_y"].shape == (0, 3)
    else:
        np.testing.assert_array_equal(transformed["train_y"], [0, 2])
        np.testing.assert_array_equal(transformed["val_y"], [1])


def test_group_symmetric_scaler_uses_requested_range():
    pytest.importorskip("bridgescaler")
    data = class_frames()
    transformed, _ = preprocess_data(data, ["x"], ["a", "b", "c"], scaler_type="symmetric", groups=[["x"]])
    np.testing.assert_allclose(transformed["train_x"], [[-1.], [1.]])


def test_quantile_groups_do_not_import_unused_bridgescaler(monkeypatch):
    import builtins
    original = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name.startswith("bridgescaler"):
            raise AssertionError("Quantile transforms must not import grouped scalers")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    transformed, _ = preprocess_data(class_frames(), ["x"], ["a", "b", "c"], scaler_type="quantile", groups=[["x"]])
    assert transformed["train_x"].shape == (2, 1)


def test_classifier_loader_honors_split_seed_and_scale_groups(monkeypatch):
    torch = pytest.importorskip("torch")
    from applications.train_classifier_torch import load_dataset_and_sampler
    import mlguess.keras.data as data_module
    captured = {}
    def load(conf, **kwargs):
        captured["split"] = kwargs["data_split"]
        return class_frames()
    def preprocess(*args, **kwargs):
        captured.update(kwargs)
        return {"train_x": pd.DataFrame(np.ones((2, 4))), "train_y": np.eye(3)[[0, 2]]}, {}
    monkeypatch.setattr(data_module, "load_ptype_uq", load)
    monkeypatch.setattr(data_module, "preprocess_data", preprocess)
    conf = {"seed": 17, "data": {"TEMP_C": ["t"], "T_DEWPOINT_C": ["d"], "UGRD_m/s": ["u"], "VGRD_m/s": ["v"],
            "ptypes": ["a", "b", "c"], "scaler_type": "standard", "scale_groups": ["TEMP_C"], "data_split": 2}}
    dataset, sampler = load_dataset_and_sampler(conf, 1, 0, "train")
    assert captured["split"] == 2 and captured["seed"] == 17
    assert captured["groups"] == [["t"], ["d", "u", "v"]]
    torch.testing.assert_close(dataset.tensors[1], torch.tensor([0, 2]))
    assert len(sampler) == 2


@pytest.mark.parametrize("module", ["train_classifier_torch", "train_regressor_torch", "predict_regressor_torch"])
def test_multi_rank_applications_reject_unsafe_csv_export_before_training(module):
    pytest.importorskip("torch")
    app = importlib.import_module(f"applications.{module}")
    with pytest.raises(NotImplementedError, match="rank-local"):
        app.main(0, 2, {"trainer": {"mode": "ddp"}})


def test_classifier_tuning_does_not_require_regression_loss_config(monkeypatch):
    pytest.importorskip("torch")
    from applications import train_classifier_torch as app
    conf = {"data": {"batch_size": 5}, "trainer": {}}
    monkeypatch.setattr(app, "main", lambda *args, **kwargs: {"valid_loss": 1.25})
    assert app._ObjectiveMixin().train(types.SimpleNamespace(number=0), conf) == {"valid_loss": 1.25}
    def bad_main(*args, **kwargs):
        raise ValueError("invalid file format")
    monkeypatch.setattr(app, "main", bad_main)
    with pytest.raises(ValueError, match="invalid file format"):
        app._ObjectiveMixin().train(types.SimpleNamespace(number=0), conf)


def test_common_distributed_setup_and_cleanup_are_guarded(monkeypatch):
    torch = pytest.importorskip("torch")
    from applications import _common
    calls = []
    monkeypatch.setattr(_common.dist, "init_process_group", lambda backend, **kwargs: calls.append((backend, kwargs)))
    _common.setup(0, 1, "ddp", torch.device("cpu"))
    assert calls[0][0] == "gloo"
    with pytest.raises(ValueError, match="FSDP requires CUDA"):
        _common.setup(0, 1, "fsdp", torch.device("cpu"))
    monkeypatch.setattr(_common.dist, "is_available", lambda: True)
    monkeypatch.setattr(_common.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(_common.dist, "destroy_process_group", lambda: calls.append("cleanup"))
    _common.cleanup_distributed()
    assert "cleanup" not in calls
    monkeypatch.setattr(_common.dist, "is_initialized", lambda: True)
    _common.cleanup_distributed()
    assert calls[-1] == "cleanup"


def test_common_optional_objective_and_array_columns(monkeypatch):
    pytest.importorskip("torch")
    from applications import _common
    with pytest.raises(AttributeError):
        _common.load_objective("missing", {"__name__": "test"}, object)
    fake = types.ModuleType("echo.src.base_objective")
    class FakeBase:
        pass
    class Mixin:
        def train(self):
            return 42
    fake.BaseObjective = FakeBase
    monkeypatch.setitem(sys.modules, "echo.src.base_objective", fake)
    namespace = {"__name__": "audit_test"}
    cls = _common.load_objective("Objective", namespace, Mixin)
    assert issubclass(cls, FakeBase) and cls().train() == 42
    frame = pd.DataFrame({"x": [1, 2]})
    result = {"mu": [[1, 2], [3, 4]], "aleatoric": None, "epistemic": None, "total": None}
    actual = _common.add_regression_columns(frame, result, np.array(["a", "b"]))
    assert actual["mu_b"].tolist() == [2, 4]
    assert list(frame.columns) == ["x"]
    with pytest.raises(ValueError, match="matching the DataFrame"):
        _common.add_regression_columns(frame, {**result, "mu": [[1, 2]]})


def test_dynamic_dataset_module_collision_without_file_is_clear(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from applications._common import import_class_from_path
    path = tmp_path / "audit_collision.py"
    path.write_text("class Dataset: pass\n")
    monkeypatch.setitem(sys.modules, path.stem, types.ModuleType(path.stem))
    with pytest.raises(ImportError, match="already in use"):
        import_class_from_path("Dataset", path)


def pbs_config(tmp_path):
    config = {"save_loc": str(tmp_path / "job outputs"), "pbs": {"name": "test", "select": 1,
              "ncpus": 2, "ngpus": 2, "mem": "4GB", "walltime": "00:10:00", "account": "example",
              "queue": "casper", "env_setup": "source ~/.bashrc", "nodes": 3, "project": "example"}}
    path = tmp_path / "job configuration.yml"
    path.write_text(yaml.safe_dump(config))
    return path, config


def test_pbs_dry_run_quotes_paths_and_never_submits(tmp_path, monkeypatch):
    from mlguess import pbs
    config_path, _ = pbs_config(tmp_path)
    monkeypatch.setattr(pbs.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry run submitted a job"))
    path = Path(pbs.launch_pbs_jobs(config_path, tmp_path / "trainer script.py", args=["--device", "cpu"], launch=False))
    content = path.read_text()
    assert content.splitlines()[1].startswith("#PBS")
    command = shlex.split(content.splitlines()[-1])
    assert command[1] == str(tmp_path / "trainer script.py")
    assert command[3] == str(config_path)
    assert command[-2:] == ["--device", "cpu"]


def test_distributed_pbs_dry_run_same_config_and_local_gpu_ids(tmp_path, monkeypatch):
    from mlguess import pbs
    source, config = pbs_config(tmp_path)
    output = Path(config["save_loc"])
    output.mkdir()
    source = output / "model.yml"
    source.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(pbs.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry run submitted a job"))
    path = Path(pbs.launch_distributed_jobs(source, tmp_path / "trainer.py", launch=False))
    assert source.is_file() and yaml.safe_load(source.read_text()) == config
    assert 'CUDA_VISIBLE_DEVICES="0,1"' in path.read_text()
    assert "--nnodes=3" in path.read_text()


def test_pbs_submission_branch_is_mocked_and_checks_exit_status(tmp_path, monkeypatch):
    from mlguess import pbs
    source, _ = pbs_config(tmp_path)
    calls = []
    def mock_run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(stdout="123.example\n")
    monkeypatch.setattr(pbs.subprocess, "run", mock_run)
    assert pbs.launch_pbs_jobs(source, tmp_path / "train.py") == "123.example"
    assert calls[0][0][0] == "qsub"
    assert calls[0][1]["check"] is True and "shell" not in calls[0][1]


@pytest.mark.parametrize("module", ["train_classifier_torch", "train_regressor_torch"])
def test_tuning_uses_small_training_batch_and_skips_postfit_test_evaluation(tmp_path, monkeypatch, module):
    torch = pytest.importorskip("torch")
    from torch.utils.data import TensorDataset
    from torch.utils.data.distributed import DistributedSampler
    app = importlib.import_module(f"applications.{module}")
    dataset = TensorDataset(torch.ones(2, 1), torch.zeros(2, 1))
    def loader(conf, world_size, rank, split, seed=42, return_df=False):
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0)
        return (dataset, sampler, {}) if return_df else (dataset, sampler)
    class SpyTrainer:
        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == torch.device("cpu")
        def fit(self, conf, train_loader, valid_loader, *args, **kwargs):
            batches = list(train_loader)
            assert len(batches) == 1 and len(batches[0][0]) == 2
            return {"valid_loss": 1.25}
        def predict(self, *args, **kwargs):
            pytest.fail("Tuning must not run post-fit test predictions")
    monkeypatch.setattr(app, "load_dataset_and_sampler", loader)
    monkeypatch.setattr(app, "Trainer", SpyTrainer)
    conf = {"save_loc": str(tmp_path), "trainer": {"mode": "none", "device": "cpu", "train_batch_size": 8,
            "valid_batch_size": 8, "start_epoch": 0, "learning_rate": .001, "weight_decay": 0., "use_scheduler": False},
            "model": {"input_size": 1, "output_size": 2, "layer_size": [4], "batch_norm": False},
            "train_loss": {}, "valid_loss": {}}
    assert app.main(0, 1, conf, trial=types.SimpleNamespace(number=0)) == {"valid_loss": 1.25}


def test_regression_cli_records_effective_overrides_and_runs_small_cpu_data(tmp_path):
    pytest.importorskip("torch")
    import os
    import subprocess
    dataset_path = tmp_path / "audit_cli_dataset.py"
    dataset_path.write_text('''import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, conf, split):
        self.x = np.linspace(0.1, 1.0, 8, dtype="float32").reshape(-1, 1)
        self.y = 1. + self.x
        self.train_data = self.valid_data = self.test_data = pd.DataFrame(self.x, columns=["x"])
    def __len__(self):
        return 8
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])
''')
    output = tmp_path / "outputs"
    config = {"save_loc": str(output), "seed": 42,
        "data": {"dataset_name": "CustomDataset", "dataset_path": str(dataset_path), "output_cols": ["target"]},
        "trainer": {"mode": "fsdp", "device": "cuda", "train_batch_size": 16, "valid_batch_size": 16,
                    "start_epoch": 0, "epochs": 1, "learning_rate": .001, "weight_decay": 0., "amp": False,
                    "use_scheduler": False, "batches_per_epoch": 0, "valid_batches_per_epoch": 0,
                    "grad_accum_every": 1, "stopping_patience": 2},
        "model": {"input_size": 1, "output_size": 1, "layer_size": [4], "dr": [0.], "batch_norm": False, "lng": True},
        "train_loss": {"coef": .1}, "valid_loss": {"coef": .1}}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.run([sys.executable, str(ROOT / "applications/train_regressor_torch.py"),
                              "-c", str(config_path), "--mode", "none", "--device", "cpu"],
                             env=env, capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    effective = yaml.safe_load((output / "model.yml").read_text())
    assert effective["trainer"]["mode"] == "none" and effective["trainer"]["device"] == "cpu"
    assert (output / "checkpoint.pt").is_file()
    assert len(pd.read_csv(output / "test.csv")) == 8


@pytest.mark.parametrize("module", ["train_classifier_torch", "train_regressor_torch", "predict_regressor_torch"])
def test_cli_pbs_receives_effective_overrides_without_submitting(tmp_path, monkeypatch, module):
    pytest.importorskip("torch")
    import runpy
    from mlguess import pbs
    source, conf = pbs_config(tmp_path)
    conf["trainer"] = {"mode": "fsdp", "device": "cuda"}
    source.write_text(yaml.safe_dump(conf))
    captured = []
    monkeypatch.setattr(pbs, "launch_pbs_jobs", lambda config, script: captured.append(yaml.safe_load(Path(config).read_text())))
    monkeypatch.setattr(sys, "argv", [module, "-c", str(source), "--mode", "none", "--device", "cpu", "-l", "1"])
    monkeypatch.syspath_prepend(str(ROOT / "applications"))
    with pytest.raises(SystemExit):
        runpy.run_path(str(ROOT / "applications" / f"{module}.py"), run_name="__main__")
    assert captured[0]["trainer"] == {"mode": "none", "device": "cpu"}


def test_pbs_example_has_one_active_block():
    conf = yaml.safe_load((ROOT / "config/pbs_example.yaml").read_text())
    assert conf["pbs"]["queue"] == "casper"
    assert conf["pbs"]["walltime"] == "12:00:00"


@pytest.mark.parametrize("module", ["train_classifier_torch", "train_regressor_torch"])
def test_application_optional_objective_delegates_without_optional_import(monkeypatch, module):
    pytest.importorskip("torch")
    app = importlib.import_module(f"applications.{module}")
    sentinel = object()
    monkeypatch.setattr(app, "load_objective", lambda name, namespace, mixin: sentinel)
    assert app.__getattr__("Objective") is sentinel


def test_regression_tuning_adapts_batch_and_validation_coefficient(monkeypatch):
    pytest.importorskip("torch")
    from applications import train_regressor_torch as app
    conf = {"data": {"batch_size": 5}, "trainer": {}, "train_loss": {"coef": .3}, "valid_loss": {"coef": .1}}
    monkeypatch.setattr(app, "main", lambda *args, **kwargs: {"valid_loss": 1.25})
    assert app._ObjectiveMixin().train(types.SimpleNamespace(number=0), conf) == {"valid_loss": 1.25}
    assert conf["trainer"]["train_batch_size"] == conf["trainer"]["valid_batch_size"] == 5
    assert conf["valid_loss"]["coef"] == .3
