"""CLI and process-portability tests for the source-tree applications."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from applications._common import (
    add_regression_columns,
    import_class_from_path,
    make_grad_scaler,
    resolve_device,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ["train_regressor_torch", "train_classifier_torch", "predict_regressor_torch"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_help_without_tuning_dependencies(script):
    code = f'''
import builtins
import runpy
import sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {{'echo', 'optuna', 'wandb', 'tensorflow', 'bridgescaler'}}:
        raise ImportError('Optional package intentionally unavailable: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
sys.path.insert(0, {str(ROOT / 'applications')!r})
sys.argv = [{script!r}, '--help']
runpy.run_path({str(ROOT / 'applications' / (script + '.py'))!r}, run_name='__main__')
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "--device" in completed.stdout


def test_explicit_cpu_never_initializes_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: pytest.fail("CPU selection touched CUDA"))
    conf = {"trainer": {"device": "cpu", "amp": True, "mode": "none"}}
    device = resolve_device(conf)
    assert device == torch.device("cpu")
    assert not make_grad_scaler(conf, device).is_enabled()


def test_custom_dataset_supports_spawn_workers(tmp_path):
    path = tmp_path / "portable_mlguess_test_dataset.py"
    path.write_text('''import torch
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __len__(self):
        return 4
    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32)
def collate_as_lists(batch):
    return [value.tolist() for value in batch]
''')
    cls = import_class_from_path("CustomDataset", path)
    try:
        # Test spawn/import portability without requiring Unix tensor-IPC sockets.
        collate = sys.modules[path.stem].collate_as_lists
        loader = DataLoader(cls(), batch_size=2, num_workers=1, multiprocessing_context="spawn", collate_fn=collate, timeout=20)
        result = torch.tensor([item for batch in loader for item in batch])
        assert result[:, 0].tolist() == [0, 1, 2, 3]
    finally:
        sys.modules.pop(path.stem, None)
        sys.path.remove(str(tmp_path))


def test_multitarget_prediction_columns_keep_each_target():
    frame = pd.DataFrame({"input": [1, 2, 3]})
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    result = dict.fromkeys(["mu", "aleatoric", "epistemic", "total"], values)
    out = add_regression_columns(frame, result, ["a", "b"])
    np.testing.assert_array_equal(out[["mu_a", "mu_b"]].values, values)
    assert list(frame.columns) == ["input"]


@pytest.mark.parametrize("script", SCRIPTS[:2])
def test_cpu_optimizer_checkpoint_resume(tmp_path, script):
    app = importlib.import_module(f"applications.{script}")
    conf = {"save_loc": str(tmp_path), "trainer": {"mode": "none", "start_epoch": 0,
            "learning_rate": 0.001, "weight_decay": 0.0, "amp": True, "use_scheduler": False}}
    model = torch.nn.Linear(2, 1)
    model, optimizer, scheduler, scaler = app.load_model_states_and_optimizer(conf, model, torch.device("cpu"))
    loss = model(torch.ones(3, 2)).square().mean()
    loss.backward()
    optimizer.step()
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None, "scaler_state_dict": scaler.state_dict()}, tmp_path / "checkpoint.pt")
    conf["trainer"].update(start_epoch=1, load_optimizer=True)
    restored = torch.nn.Linear(2, 1)
    restored, optimizer, _, scaler = app.load_model_states_and_optimizer(conf, restored, torch.device("cpu"))
    assert not scaler.is_enabled()
    torch.testing.assert_close(restored.weight, model.weight)
    assert len(optimizer.state) == 2


def test_regression_application_train_and_reload_on_cpu(tmp_path):
    from applications.train_regressor_torch import main as train
    from applications.predict_regressor_torch import main as predict

    dataset_path = tmp_path / "mlguess_application_regression_dataset.py"
    dataset_path.write_text('''import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, conf, split):
        rng = np.random.default_rng({"train": 1, "valid": 2, "test": 3}[split])
        self.x = rng.normal(size=(32, 2)).astype("float32")
        self.y = np.column_stack([2 + self.x[:, 0], 3 + self.x[:, 1]]).astype("float32")
        self.training_var = np.var(self.y, axis=0)
        self.y_scaler = None
        frame = pd.DataFrame(self.x, columns=["x0", "x1"])
        self.train_data = self.valid_data = self.test_data = frame
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])
''')
    conf = {
        "save_loc": str(tmp_path / "results"), "seed": 42,
        "data": {"dataset_path": str(dataset_path), "dataset_name": "CustomDataset", "output_cols": ["a", "b"]},
        "trainer": {"mode": "none", "device": "cpu", "start_epoch": 0, "epochs": 1,
                    "train_batch_size": 16, "valid_batch_size": 16, "thread_workers": 0,
                    "learning_rate": 0.001, "weight_decay": 0.0, "amp": True,
                    "use_scheduler": False, "batches_per_epoch": 0, "grad_accum_every": 1,
                    "stopping_patience": 3},
        "model": {"input_size": 2, "output_size": 2, "layer_size": [8], "dr": [0.0], "lng": True},
        "train_loss": {"coef": 0.1}, "valid_loss": {"coef": 0.1},
    }
    try:
        trained = train(0, 1, conf)
        reloaded = predict(0, 1, conf)
        np.testing.assert_allclose(trained["mu"], reloaded["mu"], rtol=1e-6, atol=1e-6)
        assert np.isfinite(reloaded["total"]).all()
        for split in ["train", "valid", "test"]:
            frame = pd.read_csv(tmp_path / "results" / f"{split}.csv")
            assert frame.shape[0] == 32
            assert {"mu_a", "mu_b", "total_uncertainty_a", "total_uncertainty_b"}.issubset(frame)
        assert (tmp_path / "results" / "checkpoint.pt").is_file()
    finally:
        sys.modules.pop(dataset_path.stem, None)
        sys.path.remove(str(tmp_path))


@pytest.mark.parametrize("uncertainty", [False, True])
def test_classifier_application_prediction_rows_follow_dataframe(tmp_path, monkeypatch, uncertainty):
    from applications import train_classifier_torch as app
    from torch.utils.data import TensorDataset
    from torch.utils.data.distributed import DistributedSampler
    from mlguess.torch.models import CategoricalDNN

    x = torch.linspace(-2, 2, 60).reshape(30, 2)
    labels = torch.arange(30) % 3
    frames = {split: pd.DataFrame(x.numpy(), columns=["x0", "x1"]) for split in ["train", "val", "test"]}

    def synthetic_loader(conf, world_size, rank, split, seed=42, return_df=False):
        dataset = TensorDataset(x, labels)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=(split == "train"))
        return (dataset, sampler, frames) if return_df else (dataset, sampler)

    monkeypatch.setattr(app, "load_dataset_and_sampler", synthetic_loader)
    conf = {
        "save_loc": str(tmp_path), "seed": 42,
        "trainer": {"mode": "none", "device": "cpu", "uncertainty": uncertainty,
                    "start_epoch": 0, "epochs": 1, "train_batch_size": 15, "valid_batch_size": 15,
                    "thread_workers": 0, "learning_rate": 0.001, "weight_decay": 0.0, "amp": True,
                    "use_scheduler": False, "batches_per_epoch": 0, "valid_batches_per_epoch": 0,
                    "grad_accum_every": 1, "stopping_patience": 3},
        "model": {"input_size": 2, "output_size": 3, "layer_size": [8], "dr": [0.0], "batch_norm": False},
    }
    app.main(0, 1, conf)
    restored = CategoricalDNN.from_config(conf, device="cpu")
    predictions = restored.predict(x, return_uncertainties=uncertainty)
    mu = predictions[0] if uncertainty else torch.softmax(predictions, dim=-1)
    written = pd.read_csv(tmp_path / "train.csv")
    np.testing.assert_allclose(written[[f"y_pred_prob_{i}" for i in range(3)]].values, mu.numpy(), atol=1e-6)
    assert ("u" in written) == uncertainty
