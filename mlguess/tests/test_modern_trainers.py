"""CPU trainer regressions and independently checkable metric examples."""
import copy
import importlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mlguess.torch.class_losses import EDLDigammaLoss
from mlguess.torch.metrics import MetricsCalculator
from mlguess.torch.models import DNN, CategoricalDNN
from mlguess.torch.regression_losses import EvidentialRegressionLoss
from mlguess.torch.trainer_classifier import Trainer as ClassificationTrainer
from mlguess.torch.trainer_regression import Trainer as RegressionTrainer


def config(tmp_path, **overrides):
    settings = dict(mode="none", batches_per_epoch=0, valid_batches_per_epoch=0,
                    grad_accum_every=1, amp=False, use_scheduler=False,
                    start_epoch=0, epochs=1, stopping_patience=10)
    settings.update(overrides)
    return {"save_loc": str(tmp_path / "results"), "trainer": settings}


def classification_metrics(y, predictions, split):
    assert y.shape == predictions.shape
    return {f"{split}_ave_acc": (y.argmax(1) == predictions.argmax(1)).float().mean().item()}


def regression_metrics(y, mu, total, split):
    return {f"{split}_mae": float(np.mean(np.abs(np.asarray(y) - mu)))}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU training attempted Tensor.cuda()")
    monkeypatch.setattr(torch.Tensor, "cuda", no_cuda)


@pytest.mark.parametrize("uncertainty", [False, True])
def test_classifier_cpu_train_validate_predict_fit(tmp_path, uncertainty):
    torch.manual_seed(9)
    # Five classes detects the historical hardcoded four-class one-hot encoding.
    model = CategoricalDNN(3, 5, layer_size=[], lng=False)
    loader = DataLoader(TensorDataset(torch.randn(10, 3), torch.arange(10) % 5), batch_size=5)
    criterion = EDLDigammaLoss(5, 10) if uncertainty else torch.nn.CrossEntropyLoss()
    trainer = ClassificationTrainer(model, uncertainty=uncertainty, device="cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    conf = config(tmp_path, batches_per_epoch=1, valid_batches_per_epoch=1, amp=True)
    before = model.fcn[0].weight.detach().clone()
    trained = trainer.train_one_epoch(0, conf, loader, optimizer, criterion, None, None, classification_metrics)
    assert len(trained["train_loss"]) == 1
    assert torch.isfinite(model.fcn[0].weight).all()
    assert not torch.equal(before, model.fcn[0].weight)
    validated = trainer.validate(0, conf, loader, criterion, classification_metrics)
    assert len(validated["valid_loss"]) == 1
    result = trainer.predict(conf, loader, criterion, classification_metrics, split="holdout")
    assert result["mu"].shape == (10, 5)
    assert "holdout_ave_acc" in result["metrics"]
    if uncertainty:
        np.testing.assert_allclose(result["total"], result["aleatoric"] + result["epistemic"])
    else:
        assert result["total"] is None
        assert result["dempster-shafer"] is None
    fitted = trainer.fit(conf, loader, loader, optimizer, criterion, criterion, None, None, classification_metrics)
    assert np.isfinite(fitted["valid_loss"])
    checkpoint = torch.load(tmp_path / "results" / "best.pt", weights_only=True)
    assert checkpoint["epoch"] == 0


def test_classifier_gradient_accumulation_matches_large_batch(tmp_path):
    torch.manual_seed(21)
    x, y = torch.randn(12, 3), torch.arange(12) % 3
    accumulated = torch.nn.Linear(3, 3)
    reference = copy.deepcopy(accumulated)
    optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    other_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    loss = torch.nn.CrossEntropyLoss()
    # Two groups: four samples + four samples, then a last four-sample group.
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    ClassificationTrainer(accumulated).train_one_epoch(
        0, config(tmp_path, grad_accum_every=2), loader, optimizer, loss, None, None, classification_metrics)
    for start, stop in [(0, 8), (8, 12)]:
        other_optimizer.zero_grad()
        loss(reference(x[start:stop]), y[start:stop]).backward()
        other_optimizer.step()
    for actual, expected in zip(accumulated.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_classifier_skip_validation_and_none_metric(tmp_path):
    model = torch.nn.Linear(2, 3)
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 2, 0])), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    def metric(y, pred, split):
        return {f"{split}_auc": None}
    result = ClassificationTrainer(model).fit(
        config(tmp_path, skip_validation=True), loader, None, optimizer,
        criterion, criterion, None, None, metric)
    assert result["train_loss"] == result["valid_loss"]
    assert np.isnan(result["train_auc"])


def test_regression_cpu_train_validate_predict_fit(tmp_path):
    torch.manual_seed(7)
    model = DNN(3, 2, layer_size=[], lng=True)
    loader = DataLoader(TensorDataset(torch.randn(8, 3), torch.randn(8, 2)), batch_size=4)
    trainer = RegressionTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = EvidentialRegressionLoss()
    conf = config(tmp_path, batches_per_epoch=1, valid_batches_per_epoch=1)
    before = next(model.parameters()).detach().clone()
    trained = trainer.train_one_epoch(0, conf, loader, optimizer, criterion, None, None, regression_metrics)
    assert len(trained["train_loss"]) == 1
    assert not torch.equal(before, next(model.parameters()))
    validated = trainer.validate(0, conf, loader, criterion, regression_metrics)
    assert len(validated["valid_loss"]) == 1
    prediction = trainer.predict(conf, loader, criterion, regression_metrics)
    assert prediction["mu"].shape == (8, 2)
    np.testing.assert_allclose(prediction["total"], prediction["aleatoric"] + prediction["epistemic"])
    result = trainer.fit(conf, loader, loader, optimizer, criterion, criterion, None, None, regression_metrics)
    assert np.isfinite(result["valid_loss"])
    assert (tmp_path / "results" / "checkpoint.pt").is_file()


def test_device_follows_cpu_model_even_when_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert ClassificationTrainer(torch.nn.Linear(1, 2)).device.type == "cpu"
    assert RegressionTrainer(torch.nn.Linear(1, 2)).device.type == "cpu"


def test_metrics_perfect_classification_and_calibration():
    metric = MetricsCalculator(n_bins=10)
    labels = np.array([0, 1, 0, 1])
    probabilities = np.eye(2)[labels]
    assert metric.mean_csi(probabilities, probabilities) == 1.0
    assert metric.ece(labels, probabilities) == 0.0
    assert metric.mce(labels[:, None], probabilities) == 0.0


def test_calibration_metrics_known_two_bin_example():
    # First bin: confidence .6, accuracy 1; second: confidence .9, accuracy .5.
    # Both gaps are .4: ECE=.5*.4+.5*.4=.4, MCE=.4.
    probabilities = np.array([[.6, .4], [.4, .6], [.9, .1], [.1, .9]])
    labels = np.array([0, 1, 0, 0])
    metric = MetricsCalculator(n_bins=5)
    assert metric.ece(labels, probabilities) == pytest.approx(.4)
    assert metric.mce(labels[:, None], probabilities) == pytest.approx(.4)


def test_csi_matches_independent_threshold_contingency_counts():
    rng = np.random.default_rng(5)
    probabilities = rng.dirichlet(np.ones(3), size=30)
    labels = rng.integers(0, 3, size=30)
    expected = []
    for cls in range(3):
        scores = []
        for threshold in np.arange(0, 1.01, .01):
            forecast_positive = set(np.flatnonzero(probabilities[:, cls] >= threshold))
            observed_positive = set(np.flatnonzero(labels == cls))
            union = forecast_positive | observed_positive
            if union:
                scores.append(len(forecast_positive & observed_positive) / len(union))
        expected.append(max(scores))
    assert MetricsCalculator().mean_csi(np.eye(3)[labels], probabilities) == pytest.approx(np.mean(expected))


def test_cpu_ddp_does_not_pass_cuda_device_ids(monkeypatch, tmp_path):
    module = importlib.import_module("mlguess.torch.distributed")
    captured = {}
    def fake_ddp(model, **kwargs):
        captured.update(kwargs)
        return model
    monkeypatch.setattr(module, "DDP", fake_ddp)
    model = torch.nn.Linear(2, 1)
    assert module.distributed_model_wrapper(config(tmp_path, mode="ddp"), model, torch.device("cpu")) is model
    assert captured["device_ids"] is None


def test_scheduler_modern_get_last_lr_and_phased_call(tmp_path):
    from mlguess.torch.scheduler import CosineAnnealingWarmupRestarts, phased_lr_lambda
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=.01)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=8, warmup_steps=2)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        assert scheduler.get_last_lr() == [optimizer.param_groups[0]["lr"]]
    assert np.isfinite(phased_lr_lambda(2))


def test_evidential_training_metrics_receive_probabilities(tmp_path):
    model = CategoricalDNN(2, 3, layer_size=[])
    loader = DataLoader(TensorDataset(torch.randn(6, 2), torch.arange(6) % 3), batch_size=3)
    trainer = ClassificationTrainer(model, uncertainty=True)
    criterion = EDLDigammaLoss(3, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    calls = []
    def metric(y, probability, split):
        torch.testing.assert_close(probability.sum(1), torch.ones(len(probability)))
        assert bool(torch.all(probability >= 0))
        calls.append(split)
        return {}
    trainer.train_one_epoch(0, config(tmp_path), loader, optimizer, criterion, None, None, metric)
    trainer.validate(0, config(tmp_path), loader, criterion, metric)
    assert calls == ["train", "train", "valid", "valid"]
