"""Regression tests for Python 3.11 / NumPy 2 CPU model interfaces."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlguess.torch.checkpoint import (
    load_model_state, load_state_dict, save_state_dict,
    is_dtensor_checkpoint, is_safetensor_checkpoint,
)
from mlguess.torch.class_losses import (
    edl_mse_loss, edl_log_loss, edl_digamma_loss, kl_divergence,
)
from mlguess.torch.layers import LinearNormalGamma
from mlguess.torch.mc_dropout import monte_carlo_dropout
from mlguess.torch.models import DNN, CategoricalDNN


class LegacyMetadata:
    """A trusted test-only object requiring an explicit unrestricted load."""


def small_model(**kwargs):
    return DNN(3, 2, layer_size=[8, 6], dr=[0.3, 0.4], **kwargs)


@pytest.mark.parametrize("model_class,lng", [(DNN, False), (DNN, True), (CategoricalDNN, False)])
def test_numpy_prediction_preserves_modes_and_batchnorm(model_class, lng):
    model = model_class(3, 2, layer_size=[8], dr=[0.3], lng=lng)
    # Initialize the spectral norm vectors in the usual training path.
    model(torch.randn(5, 3))
    model.fcn[1].eval()  # Verify mixed module states, not just model.training.
    before_modes = [module.training for module in model.modules()]
    before_buffers = {name: value.clone() for name, value in model.named_buffers()}
    # Single-sample inference must work with a BatchNorm model in training mode.
    x = np.arange(3, dtype=np.float64).reshape(1, 3)[:, ::-1]
    first = model.predict(x, return_uncertainties=lng or model_class is CategoricalDNN)
    second = model.predict(x, return_uncertainties=lng or model_class is CategoricalDNN)
    first = (first,) if isinstance(first, torch.Tensor) else first
    second = (second,) if isinstance(second, torch.Tensor) else second
    for actual, repeated in zip(first, second):
        assert actual.dtype == torch.float32
        assert actual.device.type == "cpu"
        assert not actual.requires_grad
        torch.testing.assert_close(actual, repeated)
    assert before_modes == [module.training for module in model.modules()]
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before_buffers[name])


def test_numpy_float64_dropout_is_stochastic_without_updating_batchnorm():
    torch.manual_seed(4)
    model = small_model(softmax=True).double().eval()
    before_buffers = {name: value.clone() for name, value in model.named_buffers()}
    outputs = model.predict_dropout(np.ones((9, 3)), mc_forward_passes=12, batch_size=4)
    pred, aleatoric, epistemic, entropy, mutual_info = outputs
    assert all(np.isfinite(value).all() for value in outputs)
    assert pred.dtype == np.float64
    np.testing.assert_allclose(pred.sum(axis=1), 1)
    assert epistemic.max() > 0
    assert (mutual_info >= -1e-12).all()
    assert not any(module.training for module in model.modules())
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before_buffers[name])


def test_loader_dropout_uses_forward_without_resetting_dropout():
    torch.manual_seed(6)
    model = CategoricalDNN(3, 2, layer_size=[8], dr=[0.5]).eval()
    loader = DataLoader(TensorDataset(torch.ones(9, 3), torch.zeros(9)), batch_size=4)
    values = monte_carlo_dropout(loader, 10, model, 2, 9, uncertainty=True)
    np.testing.assert_allclose(values["mean"].sum(axis=1), 1, rtol=1e-6)
    assert values["variance"].max() > 0
    assert all(np.isfinite(value).all() for value in values.values())
    assert not any(module.training for module in model.modules())


def test_dropout_restores_modes_on_error():
    model = small_model().train()
    before = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError):
        model.predict_dropout(np.zeros((2, 4)), mc_forward_passes=2)
    assert before == [module.training for module in model.modules()]


@pytest.mark.parametrize("name", ["plain.pt", "training.pt"])
def test_checkpoint_roundtrip_paths_and_predictions(tmp_path, name):
    model = small_model().eval()
    x = np.ones((2, 3))
    expected = model.predict(x, return_uncertainties=False)
    state = model.state_dict()
    if name == "training.pt":
        state = {"model_state_dict": state, "epoch": 2, "optimizer_state_dict": {}}
    path = tmp_path / name
    save_state_dict(state, path, use_safetensors=False)
    loaded = small_model().eval()
    loaded.load_weights(path)
    torch.testing.assert_close(loaded.predict(x, return_uncertainties=False), expected)
    assert not is_dtensor_checkpoint(path)
    assert is_safetensor_checkpoint(tmp_path / "weights.safetensors")
    with pytest.raises(FileNotFoundError):
        loaded.load_weights(tmp_path / "missing.pt")


@pytest.mark.parametrize("mode", [None, "ddp", "fsdp"])
def test_from_config_loads_cpu_without_distributed_wrapper(tmp_path, mode):
    conf = {"save_loc": str(tmp_path), "model": {"input_size": 3, "output_size": 2, "layer_size": []}}
    if mode is not None:
        conf["trainer"] = {"mode": mode}
    model = DNN(**conf["model"])
    filename = "model_checkpoint.pt" if mode == "fsdp" else "checkpoint.pt"
    state = model.state_dict() if mode == "fsdp" else {"model_state_dict": model.state_dict()}
    torch.save(state, tmp_path / filename)
    restored = DNN.from_config(conf)
    torch.testing.assert_close(restored(torch.ones(2, 3)), model(torch.ones(2, 3)))


def test_checkpoint_requires_explicit_legacy_pickle_opt_in(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"metadata": LegacyMetadata()}, path)
    with pytest.raises(Exception, match="Weights only load failed"):
        load_state_dict(path)
    assert isinstance(load_state_dict(path, weights_only=False)["metadata"], LegacyMetadata)


def test_spectral_evidence_softplus_does_not_overflow():
    layer = LinearNormalGamma(1, 1, spectral_norm=False)
    x = torch.tensor([-10.0, 0.0, 10.0, 100.0], requires_grad=True)
    result = layer.evidence(x)
    torch.testing.assert_close(result[:3], torch.log(torch.exp(x[:3]) + 1))
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("loss", [edl_mse_loss, edl_log_loss, edl_digamma_loss])
def test_classification_loss_respects_cpu_float64_inputs_even_if_cuda_exists(monkeypatch, loss):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    output = torch.tensor([[0.5, 1.2, 0.8], [0.1, 0.3, 1.1]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1, 0, 0], [0, 0, 1]])
    value = loss(output, target, 2, 3, 10)
    assert value.dtype == torch.float64
    assert value.device.type == "cpu"
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(output.grad).all()
    assert output.grad.abs().sum() > 0
    alpha = torch.ones((2, 3), dtype=torch.float64)
    torch.testing.assert_close(kl_divergence(alpha, 3), torch.zeros((2, 1), dtype=torch.float64))


def test_per_layer_dropout_and_previously_broken_initialization_flag():
    model = small_model(weight_init=True)
    assert [module.p for module in model.modules() if isinstance(module, nn.Dropout)] == [0.3, 0.4]
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    with pytest.raises(ValueError, match="dr must specify"):
        DNN(3, 2, layer_size=[4, 4, 4], dr=[0.2, 0.3])


def test_preprocessing_false_scaler_sentinel_preserves_uncertainties():
    from mlguess.preprocessing import load_preprocessing

    _, y_scaler = load_preprocessing({"data": {}})
    assert y_scaler is False
    model = DNN(3, 1, layer_size=[], lng=True)
    model(torch.randn(4, 3))
    inputs = np.ones((2, 3), dtype=np.float64)
    expected = model.predict(inputs, y_scaler=None)
    actual = model.predict(inputs, y_scaler=y_scaler)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference)
