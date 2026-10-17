"""CPU runtime audit; FSDP mock tests check routing, not distributed correctness."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from mlguess.torch import checkpoint as ckpt
from mlguess.torch import distributed, scheduler
from mlguess.torch.mixed_precision import parse_dtype


@pytest.mark.parametrize("safe", [False, True])
def test_checkpoint_roundtrip_paths(tmp_path, safe):
    if safe:
        pytest.importorskip("safetensors")
    model = torch.nn.Linear(2, 1)
    path = tmp_path / ("weights.safetensors" if safe else "weights.pt")
    ckpt.save_state_dict(model.state_dict(), path, safe)
    actual = ckpt.load_state_dict(path)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(actual[name], expected)
        assert actual[name].device.type == "cpu"


def test_checkpoint_nested_optimizer_and_restricted_loading(tmp_path):
    path = tmp_path / "nested.pt"
    value = torch.ones(2, requires_grad=True)
    ckpt.save_state_dict({"model": {"x": value}, "optimizer": [value * 2], "epoch": 3}, path, False)
    loaded = ckpt.load_state_dict(path)
    assert loaded["epoch"] == 3
    assert not loaded["model"]["x"].requires_grad
    torch.testing.assert_close(loaded["optimizer"][0], torch.full((2,), 2.))
    # PyTorch restricts nonallowlisted pickle objects by default.
    torch.save({"path": Path("checkpoint")}, path)
    with pytest.raises(Exception, match="Weights only load failed|Unsupported global"):
        ckpt.load_state_dict(path)
    assert ckpt.load_state_dict(path, weights_only=False)["path"] == Path("checkpoint")


@pytest.mark.parametrize("mode,trainer_container", [("none", True), ("none", False), ("ddp", True), ("fsdp", False)])
def test_load_model_state_into_cpu_and_wrapped_model(tmp_path, mode, trainer_container):
    original = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    path = tmp_path / ("model_checkpoint.pt" if mode == "fsdp" else "checkpoint.pt")
    state = original.state_dict()
    torch.save({"model_state_dict": state} if trainer_container else state, path)
    conf = {"save_loc": tmp_path, "trainer": {"mode": mode}}
    wrapped = ckpt.ModelWrapper(ckpt.ModelWrapper(target))
    result = ckpt.load_model_state(conf, wrapped)
    assert result is wrapped
    torch.testing.assert_close(result(torch.ones(3, 2)), original(torch.ones(3, 2)))
    # The FSDP file format is also usable without any FSDP or accelerator runtime.
    ckpt.TorchFSDPCheckpointIO().load_unsharded_model(target, path) if not trainer_container else None


def test_checkpoint_suffix_and_dependency_checks(tmp_path, monkeypatch):
    assert ckpt.is_dtensor_checkpoint(Path("model.*.bin"))
    assert ckpt.is_dtensor_checkpoint("model.*.safetensors")
    assert not ckpt.is_dtensor_checkpoint("model.bin")
    assert ckpt.is_safetensor_checkpoint(Path("model.safetensors"))
    assert not ckpt.is_safetensor_checkpoint("model.pt")
    with pytest.raises(AssertionError, match="dtensor"):
        ckpt.load_state_dict("model.*.bin")
    monkeypatch.setattr(ckpt, "find_spec", lambda name: None)
    assert not ckpt.is_safetensors_available()
    with pytest.raises(AssertionError, match="not available"):
        ckpt.save_state_dict({}, tmp_path / "x.safetensors", True)
    with pytest.raises(AssertionError, match="not available"):
        ckpt.load_state_dict(tmp_path / "x.safetensors")


def test_fsdp_import_helper_and_legacy_tensor_reshape():
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    assert ckpt._fsdp_types() == (fsdp.FullyShardedDataParallel, fsdp.FullStateDictConfig, fsdp.StateDictType)
    # This pure tensor utility does not construct FSDP or require an accelerator.
    x1 = torch.arange(48.).reshape(1, 2, 2, 3, 2, 2)
    x2 = torch.full((1, 2, 1, 2, 2), -1.)
    actual = ckpt.TorchFSDPModel.concat_and_reshape(None, x1, x2)
    assert actual.shape == (1, 7, 2, 2, 2)
    for first in range(2):
        for second in range(3):
            torch.testing.assert_close(actual[:, 3 * first + second], x1[:, :, first, second])
    torch.testing.assert_close(actual[:, -1], x2[:, :, 0])


def test_optimizer_wrapper_training_state_and_gradients():
    parameter = torch.nn.Parameter(torch.tensor([2., -3.]))
    native = torch.optim.SGD([parameter], lr=.1, momentum=.9)
    wrapper = ckpt.OptimizerWrapper(native)
    assert wrapper.unwrap() is native
    assert wrapper.param_groups is native.param_groups
    assert wrapper.defaults is native.defaults
    wrapper.backward((parameter ** 2).sum())
    wrapper.clip_grad_by_value(.5)
    assert bool(torch.all(parameter.grad.abs() <= .5))
    norm = wrapper.clip_grad_by_norm(.25)
    assert norm.item() == pytest.approx(2 ** .5 * .5)
    assert parameter.grad.norm().item() <= .250001
    before = parameter.detach().clone()
    wrapper.step()
    assert not torch.equal(before, parameter)
    saved = deepcopy(wrapper.state_dict())
    wrapper.load_state_dict(saved)
    wrapper.zero_grad(set_to_none=True)
    assert parameter.grad is None
    wrapper.backward_by_grad(parameter ** 2, torch.ones_like(parameter))
    torch.testing.assert_close(parameter.grad, 2 * parameter.detach())
    another = torch.nn.Parameter(torch.zeros(1))
    wrapper.add_param_group({"params": [another]})
    assert wrapper.parameters == [parameter, another]
    with pytest.raises(NotImplementedError, match="mixed precision"):
        wrapper.scale_loss(parameter.sum())
    with pytest.raises(NotImplementedError, match="mixed precision"):
        wrapper.unscale_grad()


def test_fsdp_optimizer_api_routing_with_mocks(tmp_path, monkeypatch):
    """No collective communication or GPU execution is implied by this test."""
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    native_model = torch.nn.Linear(2, 1)
    model = ckpt.ModelWrapper(native_model)
    native_optimizer = torch.optim.Adam(native_model.parameters(), lr=.01)
    native_model(torch.ones(2, 2)).sum().backward()
    native_optimizer.step()
    optimizer = ckpt.FSDPOptimizerWrapper(native_optimizer, model)
    assert optimizer.unwrap_model() is native_model
    state = deepcopy(native_optimizer.state_dict())
    calls = []

    class FakeFSDP:
        @staticmethod
        @contextmanager
        def state_dict_type(received_model, kind, model_config, optim_config=None):
            assert received_model is native_model
            assert kind == fsdp.StateDictType.FULL_STATE_DICT
            assert model_config.rank0_only
            if optim_config is not None:
                assert optim_config.rank0_only and optim_config.offload_to_cpu
            calls.append("full-state context")
            yield

        @staticmethod
        def optim_state_dict(received_model, optim):
            assert received_model is native_model
            assert optim is native_optimizer
            calls.append("save optimizer")
            return state

        @staticmethod
        def shard_full_optim_state_dict(received_state, received_model, optim):
            assert received_model is native_model
            assert optim is native_optimizer
            assert received_state["param_groups"] == state["param_groups"]
            calls.append("load optimizer")
            return received_state

    monkeypatch.setattr(ckpt, "_fsdp_types", lambda: (FakeFSDP, fsdp.FullStateDictConfig, fsdp.StateDictType))
    io = ckpt.TorchFSDPCheckpointIO()
    path = tmp_path / "optim.pt"
    io.save_unsharded_optimizer(optimizer, path, True, rank=1)
    assert not path.exists()
    io.save_unsharded_optimizer(optimizer, path, True, rank=0)
    io.load_unsharded_optimizer(optimizer, path)
    model_path = tmp_path / "model.pt"
    io.save_unsharded_model(model, model_path, True, False, rank=0)
    assert model_path.exists()
    assert calls.count("save optimizer") == 2
    assert calls.count("load optimizer") == 1


@pytest.mark.parametrize("name,expected", [("Float32", torch.float32), ("double", torch.float64), ("half", torch.float16), ("bfloat16", torch.bfloat16), ("long", torch.int64), ("BOOL", torch.bool)])
def test_parse_dtype(name, expected):
    assert parse_dtype(name) is expected


def test_parse_dtype_invalid():
    with pytest.raises(ValueError, match="Invalid dtype"):
        parse_dtype("float23")


def test_distributed_cpu_noop_and_clear_fsdp_error(tmp_path):
    model = torch.nn.Linear(2, 1)
    conf = {"save_loc": tmp_path, "trainer": {"mode": "none"}}
    assert distributed.distributed_model_wrapper(conf, model, "cpu") is model
    assert conf["save_loc"] == str(tmp_path)
    conf["trainer"]["mode"] = "fsdp"
    with pytest.raises(ValueError, match="FSDP requires an accelerator"):
        distributed.distributed_model_wrapper(conf, model, "cpu")


def test_distributed_cuda_device_id_routing_mocked(monkeypatch):
    captured = {}
    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(distributed, "DDP", lambda model, **kw: captured.update(kw) or model)
    conf = {"trainer": {"mode": "ddp"}}
    assert distributed.distributed_model_wrapper(conf, model, "cuda") is model
    assert captured["device_ids"] == [2]
    distributed.distributed_model_wrapper(conf, model, "cuda:3")
    assert captured["device_ids"] == [3]
    distributed.distributed_model_wrapper(conf, model, "cpu")
    assert captured["device_ids"] is None


def test_distributed_actual_single_process_cpu(tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("This torch build has no CPU Gloo distributed backend")
    if torch.distributed.is_initialized():
        pytest.skip("Another test owns the global process group")
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    try:
        torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=0, world_size=1, timeout=timedelta(seconds=20))
    except RuntimeError as exc:
        if "Operation not permitted" in str(exc) or "Permission denied" in str(exc):
            pytest.skip("Host sandbox forbids the socket creation required by CPU Gloo")
        raise
    try:
        model = torch.nn.Linear(2, 1)
        wrapped = distributed.distributed_model_wrapper({"trainer": {"mode": "ddp"}}, model, "cpu")
        values = torch.ones(3, 2)
        torch.testing.assert_close(wrapped(values), model(values))
        wrapped(values).sum().backward()
        assert model.weight.grad is not None
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("kind", ["lambda", "plateau", "cosine-annealing"])
def test_load_scheduler_wrapped_optimizer_and_resume(kind):
    native = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=.1)
    wrapper = ckpt.OptimizerWrapper(native)
    options = {"scheduler_type": kind}
    if kind == "cosine-annealing":
        options.update(first_cycle_steps=8, warmup_steps=2)
    conf = {"trainer": {"use_scheduler": True, "scheduler": options}}
    saved_conf = deepcopy(conf)
    result = scheduler.load_scheduler(wrapper, conf)
    assert conf == saved_conf
    assert result.optimizer is native
    for i in range(4):
        wrapper.step()
        result.step(1. / (i + 1)) if kind == "plateau" else result.step()
    assert np.isfinite(native.param_groups[0]["lr"])
    state = deepcopy(result.state_dict())
    result.load_state_dict(state)
    assert result.last_epoch == state["last_epoch"]


def test_load_scheduler_disabled_and_invalid():
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=.1)
    assert scheduler.load_scheduler(optimizer, {"trainer": {"use_scheduler": False}}) is None
    with pytest.raises(ValueError, match="Invalid scheduler_type"):
        scheduler.load_scheduler(optimizer, {"trainer": {"use_scheduler": True, "scheduler": {"scheduler_type": "typo"}}})


@pytest.mark.parametrize("warmup,multiplier", [(0, 1.), (2, 1.), (2, 2.), (2, 1.5), (2, 1.01)])
def test_explicit_cosine_step_matches_incremental_schedule(warmup, multiplier):
    def build():
        optim = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=.02)
        sched = scheduler.CosineAnnealingWarmupRestarts(optim, first_cycle_steps=np.int64(7), warmup_steps=warmup, cycle_mult=multiplier, max_lr=.1, min_lr=.001, gamma=.8)
        return optim, sched
    optimizer, incremental = build()
    _, explicit = build()
    for epoch in range(1, 85):
        optimizer.step()
        incremental.step()
        explicit.step(epoch)
        assert explicit.cycle == incremental.cycle
        assert explicit.cur_cycle_steps == incremental.cur_cycle_steps
        assert explicit.step_in_cycle == incremental.step_in_cycle
        assert explicit.get_last_lr() == pytest.approx(incremental.get_last_lr())
    explicit.step(1)
    assert explicit.cycle == 0
    assert explicit.cur_cycle_steps == 7


@pytest.mark.parametrize("options", [{"first_cycle_steps": 0}, {"first_cycle_steps": 4, "warmup_steps": -1}, {"first_cycle_steps": 4, "warmup_steps": 4}, {"first_cycle_steps": 4, "cycle_mult": .5}])
def test_invalid_cosine_configuration(options):
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=.1)
    with pytest.raises(ValueError):
        scheduler.CosineAnnealingWarmupRestarts(optimizer, **options)


def test_phase_functions_and_annealing_reference_values():
    assert scheduler.lr_lambda_phase1(0) == 0
    assert scheduler.lr_lambda_phase1(10) == 1
    assert scheduler.lr_lambda_phase1(100) == 0
    assert scheduler.lr_lambda_phase2(0).item() == 1
    assert scheduler.phased_lr_lambda(2) == scheduler.lr_lambda_phase1(2, num_epochs=1000)
    torch.testing.assert_close(scheduler.phased_lr_lambda(1000), torch.tensor(1.))
    assert scheduler.annealed_probability(0) == 1
    assert scheduler.annealed_probability(100) == pytest.approx(.01)
    assert scheduler.annealed_probability(1000) == .01
