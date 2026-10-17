"""Portable execution helpers for the source-tree PyTorch applications."""

import importlib
import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist



def validate_application_run(conf, world_size):
    """Reject unsupported multi-rank CSV exports before training starts."""
    mode = conf.get("trainer", {}).get("mode", "none")
    if mode not in {"none", "ddp", "fsdp"}:
        raise ValueError("trainer.mode must be 'none', 'ddp', or 'fsdp'")
    if world_size != 1:
        raise NotImplementedError(
            "Multi-rank application prediction export is unsupported: rank-local "
            "predictions cannot be attached to a full DataFrame. Use one process "
            "for these applications; library distributed wrappers remain available."
        )


def resolve_device(conf, rank=0):
    """Select CPU or CUDA explicitly; ``auto`` preserves GPU use when available."""
    requested = str(conf.get("trainer", {}).get("device", "auto")).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("The applications support trainer.device='cpu', 'cuda', or 'auto'.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
        if device.index is None:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            device = torch.device("cuda", local_rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    return device


def setup(rank, world_size, mode, device=None):
    """Initialize optional distributed execution with a platform-supported backend."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if mode == "fsdp" and device.type != "cuda":
        raise ValueError("FSDP requires CUDA; use trainer.mode='none' for ordinary CPU training.")
    backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
    logging.info("Running %s on rank %s with world size %s (%s).", mode.upper(), rank, world_size, backend)
    dist.init_process_group(backend, rank=rank, world_size=world_size)



def cleanup_distributed():
    """Release a group after the application's final prediction pass."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def make_grad_scaler(conf, device):
    """Keep CPU computation in float32; use gradient scaling for CUDA AMP only."""
    enabled = bool(conf["trainer"].get("amp", False)) and device.type == "cuda"
    if conf["trainer"].get("mode", "none") == "fsdp":
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        return ShardedGradScaler(enabled=enabled)
    return torch.amp.GradScaler("cuda", enabled=enabled)


def import_class_from_path(class_name, file_path):
    """Import a dataset by its real module name so spawned workers can import it.

    macOS and Windows use ``spawn`` for DataLoader workers. Registering a
    file only under a synthetic name in the parent process is not sufficient.
    """
    path = Path(os.path.expandvars(file_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset module does not exist: {path}")
    if not path.stem.isidentifier():
        raise ValueError("Dataset filenames must be valid Python module names.")
    existing = sys.modules.get(path.stem)
    if existing is not None and Path(getattr(existing, "__file__", None) or "").resolve() != path:
        raise ImportError(f"Dataset module name {path.stem!r} is already in use; rename the dataset file.")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    importlib.invalidate_caches()
    module = importlib.import_module(path.stem)
    return getattr(module, class_name)


def load_objective(name, namespace, mixin):
    """Load the legacy ECHO tuning interface only when explicitly requested."""
    if name != "Objective":
        raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
    try:
        from echo.src.base_objective import BaseObjective
    except ImportError as exc:
        raise ImportError(
            "Hyperparameter tuning requires the optional ECHO and Optuna packages. "
            "Normal training and prediction do not require them."
        ) from exc
    objective = type("Objective", (mixin, BaseObjective), {"__module__": namespace["__name__"]})
    namespace["Objective"] = objective
    return objective


def add_regression_columns(frame, result, output_names=None):
    """Write scalar or multi-target predictions without discarding target columns."""
    columns = {"mu": "mu", "aleatoric": "aleatoric", "epistemic": "epistemic", "total_uncertainty": "total"}
    frame = frame.copy()
    for prefix, key in columns.items():
        values = result[key]
        if values is None:
            continue
        values = np.asarray(values)
        if values.ndim not in (1, 2) or len(values) != len(frame):
            raise ValueError("Prediction arrays must have shape (rows,) or (rows, targets) matching the DataFrame")
        if values.ndim == 1 or values.shape[1] == 1:
            frame[prefix] = values.reshape(-1)
        else:
            names = list(range(values.shape[1])) if output_names is None else list(output_names)
            if len(names) != values.shape[1]:
                raise ValueError("The number of output column names does not match the predictions.")
            frame[[f"{prefix}_{name}" for name in names]] = values
    return frame
