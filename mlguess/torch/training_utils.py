"""Shared bookkeeping for CPU/GPU training; no model or loss formulas."""

from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist


class BatchResults(defaultdict):
    """Batch diagnostics plus sample/criterion weights for epoch summaries."""

    def __init__(self, loss_reduction=None):
        super().__init__(list)
        self.loss_reduction = loss_reduction
        self.batch_sizes = []
        self.loss_weights = []


def batch_limit(loader, requested=0):
    count = len(loader)
    if count < 1:
        raise ValueError("The data loader must contain at least one batch")
    if not isinstance(requested, (int, np.integer)):
        raise ValueError("The batch limit must be an integer")
    return min(requested, count) if requested > 0 else count


def underlying_model(model):
    """Access custom inference methods without bypassing wrapped forward calls."""
    while hasattr(model, "module"):
        model = model.module
    return model


def class_labels(labels, device):
    labels = labels.to(device=device).reshape(-1)
    if labels.is_floating_point() and (
        not torch.isfinite(labels).all() or not torch.equal(labels, labels.round())
    ):
        raise ValueError("Class labels must be finite integers")
    return labels.to(dtype=torch.long)


def loss_weight(criterion, targets):
    """Denominator of standard mean-reduced losses, including weighted CE."""
    if getattr(criterion, "reduction", None) == "sum":
        return 1.0
    if isinstance(criterion, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss)) and criterion.weight is not None:
        weights = criterion.weight.to(targets.device)
        keep = targets != criterion.ignore_index
        return float(weights[targets[keep]].sum().item())
    return float(len(targets))


def normalize_gradients(optimizer, denominator):
    if denominator <= 0:
        raise ValueError("A gradient accumulation group must have positive loss weight")
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.div_(denominator)


def result_mean(results, name):
    values = results[name]
    if name.endswith("_loss") and getattr(results, "loss_reduction", None) == "sum":
        return float(np.sum(values))
    weights = getattr(results, "loss_weights" if name.endswith("_loss") else "batch_sizes", [])
    if len(weights) == len(values) and len(values):
        return float(np.average(values, weights=weights))
    return float(np.mean(values))


def renamed_results(results, old="train_", new="valid_"):
    renamed = BatchResults(loss_reduction=getattr(results, "loss_reduction", None))
    renamed.update({key.replace(old, new, 1): value for key, value in results.items()})
    renamed.batch_sizes = list(getattr(results, "batch_sizes", []))
    renamed.loss_weights = list(getattr(results, "loss_weights", []))
    return renamed


def best_metric_index(values, direction="min"):
    if direction not in {"min", "max"}:
        raise ValueError("training_metric_direction must be 'min' or 'max'")
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).any():
        raise ValueError("The selected training metric has no finite observations")
    array = np.where(np.isfinite(array), array, np.inf if direction == "min" else -np.inf)
    return int(np.argmin(array) if direction == "min" else np.argmax(array))


def distributed_enabled(conf):
    """Permit the supported single-process path, with explicit multi-rank limits.

    Global sample gathering, sampler-padding deduplication, weighted gradients
    and stopping decisions are not implemented for multi-rank trainers.
    Reject that mode rather than report rank-local values as global results.
    """
    if conf.get("trainer", {}).get("mode", "none") not in {"ddp", "fsdp"}:
        return False
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed training requires an initialized process group")
    if dist.get_world_size() != 1:
        raise NotImplementedError("Multi-rank Trainer execution is not supported: global metrics and weighted gradient aggregation require validation")
    return True


def prepare_model_device(model, device=None):
    """Respect an initialized FSDP wrapper's compute device and CPU offload."""
    parameter = next(model.parameters(), None)
    inferred = parameter.device if parameter is not None else torch.device("cpu")
    current, fsdp = model, False
    while True:
        if hasattr(current, "_fsdp_wrapped_module"):
            fsdp = True
            inferred = getattr(current, "compute_device", inferred)
        if not hasattr(current, "module"):
            break
        current = current.module
    selected = torch.device(device) if device is not None else torch.device(inferred)
    # Moving initialized FSDP modules bypasses their sharding/offload contract.
    if not fsdp:
        model.to(selected)
    return selected
