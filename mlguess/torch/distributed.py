
import torch
import os

from torch.nn.parallel import DistributedDataParallel as DDP
from mlguess.torch.mixed_precision import parse_dtype
import functools
import logging


def distributed_model_wrapper(conf, neural_network, device):
    """Wraps a neural network model in a distributed training wrapper (FSDP or DDP) based on configuration.

    Args:
        conf (dict): Configuration dictionary specifying the training setup, including the model type,
                     and training options such as mixed precision, CPU offloading, and activation checkpointing.
        neural_network (torch.nn.Module): The neural network model to be wrapped.
        device (torch.device): The device to which the model will be moved, usually a CUDA device.

    Returns:
        torch.nn.Module: The distributed model wrapped according to the configuration.
    """
    # convert $USER to the actual user name
    if 'save_loc' in conf:
        conf['save_loc'] = os.path.expanduser(os.path.expandvars(os.fspath(conf['save_loc'])))

    # FSDP polices
    if conf["trainer"]["mode"] == "fsdp":
        if torch.device(device).type == "cpu":
            raise ValueError("FSDP requires an accelerator; use mode=none or CPU DDP for CPU training.")

        from torch.distributed.fsdp import MixedPrecision, CPUOffload
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
        )
        from mlguess.torch.checkpoint import TorchFSDPModel

        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100_000
        )

        # Mixed precision

        use_mixed_precision = conf["trainer"]["use_mixed_precision"] if "use_mixed_precision" in conf["trainer"] else False

        logging.info(f"Using mixed_precision: {use_mixed_precision}")

        if use_mixed_precision:
            mixed_precision_options = {
                key: parse_dtype(val) if isinstance(val, str) else val
                for key, val in conf["trainer"]["mixed_precision"].items()
            }
            mixed_precision_policy = MixedPrecision(**mixed_precision_options)
        else:
            mixed_precision_policy = None

        # CPU offloading

        cpu_offload = conf["trainer"]["cpu_offload"] if "cpu_offload" in conf["trainer"] else False

        logging.info(f"Using CPU offloading: {cpu_offload}")

        # FSDP module

        model = TorchFSDPModel(
            neural_network,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=cpu_offload)
        )

        # activation checkpointing on the transformer blocks

        activation_checkpoint = conf["trainer"]["activation_checkpoint"] if "activation_checkpoint" in conf["trainer"] else False

        logging.info(f"Activation checkpointing: {activation_checkpoint}")

        if activation_checkpoint:

            # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/

            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

            # The size-based wrap policy is a callable, not an iterable of classes.
            check_fn = lambda submodule: size_based_auto_wrap_policy(
                submodule, recurse=False,
                nonwrapped_numel=sum(p.numel() for p in submodule.parameters()),
                min_num_params=100_000,
            )

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn
            )

        # attempting to get around the launch issue we are having
        torch.distributed.barrier()

    elif conf["trainer"]["mode"] == "ddp":
        device = torch.device(device)
        device_ids = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else None
        model = DDP(neural_network, device_ids=device_ids)
    else:
        model = neural_network

    return model
