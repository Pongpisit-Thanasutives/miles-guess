import os
import sys
import copy
import yaml
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.utils.data.distributed import DistributedSampler

from mlguess.torch.distributed import distributed_model_wrapper
from mlguess.torch.scheduler import load_scheduler
from mlguess.torch.trainer_classifier import Trainer
from mlguess.torch.class_losses import EDLDigammaLoss
from mlguess.torch.models import seed_everything, CategoricalDNN
from mlguess.torch.metrics import MetricsCalculator
from torch.utils.data import TensorDataset

if __package__:
    from ._common import (
        resolve_device,
        validate_application_run,
        setup,
        import_class_from_path,
        cleanup_distributed,
        make_grad_scaler,
        load_objective,
    )
else:
    from _common import (
        resolve_device,
        validate_application_run,
        setup,
        import_class_from_path,
        cleanup_distributed,
        make_grad_scaler,
        load_objective,
    )


def load_dataset_and_sampler(conf, world_size, rank, split, seed=42, return_df=False):
    """
    Load a dataset and its corresponding distributed sampler based on the configuration.

    Args:
        conf (dict): Configuration dictionary specifying dataset and sampler settings.
        world_size (int): The total number of processes involved in distributed training.
        rank (int): The rank of the current process.
        is_train (bool): Whether to load the training or validation dataset.
        seed (int, optional): Random seed for sampling. Defaults to 42.

    Returns:
        tuple: A tuple containing the dataset and the distributed sampler.

    Logs:
        Information about the dataset and sampler loaded.
    """

    from mlguess.keras.data import load_ptype_uq, preprocess_data

    # Z-score
    # torch_dataset = import_class_from_path(conf["data"]["dataset_name"], conf["data"]["dataset_path"])
    # dataset = torch_dataset(conf, split='train' if is_train else 'val')

    input_features = []
    for features in ["TEMP_C", "T_DEWPOINT_C", "UGRD_m/s", "VGRD_m/s"]:
        input_features += conf["data"][features]
    output_features = conf["data"]["ptypes"]
    is_train = split == "train"

    # Load data
    _conf = copy.deepcopy(conf)
    _conf.update(conf["data"])
    data = load_ptype_uq(_conf, data_split=_conf.get("data_split", 0), verbose=1, drop_mixed=False)

    # check if we should scale the input data by groups
    scale_groups = [] if "scale_groups" not in conf["data"] else conf["data"]["scale_groups"]
    groups = [list(conf["data"][g]) for g in scale_groups]
    grouped_features = {feature for group in groups for feature in group}
    leftovers = [feature for feature in input_features if feature not in grouped_features]
    if groups and leftovers:
        groups.append(leftovers)
    # scale the data
    scaled_data, scalers = preprocess_data(
        data,
        input_features,
        output_features,
        scaler_type=conf["data"]["scaler_type"],
        encoder_type="onehot",
        groups=groups,
        seed=conf.get("seed", seed),
    )

    X = torch.FloatTensor(scaled_data[f"{split}_x"].values)
    y = torch.LongTensor(np.argmax(scaled_data[f"{split}_y"], axis=-1))

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)

    # Pytorch sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=is_train,
        drop_last=False
    )
    logging.info(f"Loaded a {split} torch dataset, and a distributed sampler")

    if return_df:
        return dataset, sampler, data

    return dataset, sampler


def load_model_states_and_optimizer(conf, model, device):
    """
    Load the model, optimizer, scheduler, and scaler states from checkpoint files based on the configuration.

    Args:
        conf (dict): Configuration dictionary specifying the checkpoint paths and training hyperparameters.
        model (torch.nn.Module): The model to load states into.
        device (torch.device): The device to map the checkpoint to.

    Returns:
        tuple: A tuple containing the model, optimizer, scheduler, and scaler.
    """

    # convert $USER to the actual user name
    conf['save_loc'] = save_loc = os.path.expanduser(os.path.expandvars(conf['save_loc']))

    if conf["trainer"]["mode"] == "fsdp":
        from mlguess.torch.checkpoint import FSDPOptimizerWrapper, TorchFSDPCheckpointIO

    # training hyperparameters
    start_epoch = conf['trainer']['start_epoch']
    learning_rate = float(conf['trainer']['learning_rate'])
    weight_decay = float(conf['trainer']['weight_decay'])

    # load weights falg
    load_weights = False if 'load_weights' not in conf['trainer'] else conf['trainer']['load_weights']

    #  Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    if start_epoch == 0 and not load_weights:  # Loaded after loading model weights when reloading
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = FSDPOptimizerWrapper(optimizer, model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = make_grad_scaler(conf, device)

    # load optimizer and grad scaler states
    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device, weights_only=True)

        # FSDP checkpoint settings
        if conf["trainer"]["mode"] == "fsdp":
            logging.info(f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            optimizer = FSDPOptimizerWrapper(optimizer, model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
            if 'load_optimizer' in conf['trainer'] and conf['trainer']['load_optimizer']:
                checkpoint_io.load_unsharded_optimizer(optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt"))

        else:
            # DDP settings
            if conf["trainer"]["mode"] == "ddp":
                logging.info(f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                logging.info(f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            if 'load_optimizer' in conf['trainer'] and conf['trainer']['load_optimizer']:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = load_scheduler(optimizer, conf)
        scaler = make_grad_scaler(conf, device)

        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler.is_enabled() and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Enable updating the lr if not using a policy
    if (conf["trainer"]["update_learning_rate"] if "update_learning_rate" in conf["trainer"] else False):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    return model, optimizer, scheduler, scaler


def main(rank, world_size, conf, trial=False):
    """
    Main function to set up and run the training and validation process.

    Args:
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes involved in distributed training.
        conf (dict): Configuration dictionary containing settings for the model, training, and data.
        trial (bool, optional): Flag to indicate if the function is being used in a hyperparameter tuning trial. Defaults to False.

    Returns:
        result: The result of the training process.

    Logs:
        Information about setup, dataset loading, model wrapping, and training process.
    """

    validate_application_run(conf, world_size)

    # convert $USER to the actual user name
    conf['save_loc'] = os.path.expanduser(os.path.expandvars(conf['save_loc']))
    # Assuming `conf` is a dictionary containing your configuration
    trainer_config = conf.get('trainer', {})
    # Get KL loss coefficient with a default value of 10 if not present
    kl_loss_coefficient = trainer_config.get('kl_loss_coefficient', 10)
    # Get uncertainty with a default value of False if not present
    uncertainty = trainer_config.get('uncertainty', False)
    # Number of class labels present
    outputs = conf["model"]["output_size"]
    num_classes = len(outputs) if isinstance(outputs, (list, tuple)) else outputs

    conf["trainer"].setdefault("mode", "none")
    device = resolve_device(conf, rank)
    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"], device)

    # Config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    train_batch_size = conf['trainer']['train_batch_size']
    valid_batch_size = conf['trainer']['valid_batch_size']
    thread_workers = conf['trainer'].get('thread_workers', 0)
    valid_thread_workers = conf['trainer']['valid_thread_workers'] if 'valid_thread_workers' in conf['trainer'] else thread_workers

    # load dataset and sampler

    train_dataset, train_sampler, data = load_dataset_and_sampler(conf, world_size, rank, split="train", return_df=True)
    valid_dataset, valid_sampler = load_dataset_and_sampler(conf, world_size, rank, split="val")
    test_dataset, test_sampler = load_dataset_and_sampler(conf, world_size, rank, split="test")

    # setup the dataloder for this process

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,  # sampler handles shuffling
        sampler=train_sampler,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True if thread_workers > 0 else False,
        num_workers=thread_workers,
        drop_last=(len(train_sampler) >= train_batch_size)
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=(device.type == "cuda"),
        num_workers=valid_thread_workers,
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=(device.type == "cuda"),
        num_workers=valid_thread_workers,
        drop_last=False
    )

    # model

    m = CategoricalDNN(**conf["model"])

    # have to send the module to the correct device first

    m.to(device)

    # Wrap in DDP or FSDP module, or none

    model = distributed_model_wrapper(conf, m, device)

    # Load model weights (if any), an optimizer, scheduler, and gradient scaler

    model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

    # Train and validation losses

    if uncertainty:
        train_criterion = EDLDigammaLoss(num_classes, kl_loss_coefficient)
        valid_criterion = EDLDigammaLoss(num_classes, kl_loss_coefficient)
    else:
        train_criterion = torch.nn.CrossEntropyLoss()
        valid_criterion = torch.nn.CrossEntropyLoss()

    # Initialize a trainer object

    trainer = Trainer(
        model,
        rank,
        module=(conf["trainer"]["mode"] == "ddp"),
        uncertainty=uncertainty,
        device=device,
    )

    # Fit the model
    classifier_metrics = MetricsCalculator(use_uncertainty=uncertainty)

    fit_result = trainer.fit(
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        classifier_metrics,
        trial=trial
    )

    if trial:
        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup_distributed()
        return fit_result

    # Predict with the model
    logging.info("Predicting on the data splits with the trained model. This may take some time")

    # Predict in the original DataFrame order, without the training shuffle.
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,  # sampler handles shuffling
        sampler=train_sampler,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True if thread_workers > 0 else False,
        num_workers=thread_workers,
        drop_last=False
    )

    splits = ["train", "val", "test"]
    loaders = [train_loader, valid_loader, test_loader]
    all_results = defaultdict(dict)

    for split, dataloader in zip(splits, loaders):

        result = trainer.predict(
            conf,
            dataloader,
            valid_criterion,
            classifier_metrics,
            split
        )

        metrics = result["metrics"]
        mu = result["mu"]
        u = result["dempster-shafer"]
        aleatoric = result["aleatoric"]
        epistemic = result["epistemic"]
        total = result["total"]

        # Add predictions back to the DataFrame
        df = data[split].copy()

        df[[f"y_pred_prob_{label}" for label in range(num_classes)]] = mu
        if uncertainty:
            df["u"] = u
            df[[f"aleatorc_{label}" for label in range(num_classes)]] = aleatoric
            df[[f"epistemic_{label}" for label in range(num_classes)]] = epistemic
            df[[f"total_{label}" for label in range(num_classes)]] = total

        for key, values in metrics.items():
            if isinstance(values, list):
                values = values[0]
            all_results[key][split] = values

        df.to_csv(os.path.join(conf['save_loc'], f"{split}.csv"))

    mets = pd.DataFrame.from_dict(all_results, orient='index')
    mets.to_csv(os.path.join(conf['save_loc'], "metrics.csv"))

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        cleanup_distributed()
    return result


class _ObjectiveMixin:
    """
    Objective class for hyperparameter tuning with Optuna.

    Args:
        config (dict): Configuration dictionary for the objective.
        metric (str, optional): The metric to optimize. Defaults to "val_loss".
        device (str, optional): The device to use. Defaults to "cpu".
    """

    def train(self, trial, conf):
        """
        Train the model with the given trial configuration.

        Args:
            trial (optuna.Trial): The Optuna trial object for hyperparameter tuning.
            conf (dict): Configuration dictionary for the model and training.

        Returns:
            float: The result of the training process.

        Raises:
            optuna.TrialPruned: If the trial is to be pruned due to various reasons.
        """

        conf['trainer']['train_batch_size'] = conf['data']['batch_size']
        conf['trainer']['valid_batch_size'] = conf['data']['batch_size']

        try:
            return main(0, 1, conf, trial=trial)

        except Exception as E:
            if "CUDA" in str(E):
                import optuna
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}."
                )
                raise optuna.TrialPruned()
            elif "non-singleton" in str(E):
                import optuna
                logging.warning(
                    f"Pruning trial {trial.number} due to shape mismatch: {str(E)}."
                )
                raise optuna.TrialPruned()
            elif "infinity" in str(E).lower() or "nan" in str(E).lower():
                import optuna
                logging.warning(
                    f"Pruning trial {trial.number} due to encountering a nan in metrics calculations: {str(E)}."
                )
                raise optuna.TrialPruned()
            else:
                logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E


def __getattr__(name):
    return load_objective(name, globals(), _ObjectiveMixin)


if __name__ == "__main__":

    description = "Train a PyTorch classifier"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        dest="model_config",
        type=str,
        required=True,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=None,
        choices=["none", "ddp", "fsdp"],
        help="Update the config to use none, DDP, or FSDP",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], help="Override trainer.device (CPU is supported on macOS and Windows).")
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = args_dict.pop("mode")
    device_override = args_dict.pop("device")

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.safe_load(cf)

    # Create directories if they do not exist and copy yml file
    save_loc = os.path.expanduser(os.path.expandvars(conf["save_loc"]))
    os.makedirs(save_loc, exist_ok=True)

    # Update config using override options
    conf["trainer"].setdefault("mode", "none")
    if device_override is not None:
        conf["trainer"]["device"] = device_override
    if mode and mode in ["none", "ddp", "fsdp"]:
        logging.info(f"Setting the running mode to {mode}")
        conf["trainer"]["mode"] = mode

    # Record effective overrides, so checkpoint reload uses the same mode/device.
    effective_config = os.path.join(save_loc, "model.yml")
    with open(effective_config, "w", encoding="utf-8") as stream:
        yaml.safe_dump(conf, stream, sort_keys=False)
    if launch:
        config = effective_config

    # Launch PBS jobs
    if launch:
        from mlguess.pbs import launch_pbs_jobs, launch_distributed_jobs
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf['pbs']['queue'] == 'casper':
            logging.info("Launching to PBS on Casper")
            launch_pbs_jobs(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_distributed_jobs(config, script_path)
        sys.exit()

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        main(0, 1, conf)
