from keras.callbacks import (
    Callback,
    ModelCheckpoint,
    CSVLogger,
    EarlyStopping,
)
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from typing import List, Dict
import logging
import copy
import inspect
from functools import partial
import math
import os
import keras
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

logger = logging.getLogger(__name__)


def get_callbacks(config: Dict[str, str], path_extend=False) -> List[Callback]:
    callbacks = []
    config = copy.deepcopy(config)

    if "callbacks" in config:
        if path_extend:
            save_data = os.path.join(config["save_loc"], path_extend)
        else:
            save_data = config["save_loc"]
        os.makedirs(save_data, exist_ok=True)
        config = config["callbacks"]
    else:
        return []

    if "ModelCheckpoint" in config:
        config["ModelCheckpoint"]["filepath"] = os.path.join(
            save_data, config["ModelCheckpoint"]["filepath"]
        )
        callbacks.append(ModelCheckpoint(**config["ModelCheckpoint"]))
        logger.info("... loaded Checkpointer")

    if "EarlyStopping" in config:
        callbacks.append(EarlyStopping(**config["EarlyStopping"]))
        logger.info("... loaded EarlyStopping")

    # LearningRateTracker(),  ## ReduceLROnPlateau does this already, use when supplying custom LR annealer

    if "ReduceLROnPlateau" in config:
        callbacks.append(ReduceLROnPlateau(**config["ReduceLROnPlateau"]))
        logger.info("... loaded ReduceLROnPlateau")

    if "CSVLogger" in config:
        config["CSVLogger"]["filename"] = os.path.join(
            save_data, config["CSVLogger"]["filename"]
        )
        callbacks.append(CSVLogger(**config["CSVLogger"]))
        logger.info("... loaded CSVLogger")

    if "LearningRateScheduler" in config:
        drop = config["LearningRateScheduler"]["drop"]
        epochs_drop = config["LearningRateScheduler"]["epochs_drop"]
        f = partial(step_decay, drop=drop, epochs_drop=epochs_drop,
                    init_lr=config["LearningRateScheduler"].get("init_lr", 0.001))
        callbacks.append(LearningRateScheduler(f))
        callbacks.append(LearningRateTracker())

    return callbacks


def step_decay(epoch, drop=0.2, epochs_drop=5.0, init_lr=0.001):
    if epochs_drop <= 0:
        raise ValueError("epochs_drop must be positive")
    lrate = init_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        if logs is None:
            logs = {}
        logs["lr"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))

@keras.saving.register_keras_serializable()
class ReportEpoch(keras.callbacks.Callback):
    def __init__(self, epoch_var=None, initial_epoch=1.0):
        super().__init__()
        self.epoch_var = keras.Variable(initial_epoch, trainable=False) if epoch_var is None else epoch_var

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_var.assign_add(1)

    def get_config(self):

        return {"initial_epoch": float(keras.ops.convert_to_numpy(self.epoch_var))}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, x, y, name="val", n_bins=10, use_uncertainty=False, **kwargs):
        super().__init__()
        self.x = x
        self.y = y
        self.name = name
        self.use_uncertainty = use_uncertainty

        if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        options = {"verbose": 0}
        if "return_uncertainties" in inspect.signature(self.model.predict).parameters:
            options["return_uncertainties"] = self.use_uncertainty
        output = self.model.predict(self.x, **options)
        if self.use_uncertainty:
            if isinstance(output, (tuple, list)) and len(output) == 4:
                output = output[0]
            else:
                alpha = np.maximum(keras.ops.convert_to_numpy(output), 0) + 1
                output = alpha / alpha.sum(axis=-1, keepdims=True)
        pred_probs = keras.ops.convert_to_numpy(output)
        logs[f"{self.name}_csi"] = self.mean_csi(pred_probs)
        true_labels = np.argmax(self.y, 1)
        pred_labels = np.argmax(pred_probs, 1)
        prec, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0)
        logs[f"{self.name}_ave_acc"] = self.ave_acc(true_labels, pred_labels)
        logs[f"{self.name}_prec"] = prec
        logs[f"{self.name}_recall"] = recall
        logs[f"{self.name}_f1"] = f1
        try:
            logs[f"{self.name}_auc"] = roc_auc_score(self.y, pred_probs, multi_class="ovr")
        except ValueError:
            logs[f"{self.name}_auc"] = np.nan

    def mean_csi(self, pred_probs):
        scores = []
        labels = np.argmax(self.y, axis=1)
        for index in range(pred_probs.shape[1]):
            observed = labels == index
            class_scores = []
            for threshold in np.arange(0, 1.01, .01):
                forecast = pred_probs[:, index] >= threshold
                denominator = np.count_nonzero(forecast | observed)
                if denominator:
                    class_scores.append(np.count_nonzero(forecast & observed) / denominator)
            scores.append(max(class_scores) if class_scores else np.nan)
        return float(np.mean(scores))

    def ave_acc(self, true_labels, pred_labels):
        return np.mean(
            [
                (
                        true_labels[np.where(true_labels == _label)]
                        == pred_labels[np.where(true_labels == _label)]
                ).mean()
                for _label in np.unique(true_labels)
            ]
        )

    def mce(self, true_labels, pred_probs):
        gaps, _ = self._calibration_bins(true_labels, pred_probs)
        return max(gaps, default=0.0)

    def ece(self, true_labels, pred_probs):
        gaps, proportions = self._calibration_bins(true_labels, pred_probs)
        return float(np.dot(gaps, proportions))

    def _calibration_bins(self, true_labels, pred_probs):
        probabilities = np.asarray(pred_probs)
        labels = np.asarray(true_labels).reshape(-1)
        if probabilities.ndim != 2 or len(labels) != len(probabilities):
            raise ValueError("Expected (samples, classes) probabilities and one label per sample")
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == labels
        gaps, proportions = [], []
        for index, (lower, upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            mask = (confidence <= upper) & ((confidence >= lower) if index == 0 else (confidence > lower))
            if np.any(mask):
                gaps.append(float(abs(confidence[mask].mean() - correct[mask].mean())))
                proportions.append(float(mask.mean()))
        return gaps, proportions
