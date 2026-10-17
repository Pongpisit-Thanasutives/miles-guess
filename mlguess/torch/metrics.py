import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class MetricsCalculator:
    """A class to calculate various metrics for model evaluation, including CSI, average accuracy,
    precision, recall, F1 score, AUC, MCE, and ECE.

    Args:
        n_bins (int, optional): Number of bins for MCE and ECE calculations. Default is 10.
        use_uncertainty (bool, optional): If True, the model's uncertainty is used in predictions. Default is False.

    Methods:
        __call__(y_true, y_pred, split="train"):
            Computes and returns a dictionary of metrics based on the true and predicted values.

        mean_csi(y, pred_probs):
            Computes the mean Critical Success Index (CSI) for the predicted probabilities.

        ave_acc(true_labels, pred_labels):
            Computes the average accuracy for the true and predicted labels.

        mce(true_labels, pred_probs):
            Computes the Maximum Calibration Error (MCE) for the predicted probabilities.

        ece(true_labels, pred_probs):
            Computes the Expected Calibration Error (ECE) for the predicted probabilities.
    """

    def __init__(self, n_bins=10, use_uncertainty=False):
        """Initializes the MetricsCalculator with the specified number of bins and uncertainty flag.

        Args:
            n_bins (int, optional): Number of bins for MCE and ECE calculations. Default is 10.
            use_uncertainty (bool, optional): If True, the model's uncertainty is used in predictions. Default is False.
        """
        if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        self.use_uncertainty = use_uncertainty
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __call__(self, y_true, y_pred, split="train"):
        """Computes various metrics based on the true and predicted values.

        Args:
            y_true (torch.Tensor): Tensor of true labels (one-hot encoded).
            y_pred (torch.Tensor): Tensor of predicted probabilities.
            split (str, optional): Indicates which data split is being evaluated (e.g., "train", "val"). Default is "train".

        Returns:
            dict: A dictionary containing the computed metrics, including CSI, average accuracy, precision, recall,
                  F1 score, and AUC.
        """
        logs = {}

        y_true = y_true.detach().cpu().numpy() if hasattr(y_true, "detach") else np.asarray(y_true)
        y_pred = y_pred.detach().cpu().numpy() if hasattr(y_pred, "detach") else np.asarray(y_pred)
        self._validate_probabilities(y_pred)
        if y_true.shape != y_pred.shape or not np.all((y_true == 0) | (y_true == 1)) or not np.all(y_true.sum(axis=1) == 1):
            raise ValueError("Targets must be one-hot encoded with the same shape as probabilities")

        true_labels = np.argmax(y_true, axis=1)
        pred_probs = y_pred if self.use_uncertainty else y_pred
        pred_labels = np.argmax(pred_probs, axis=1)

        logs[f"{split}_csi"] = self.mean_csi(y_true, pred_probs)
        prec, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0
        )
        logs[f"{split}_ave_acc"] = self.ave_acc(true_labels, pred_labels)
        logs[f"{split}_prec"] = prec
        logs[f"{split}_recall"] = recall
        logs[f"{split}_f1"] = f1
        try:
            logs[f"{split}_auc"] = roc_auc_score(y_true, pred_probs, multi_class='ovr')
        except ValueError:
            # AUC computation might fail if there's only one class in either true labels or predicted labels
            logs[f"{split}_auc"] = np.nan

        return logs

    def mean_csi(self, y, pred_probs):
        """Computes the mean Critical Success Index (CSI) for the predicted probabilities.

        Args:
            y (numpy.ndarray): Array of true labels (one-hot encoded).
            pred_probs (numpy.ndarray): Array of predicted probabilities.

        Returns:
            float: The mean Critical Success Index.
        """
        # Same threshold grid and TP/(TP+FP+FN) statistic as DistributedROC.
        # Each one-vs-rest event must use its own class probability, not the
        # maximum probability assigned to any class.
        pred_probs = np.asarray(pred_probs)
        y = np.asarray(y)
        self._validate_probabilities(pred_probs)
        if y.shape != pred_probs.shape or not np.all((y == 0) | (y == 1)) or not np.all(y.sum(axis=1) == 1):
            raise ValueError("Targets must be one-hot encoded with the same shape as probabilities")
        true_labels = np.argmax(y, axis=1)
        thresholds = np.arange(0.0, 1.01, 0.01)
        scores = []
        for i in range(pred_probs.shape[1]):
            positive = true_labels == i
            forecasts = np.asarray(pred_probs)[:, i]
            class_scores = []
            for threshold in thresholds:
                predicted = forecasts >= threshold
                tp = np.count_nonzero(predicted & positive)
                fp = np.count_nonzero(predicted & ~positive)
                fn = np.count_nonzero((forecasts < threshold) & positive)
                denominator = tp + fp + fn
                if denominator:
                    class_scores.append(tp / denominator)
            scores.append(max(class_scores) if class_scores else np.nan)
        return np.mean(scores)

    def ave_acc(self, true_labels, pred_labels):
        """Computes the average accuracy for the true and predicted labels.

        Args:
            true_labels (numpy.ndarray): Array of true labels.
            pred_labels (numpy.ndarray): Array of predicted labels.

        Returns:
            float: The average accuracy.
        """
        true_labels, pred_labels = np.asarray(true_labels), np.asarray(pred_labels)
        if true_labels.ndim != 1 or true_labels.size == 0 or pred_labels.shape != true_labels.shape:
            raise ValueError("Provide matching nonempty label vectors")
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
        """Computes the Maximum Calibration Error (MCE) for the predicted probabilities.

        Args:
            true_labels (numpy.ndarray): Array of true labels.
            pred_probs (numpy.ndarray): Array of predicted probabilities.

        Returns:
            float: The Maximum Calibration Error.
        """
        errors, _ = self._calibration_bins(true_labels, pred_probs)
        return float(np.max(errors)) if errors else 0.0

    def ece(self, true_labels, pred_probs):
        """Computes the Expected Calibration Error (ECE) for the predicted probabilities.

        Args:
            true_labels (numpy.ndarray): Array of true labels.
            pred_probs (numpy.ndarray): Array of predicted probabilities.

        Returns:
            float: The Expected Calibration Error.
        """
        errors, weights = self._calibration_bins(true_labels, pred_probs)
        return float(np.dot(errors, weights))

    def _calibration_bins(self, true_labels, pred_probs):
        """Return absolute mean calibration gaps and fractions in nonempty bins."""
        pred_probs = np.asarray(pred_probs)
        self._validate_probabilities(pred_probs)
        labels = np.asarray(true_labels)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1 or len(labels) != len(pred_probs) or not np.isin(labels, np.arange(pred_probs.shape[1])).all():
            raise ValueError("Expected class probabilities (samples, classes) and one label per sample")
        confidences = np.max(pred_probs, axis=1)
        accuracies = np.argmax(pred_probs, axis=1) == labels
        errors, weights = [], []
        for index, (lower, upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = (confidences <= upper) & (
                (confidences >= lower) if index == 0 else (confidences > lower)
            )
            if np.any(in_bin):
                errors.append(abs(confidences[in_bin].mean() - accuracies[in_bin].mean()))
                weights.append(in_bin.mean())
        return errors, weights

    @staticmethod
    def _validate_probabilities(probabilities):
        if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
            raise ValueError("Provide a nonempty (samples, classes) probability matrix")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)) or not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-5, atol=1e-7):
            raise ValueError("Predictions must be finite nonnegative probabilities summing to one")
