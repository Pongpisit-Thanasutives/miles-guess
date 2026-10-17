import os
import sys
import glob
import keras
import keras.ops as ops
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.regularizers import L1, L2, L1L2
from keras.layers import Dense, LeakyReLU, GaussianNoise, Dropout
from keras.optimizers import Adam, SGD
from mlguess.keras.layers import DenseNormalGamma, DenseNormal
from mlguess.keras.deprecated.losses import EvidentialRegressionLoss, EvidentialRegressionCoupledLoss, gaussian_nll_per_sample
from mlguess.keras.deprecated.losses import DirichletEvidentialLoss
from mlguess.keras.callbacks import ReportEpoch
from collections import defaultdict
import logging
import re
import warnings
from mlguess.uncertainty import affine_inverse_scale, inverse_scale_moments, nig_uncertainties
from mlguess.keras.monte_carlo import mc_raw_predictions, categorical_mc_moments

class BaseRegressor(object):
    """A base class for regression models.

    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object
        use_noise: Whether additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        use_dropout: Whether Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """

    _uncertainty_kind = "deterministic"

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        optimizer="adam",
        loss="mse",
        loss_weights=None,
        use_noise=False,
        noise_sd=0.01,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.1,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.01,
        l2_weight=0.01,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        verbose=0,
        save_path=".",
        model_name="model.h5",
        metrics=None,
        eps=1e-7
    ):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.loss = loss
        self.loss_weights = loss_weights
        self.lr = lr
        self.kernel_reg = kernel_reg
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.verbose = verbose
        self.save_path = save_path
        self.model_name = model_name
        self.model = None
        self.optimizer_obj = None
        self.training_std = None
        self.training_var = []
        self.metrics = metrics
        self.eps = eps
        self.ensemble_member_files = []
        self.ensemble_member_scalers = None
        self.ensemble_member_input_scalers = None
        self.history = None

    def build_neural_network(self, inputs, outputs, last_layer="Dense"):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables.
            outputs (int): Number of output predictor variables.
        """
        nn_input = Input(shape=(inputs,), name="input")
        nn_model = nn_input

        if self.activation == "leaky":
            self.activation = LeakyReLU()
        if self.kernel_reg == "l1":
            self.kernel_reg = L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = keras.regularizers.get(self.kernel_reg)

        for h in range(self.hidden_layers):
            nn_model = Dense(
                self.hidden_neurons,
                activation=self.activation,
                kernel_regularizer=self.kernel_reg,
                name=f"dense_{h:02d}",
            )(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(
                    nn_model
                )
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(
                    nn_model
                )

        if last_layer == "Dense":
            nn_model = Dense(outputs, name="dense_last")(nn_model)
        elif last_layer == "DenseNormal":
            nn_model = DenseNormal(outputs, name="DenseNormal", eps=self.eps)(nn_model)
        elif last_layer == "DenseNormalGamma":
            nn_model = DenseNormalGamma(outputs, name="DenseNormalGamma", eps=self.eps)(
                nn_model
            )
        else:
            raise ValueError("Invalid last_layer type. Use 'Dense', 'DenseNormal', or 'DenseNormalGamma'.")

        self.model = Model(nn_input, nn_model)

        if self.optimizer == "adam":
            self.optimizer_obj = Adam(learning_rate=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        else:
            self.optimizer_obj = keras.optimizers.get(self.optimizer)

        if self.metrics == "mae":
            metrics = [self.mae]
        elif self.metrics == "mse":
            metrics = [self.mse]
        else:
            metrics = self.metrics

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=metrics,
            run_eagerly=False,
        )

    def build_from_sequential(self, model, optimizer="adam", loss="mse", metrics=None):
        """Build the neural network model using a Keras Sequential model.

        Args:
            model (tf.keras.Sequential): Keras Sequential model to use.
            optimizer (str or tf.keras.optimizers.Optimizer): Optimizer for the model.
            loss (str or tf.keras.losses.Loss): Loss function for the model.
            metrics (list of str or tf.keras.metrics.Metric): Metrics for the model.
        """
        self.model = model

        if optimizer == "adam":
            self.optimizer_obj = Adam(learning_rate=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2)
        elif optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        else:
            self.optimizer_obj = keras.optimizers.get(optimizer)

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=loss,
            metrics=metrics
        )

    def fit(
        self,
        x,
        y,
        validation_data=None,
        callbacks=None,
        initial_epoch=0,
        steps_per_epoch=None,
        workers=0,
        use_multiprocessing=False,
        shuffle=True,
        **kwargs,
    ):
        """Fit the regression model.

        Args:
            x: Input data
            y: Target data
            validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch
            callbacks: List of callbacks to apply during training
            initial_epoch: Epoch at which to start training (useful for resuming a previous training run)
            steps_per_epoch: Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
            workers: Number of workers to use for data loading
            use_multiprocessing: If True, use ProcessPoolExecutor to load data, which is faster but can cause issues with certain GPU setups. If False, use a ThreadPoolExecutor.
            **kwargs: Additional arguments to be passed to the `fit` method
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_neural_network first.")
        if self.verbose:
            self.model.summary()
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        if validation_data is not None and isinstance(validation_data, tuple):
            validation_y = np.asarray(validation_data[1])
            if validation_y.ndim == 1:
                validation_data = (validation_data[0], validation_y[:, None], *validation_data[2:])
        self.training_var = np.var(y, axis=0)
        if workers or use_multiprocessing:
            warnings.warn("Keras 3 configures workers on a PyDataset, not Model.fit; legacy worker arguments are ignored", DeprecationWarning, stacklevel=2)
        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("epochs", self.epochs)
        kwargs.setdefault("verbose", self.verbose)
        self.history = self.model.fit(
            x,
            y,
            callbacks=callbacks,
            validation_data=validation_data,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            **kwargs,
        )
        return self.history

    def save_model(self):
        """Save the trained model to a file.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        model_path = os.path.join(self.save_path, self.model_name)
        keras.models.save_model(
            self.model, model_path
        )
        logging.info(f"Saved model to {model_path}")

        # Save the training variances
        np.savetxt(
            os.path.join(self.save_path, os.path.splitext(self.model_name)[0] + "_training_var.txt"),
            np.array(self.training_var),
        )

    @classmethod
    def load_model(cls, conf):
        """Load a trained model using args from a configuration
        """
        # Check if weights file exists
        weights = os.path.join(conf["model"]["save_path"], "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                f"No saved model exists at {weights}. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading a DNN with pre-trained weights from path {weights}"
        )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)

        # Load ensemble weights
        save_loc = conf["save_loc"]
        n_models = conf["ensemble"]["n_models"]
        n_splits = conf["ensemble"]["n_splits"]
        if n_splits > 1 and n_models == 1:
            mode = "cv_ensemble"
        elif n_splits == 1 and n_models > 1:
            mode = "deep_ensemble"
        elif n_splits == 1 and n_models == 1:
            mode = "single"
        elif n_splits > 1 and n_models > 1:
            mode = "multi_ensemble"
        else:
            raise ValueError(
                "Incorrect selection of n_splits or n_models. Both must be at greater than or equal to 1."
            )

        model_class.ensemble_member_files = []
        if mode != "single":
            for i in range(n_models):
                for j in range(n_splits):
                    model_class.ensemble_member_files.append(
                        os.path.join(save_loc, mode, "models", f"model_seed{i}_split{j}.h5")
                    )

        return model_class

    def mae(self, y_true, y_pred):
        """MAE of the predictive mean, including multiple target columns."""
        parts = {"deterministic": 1, "gaussian": 2, "evidential": 4}[self._uncertainty_kind]
        mu = ops.split(y_pred, parts, axis=-1)[0]
        return keras.metrics.mean_absolute_error(y_true, mu)

    def mse(self, y_true, y_pred):
        """MSE of the predictive mean, including multiple target columns."""
        parts = {"deterministic": 1, "gaussian": 2, "evidential": 4}[self._uncertainty_kind]
        mu = ops.split(y_pred, parts, axis=-1)[0]
        return keras.metrics.mean_squared_error(y_true, mu)

    def predict(self, x, scaler=None, batch_size=None):
        """Predict target values for input data.

        Args:
            x (numpy.ndarray): Input data.
            scaler (optional): Scaler object for preprocessing input data (default: None).
            batch_size (optional): Batch size for prediction (default: None).
            y_scaler (optional): Scaler object for post-processing predicted target values (default: None).

        Returns:
            numpy.ndarray: Predicted target values.
        """
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        if scaler:
            if len(y_out.shape) == 1:
                y_out = np.expand_dims(y_out, 1)
            y_out = scaler.inverse_transform(y_out)
        return y_out

    def predict_ensemble(self, x, batch_size=None, scaler=None, num_outputs=1, member_scalers=None, member_input_scalers=None):
        """Return individual ensemble moments with shape ``(members, samples, targets)``.

        Supply ``member_scalers`` (one fitted target scaler per weight file) when
        members used different target transforms. A shared ``scaler`` is valid
        only when all members used that same transform. Cross-validation files
        naming distinct ``_splitN`` values require explicit target AND input
        member scalers. With ``member_input_scalers``, x contains raw features;
        otherwise x must already be in the common feature space of all members.
        No scaler means that predictions already share the desired target units.
        Training-variance sidecars are retained as metadata, never guessed to be
        an inverse-transform scale. Original weights and metadata are restored.
        """
        expected_outputs = {"deterministic": 1, "gaussian": 2, "evidential": 3}[self._uncertainty_kind]
        if num_outputs != expected_outputs:
            raise ValueError(f"num_outputs must be {expected_outputs} for {self._uncertainty_kind} predictions")
        files = self.ensemble_member_files
        count = len(files)
        if member_scalers is None:
            member_scalers = getattr(self, "ensemble_member_scalers", None)
        if member_scalers is not None and scaler is not None:
            raise ValueError("Supply a shared scaler or member_scalers, not both")
        splits = {match.group(1) for path in files
                  if (match := re.search(r"_split(\d+)", str(path))) is not None}
        if member_scalers is None:
            if len(splits) > 1:
                raise ValueError("Cross-validation ensemble members require their fitted target member_scalers")
            member_scalers = [scaler] * count
        elif len(member_scalers) != count:
            raise ValueError("member_scalers must contain one target scaler per ensemble weight file")
        if len(splits) > 1 and any(item is None or item is False for item in member_scalers):
            raise ValueError("A fitted target scaler is required for every cross-validation member")
        if member_input_scalers is None:
            member_input_scalers = getattr(self, "ensemble_member_input_scalers", None)
        if member_input_scalers is None:
            if len(splits) > 1:
                raise ValueError("Cross-validation ensemble members require their fitted member_input_scalers")
            member_input_scalers = [None] * count
        elif len(member_input_scalers) != count:
            raise ValueError("member_input_scalers must contain one input scaler per ensemble weight file")
        if len(splits) > 1 and any(item is None for item in member_input_scalers):
            raise ValueError("A fitted input scaler is required for every cross-validation member")
        if count == 0:
            parameter_count = 1 if num_outputs == 1 else (2 if num_outputs == 2 else 4)
            n_targets = self.model.output_shape[-1] // parameter_count
            empty = tuple(np.empty((0, len(x), n_targets)) for _ in range(num_outputs))
            return empty[0] if num_outputs == 1 else empty
        original_weights = self.model.get_weights()
        original_variance = self.training_var
        collected = [[] for _ in range(num_outputs)]
        try:
            for path, target_scaler, input_scaler in zip(files, member_scalers, member_input_scalers):
                self.model.load_weights(path)
                member_x = x if input_scaler is None else input_scaler.transform(x)
                variance_path = os.path.splitext(str(path))[0] + "_training_var.txt"
                self.training_var = np.atleast_1d(np.loadtxt(variance_path)) if os.path.isfile(variance_path) else []
                if num_outputs == 1:
                    components = (self.predict(member_x, batch_size=batch_size, scaler=target_scaler),)
                else:
                    components = self.predict_uncertainty(member_x, batch_size=batch_size, scaler=target_scaler)
                if len(components) != num_outputs:
                    raise ValueError("The uncertainty helper returned an unexpected number of components")
                for values, component in zip(collected, components):
                    values.append(component)
        finally:
            self.model.set_weights(original_weights)
            self.training_var = original_variance
        output = tuple(np.stack(values, axis=0) for values in collected)
        return output[0] if num_outputs == 1 else output

    def predict_monte_carlo(self, x_test, forward_passes, scaler=None, batch_size=None, num_outputs=1):
        """Return per-pass target moments; only dropout is stochastic.

        Gaussian and NIG parameter tensors are decoded before an affine inverse
        transform is applied once. Every returned array has shape
        ``(forward_passes, samples, targets)``, not the raw parameter width.
        Use ``monte_carlo_ensemble`` to aggregate these conditional moments.
        """
        expected_outputs = {"deterministic": 1, "gaussian": 2, "evidential": 3}[self._uncertainty_kind]
        if num_outputs != expected_outputs:
            raise ValueError(f"num_outputs must be {expected_outputs} for {self._uncertainty_kind} predictions")
        size = self.batch_size if batch_size is None else batch_size
        raw = mc_raw_predictions(self.model, x_test, forward_passes, size)
        outputs = [[] for _ in range(num_outputs)]
        for draw in raw:
            if num_outputs == 1:
                zeros = np.zeros_like(draw)
                mean, _, _ = inverse_scale_moments(draw, zeros, zeros, scaler)
                components = (mean,)
            else:
                components = self.calc_uncertainties(draw, scaler)
            if len(components) != num_outputs:
                raise ValueError("The uncertainty helper returned an unexpected number of components")
            for values, component in zip(outputs, components):
                values.append(component)
        result = tuple(np.stack(values, axis=0) for values in outputs)
        return result[0] if num_outputs == 1 else result

    def calc_uncertainties(self, output, scaler=None):
        raise NotImplementedError

    def predict_uncertainty(self, x, scaler=None, batch_size=None):
        raise NotImplementedError


class RegressorDNN(BaseRegressor):
    _uncertainty_kind = "deterministic"

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        optimizer="adam",
        loss="mse",
        loss_weights=None,
        use_noise=False,
        noise_sd=0.01,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.1,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.01,
        l2_weight=0.01,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        verbose=0,
        save_path=".",
        model_name="model.h5",
        metrics=None,
    ):
        super().__init__(
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            activation=activation,
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            use_noise=use_noise,
            noise_sd=noise_sd,
            lr=lr,
            use_dropout=use_dropout,
            dropout_alpha=dropout_alpha,
            batch_size=batch_size,
            epochs=epochs,
            kernel_reg=kernel_reg,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            sgd_momentum=sgd_momentum,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            verbose=verbose,
            save_path=save_path,
            model_name=model_name,
            metrics=metrics,
        )


class GaussianRegressorDNN(BaseRegressor):
    _uncertainty_kind = "gaussian"

    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers
    and provides evidential uncertainty estimation.
    Inherits from BaseRegressor.

    Attributes:
        hidden_layers: Number of hidden layers.
        hidden_neurons: Number of neurons in each hidden layer.
        activation: Type of activation function.
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object.
        use_noise: Whether additive Gaussian noise layers are included in the network.
        noise_sd: The standard deviation of the Gaussian noise layers.
        use_dropout: Whether Dropout layers are added to the network.
        dropout_alpha: Proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch.
        epochs: Number of epochs to train.
        verbose: Level of detail to provide during training.
        model: Keras Model object.
        evidential_coef: Evidential regularization coefficient.
        metrics: Optional list of metrics to monitor during training.

    """

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        loss="",
        optimizer="adam",
        loss_weights=None,
        use_noise=False,
        noise_sd=0.01,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.1,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.01,
        l2_weight=0.01,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        verbose=0,
        save_path=".",
        model_name="model.h5",
        metrics=None,
        eps=1e-7
    ):
        """
        Initialize the EvidentialRegressorDNN.

        Args:
            coupling_coef: Coupling coeffient for loss fix
            evidential_coef: Evidential regularization coefficient.

        """
        super().__init__(  # Call the constructor of the base class
            hidden_layers,
            hidden_neurons,
            activation,
            optimizer,
            loss,
            loss_weights,
            use_noise,
            noise_sd,
            lr,
            use_dropout,
            dropout_alpha,
            batch_size,
            epochs,
            kernel_reg,
            l1_weight,
            l2_weight,
            sgd_momentum,
            adam_beta_1,
            adam_beta_2,
            verbose,
            save_path,
            model_name,
            metrics,
        )
        self.eps = eps
        self.loss = gaussian_nll_per_sample

    def build_neural_network(self, inputs, outputs, last_layer="DenseNormal"):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables.
            outputs (int): Number of output predictor variables.
        """
        super().build_neural_network(inputs, outputs, last_layer=last_layer)

    @classmethod
    def load_model(cls, conf):
        # Load ensemble weights
        save_loc = conf["save_loc"]
        n_models = conf["ensemble"]["n_models"]
        n_splits = conf["ensemble"]["n_splits"]
        if n_splits > 1 and n_models == 1:
            mode = "cv_ensemble"
        elif n_splits == 1 and n_models > 1:
            mode = "deep_ensemble"
        elif n_splits == 1 and n_models == 1:
            mode = "single"
        elif n_splits > 1 and n_models > 1:
            mode = "multi_ensemble"
        else:
            raise ValueError("n_models and n_splits must both be positive")
        save_loc = conf["save_loc"]
        # Check if weights file exists
        weights = os.path.join(save_loc, f"{mode}/models", "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                f"No saved model exists at {weights}. You must train a model first. Exiting."
            )
        if conf["model"]["verbose"]:
            logging.info(
                f"Loading a Gaussian DNN with pre-trained weights from path {weights}"
            )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)

        # Load the variances
        variance_file = os.path.join(save_loc, mode, "models", "best_training_var.txt")
        model_class.training_var = np.atleast_1d(np.loadtxt(variance_file)) if os.path.isfile(variance_file) else []

        # Load ensemble weights
        model_class.ensemble_member_files = []
        if mode != "single":
            for i in range(n_models):
                for j in range(n_splits):
                    model_class.ensemble_member_files.append(
                        os.path.join(save_loc, mode, "models", f"model_seed{i}_split{j}.h5")
                    )

        return model_class

    def calc_uncertainties(self, preds, y_scaler=None):
        """Decode Gaussian ``[mean, sigma]`` into mean and variance in target units."""
        mu, sigma = np.split(np.atleast_2d(preds), 2, axis=-1)
        if not np.all(np.isfinite(preds)) or np.any(sigma < 0):
            raise ValueError("Gaussian means and standard deviations must be finite, with nonnegative standard deviations")
        aleatoric = np.square(sigma)
        mu, aleatoric, _ = inverse_scale_moments(mu, aleatoric, np.zeros_like(aleatoric), y_scaler)
        return mu, aleatoric

    def predict_uncertainty(self, x, scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        y_out = self.calc_uncertainties(y_out, scaler)
        return y_out

    def predict_dist_params(self, x, scaler=None, batch_size=None):
        """Return Gaussian ``(mean, standard_deviation)`` in target units."""
        _batch_size = self.batch_size if batch_size is None else batch_size
        preds = self.model.predict(x, batch_size=_batch_size)
        mu, sigma = np.split(np.atleast_2d(preds), 2, axis=-1)
        if not np.all(np.isfinite(preds)) or np.any(sigma < 0):
            raise ValueError("Gaussian means and standard deviations must be finite, with nonnegative standard deviations")
        slope = affine_inverse_scale(scaler, mu.shape[-1])
        mu, _, _ = inverse_scale_moments(mu, np.zeros_like(mu), np.zeros_like(mu), scaler)
        return mu, sigma * np.abs(slope)

    def predict_ensemble(self, x_test, scaler=None, batch_size=None, num_outputs=2, member_scalers=None, member_input_scalers=None):
        return super().predict_ensemble(x_test, scaler=scaler, batch_size=batch_size,
                                       num_outputs=num_outputs, member_scalers=member_scalers,
                                       member_input_scalers=member_input_scalers)

    def predict_monte_carlo(self, x_test, forward_passes, scaler=None, batch_size=None, num_outputs=2):
        return super().predict_monte_carlo(x_test, forward_passes, scaler=scaler,
                                           batch_size=batch_size, num_outputs=num_outputs)


class EvidentialRegressorDNN(BaseRegressor):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers
    and provides evidential uncertainty estimation.
    Inherits from BaseRegressor.

    Attributes:
        hidden_layers: Number of hidden layers.
        hidden_neurons: Number of neurons in each hidden layer.
        activation: Type of activation function.
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object.
        use_noise: Whether additive Gaussian noise layers are included in the network.
        noise_sd: The standard deviation of the Gaussian noise layers.
        use_dropout: Whether Dropout layers are added to the network.
        dropout_alpha: Proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch.
        epochs: Number of epochs to train.
        verbose: Level of detail to provide during training.
        model: Keras Model object.
        evidential_coef: Evidential regularization coefficient.
        metrics: Optional list of metrics to monitor during training.

    """
    _uncertainty_kind = "evidential"

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        loss="evidentialReg",
        coupling_coef=1.0,  # right now we have alpha = ... v.. so alpha will be coupled in new loss
        evidential_coef=0.05,
        optimizer="adam",
        loss_weights=None,
        use_noise=False,
        noise_sd=0.01,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.1,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.01,
        l2_weight=0.01,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        verbose=0,
        save_path=".",
        model_name="model.h5",
        metrics=None,
        eps=1e-7
    ):
        """Initialize the EvidentialRegressorDNN.

        Args:
            coupling_coef: Coupling coeffient for loss fix
            evidential_coef: Evidential regularization coefficient.
        """
        super().__init__(  # Call the constructor of the base class
            hidden_layers,
            hidden_neurons,
            activation,
            optimizer,
            loss,
            loss_weights,
            use_noise,
            noise_sd,
            lr,
            use_dropout,
            dropout_alpha,
            batch_size,
            epochs,
            kernel_reg,
            l1_weight,
            l2_weight,
            sgd_momentum,
            adam_beta_1,
            adam_beta_2,
            verbose,
            save_path,
            model_name,
            metrics,
        )
        self.coupling_coef = coupling_coef
        self.evidential_coef = evidential_coef
        self.eps = eps

        if (
            loss == "evidentialReg"
        ):  # retains backwards compatibility since default without loss arg is original loss
            self.loss = EvidentialRegressionLoss(coeff=self.evidential_coef)
        elif (
            loss == "evidentialFix"
        ):  # by default we do not regularize this loss as per meinert and lavin
            self.loss = EvidentialRegressionCoupledLoss(
                coeff=self.evidential_coef, r=self.coupling_coef
            )
        else:
            raise ValueError("loss needs to be one of 'evidentialReg' or 'evidentialFix'")

        logging.info(f"Using loss: {loss}")

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables.
            outputs (int): Number of output predictor variables.
        """
        super().build_neural_network(inputs, outputs, last_layer="DenseNormalGamma")

    @classmethod
    def load_model(cls, conf):
        # Check if weights file exists
        weights = os.path.join(conf["model"]["save_path"], "models", "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                f"No saved model exists at {weights}. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading an evidential DNN with pre-trained weights from path {weights}"
        )
        model_class = cls(**conf["model"])
        model_class.build_neural_network(
            len(conf["data"]["input_cols"]), len(conf["data"]["output_cols"])
        )
        model_class.model.load_weights(weights)

        # Load the variances
        variance_file = os.path.join(conf["model"]["save_path"], "models", "best_training_var.txt")
        model_class.training_var = np.atleast_1d(np.loadtxt(variance_file)) if os.path.isfile(variance_file) else []

        # Load ensemble if there is one
        save_loc = conf["save_loc"]
        n_models = conf["ensemble"]["n_models"]
        n_splits = conf["ensemble"]["n_splits"]
        if n_splits > 1 and n_models == 1:
            mode = "cv_ensemble"
        elif n_splits == 1 and n_models > 1:
            mode = "deep_ensemble"
        elif n_splits == 1 and n_models == 1:
            mode = "single"
        elif n_splits > 1 and n_models > 1:
            mode = "multi_ensemble"
        else:
            raise ValueError("n_models and n_splits must both be positive")
        model_class.ensemble_member_files = []
        if mode != "single":
            for i in range(n_models):
                for j in range(n_splits):
                    model_class.ensemble_member_files.append(
                        os.path.join(save_loc, "models", f"model_seed{i}_split{j}.h5")
                    )

        return model_class

    def calc_uncertainties(self, preds, y_scaler=None):
        """Decode NIG moments and transform both variances with the affine slope squared."""
        mu, v, alpha, beta = np.split(np.atleast_2d(preds), 4, axis=-1)
        if isinstance(self.loss, EvidentialRegressionCoupledLoss):
            v = 2 * (alpha - 1) / self.coupling_coef
        mu, aleatoric, epistemic, _ = nig_uncertainties(mu, v, alpha, beta)
        return inverse_scale_moments(mu, aleatoric, epistemic, y_scaler)

    def predict_uncertainty(self, x, scaler=None, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_out = self.model.predict(x, batch_size=_batch_size)
        y_out = self.calc_uncertainties(
            y_out, scaler
        )  # todo calc uncertainty for coupled params
        return y_out

    def predict_dist_params(self, x, y_scaler=None, batch_size=None):
        """Return NIG parameters in target units (beta scales by inverse slope squared)."""
        _batch_size = self.batch_size if batch_size is None else batch_size
        preds = self.model.predict(x, batch_size=_batch_size)
        mu, v, alpha, beta = np.split(np.atleast_2d(preds), 4, axis=-1)
        if isinstance(self.loss, EvidentialRegressionCoupledLoss):
            v = 2 * (alpha - 1) / self.coupling_coef
        nig_uncertainties(mu, v, alpha, beta)
        slope = affine_inverse_scale(y_scaler, mu.shape[-1])
        mu, _, _ = inverse_scale_moments(mu, np.zeros_like(mu), np.zeros_like(mu), y_scaler)
        return mu, v, alpha, beta * np.square(slope)

    def predict_ensemble(self, x_test, scaler=None, batch_size=None, member_scalers=None, member_input_scalers=None):
        return super().predict_ensemble(x_test, scaler=scaler, batch_size=batch_size,
                                       num_outputs=3, member_scalers=member_scalers,
                                       member_input_scalers=member_input_scalers)

    def predict_monte_carlo(self, x_test, forward_passes, scaler=None, batch_size=None):
        return super().predict_monte_carlo(x_test, forward_passes,
                                           scaler=scaler, batch_size=batch_size, num_outputs=3)


class CategoricalDNN(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.

    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss functions or loss objects (can match up to number of output layers)
        loss_weights: Weights to be assigned to respective loss/output layer
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        lr: Learning rate for optimizer
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        l2_weight: L2 weight parameter
        sgd_momentum: SGD optimizer momentum parameter
        adam_beta_1: Adam optimizer beta_1 parameter
        adam_beta_2: Adam optimizer beta_2 parameter
        decay: Level of decay to apply to learning rate
        verbose: Level of detail to provide during training (0 = None, 1 = Minimal, 2 = All)
        classifier: (boolean) If training on classes

    """
    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        output_activation="softmax",
        optimizer="adam",
        loss="categorical_crossentropy",
        loss_weights=None,
        annealing_coeff=None,
        use_noise=False,
        noise_sd=0.0,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.2,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.0,
        l2_weight=0.0,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        epsilon=1e-7,
        decay=0,
        verbose=0,
        random_state=1000,
        callbacks=None,
        balanced_classes=0,
        steps_per_epoch=0,
    ):

        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.loss = loss
        self.loss_weights = loss_weights
        self.annealing_coeff = annealing_coeff
        self.lr = lr
        self.kernel_reg = kernel_reg
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.callbacks = [] if callbacks is None else list(callbacks)
        self.decay = decay
        self.verbose = verbose
        self.y_labels = None
        self.model = None
        self.random_state = random_state
        self.balanced_classes = balanced_classes
        self.steps_per_epoch = steps_per_epoch
        self.ensemble_member_files = []

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        if self.loss == "evidential":
            epoch = keras.Variable(0.0, trainable=False)
            report = ReportEpoch(epoch)
            self.callbacks.append(report)
            self.loss = DirichletEvidentialLoss(callback=report, this_epoch_num=epoch,
                                                annealing_coef=1.0 if self.annealing_coeff is None else self.annealing_coeff)
            self.output_activation = "linear"
        if self.activation == "leaky":
            self.activation = LeakyReLU()
        if self.kernel_reg == "l1":
            self.kernel_reg = L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = keras.regularizers.get(self.kernel_reg)

        self.model = keras.models.Sequential()
        self.model.add(
            Dense(
                inputs,
                activation=self.activation,
                kernel_regularizer=self.kernel_reg,
                name="dense_input",
            )
        )

        for h in range(self.hidden_layers):
            self.model.add(
                Dense(
                    self.hidden_neurons,
                    activation=self.activation,
                    kernel_regularizer=self.kernel_reg,
                    name=f"dense_{h:02d}",
                )
            )
            if self.use_dropout:
                self.model.add(Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model.add(GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        self.model.add(
            Dense(outputs, activation=self.output_activation, name="dense_output")
        )

        if self.optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
                epsilon=self.epsilon,
            )
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        else:
            self.optimizer_obj = keras.optimizers.get(self.optimizer)

        self.model.build((self.batch_size, inputs))
        self.model.compile(optimizer=self.optimizer_obj, loss=self.loss)

    def build_from_sequential(self, model, optimizer="adam", loss="mse", metrics=None):
        """
        Build the neural network model using a Keras Sequential model.

        Args:
            model (tf.keras.Sequential): Keras Sequential model to use.
            optimizer (str or tf.keras.optimizers.Optimizer): Optimizer for the model.
            loss (str or tf.keras.losses.Loss): Loss function for the model.
            metrics (list of str or tf.keras.metrics.Metric): Metrics for the model.

        """
        self.model = model

        if optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
                epsilon=self.epsilon,
            )
        elif optimizer == "sgd":
            self.optimizer_obj = SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        else:
            self.optimizer_obj = keras.optimizers.get(optimizer)

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=loss,
            loss_weights=self.loss_weights,
            metrics=metrics,
        )

    def fit(self, x_train, y_train, validation_data=None):
        y_train = np.asarray(y_train)
        if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
            if np.any(y_train != y_train.astype(int)) or np.any(y_train < 0):
                raise ValueError("Class labels must be nonnegative integer indices")
            y_train = keras.utils.to_categorical(y_train.reshape(-1))
        inputs = x_train.shape[-1]
        outputs = y_train.shape[-1]

        evidential = self.loss == "evidential" or isinstance(self.loss, DirichletEvidentialLoss)
        if evidential:
            this_epoch_num = keras.Variable(0.0, trainable=False)
            report_epoch_callback = ReportEpoch(this_epoch_num)
            self.callbacks = [item for item in self.callbacks if not isinstance(item, ReportEpoch)]
            self.callbacks.insert(0, report_epoch_callback)
            self.loss = DirichletEvidentialLoss(
                callback=report_epoch_callback, this_epoch_num=this_epoch_num,
                annealing_coef=1.0 if self.annealing_coeff is None else self.annealing_coeff,
            )
            self.output_activation = "linear"

        self.build_neural_network(inputs, outputs)
        if self.balanced_classes:
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.tensorflow import balanced_batch_generator
            train_idx = np.argmax(y_train, 1)
            training_generator, steps_per_epoch = balanced_batch_generator(
                x_train,
                y_train,
                sample_weight=None if self.loss_weights is None else np.array([self.loss_weights[_] for _ in train_idx]),
                sampler=RandomUnderSampler(),
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
            history = self.model.fit(
                training_generator,
                validation_data=validation_data,
                steps_per_epoch=steps_per_epoch,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                shuffle=True,
            )
        elif self.loss_weights is not None and not evidential:
            # Allow weights to be used with ev model but they need loaded into the custom loss first
            if self.loss == "evidential":
                logging.warning("Class weights are not being used with the evidential model. They will be supported in a future version.")

            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                class_weight={k: v for k, v in enumerate(self.loss_weights)},
                shuffle=True,
            )
        else:
            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                shuffle=True,
            )
        return history

    @classmethod
    def load_model(cls, conf):
        weights = os.path.join(conf["save_loc"], "models", "best.h5")
        if not os.path.isfile(weights):
            raise ValueError(
                "No saved model exists. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading a CategoricalDNN with pre-trained weights from path {weights}"
        )
        input_features = conf["input_features"]
        output_features = conf["output_features"]

        # flag for our ptype model
        if all([x in conf for x in input_features]):
            input_features = [conf[x] for x in input_features]
            input_features = [item for sublist in input_features for item in sublist]

        model_class = cls(**conf["model"])
        model_class.build_neural_network(len(input_features), len(output_features))
        model_class.model.load_weights(weights)

        # Load the path to ensemble weights
        model_class.ensemble_member_files = []
        if conf["ensemble"]["n_splits"] > 1:
            for j in range(conf["ensemble"]["n_splits"]):
                model_class.ensemble_member_files.append(
                    os.path.join(conf["save_loc"], "models", f"model_{j}.h5")
                )

        return model_class

    def save_model(self, model_path):
        keras.models.save_model(self.model, model_path)
        return

    def predict(self, x, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = self.model.predict(x, batch_size=_batch_size, verbose=self.verbose)
        return y_prob

    def predict_proba(self, x, batch_size=None):
        _batch_size = self.batch_size if batch_size is None else batch_size
        y_prob = self.model.predict(x, batch_size=_batch_size, verbose=self.verbose)
        return y_prob

    def predict_dropout(self, x, mc_forward_passes=10, batch_size=None):
        """Return probabilities, marginal variances, entropy, and mutual information.

        Only dropout is stochastic; BatchNorm and additive noise remain in
        inference mode. Evidential predictions include both within-Dirichlet
        and between-pass epistemic uncertainty. Inputs retain shape
        (samples, features), including batches smaller than ``batch_size``.
        """
        size = self.batch_size if batch_size is None else batch_size
        x = ops.convert_to_numpy(x)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("x must be a nonempty (samples, features) array")
        evidential = self.loss == "evidential" or isinstance(self.loss, DirichletEvidentialLoss)
        activation = keras.activations.serialize(keras.activations.get(self.output_activation))
        if not evidential and activation not in ("softmax", "linear"):
            raise ValueError("Categorical dropout requires softmax probabilities or linear logits")
        samples = mc_raw_predictions(self.model, x, mc_forward_passes, size)
        return categorical_mc_moments(samples, evidential=evidential,
                                      from_logits=(not evidential and activation == "linear"))

    def predict_ensemble(self, x, batch_size=None):
        """Return member predictions while preserving the loaded model weights."""
        original = self.model.get_weights()
        predictions = []
        try:
            for path in self.ensemble_member_files:
                self.model.load_weights(path)
                predictions.append(self.predict(x, batch_size=batch_size))
        finally:
            self.model.set_weights(original)
        if not predictions:
            return np.empty((0, len(x), self.model.output_shape[-1]))
        return np.stack(predictions)

    def predict_uncertainty(self, x):
        if not (self.loss == "evidential" or isinstance(self.loss, DirichletEvidentialLoss)):
            raise ValueError("Dirichlet uncertainty requires an evidential classifier")
        num_classes = self.model.output_shape[-1]
        y_pred = self.predict(x)
        evidence = ops.relu(y_pred)
        alpha = evidence + 1
        S = ops.sum(alpha, axis=1, keepdims=True)
        u = num_classes / S
        prob = alpha / S
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob**2 - epistemic
        return prob, u, aleatoric, epistemic


def locate_best_model(filepath, metric="val_ave_acc", direction="max"):
    """Select a finite best metric from saved training logs."""
    if direction not in ("min", "max"):
        raise ValueError("direction must be 'min' or 'max'")
    files = glob.glob(os.path.join(filepath, "models", "training_log_*.csv"))
    candidates = []
    for filename in files:
        values = pd.read_csv(filename)[metric]
        finite = values[np.isfinite(values)]
        if len(finite):
            member = int(os.path.basename(filename).split("_log_")[1].replace(".csv", ""))
            candidates.append((float(finite.min() if direction == "min" else finite.max()), member))
    if not candidates:
        raise ValueError("No training logs contain finite values for the requested metric")
    return (min(candidates) if direction == "min" else max(candidates))[1]
