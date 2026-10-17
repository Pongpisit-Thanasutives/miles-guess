import keras
import keras.ops as ops
import keras.layers as layers
import keras.optimizers as optimizers
import numpy as np
from scipy.special import digamma
from mlguess.uncertainty import inverse_scale_moments, nig_uncertainties
# from keras.layers import Dense, GaussianNoise, Dropout
from mlguess.keras.layers import DenseNormalGamma, DenseNormal
from mlguess.keras.losses import evidential_cat_loss, evidential_reg_loss, gaussian_nll_per_sample
from mlguess.keras.callbacks import ReportEpoch
# from keras.optimizers import Adam, SGD



@keras.saving.register_keras_serializable()
class CategoricalDNN(keras.models.Model):
    """A Categorical Dense Neural Network Model that can support arbitrary numbers of hidden layers
    and the ability to provide evidential uncertainty estimation.

    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        evidential (bool): Whether to use evidential approach (custom loss)
        annealing_coeff: Annealing coefficient for evidential loss (ignored if evidential==False)
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

    Example:
        When evidential==True, the output activation and the loss function will be overridden under the hood. When
        evidential==False, it will use the parameters specified and ignore the annealing_coeff.
        Note: Model compilation happens under the hood when .fit() is called.::

            n_samples = 1000
            n_features = 23
            n_classes = 5
            x_train = np.random.random(size=(n_samples, n_features))
            y_train = np.random.randint(low=0, high=n_classes, size=n_samples)

            ### Evidential
            model = CategoricalDNN(hidden_layers=2,
                                   evidential=True,
                                   activation='relu',
                                   n_classes=n_classes,
                                   n_inputs=n_features,
                                   epochs=10,
                                   annealing_coeff=1.5,
                                   lr=0.0001)
            hist = model.fit(x_train, y_train)
            p_with_uncertainty = model.predict(x_train, return_uncertainties=True, batch_size=5000)

            ### Vanilla DNN
            model = CategoricalDNN(hidden_layers=2,
                                   evidential=False,
                                   activation='relu',
                                   output_activation='softmax',
                                   loss='categorical_crossentropy',
                                   n_classes=n_classes,
                                   n_inputs=n_features,
                                   epochs=10,
                                   lr=0.0001)
            hist = model.fit(x_train, y_train)
            p = model.predict(x_train, return_uncertainties=False, batch_size=5000)

    """

    def __init__(self, hidden_layers=2, hidden_neurons=64, evidential=False, activation="relu",
                 output_activation="softmax", optimizer="adam", loss="categorical_crossentropy", loss_weights=None,
                 annealing_coeff=1.0, use_noise=False, noise_sd=0.0, lr=0.001, use_dropout=False, dropout_alpha=0.2,
                 batch_size=128, epochs=2, kernel_reg=None, l1_weight=0.0, l2_weight=0.0, sgd_momentum=0.9,
                 adam_beta_1=0.9, adam_beta_2=0.999, epsilon=1e-7, decay=0, verbose=0, random_state=1000, n_classes=2,
                 n_inputs=42, callbacks=None, **kwargs):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.evidential = evidential
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
        self.decay = decay
        self.verbose = verbose
        self.random_state = random_state
        self.n_classes = n_classes
        self.n_inputs = n_inputs
        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

        self.hyperparameters = ["hidden_layers", "hidden_neurons", "evidential", "activation", "output_activation",
                                "optimizer", "sgd_momentum", "adam_beta_1", "adam_beta_2", "epsilon", "loss",
                                "loss_weights", "annealing_coeff", "lr", "kernel_reg", "l1_weight", "l2_weight",
                                "batch_size", "use_noise", "noise_sd", "use_dropout", "dropout_alpha", "epochs",
                                "callbacks", "decay", "verbose", "random_state", "n_classes", "n_inputs"]


        if self.kernel_reg == "l1":
            self.kernel_reg = keras.regularizers.L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = keras.regularizers.L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = keras.regularizers.L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = keras.regularizers.get(self.kernel_reg)

        if self.optimizer == "adam":
            self.optimizer_obj = optimizers.Adam(learning_rate=self.lr,
                                      beta_1=self.adam_beta_1,
                                      beta_2=self.adam_beta_2,
                                      epsilon=self.epsilon)
        elif self.optimizer == "sgd":
            self.optimizer_obj = optimizers.SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        else:
            self.optimizer_obj = optimizers.get(self.optimizer)

        if self.evidential:
            self.output_activation = "linear"

        self.model_layers = []
        self.model_layers.append(layers.Dense(self.n_inputs,
                                       activation=self.activation,
                                       kernel_regularizer=self.kernel_reg,
                                       name="input_dense"))
        for h in range(self.hidden_layers):
            self.model_layers.append(layers.Dense(self.hidden_neurons,
                                           activation=self.activation,
                                           kernel_regularizer=self.kernel_reg,
                                           name=f"dense_{h:02d}"))
            if self.use_dropout:
                self.model_layers.append(layers.Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model_layers.append(layers.GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        self.model_layers.append(layers.Dense(self.n_classes,
                                       activation=self.output_activation,
                                       name="dense_output"))

    def call(self, inputs):

        mod = self.model_layers[0](inputs)
        for layer in range(1, len(self.model_layers)):
            mod = self.model_layers[layer](mod)

        return mod

    def fit(self, x=None, y=None, **kwargs):
        if y is not None and (self.evidential or self.loss == "categorical_crossentropy"):
            labels = ops.convert_to_numpy(y)
            if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
                if np.any(labels != labels.astype(int)) or np.any(labels < 0) or np.any(labels >= self.n_classes):
                    raise ValueError("Class labels must be integer indices in [0, n_classes)")
                y = keras.utils.to_categorical(labels.reshape(-1), self.n_classes)
        callbacks = list(kwargs.pop("callbacks", self.callbacks))
        if self.evidential:
            callbacks = [callback for callback in callbacks if not isinstance(callback, ReportEpoch)]
            report_epoch_callback = ReportEpoch(keras.Variable(1.0, trainable=False))
            self.loss = evidential_cat_loss(evi_coef=self.annealing_coeff,
                                            epoch_callback=report_epoch_callback)
            callbacks.append(report_epoch_callback)
        super().compile(loss=self.loss, optimizer=self.optimizer_obj,
                        loss_weights=self.loss_weights, run_eagerly=False)
        kwargs.setdefault("epochs", self.epochs)
        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("verbose", self.verbose)
        return super().fit(x, y, callbacks=callbacks, **kwargs)

    def predict(self, x, return_uncertainties=True, **kwargs):
        """
        Make a prediction with the trained model.

        Args:
            x: Input data
            batch_size: Size of batch to predict
            return_uncertainties: Returns derived uncertainties from evidential distribution parameters.
                                  If False, return the probabilities only.

        Returns:
            If return_uncertainties is True (tuple): (probs, u (evidential uncertainty), aleatoric, epistemic)
            Else If return_uncertainties is False: probs
        """
        if (not self.evidential) and return_uncertainties:
            raise NotImplementedError("You can only return uncertainty estimates when 'evidential' is True. Otherwise "
                                      "you can set 'return_uncertainties' to False to return probabilities.")
        output = super().predict(x, **kwargs)
        if return_uncertainties:
            return self.calc_uncertainty(output)
        else:
            return output

    @staticmethod
    def calc_uncertainty(y_pred):
        num_classes = y_pred.shape[-1]
        evidence = ops.relu(y_pred)
        alpha = evidence + 1
        S = ops.sum(alpha, axis=1, keepdims=True)
        u = num_classes / S
        prob = alpha / S
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob ** 2 - epistemic
        return prob, u, aleatoric, epistemic

    def predict_dropout(self, x, mc_forward_passes=10, batch_size=None):
        """Return categorical moments from equally weighted stochastic passes.

        Returns ``(probability, aleatoric_variance, epistemic_variance,
        entropy, mutual_information)``. Variances describe a one-hot class
        indicator, so their sum is ``probability * (1 - probability)``.
        For evidential networks the epistemic variance includes both the
        within-pass Dirichlet variance and variance of the pass means.
        Mutual information likewise includes both levels of uncertainty:
        ``H(mean probability) - E[H(categorical probability)]``; the inner
        expectation is taken over the Dirichlet distribution when applicable.
        Standard networks may emit softmax probabilities or linear logits.
        Batch-normalisation and additive-noise layers remain in inference mode.
        Sampling temporarily overrides dropout calls; do not run concurrent
        calls on this model while sampling.
        """
        if not isinstance(mc_forward_passes, (int, np.integer)) or mc_forward_passes < 1:
            raise ValueError("mc_forward_passes must be a positive integer")
        size = self.batch_size if batch_size is None else batch_size
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("batch_size must be a positive integer")
        x = ops.convert_to_numpy(x)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("x must be a nonempty (samples, features) array")
        activation = keras.activations.serialize(keras.activations.get(self.output_activation))
        if not self.evidential and activation not in ("softmax", "linear"):
            raise ValueError("Categorical dropout requires softmax probabilities or linear logits")

        from mlguess.keras.monte_carlo import mc_raw_predictions

        probabilities, aleatoric_passes, epistemic_passes, expected_entropies = [], [], [], []
        for output in mc_raw_predictions(self, x, mc_forward_passes, size):
            output = output.astype(np.float64, copy=False)
            if self.evidential:
                alpha = np.maximum(output, 0) + 1
                concentration = alpha.sum(axis=-1, keepdims=True)
                probability = alpha / concentration
                within_epistemic = probability * (1 - probability) / (concentration + 1)
                aleatoric = probability * (1 - probability) - within_epistemic
                expected_entropy = (digamma(concentration + 1).squeeze(-1)
                                    - np.sum(probability * digamma(alpha + 1), axis=-1))
            else:
                if activation == "linear":
                    exponential = np.exp(output - np.max(output, axis=-1, keepdims=True))
                    probability = exponential / exponential.sum(axis=-1, keepdims=True)
                else:
                    probability = output
                within_epistemic = np.zeros_like(probability)
                aleatoric = probability * (1 - probability)
                expected_entropy = -np.sum(
                    probability * np.log(np.maximum(probability, np.finfo(float).tiny)), axis=-1)
            probabilities.append(probability)
            aleatoric_passes.append(aleatoric)
            epistemic_passes.append(within_epistemic)
            expected_entropies.append(expected_entropy)
        probabilities = np.stack(probabilities)
        pred_probs = probabilities.mean(axis=0)
        aleatoric = np.mean(aleatoric_passes, axis=0)
        epistemic = np.mean(epistemic_passes, axis=0) + probabilities.var(axis=0, ddof=0)
        entropy = -np.sum(
            pred_probs * np.log(np.maximum(pred_probs, np.finfo(float).tiny)), axis=-1)
        mutual_info = entropy - np.mean(expected_entropies, axis=0)
        return pred_probs, aleatoric, epistemic, entropy, mutual_info

    def compile_from_config(self, config):
        super().compile_from_config(config)
        self.optimizer_obj = self.optimizer

    def get_config(self):
        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        parameter_config['callbacks'] = []
        return {**base_config, **parameter_config}


@keras.saving.register_keras_serializable()
class RegressorDNN(keras.models.Model):
    """A Dense Neural Network Model that can support arbitrary numbers of hidden layers
    and the ability to provide evidential uncertainty estimation or uncertainty estimation through
    a gaussian parametric approach.

    Attributes:
        hidden_layers: Number of hidden layers.
        hidden_neurons: Number of neurons in each hidden layer.
        activation: Type of activation function.
        evidential (bool): Whether to use evidential model (outputs [mean, aleatoric, epistemic])
        uncertainty (bool): Whether to use the Gaussian Parametric approach (outputs [mean, std])
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object.
        use_noise: Whether additive Gaussian noise layers are included in the network.
        noise_sd: The standard deviation of the Gaussian noise layers.
        use_dropout: Whether Dropout layers are added to the network.
        dropout_alpha: Proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch.
        epochs: Number of epochs to train.
        verbose: Level of detail to provide during training.
        evi_coeff: Evidential regularization coefficient.
        metrics: Optional list of metrics to monitor during training.

    Example:
        When evidential==True or uncertainty==True, the output activation and the loss function will be overridden
        under the hood. If both are True, the evidential model will override. When both are set to False,
        it will train a generic DNN with a linear output activation and the specified loss function.
        'evi_coeff' is only used when evidential==True and is otherwise ignored.
        Note: Model compilation happens under the hood when .fit() is called.::

            n_samples = 1000
            n_features = 23
            n_output_tasks = 1
            x_train = np.random.random(size=(n_samples, n_features))
            y_train = np.random.random(size=(n_samples, n_output_tasks)

            ### Evidential
            model = RegressorDNN(hidden_layers=2,
                                 n_output_tasks=n_output_tasks,
                                 n_inputs=n_features,
                                 evidential=True,
                                 epochs=10)
            model.fit(x_train, y_train)
            p_with_uncertainty = model.predict(x_train, return_uncertainties=True)


            ### Gaussian Parametric
            model = RegressorDNN(hidden_layers=2,
                                 n_output_tasks=n_output_tasks,
                                 n_inputs=n_features,
                                 evidential=False,
                                 uncertainty=True,
                                 epochs=10)
            model.fit(x_train, y_train)
            p_with_uncertainty = model.predict(x_train, return_uncertainties=True)


            ### Vanilla DNN
            model = RegressorDNN(hidden_layers=2,
                                 n_output_tasks=n_output_tasks,
                                 n_inputs=n_features,
                                 evidential=False,
                                 uncertainty=False,
                                 epochs=10)
            model.fit(x_train, y_train)
            p = model.predict(x_train, return_uncertainties=False)

    """
    def __init__(self, hidden_layers=2, hidden_neurons=64, evidential=False, activation="relu", optimizer="adam",
                 loss_weights=None, use_noise=False, noise_sd=0.01, lr=0.00001, use_dropout=False, dropout_alpha=0.1,
                 batch_size=128, loss="mse", epochs=2, kernel_reg=None, l1_weight=0.01, l2_weight=0.01,
                 sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999, epsilon=1e-7, verbose=1, training_var=None,
                 n_inputs=10, n_output_tasks=1, evi_coeff=1.0, uncertainty=False, **kwargs):

        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.evidential = evidential
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.loss_weights = loss_weights
        self.loss = loss
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
        self.optimizer_obj = None
        self.training_var = training_var
        self.epsilon = epsilon
        self.n_output_tasks = n_output_tasks
        self.n_inputs = n_inputs
        self.evi_coeff = evi_coeff
        self.uncertainty = uncertainty
        self.N_EVI_OUTPUT_PARAMS = 4
        self.N_DENSE_NORMAL_OUTPUT_PARAMS = 2
        self.hyperparameters = ["hidden_layers", "evidential", "hidden_neurons", "activation", "training_var",
                                "optimizer", "sgd_momentum", "adam_beta_1", "adam_beta_2", "epsilon",
                                "loss_weights", "lr", "kernel_reg", "l1_weight", "l2_weight", "loss",
                                "batch_size", "use_noise", "noise_sd", "use_dropout", "dropout_alpha", "epochs",
                                "verbose", "n_inputs", "n_output_tasks", "epsilon", "evi_coeff", "uncertainty"]

        if self.kernel_reg == "l1":
            self.kernel_reg = keras.regularizers.L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = keras.regularizers.L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = keras.regularizers.L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = keras.regularizers.get(self.kernel_reg)

        if self.optimizer == "adam":
            self.optimizer_obj = optimizers.Adam(learning_rate=self.lr,
                                      beta_1=self.adam_beta_1,
                                      beta_2=self.adam_beta_2,
                                      epsilon=self.epsilon)
        elif self.optimizer == "sgd":
            self.optimizer_obj = optimizers.SGD(learning_rate=self.lr, momentum=self.sgd_momentum)
        else:
            self.optimizer_obj = optimizers.get(self.optimizer)

        self.model_layers = []
        self.model_layers.append(layers.Dense(self.n_inputs,
                                       activation=self.activation,
                                       kernel_regularizer=self.kernel_reg,
                                       name="input_dense"))
        for h in range(self.hidden_layers):
            self.model_layers.append(layers.Dense(self.hidden_neurons,
                                           activation=self.activation,
                                           kernel_regularizer=self.kernel_reg,
                                           name=f"dense_{h:02d}"))
            if self.use_dropout:
                self.model_layers.append(layers.Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model_layers.append(layers.GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        if self.evidential:
            self.model_layers.append(DenseNormalGamma(self.n_output_tasks, name="dense_output"))
        elif self.uncertainty:
            self.model_layers.append(DenseNormal(self.n_output_tasks, name="dense_output"))
        else:
            self.model_layers.append(layers.Dense(self.n_output_tasks, name="dense_output"))

    def call(self, inputs):

        layer_output = self.model_layers[0](inputs)
        for layer in range(1, len(self.model_layers)):
            layer_output = self.model_layers[layer](layer_output)

        return layer_output

    def fit(self, x=None, y=None, **kwargs):
        # A single target still needs a target axis for the distribution losses.
        if y is not None:
            y = ops.convert_to_numpy(y)
            if y.ndim == 1 and self.n_output_tasks == 1:
                y = y[:, None]
            if y.ndim != 2 or y.shape[1] != self.n_output_tasks:
                raise ValueError("y must have shape (samples, n_output_tasks)")

        validation_data = kwargs.get("validation_data")
        if isinstance(validation_data, tuple) and len(validation_data) in (2, 3) and self.n_output_tasks == 1:
            validation_y = ops.convert_to_numpy(validation_data[1])
            if validation_y.ndim == 1:
                kwargs["validation_data"] = (validation_data[0], validation_y[:, None], *validation_data[2:])

        if self.evidential:
            self.loss = evidential_reg_loss(evi_coef=self.evi_coeff)
        elif self.uncertainty:
            self.loss = gaussian_nll_per_sample
        super().compile(optimizer=self.optimizer_obj, loss=self.loss)
        kwargs.setdefault("epochs", self.epochs)
        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("verbose", self.verbose)
        hist = super().fit(x, y, **kwargs)
        if y is not None:
            # Descriptive metadata only: the likelihood already learns variance
            # in the units of y, so multiplying by this again is incorrect.
            self.training_var = np.var(y, axis=0)
        return hist

    def predict(self, x, return_uncertainties=True, batch_size=1000, y_scaler=None, **kwargs):
        """Predict with the existing output contract.

        Evidential models return concatenated ``[mean, aleatoric_variance,
        epistemic_variance]`` when ``return_uncertainties=True``, or raw
        ``[mean, v, alpha, beta]`` otherwise. Gaussian models retain their
        historical raw ``[mean, standard_deviation]`` output for either flag;
        use :meth:`predict_uncertainties` to obtain Gaussian variances.
        Ordinary regressors return means with ``return_uncertainties=False``.

        ``y_scaler`` is accepted only for derived evidential moments. Its
        inverse transform must be affine; variances use the squared inverse
        slope. Raw distribution parameters remain in training-target units.
        """
        if not isinstance(return_uncertainties, bool):
            raise ValueError("return_uncertainties must be a boolean")
        if not self.evidential and not self.uncertainty and return_uncertainties:
            raise NotImplementedError("Set return_uncertainties=False for a deterministic regressor")
        if y_scaler is not None and y_scaler is not False and not (self.evidential and return_uncertainties):
            raise ValueError("Use predict_uncertainties(..., y_scaler=...) to transform distribution moments")
        output = super().predict(x, batch_size=batch_size, **kwargs)
        if self.evidential and return_uncertainties:
            return self.calc_uncertainties(output, y_scaler=y_scaler)
        return output

    def predict_uncertainties(self, x, y_scaler=None, batch_size=1000, **kwargs):
        """Return ``[mean, aleatoric_variance, epistemic_variance]`` for either distribution.

        Gaussian epistemic variance is zero for a single fitted model. Between
        model or dropout variation must be added by an ensemble calculation.
        """
        if not (self.evidential or self.uncertainty):
            raise NotImplementedError("Distribution moments require an evidential or Gaussian model")
        output = super().predict(x, batch_size=batch_size, **kwargs)
        return self.calc_uncertainties(output, y_scaler=y_scaler)

    def calc_uncertainties(self, preds, y_scaler=None):
        """Convert raw distribution parameters to mean and variance components.

        Raw NIG parameters are ``[mean, v, alpha, beta]``; raw Gaussian
        parameters are ``[mean, sigma]`` with sigma a standard deviation.
        Return concatenated ``[mean, aleatoric_variance, epistemic_variance]``.
        ``training_var`` is metadata and is never an uncertainty multiplier.
        Without ``y_scaler``, all moments remain in training-target units.
        Parameters must be finite; NIG moments require ``v > 0``,
        ``alpha > 1``, ``beta > 0`` and Gaussian sigma must be nonnegative.
        """
        preds = np.asarray(ops.convert_to_numpy(preds))
        if preds.ndim != 2:
            raise ValueError("preds must have shape (samples, distribution parameters)")
        if self.evidential:
            if preds.shape[-1] != 4 * self.n_output_tasks:
                raise ValueError("Expected four NIG parameters per target")
            mu, v, alpha, beta = np.split(preds, 4, axis=-1)
            mu, aleatoric, epistemic, _ = nig_uncertainties(mu, v, alpha, beta)
        elif self.uncertainty:
            if preds.shape[-1] != 2 * self.n_output_tasks:
                raise ValueError("Expected Gaussian mean and standard deviation per target")
            mu, sigma = np.split(preds, 2, axis=-1)
            if not np.all(np.isfinite(preds)) or np.any(sigma < 0):
                raise ValueError("Gaussian moments require finite means and nonnegative finite standard deviations")
            aleatoric = np.square(sigma)
            epistemic = np.zeros_like(aleatoric)
        else:
            raise NotImplementedError("Distribution moments require an evidential or Gaussian model")
        mu, aleatoric, epistemic = inverse_scale_moments(mu, aleatoric, epistemic, y_scaler)
        return np.concatenate([mu, aleatoric, epistemic], axis=-1)

    def compile_from_config(self, config):
        super().compile_from_config(config)
        self.optimizer_obj = self.optimizer

    def get_config(self):

        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        return {**base_config, **parameter_config}