import numpy as np
import keras
import keras.ops as ops

backend = keras.backend.backend()
if backend == "tensorflow":
    try:
        from tensorflow.math import digamma, lgamma
    except ImportError:
        print("Tensorflow not available")
elif backend == "jax":
    try:
        from jax.scipy.special import digamma
        from jax.lax import lgamma
    except ImportError:
        print("jax not available")
elif backend == "torch":
    try:
        from torch.special import digamma
        from torch import lgamma
    except ImportError:
        print("pytorch not available")

def _categorical_loss_function(evi_coef, epoch_callback, class_weights=None):
    if evi_coef <= 0:
        raise ValueError("evi_coef must be positive")
    if class_weights is not None:
        class_weights = ops.convert_to_tensor(class_weights, dtype="float32")

    def calc_kl(alpha):
        beta = ops.ones_like(alpha[:1])
        S_alpha = ops.sum(alpha, axis=1, keepdims=True)
        S_beta = ops.sum(beta, axis=1, keepdims=True)
        lnB = lgamma(S_alpha) - ops.sum(lgamma(alpha), axis=1, keepdims=True)
        lnB_uni = ops.sum(lgamma(beta), axis=1, keepdims=True) - lgamma(S_beta)
        dg0 = digamma(S_alpha)
        dg1 = digamma(alpha)
        if class_weights is not None:
            kl = (ops.sum(class_weights * (alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB +
                  lnB_uni)
        else:
            kl = (ops.sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni)
        return kl

    def loss(y, y_pred):
        current_epoch = epoch_callback.epoch_var
        evidence = ops.relu(y_pred)
        alpha = evidence + 1
        s = ops.sum(alpha, axis=1, keepdims=True)
        m = alpha / s

        if class_weights is not None:
            a = ops.sum(class_weights * (y - m) ** 2, axis=1, keepdims=True)
            b = ops.sum(class_weights * alpha * (s - alpha) / (s * s * (s + 1)), axis=1, keepdims=True)
        else:
            a = ops.sum((y - m) ** 2, axis=1, keepdims=True)
            b = ops.sum(alpha * (s - alpha) / (s * s * (s + 1)), axis=1, keepdims=True)

        annealing_coef = ops.minimum(1.0, current_epoch / evi_coef)
        alpha_hat = y + (1 - y) * alpha
        c = annealing_coef * calc_kl(alpha_hat)
        return ops.squeeze(a + b + c, axis=-1)

    return loss


def _regression_loss_function(evi_coef):
    """Loss function for an evidential regression model. The total loss is the Negative Log Likelihood of the
    Normal Inverse Gamma summed with the error and scaled by the evidential coefficient. The coefficient has a strong
    influence on the uncertainty predictions (less so for the predictions themselves) of the model and must be tuned
    for individual datasets.
    Loss = loss_nll + coeff * loss_reg
    Args:
        coeff (float): Evidential Coefficient
    """

    def nig_nll(y, gamma, v, alpha, beta, reduce=True):
        if len(y.shape) == 1:
            y = ops.expand_dims(y, axis=-1)
        floor = ops.cast(keras.backend.epsilon(), v.dtype)
        v = ops.where(v < floor, floor, v)
        twoBlambda = 2 * beta * (1 + v)
        nll = (0.5 * ops.log(np.pi / v)
               - alpha * ops.log(twoBlambda)
               + (alpha + 0.5) * ops.log(v * (y - gamma) ** 2 + twoBlambda)
               + lgamma(alpha)
               - lgamma(alpha + 0.5))

        return ops.mean(nll) if reduce else nll

    def nig_reg(y, gamma, v, alpha, reduce=True):
        if len(y.shape) == 1:
            y = ops.expand_dims(y, axis=-1)
        error = ops.abs(y - gamma)
        evi = 2 * v + alpha
        reg = error * evi

        return ops.mean(reg) if reduce else reg
    def loss(y, y_pred):

        gamma, v, alpha, beta = ops.split(y_pred, 4, axis=-1)
        loss_nll = nig_nll(y, gamma, v, alpha, beta, reduce=False)
        loss_reg = nig_reg(y, gamma, v, alpha, reduce=False)

        return ops.mean(loss_nll + evi_coef * loss_reg, axis=-1)

    return loss

@keras.saving.register_keras_serializable(package="mlguess")
class EvidentialCategoricalLoss(keras.losses.Loss):
    """Serializable Dirichlet loss with per-sample values for sample weights."""
    def __init__(self, evi_coef=1.0, epoch=1.0, class_weights=None, epoch_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.evi_coef = float(evi_coef)
        self.class_weights = None if class_weights is None else np.asarray(class_weights).tolist()
        if epoch_callback is None:
            from mlguess.keras.callbacks import ReportEpoch
            epoch_callback = ReportEpoch(initial_epoch=epoch)
        self.epoch_callback = epoch_callback
        self._loss_function = _categorical_loss_function(self.evi_coef, epoch_callback, self.class_weights)

    def call(self, y_true, y_pred):
        return self._loss_function(y_true, y_pred)

    def get_config(self):
        return {**super().get_config(), "evi_coef": self.evi_coef,
                "epoch": float(ops.convert_to_numpy(self.epoch_callback.epoch_var)),
                "class_weights": self.class_weights}


@keras.saving.register_keras_serializable(package="mlguess")
class EvidentialRegressionLoss(keras.losses.Loss):
    """Serializable NIG loss retaining the original unweighted objective."""
    def __init__(self, evi_coef=1.0, **kwargs):
        super().__init__(**kwargs)
        self.evi_coef = float(evi_coef)
        self._loss_function = _regression_loss_function(self.evi_coef)

    def call(self, y_true, y_pred):
        return self._loss_function(y_true, y_pred)

    def get_config(self):
        return {**super().get_config(), "evi_coef": self.evi_coef}


def evidential_cat_loss(evi_coef, epoch_callback, class_weights=None):
    """Create a serializable evidential classification loss."""
    return EvidentialCategoricalLoss(evi_coef=evi_coef, epoch_callback=epoch_callback, class_weights=class_weights)


def evidential_reg_loss(evi_coef):
    """Create a serializable evidential regression loss."""
    return EvidentialRegressionLoss(evi_coef=evi_coef)


@keras.saving.register_keras_serializable()
def gaussian_nll(y, y_pred, reduce=True):
    """Loss function for a parametric Gaussian Loss.

    Args:
        y: Training data targets
        y_pred: Model predicitons
    Returns:
        Mean Negative log likelihood
    """
    if len(y.shape) == 1:
        y = ops.expand_dims(y, axis=-1)
    ax = list(range(1, len(y.shape)))
    mu, sigma = ops.split(y_pred, 2, axis=-1)
    logprob = (-ops.log(sigma)
               - 0.5 * ops.log(2 * np.pi)
               - 0.5 * ((y - mu) / sigma) ** 2)
    loss = ops.mean(-logprob, axis=ax)
    return ops.mean(loss) if reduce else loss


@keras.saving.register_keras_serializable(package="mlguess")
def gaussian_nll_per_sample(y, y_pred):
    """Gaussian NLL per observation, for Keras compile/sample weighting.

    The public gaussian_nll function retains its scalar reduction default.
    """
    return gaussian_nll(y, y_pred, reduce=False)


@keras.saving.register_keras_serializable(package="mlguess")
class EvidentialRegressionCoupledLoss(keras.losses.Loss):
    def __init__(self, r=1.0, coeff=1.0, **kwargs):
        """Implementation of the loss from meinert and lavin that fixes issues with the original
        evidential loss for regression. The loss couples the virtual evidence values with coefficient r.
        In this new loss, the regularizer is unnecessary.
        """
        super(EvidentialRegressionCoupledLoss, self).__init__(**kwargs)
        if r <= 0:
            raise ValueError("r must be positive")
        self.coeff = coeff
        self.r = r

    def nig_nll(self, y, gamma, v, alpha, beta, reduce=True):
        if len(y.shape) == 1:
            y = ops.expand_dims(y, axis=-1)
        # couple the parameters as per meinert and lavin

        twoBlambda = 2 * beta * (1 + v)
        nll = (0.5 * ops.log(np.pi / v)
               - alpha * ops.log(twoBlambda)
               + (alpha + 0.5) * ops.log(v * (y - gamma) ** 2 + twoBlambda)
               + lgamma(alpha)
               - lgamma(alpha + 0.5))

        return ops.mean(nll) if reduce else nll

    def nig_reg(self, y, gamma, v, alpha, reduce=True):
        if len(y.shape) == 1:
            y = ops.expand_dims(y, axis=-1)
        error = ops.abs(y - gamma)  # can try squared loss here to target the right minimizer
        evi = (v + 2 * alpha)  # new paper: = v + 2 * alpha, can try to change this to just 2alpha
        reg = error * evi

        return ops.mean(reg) if reduce else reg

    def call(self, y_true, evidential_output):
        gamma, v, alpha, beta = ops.split(evidential_output, 4, axis=-1)
        v = (2 * (alpha - 1) / self.r)  # need to couple this way otherwise alpha could be negative

        loss_nll = self.nig_nll(y_true, gamma, v, alpha, beta, reduce=False)
        loss_reg = self.nig_reg(y_true, gamma, v, alpha, reduce=False)

        return ops.mean(loss_nll + self.coeff * loss_reg, axis=-1)

    def get_config(self):
        config = super(EvidentialRegressionCoupledLoss, self).get_config()
        config.update({"r": self.r, "coeff": self.coeff})
        return config



