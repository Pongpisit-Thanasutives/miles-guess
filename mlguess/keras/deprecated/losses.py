import numpy as np
import logging
import keras
import keras.ops as ops

backend = keras.backend.backend()
if backend == "tensorflow":
    from tensorflow.math import digamma, lgamma
elif backend == "jax":
    from jax.scipy.special import digamma
    from jax.lax import lgamma
elif backend == "torch":
    from torch.special import digamma
    from torch import lgamma


@keras.saving.register_keras_serializable(package="mlguess_legacy")
class DirichletEvidentialLoss(keras.losses.Loss):
    """Loss function for an evidential categorical model.

    Args:
        callback (list): List of callbacks.
        name (str): reference name
        this_epoch_num (int):  Epoch callback
        class_weights (list): List of class weights (experimental)
    """
    def __init__(self, callback=None, name="dirichlet", this_epoch_num=None,
                 class_weights=None, annealing_coef=1.0, epoch=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.callback = callback
        self.__name__ = name
        self.annealing_coef = float(getattr(callback, "annealing_coef", annealing_coef))
        if self.annealing_coef <= 0:
            raise ValueError("annealing_coef must be positive")
        self.class_weights = None if class_weights is None else ops.convert_to_tensor(class_weights, dtype="float32")
        self.this_epoch_num = (this_epoch_num if this_epoch_num is not None else
                               getattr(callback, "epoch_var", keras.Variable(epoch, trainable=False)))
        if self.class_weights is not None:
            logging.warning("The application of class weights to this loss is experimental.")

    def kl(self, alpha):
        beta = ops.ones_like(alpha[:1])
        S_alpha = ops.sum(alpha, axis=1, keepdims=True)
        S_beta = ops.sum(beta, axis=1, keepdims=True)
        lnB = lgamma(S_alpha) - ops.sum(
            lgamma(alpha), axis=1, keepdims=True
        )
        lnB_uni = ops.sum(
            lgamma(beta), axis=1, keepdims=True
        ) - lgamma(S_beta)

        dg0 = digamma(S_alpha)
        dg1 = digamma(alpha)

        if self.class_weights is not None:
            kl = (ops.sum(self.class_weights * (alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB +
                  lnB_uni)
        else:
            kl = (ops.sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni)
        return kl

    def call(self, y, output):
        evidence = ops.relu(output)
        alpha = evidence + 1

        S = ops.sum(alpha, axis=1, keepdims=True)
        m = alpha / S

        if self.class_weights is not None:
            A = ops.sum(self.class_weights * (y - m) ** 2, axis=1, keepdims=True)
            B = ops.sum(self.class_weights * alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)
        else:
            A = ops.sum((y - m) ** 2, axis=1, keepdims=True)
            B = ops.sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)

        annealing_coef = ops.minimum(1.0, self.this_epoch_num / self.annealing_coef)
        alpha_hat = y + (1 - y) * alpha
        C = annealing_coef * self.kl(alpha_hat)
        return ops.squeeze(A + B + C, axis=-1)

    def get_config(self):
        return {**super().get_config(), "annealing_coef": self.annealing_coef,
                "epoch": float(ops.convert_to_numpy(self.this_epoch_num)),
                "class_weights": None if self.class_weights is None else ops.convert_to_numpy(self.class_weights).tolist()}


@keras.saving.register_keras_serializable(package="mlguess_legacy")
class EvidentialRegressionLoss(keras.losses.Loss):
    """Loss function for an evidential regression model. The total loss is the Negative Log Likelihood of the
    Normal Inverse Gamma summed with the error and scaled by the evidential coefficient. The coefficient has a strong
    influence on the uncertainty predictions (less so for the predictions themselves) of the model and must be tuned
    for individual datasets.
    Loss = loss_nll + coeff * loss_reg
    Args:
        coeff (float): Evidential Coefficient
    """
    def __init__(self, coeff=1.0, **kwargs):
        super(EvidentialRegressionLoss, self).__init__(**kwargs)
        self.coeff = coeff

    def nig_nll(self, y, gamma, v, alpha, beta, reduce=True):
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

    def nig_reg(self, y, gamma, v, alpha, reduce=True):
        if len(y.shape) == 1:
            y = ops.expand_dims(y, axis=-1)
        error = ops.abs(y - gamma)
        evi = 2 * v + alpha
        reg = error * evi

        return ops.mean(reg) if reduce else reg

    def call(self, y_true, evidential_output):
        gamma, v, alpha, beta = ops.split(evidential_output, 4, axis=-1)
        loss_nll = self.nig_nll(y_true, gamma, v, alpha, beta, reduce=False)
        loss_reg = self.nig_reg(y_true, gamma, v, alpha, reduce=False)

        return ops.mean(loss_nll + self.coeff * loss_reg, axis=-1)

    def get_config(self):
        config = super(EvidentialRegressionLoss, self).get_config()
        config.update({"coeff": self.coeff})
        return config

@keras.saving.register_keras_serializable(package="mlguess_legacy")
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


@keras.saving.register_keras_serializable(package="mlguess_legacy")
def gaussian_nll_per_sample(y, y_pred):
    """Gaussian NLL per observation, for Keras compile/sample weighting.

    The public gaussian_nll function retains its scalar reduction default.
    """
    return gaussian_nll(y, y_pred, reduce=False)


@keras.saving.register_keras_serializable(package="mlguess_legacy")
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