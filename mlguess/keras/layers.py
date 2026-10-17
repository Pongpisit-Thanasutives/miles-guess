import keras
import numpy as np
import keras.layers as layers
import keras.ops as ops


@keras.saving.register_keras_serializable()
class DenseNormalGamma(layers.Layer):
    """Implements dense output layer for a deep evidential regression model.
    Reference: https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf
    Source: https://github.com/aamini/evidential-deep-learning

    Args:
        units (int): Output size of regression tasks
        name (str): Name of the layer
        spectral_normalization (bool): Whether to use Spectral Normalization
        eps (float): Minimum value for evidence (prevents potential division by zero)

    Returns:
        [mu, lambda, alpha, beta] (see reference documentation for more)
    """
    NUM_OUTPUT_PARAMS = 4

    def __init__(self, units: int,
                 spectral_normalization: bool = False,
                 eps: float = 1e-12, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.spectral_normalization = spectral_normalization
        if self.units < 1 or eps <= 0:
            raise ValueError("units and eps must be positive")
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(layers.Dense(DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units,
                                                                   activation=None, dtype=self.dtype_policy))
        else:
            self.dense = layers.Dense(DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units, activation=None, dtype=self.dtype_policy)
        self.eps = eps

    def evidence(self, x):
        """Converts values from continuous space to greater than 0 using a softplus activation function.

        Args:
            x: input value

        Returns:
            Tranformed values
        """
        value = ops.softplus(x)
        floor = ops.cast(self.eps, x.dtype)
        return ops.where(value < floor, floor, value)

    def build(self, input_shape):
        self.dense.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = ops.split(output, 4, axis=-1)
        v = self.evidence(logv)
        # Keep alpha strictly above one after finite-precision rounding.
        machine_epsilon = 2 ** -7 if self.compute_dtype == "bfloat16" else np.finfo(self.compute_dtype).eps
        alpha_evidence = self.evidence(logalpha)
        alpha_floor = ops.cast(max(self.eps, machine_epsilon), logalpha.dtype)
        alpha = ops.where(alpha_evidence < alpha_floor, alpha_floor, alpha_evidence) + 1
        beta = self.evidence(logbeta)
        return ops.concatenate([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (DenseNormalGamma.NUM_OUTPUT_PARAMS * self.units,)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        base_config['spectral_normalization'] = self.spectral_normalization
        base_config['eps'] = self.eps
        return base_config
      

@keras.saving.register_keras_serializable()
class DenseNormal(layers.Layer):
    """Dense output layer for a Gaussian distribution regression neural network.

    Args:
        units (int): Output size of regression tasks
        name (str): Name of the layer
        spectral_normalization (bool): Whether to use Spectral Normalization
        eps (float): Minimum value for evidence (prevents potential division by zero)

    Returns:
        mu, sigma: mean and standard deviation of output
    """
    NUM_OUTPUT_PARAMS = 2

    def __init__(self, units: int,
                 spectral_normalization: bool = False,
                 eps: float = 1e-12, output_activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.spectral_normalization = spectral_normalization
        if self.units < 1 or eps <= 0:
            raise ValueError("units and eps must be positive")
        self.output_activation = output_activation
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(layers.Dense(DenseNormal.NUM_OUTPUT_PARAMS * self.units,
                                                                   activation=self.output_activation, dtype=self.dtype_policy))
        else:
            self.dense = layers.Dense(DenseNormal.NUM_OUTPUT_PARAMS * self.units,
                                      activation=self.output_activation, dtype=self.dtype_policy)
        self.eps = eps

    def build(self, input_shape):
        self.dense.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        output = self.dense(x)
        mu, sigma = ops.split(output, 2, axis=-1)
        floor = ops.cast(self.eps, sigma.dtype)
        sigma = ops.where(sigma < floor, floor, sigma)
        return ops.concatenate([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (DenseNormal.NUM_OUTPUT_PARAMS * self.units,)

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config['units'] = self.units
        base_config['eps'] = self.eps
        base_config['spectral_normalization'] = self.spectral_normalization
        base_config["output_activation"] = self.output_activation
        return base_config