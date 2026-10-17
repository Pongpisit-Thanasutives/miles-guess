from torch import nn
from torch.nn import init
import torch
import logging
from torch.nn.utils import spectral_norm as SpectralNorm
import random
import numpy as np
import os
from mlguess.torch.checkpoint import load_model_state, load_state_dict
from mlguess.torch.layers import LinearNormalGamma
from mlguess.torch.mc_dropout import as_model_input, inference_context, monte_carlo_dropout, _scale_moments, _output_moments

logger = logging.getLogger(__name__)


def get_device():
    """Determine the computing device to use.

    Checks if CUDA is available and returns the appropriate device
    (either "cuda" or "cpu").

    Returns:
        torch.device: The device to be used for computation.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def seed_everything(seed=1234):
    """Seed all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators. Default is 1234.

    Notes:
        This function seeds:
        - Python's random module
        - NumPy's random module
        - PyTorch's CPU and GPU random number generators
        - Sets environment variable to control hash seed
        - Configures PyTorch's cuDNN to be deterministic and benchmark mode.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(net, init_type='normal', init_gain=0.0, verbose=True):
    """Initialize network weights using the specified method.

    Args:
        net (nn.Module): The network whose weights are to be initialized.
        init_type (str): The type of initialization method to use. Options are:
                         'normal', 'xavier', 'kaiming', 'orthogonal'. Default is 'normal'.
        init_gain (float): Scaling factor for 'normal', 'xavier', and 'orthogonal' methods. Default is 0.0.
        verbose (bool): If True, logs the initialization method. Default is True.

    Raises:
        NotImplementedError: If an unsupported initialization method is specified.

    Notes:
        - 'normal': Gaussian distribution with mean 0 and specified standard deviation.
        - 'xavier': Xavier initialization.
        - 'kaiming': Kaiming initialization.
        - 'orthogonal': Orthogonal initialization.
    """

    def init_func(m):
        """Initialization function for network layers.

        Args:
            m (nn.Module): The module to be initialized.

        Notes:
            - Applies the specified initialization method to Conv and Linear layers.
            - Initializes BatchNorm2d layers with a normal distribution.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            weight = getattr(m, 'weight_orig', m.weight)
            if init_type == 'normal':
                init.normal_(weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    if verbose:
        logging.info('Initializing network with %s' % init_type)
    net.apply(init_func)


class DNN(nn.Module):
    """Initialize the Deep Neural Network (DNN) model.

    Args:
        input_size (int or list of int): Number of input features or a list of sizes for each input.
        output_size (int or list of int): Number of output features or a list of sizes for each output.
        layer_size (list of int): List of sizes for hidden layers. Default is [1000].
        dr (list of float): Dropout rates for each layer. Default is [0.5].
        batch_norm (bool): Whether to use batch normalization. Default is True.
        lng (bool): Whether to use LinearNormalGamma layer at the end. Default is False.
        weight_init (bool): Use Kaiming initialization when True. Default is False.
        num_layers (int): Number of layers to create if layer_size is a single number. Default is None.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 layer_size=[1000],
                 dr=[0.5],
                 batch_norm=True,
                 lng=False,
                 softmax=False,
                 weight_init=False,
                 num_layers=None):

        input_size = len(input_size) if isinstance(input_size, (list, tuple)) else input_size
        output_size = len(output_size) if isinstance(output_size, (list, tuple)) else output_size

        super(DNN, self).__init__()
        self.lng = lng
        self.mc_output_kind = "nig" if lng else ("probabilities" if softmax else "regression")

        if isinstance(layer_size, int):
            layer_size = [layer_size] * (1 if num_layers is None else num_layers)
        else:
            layer_size = list(layer_size)
        if isinstance(dr, (int, float)):
            dr = [dr] * len(layer_size)
        else:
            dr = list(dr)
            if len(dr) == 1:
                dr *= len(layer_size)
        if len(dr) < len(layer_size):
            raise ValueError("dr must specify one dropout rate or a rate for each hidden layer")

        if len(layer_size) > 0:
            blocks = self.block(input_size, layer_size[0], dr[0], batch_norm)
            if len(layer_size) > 1:
                for i in range(len(layer_size) - 1):
                    blocks += self.block(layer_size[i], layer_size[i + 1], dr[i + 1], batch_norm)
            if lng:
                blocks.append(LinearNormalGamma(layer_size[-1], output_size))
            else:
                blocks.append(nn.Linear(layer_size[-1], output_size))
                if softmax:
                    blocks.append(nn.Softmax(dim=-1))
        else:
            if lng:
                blocks = [LinearNormalGamma(input_size, output_size)]
            else:
                blocks = [nn.Linear(input_size, output_size)]
                if softmax:
                    blocks.append(nn.Softmax(dim=-1))

        self.fcn = nn.Sequential(*blocks)
        if weight_init:
            init_weights(self, init_type="kaiming", verbose=False)

    def block(self, input_size, output_size, dr, batch_norm):
        """Create a block of layers for the network.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            dr (float): Dropout rate.
            batch_norm (bool): Whether to apply batch normalization.

        Returns:
            list: A list of layers constituting the block.
        """
        block = [SpectralNorm(nn.Linear(input_size, output_size))]
        if batch_norm:
            block.append(nn.BatchNorm1d(output_size))
        block.append(nn.LeakyReLU())
        if dr > 0.0:
            block.append(nn.Dropout(dr))
        return block

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.fcn(x)
        return x

    def load_weights(self, weights_path: str, *, weights_only=True) -> None:
        """Load model weights from a file.

        Args:
            weights_path (str or Path): A state dict or trainer checkpoint (.pt).
            weights_only (bool): Use PyTorch's restricted loader by default. Set
                False only for trusted legacy checkpoints requiring pickle.

        Raises:
            FileNotFoundError or RuntimeError: If loading fails. Errors are not
                suppressed, so inference cannot silently use random weights.

        Returns:
            None
        """
        logger.info(f"Loading model weights from {weights_path}")

        checkpoint = load_state_dict(weights_path, weights_only=weights_only)
        self.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    def predict(self, input, y_scaler=None, return_uncertainties=True, return_tuple=False):
        """Make predictions in evaluation mode without recording gradients.

        NumPy inputs are converted to the model's dtype and device. Every
        module's original training/evaluation state is restored afterwards.

        Args:
            input (torch.Tensor): Input tensor.
            y_scaler (optional, callable): A scaler to inverse transform predictions. Default is None.
            return_uncertainties (bool): Whether to return uncertainties along with predictions. Default is True.
            return_tuple (bool): Retained for compatibility. Evidential outputs
                remain tuples and deterministic outputs remain tensors.

        Returns:
            tuple or torch.Tensor:
                Evidential uncertainty predictions return (mu, aleatoric,
                epistemic, total). With return_uncertainties=False, return the
                raw forward output.
        """
        with inference_context(self):
            output = self(as_model_input(input, self))
            if return_uncertainties:
                if not self.lng:
                    raise ValueError("Uncertainty prediction requires lng=True; use return_uncertainties=False")
                return self.predict_uncertainty(output, y_scaler=y_scaler)
            return output

    def predict_uncertainty(self, input, y_scaler=None):
        """Estimate uncertainties of predictions.

        Args:
            input: Iterable of (mu, v, alpha, beta) tensors, or a tensor
                containing the four concatenated parameter blocks.
            y_scaler: Fitted affine sklearn scaler for the target. The mean is
                inverse-transformed and variances use the squared inverse slope.
                Unsupported or nonlinear scalers raise TypeError; they require
                distribution-specific uncertainty transformations.

        Returns:
            tuple: A tuple containing:
                - mu (torch.Tensor): Mean predictions.
                - aleatoric (torch.Tensor): Aleatoric uncertainty.
                - epistemic (torch.Tensor): Epistemic uncertainty.
                - total (torch.Tensor): Total uncertainty (aleatoric + epistemic).
        """
        # Trainers historically pass a generator of detached parameters.
        # Materialize only parameter iterables; never split a Tensor by rows.
        if not isinstance(input, torch.Tensor):
            input = tuple(input)
        mu, aleatoric, epistemic, _ = _output_moments(input, "nig")
        mu, aleatoric, epistemic = _scale_moments(mu, aleatoric, epistemic, y_scaler)
        total = aleatoric + epistemic
        if not all(torch.isfinite(value).all() for value in (mu, aleatoric, epistemic, total)):
            raise ValueError("Predictive uncertainty moments must be finite; check parameter magnitudes and dtype")
        return mu, aleatoric, epistemic, total

    def predict_dropout(self, x, mc_forward_passes=10, batch_size=16, *,
                        output_kind=None, y_scaler=None, return_dict=False):
        """Predict an equal-weight Monte Carlo dropout mixture (ddof=0).

        The architecture declares the default output kind: ``lng=True`` is
        NIG regression, ``softmax=True`` is categorical probabilities, and
        other DNNs are point regression. Set ``output_kind='logits'`` explicitly
        for a classifier without a softmax output. See ``monte_carlo_dropout``
        for all supported output distributions.

        Classification retains the five-tuple (mean, aleatoric, epistemic,
        entropy, mutual_info). Regression returns (mean, aleatoric, epistemic,
        total). ``return_dict=True`` exposes the within/between decomposition.
        Estimated outputs are NumPy arrays. Point regression does not estimate
        aleatoric or total predictive uncertainty: those outputs are None. Its
        dictionary has aleatoric_available=False, with epistemic equal to the
        between-pass variance.
        """
        if batch_size < 1 or len(x) < 1:
            raise ValueError("batch_size and the number of samples must be positive")
        inputs = as_model_input(x, self)
        result = monte_carlo_dropout(
            inputs.split(batch_size), mc_forward_passes, self,
            n_samples=len(inputs), output_kind=output_kind, y_scaler=y_scaler,
        )
        if return_dict:
            return result
        if "entropy" in result:
            return tuple(result[name] for name in ("mean", "aleatoric", "epistemic", "entropy", "mutual_info"))
        return tuple(result[name] for name in ("mean", "aleatoric", "epistemic", "total"))

    @classmethod
    def from_config(cls, conf, device="cpu"):
        """Create a model instance from configuration.

        Args:
            conf (dict): Configuration dictionary with model parameters.
            device (str): Device to load the model onto. Default is "cpu".

        Returns:
            DNN: The initialized model.
        """
        # init the model
        model = cls(**conf["model"])

        # load model weights
        model = load_model_state(conf, model, device)

        return model


class CategoricalDNN(DNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_output_kind = "evidence"

    def calc_uncertainty(self, y_pred):
        prob, aleatoric, epistemic, _ = _output_moments(y_pred, "evidence")
        alpha = torch.relu(y_pred) + 1
        u = y_pred.shape[-1] / alpha.sum(dim=-1, keepdim=True)
        return prob, u, aleatoric, epistemic

    def predict(self, x, return_uncertainties=True, **kwargs):
        with inference_context(self):
            output = self(as_model_input(x, self))
            if return_uncertainties:
                return self.calc_uncertainty(output)
            else:
                return output

    def predict_uncertainty(self, input):
        return self.predict(input, return_uncertainties=True)



if __name__ == "__main__":
    # Model parameters
    input_size = 10
    output_size = 4
    batch_size = 16

    # Initialize the model
    model = CategoricalDNN(input_size=input_size, output_size=output_size)

    # Example input data (batch size 1, input size 10)
    example_data = torch.rand((batch_size, input_size), dtype=torch.float32)

    # Pass data through the model
    predictions = model(example_data)
    print("Predictions:", predictions)

    prob, u, aleatoric, epistemic = model.predict_uncertainty(example_data)
    print("Predictions:", prob)
    print("Evidential u:", u)
    print("Aleatoric u:", aleatoric)
    print("Epistemic u:", epistemic)
