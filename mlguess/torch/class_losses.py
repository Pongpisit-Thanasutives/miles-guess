import torch.nn.functional as F
import torch

# Adapted from https://github.com/dougbrion/pytorch-classification-uncertainty/blob/master/losses.py

def _annealing_coefficient(epoch_num, annealing_step):
    if annealing_step <= 0 or epoch_num < 0:
        raise ValueError("annealing_step must be positive and epoch_num must be nonnegative")
    return min(1.0, epoch_num / annealing_step)


def _class_weights(weights, alpha):
    weights = torch.as_tensor(weights, dtype=alpha.dtype, device=alpha.device)
    if weights.ndim != 1 or weights.shape[0] != alpha.shape[-1] or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Class weights must be a finite nonnegative vector with one entry per class")
    return weights


def get_device():
    """Get the device for PyTorch operations.

    Returns:
        torch.device: The device to use, either "cuda" if CUDA is available, otherwise "cpu".
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def relu_evidence(y):
    """Apply the ReLU activation function to the input tensor.

    Args:
        y (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying the ReLU activation function.
    """
    return F.relu(y)


def exp_evidence(y):
    """Apply the exponential function to the input tensor with clamping.

    Args:
        y (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying the exponential function with clamping.
    """
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    """Apply the Softplus activation function to the input tensor.

    Args:
        y (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying the Softplus activation function.
    """
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    """Compute the Kullback-Leibler divergence for a Dirichlet distribution.

    Args:
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        num_classes (int): The number of classes.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The KL divergence between the given Dirichlet distribution and a uniform distribution.
    """
    if device is None:
        device = alpha.device
    alpha = alpha.to(device)
    if alpha.ndim != 2 or alpha.shape[-1] != num_classes or num_classes < 1:
        raise ValueError("alpha must have shape (samples, num_classes)")
    if not torch.isfinite(alpha).all() or (alpha <= 0).any():
        raise ValueError("Dirichlet concentrations must be finite and positive")
    ones = alpha.new_ones((1, num_classes))
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    """Compute the log-likelihood loss for a Dirichlet distribution.

    Args:
        y (torch.Tensor): Target values.
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed log-likelihood loss.
    """
    if device is None:
        device = alpha.device
    alpha = alpha.to(device)
    y = y.to(device=device, dtype=alpha.dtype)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    """Compute the mean squared error loss with KL divergence for Dirichlet distributions.

    Args:
        y (torch.Tensor): Target values.
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed MSE loss with KL divergence.
    """
    if device is None:
        device = alpha.device
    alpha = alpha.to(device)
    y = y.to(device=device, dtype=alpha.dtype)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = alpha.new_tensor(_annealing_coefficient(epoch_num, annealing_step))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, weights=None, device=None):
    """Compute the Evidence Deep Learning (EDL) loss.

    Args:
        func (callable): Function to apply to alpha (e.g., log, softplus).
        y (torch.Tensor): Target values.
        alpha (torch.Tensor): The Dirichlet parameters (alpha).
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        weights (optional, torch.Tensor): Weights to apply to the loss. Defaults to None.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed EDL loss.
    """
    if device is None:
        device = alpha.device
    alpha = alpha.to(device)
    y = y.to(device=device, dtype=alpha.dtype)
    S = torch.sum(alpha, dim=1, keepdim=True)
    if weights is None:
        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    else:
        weights = _class_weights(weights, alpha)
        A = torch.sum(weights * y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = alpha.new_tensor(_annealing_coefficient(epoch_num, annealing_step))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, weights=None, device=None):
    """Compute the Evidence Deep Learning (EDL) loss with mean squared error.

    Args:
        output (torch.Tensor): Model output tensor.
        target (torch.Tensor): Target values.
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        weights: Must be None; class weighting is not defined for this loss.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed EDL loss with MSE.
    """
    if device is None:
        device = output.device
    if weights is not None:
        raise NotImplementedError("Class weights are not defined for the evidential MSE loss")
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, weights=None, device=None):
    """Compute the Evidence Deep Learning (EDL) loss with the logarithm of evidence.

    Args:
        output (torch.Tensor): Model output tensor.
        target (torch.Tensor): Target values.
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        weights (optional, torch.Tensor): Weights to apply to the loss. Defaults to None.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed EDL loss with logarithmic evidence.
    """
    if device is None:
        device = output.device
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, weights, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, weights=None, device=None
):
    """Compute the Evidence Deep Learning (EDL) loss with the digamma function of evidence.

    Args:
        output (torch.Tensor): Model output tensor.
        target (torch.Tensor): Target values.
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The step at which annealing occurs.
        weights (optional, torch.Tensor): Weights to apply to the loss. Defaults to None.
        device (optional, torch.device): Device to perform computation on. Defaults to None, which uses the default device.

    Returns:
        torch.Tensor: The computed EDL loss with digamma evidence.
    """
    if device is None:
        device = output.device
    evidence = relu_evidence(output)
    alpha = evidence + 1

    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, weights, device
        )
    )
    return loss


class EDLLoss(torch.nn.Module):
    def __init__(self, func, num_classes, annealing_step, weights=None):
        super(EDLLoss, self).__init__()
        self.func = func
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.weights = weights

    def forward(self, y, alpha, epoch_num, device=None):
        if device is None:
            device = alpha.device
        alpha = alpha.to(device)
        y = y.to(device=device, dtype=alpha.dtype)
        S = torch.sum(alpha, dim=1, keepdim=True)
        if self.weights is None:
            A = torch.sum(y * (self.func(S) - self.func(alpha)), dim=1, keepdim=True)
        else:
            weights = _class_weights(self.weights, alpha)
            A = torch.sum(weights * y * (self.func(S) - self.func(alpha)), dim=1, keepdim=True)

        annealing_coef = _annealing_coefficient(epoch_num, self.annealing_step)
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, self.num_classes, device=device)
        return A + kl_div


class EDLDigammaLoss(torch.nn.Module):
    def __init__(self, num_classes, annealing_step, weights=None):
        super(EDLDigammaLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.weights = weights

    def forward(self, output, target, epoch_num, device=None):
        if device is None:
            device = output.device
        evidence = F.relu(output)  # Apply ReLU to output
        alpha = evidence + 1
        edl_loss_fn = EDLLoss(torch.digamma, self.num_classes, self.annealing_step, self.weights)
        loss = torch.mean(edl_loss_fn(target, alpha, epoch_num, device))
        return loss
