"""Independent mathematical checks for the scientific quantities exposed by GUESS.

The evidential objectives are checked against probability distributions, not
training quality on a particular random data set. Numerical tolerances allow
the original small epsilon stabilizers in the regression objectives.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.stats import norm
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from mlguess.pit import pit_deviation_skill_score, pit_histogram
from mlguess.regression_metrics import regression_metrics
from mlguess.reliability import compute_calibration, reliability_diagram, reliability_diagrams
from mlguess.torch.class_losses import (
    edl_digamma_loss,
    edl_log_loss,
    edl_mse_loss,
    kl_divergence,
    loglikelihood_loss,
)
from mlguess.torch.layers import LinearNormalGamma
from mlguess.torch.models import CategoricalDNN, DNN
from mlguess.torch.regression_losses import (
    EvidenceRegularizer,
    EvidentialMarginalLikelihood,
    EvidentialRegressionLoss,
    get_mse_coef,
    modified_mse,
)


def nig_parameters():
    """Nondegenerate parameters for multiple observations and response variables."""
    return tuple(
        torch.tensor(value, dtype=torch.float64)
        for value in (
            [[-1.0, 0.5], [2.0, -0.4]],
            [[0.5, 1.0], [2.0, 4.0]],
            [[1.5, 2.0], [3.0, 5.0]],
            [[0.8, 2.0], [1.0, 3.0]],
            [[-0.2, 1.3], [0.0, -0.5]],
        )
    )


def test_nig_negative_log_likelihood_matches_student_t():
    gamma, nu, alpha, beta, target = nig_parameters()
    predictive = torch.distributions.StudentT(
        df=2 * alpha,
        loc=gamma,
        scale=torch.sqrt(beta * (1 + nu) / (alpha * nu)),
    )
    expected = -predictive.log_prob(target)
    actual = EvidentialRegressionLoss().normal_inverse_gamma_nll(
        target, gamma, nu, alpha, beta
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for reduction in ("none", "sum", "mean"):
        result = EvidentialMarginalLikelihood(reduction=reduction)(
            gamma, nu, alpha, beta, target
        )
        reference = expected if reduction == "none" else getattr(expected, reduction)()
        torch.testing.assert_close(result, reference, atol=3e-6, rtol=2e-6)


def test_regression_regularizer_and_total_objective():
    gamma, nu, alpha, beta, target = nig_parameters()
    coef = 0.37
    expected_regularizer = (target - gamma).abs() * (2 * nu + alpha)
    loss = EvidentialRegressionLoss(coef=coef)
    torch.testing.assert_close(
        loss.normal_inverse_gamma_reg(target, gamma, nu, alpha, beta),
        expected_regularizer,
    )
    for reduction in ("none", "sum", "mean"):
        result = EvidenceRegularizer(coef=coef, reduction=reduction)(gamma, nu, alpha, target)
        expected = expected_regularizer * coef
        if reduction != "none":
            expected = getattr(expected, reduction)()
        torch.testing.assert_close(result, expected)
    expected_nll = -torch.distributions.StudentT(
        2 * alpha, gamma, torch.sqrt(beta * (1 + nu) / (alpha * nu))
    ).log_prob(target)
    torch.testing.assert_close(
        loss(gamma, nu, alpha, beta, target),
        expected_nll.mean() + coef * expected_regularizer.mean(),
        atol=2e-6,
        rtol=2e-6,
    )


def test_modified_mse_keeps_coefficient_detached():
    gamma, nu, alpha, beta, target = nig_parameters()
    gamma, nu, alpha, beta = [value.requires_grad_() for value in (gamma, nu, alpha, beta)]
    coefficient = get_mse_coef(gamma, nu, alpha, beta, target)
    assert not coefficient.requires_grad
    assert torch.all((coefficient >= 0) & (coefficient <= 1))
    actual = modified_mse(gamma, nu, alpha, beta, target)
    actual.backward()
    torch.testing.assert_close(gamma.grad, 2 * (gamma - target) * coefficient / gamma.numel())
    assert nu.grad is None and alpha.grad is None and beta.grad is None


def test_regression_objective_has_finite_gradients():
    gamma, nu, alpha, beta, target = nig_parameters()
    params = [value.requires_grad_() for value in (gamma, nu, alpha, beta)]
    loss = EvidentialRegressionLoss(coef=0.1)(*params, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in params)


def test_dirichlet_kl_matches_torch_distribution():
    alpha = torch.tensor([[1.0, 1.0, 1.0], [2.0, 3.0, 7.0]], dtype=torch.float64)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Dirichlet(alpha),
        torch.distributions.Dirichlet(torch.ones_like(alpha)),
    )
    result = kl_divergence(alpha, num_classes=3, device=alpha.device)
    torch.testing.assert_close(result[:, 0], expected)


def test_classification_mse_equals_squared_bias_plus_dirichlet_variance():
    alpha = torch.tensor([[2.0, 3.0, 7.0], [1.0, 4.0, 2.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    distribution = torch.distributions.Dirichlet(alpha)
    expected = ((target - distribution.mean) ** 2 + distribution.variance).sum(dim=1)
    torch.testing.assert_close(
        loglikelihood_loss(target, alpha, device=alpha.device)[:, 0], expected
    )


@pytest.mark.parametrize(
    "loss_fn,expected",
    [(edl_log_loss, math.log(3)), (edl_digamma_loss, 1.5), (edl_mse_loss, 5 / 6)],
)
@pytest.mark.parametrize("epoch_num", [0, 10])
def test_zero_evidence_classification_objectives(loss_fn, expected, epoch_num):
    # Dirichlet(1,1,1) is uniform; its KL penalty is zero at every epoch.
    output = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    result = loss_fn(output, target, epoch_num, 3, 10, device=torch.device("cpu"))
    assert result.item() == pytest.approx(expected, abs=1e-7)
    result.backward()
    assert torch.isfinite(output.grad).all()


def test_linear_normal_gamma_preserves_interleaved_parameter_order():
    layer = LinearNormalGamma(1, 2, spectral_norm=False).double()
    with torch.no_grad():
        layer.linear.weight.zero_()
        layer.linear.bias.copy_(torch.tensor([2.0, 0.0, 1.0, -1.0, -3.0, 2.0, -2.0, 0.5]))
    mu, nu, alpha, beta = layer(torch.zeros((3, 1), dtype=torch.float64))
    torch.testing.assert_close(mu, torch.tensor([[2.0, -3.0]] * 3, dtype=torch.float64))
    for result, logits, shift in (
        (nu, [0.0, 2.0], 0), (alpha, [1.0, -2.0], 1), (beta, [-1.0, 0.5], 0)
    ):
        expected = np.logaddexp(0, np.asarray(logits)) + shift
        np.testing.assert_allclose(result.detach().numpy(), np.tile(expected, (3, 1)), atol=1e-7)


def test_evidence_activation_is_stable_for_large_positive_input():
    layer = LinearNormalGamma(1, 1, spectral_norm=False)
    logits = torch.tensor([0.0, 10.0, 1000.0], requires_grad=True)
    actual = layer.evidence(logits)
    assert torch.isfinite(actual).all()
    np.testing.assert_allclose(actual.detach().numpy(), np.logaddexp(0, logits.detach().numpy()), rtol=1e-6)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_nig_uncertainty_matches_predictive_variance():
    gamma, nu, alpha, beta, _ = nig_parameters()
    model = DNN(2, 2, layer_size=[], dr=[], lng=True)
    mu, aleatoric, epistemic, total = model.predict_uncertainty((gamma, nu, alpha, beta))
    predictive = torch.distributions.StudentT(
        2 * alpha, gamma, torch.sqrt(beta * (1 + nu) / (alpha * nu))
    )
    torch.testing.assert_close(mu, gamma)
    torch.testing.assert_close(aleatoric, beta / (alpha - 1))
    torch.testing.assert_close(epistemic, aleatoric / nu)
    torch.testing.assert_close(total, predictive.variance)


@pytest.mark.parametrize(
    "scaler",
    [StandardScaler(), StandardScaler(with_std=False), MinMaxScaler(),
     MinMaxScaler(feature_range=(-1, 1)), RobustScaler(), MaxAbsScaler()],
)
def test_affine_inverse_scaling_applies_squared_slope_to_variance(scaler):
    scaler.fit(np.array([[0.0, -4.0], [2.0, 1.0], [10.0, 8.0], [4.0, 2.0]]))
    gamma, nu, alpha, beta, _ = nig_parameters()
    model = DNN(2, 2, layer_size=[], dr=[], lng=True)
    # Compute the inverse slope independently from the public transformer.
    slope = scaler.inverse_transform(np.ones((1, 2))) - scaler.inverse_transform(np.zeros((1, 2)))
    mu, aleatoric, epistemic, total = model.predict_uncertainty(
        (gamma, nu, alpha, beta), y_scaler=scaler
    )
    expected_a = beta.numpy() / (alpha.numpy() - 1) * slope**2
    expected_e = expected_a / nu.numpy()
    np.testing.assert_allclose(mu.numpy(), scaler.inverse_transform(gamma.numpy()))
    np.testing.assert_allclose(aleatoric.numpy(), expected_a)
    np.testing.assert_allclose(epistemic.numpy(), expected_e)
    np.testing.assert_allclose(total.numpy(), expected_a + expected_e)


def test_dirichlet_uncertainty_decomposes_categorical_variance():
    model = CategoricalDNN(1, 3, layer_size=[], dr=[])
    output = torch.tensor([[0.0, 0.0, 0.0], [1.0, 4.0, 8.0]], dtype=torch.float64)
    probability, uncertainty, aleatoric, epistemic = model.calc_uncertainty(output)
    distribution = torch.distributions.Dirichlet(output + 1)
    torch.testing.assert_close(probability, distribution.mean)
    torch.testing.assert_close(epistemic, distribution.variance)
    torch.testing.assert_close(aleatoric + epistemic, probability * (1 - probability))
    torch.testing.assert_close(uncertainty[:, 0], 3 / (output + 1).sum(dim=1))


def test_crps_uses_standard_deviation_from_total_variance():
    # At the predictive mean, CRPS(N(mu,sigma**2),mu)=sigma*(sqrt(2)-1)/sqrt(pi).
    truth = np.arange(1.0, 17.0).reshape(-1, 1)
    variance = np.linspace(1.0, 9.0, len(truth)).reshape(-1, 1)
    metrics = regression_metrics(truth, truth, total=variance)
    expected = np.sqrt(variance).mean() * (math.sqrt(2) - 1) / math.sqrt(math.pi)
    assert metrics["val_crps"] == pytest.approx(expected, rel=1e-12)


def test_reliability_keeps_zero_confidence_and_computes_weighted_ece():
    result = compute_calibration(
        np.array([1, 0, 1, 0]),
        np.array([1, 1, 0, 0]),
        np.array([0.0, 0.25, 0.75, 1.0]),
        num_bins=2,
    )
    np.testing.assert_array_equal(result["counts"], [2, 2])
    np.testing.assert_allclose(result["accuracies"], [0.5, 0.5])
    assert result["avg_accuracy"] == pytest.approx(0.5)
    assert result["expected_calibration_error"] == pytest.approx(0.375)


def test_reliability_single_plot_and_grid_work_with_numpy2():
    data = dict(
        true_labels=np.array([0, 1, 1, 0]),
        pred_labels=np.array([0, 1, 0, 0]),
        confidences=np.array([0.4, 0.6, 0.8, 0.9]),
    )
    figures = []
    try:
        figures.append(reliability_diagram(**data, num_bins=2, return_fig=True))
        figures.append(reliability_diagrams({"one": data}, num_cols=1, return_fig=True))
        figures.append(reliability_diagrams({"one": data, "two": data}, num_cols=2, return_fig=True))
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)


def test_pit_integer_bins_cover_probability_interval():
    quantiles = np.array([0.91, 0.92, 0.93, 0.94])
    truth = norm.ppf(quantiles)
    prediction = np.column_stack((np.zeros(4), np.ones(4)))
    counts, edges = pit_histogram(truth, prediction, pred_type="gaussian", bins=10)
    np.testing.assert_allclose(edges, np.linspace(0, 1, 11))
    np.testing.assert_array_equal(counts, [0] * 9 + [4])


def test_pit_skill_score_supports_explicit_edges():
    truth = norm.ppf(np.linspace(0.05, 0.95, 10))
    prediction = np.column_stack((np.zeros(10), np.ones(10)))
    result = pit_deviation_skill_score(
        truth, prediction, pred_type="gaussian", bins=np.linspace(0, 1, 11)
    )
    assert np.ndim(result) == 0
    assert result == pytest.approx(1.0)
