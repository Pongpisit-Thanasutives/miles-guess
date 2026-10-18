"""Independent mathematical checks for NIG component CDF diagnostics.

Only the total CDF is a predictive PIT for observed targets.  The component
CDFs describe the two marginal residual distributions in the NIG hierarchy.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.special import gammaln, ndtr
from scipy.stats import t
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mlguess.evaluation import evaluate_nig


def _normal_inverse_gamma_cdf(residual, alpha, beta, variance_multiplier):
    """Integrate the conditional Normal CDF against the precision density.

    If sigma² ~ InvGamma(alpha, beta), then x = beta / sigma² is
    Gamma(alpha, 1).  The conditional residual variance is
    variance_multiplier * sigma².  This oracle does not use a Student-t CDF.
    """
    def integrand(x):
        if x == 0:
            return 0.0
        log_weight = (alpha - 1) * np.log(x) - x - gammaln(alpha)
        z = residual * np.sqrt(x / (beta * variance_multiplier))
        return np.exp(log_weight) * ndtr(z)

    value, error = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-12)
    assert error < 1e-10
    return value


def test_component_cdfs_match_integrated_normal_inverse_gamma_hierarchy():
    mu = np.array([[0.3, -2.0], [1.1, 4.0], [-0.5, 0.2]])
    residual = np.array([[-1.2, 0.7], [0.0, -0.3], [2.1, 0.0]])
    nu = np.array([[0.25, 2.0], [1.0, 0.5], [8.0, 3.0]])
    alpha = np.array([[1.25, 2.0], [3.5, 6.0], [2.4, 1.1]])
    beta = np.array([[0.8, 3.0], [1.5, 0.4], [2.0, 0.6]])
    result = evaluate_nig(mu + residual, (mu, nu, alpha, beta))
    multipliers = {"aleatoric": np.ones_like(nu),
                   "epistemic": 1 / nu, "total": 1 + 1 / nu}

    assert set(result["component_cdf"]) == set(multipliers)
    for component, multiplier in multipliers.items():
        expected = np.empty_like(mu)
        for index in np.ndindex(mu.shape):
            expected[index] = _normal_inverse_gamma_cdf(
                residual[index], alpha[index], beta[index], multiplier[index])
        actual = result["component_cdf"][component]
        assert actual.shape == mu.shape
        assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)

    assert_array_equal(result["component_cdf"]["total"], result["pit"])


@pytest.mark.parametrize("component,multiplier", [
    ("aleatoric", 1.0), ("epistemic", 0.25), ("total", 1.25),
])
def test_component_cdfs_have_closed_form_t4_values(component, multiplier):
    # alpha = beta = 2, nu = 4 gives df = 4 and scale² = multiplier.
    # Analytically, F_t4(2) = 1/2 + 5/(8 sqrt(2)).
    mu = np.array([3.0, -2.0, 8.0])
    y = mu + np.array([-2.0, 0.0, 2.0]) * np.sqrt(multiplier)
    result = evaluate_nig(y, (mu, np.full(3, 4.0),
                              np.full(3, 2.0), np.full(3, 2.0)))
    tail_offset = 5 / (8 * np.sqrt(2))
    assert_allclose(result["component_cdf"][component][:, 0],
                    [0.5 - tail_offset, 0.5, 0.5 + tail_offset],
                    rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize("scaler", [
    StandardScaler(), MinMaxScaler(feature_range=(-2, 3)),
])
def test_component_cdfs_preserve_affine_target_transformations(scaler):
    # Distinct target scales and offsets detect using the wrong target column
    # or scaling variances linearly rather than quadratically.
    training_targets = np.array([[10., -90.], [20., -30.],
                                 [35., 50.], [60., 170.]])
    scaler.fit(training_targets)
    y = np.array([[15., 100.], [40., -20.], [55., 160.]])
    z = scaler.transform(y)
    mu = z + np.array([[-0.2, 0.4], [0.5, -0.1], [-0.3, -0.6]])
    nu = np.array([[1., 4.], [0.5, 2.], [3., 0.7]])
    alpha = np.array([[2., 3.], [1.5, 2.5], [5., 1.2]])
    beta = np.array([[0.4, 2.], [3., 1.], [0.8, 1.4]])
    parameters = (mu, nu, alpha, beta)
    training_units = evaluate_nig(z, parameters)
    physical_units = evaluate_nig(y, parameters, y_scaler=scaler)

    for component in ("aleatoric", "epistemic", "total"):
        assert_allclose(physical_units["component_cdf"][component],
                        training_units["component_cdf"][component],
                        rtol=2e-13, atol=2e-14)
    assert_array_equal(physical_units["component_cdf"]["total"],
                       physical_units["pit"])


def test_large_nu_removes_epistemic_spread_from_total_distribution():
    y = np.array([-1., 0., 1.])
    parameters = (np.zeros(3), np.full(3, 1e12),
                  np.full(3, 3.), np.full(3, 2.))
    result = evaluate_nig(y, parameters)
    component = result["component_cdf"]
    assert_allclose(component["total"], component["aleatoric"],
                    rtol=0, atol=2e-13)
    assert_allclose(component["epistemic"][:, 0], [0., 0.5, 1.],
                    rtol=0, atol=1e-15)


def test_small_positive_component_variance_does_not_underflow_its_scale():
    # E is a positive subnormal float; multiplying it by (alpha-1)/alpha
    # before taking sqrt would round to zero although the scale is representable.
    result = evaluate_nig([0.], ([0.], [2000.], [2.], [1e-320]))
    assert result["epistemic"][0, 0] > 0
    for values in result["component_cdf"].values():
        assert_array_equal(values, [[0.5]])


def test_component_cdfs_are_not_additive_or_uniform_predictive_pits():
    # Deterministic midpoint quantiles replace random draws.  These observations
    # follow the total distribution exactly on a finite probability grid.
    n = 1000
    probabilities = (np.arange(n) + 0.5) / n
    # alpha = beta = 2, nu = 1 => component scale² = 1, total scale² = 2.
    y = t.ppf(probabilities, df=4, scale=np.sqrt(2))
    parameters = (np.zeros(n), np.ones(n),
                  np.full(n, 2.), np.full(n, 2.))
    result = evaluate_nig(y, parameters)
    cdf = result["component_cdf"]
    assert_allclose(cdf["total"][:, 0], probabilities, rtol=0, atol=2e-11)
    assert_allclose(cdf["aleatoric"], cdf["epistemic"])
    assert not np.allclose(cdf["aleatoric"] + cdf["epistemic"], cdf["total"])

    # A flat total histogram is compatible with U-shaped component histograms:
    # observed targets include both residual sources.
    bins = np.linspace(0, 1, 11)
    total_counts = np.histogram(cdf["total"], bins=bins)[0]
    component_counts = np.histogram(cdf["aleatoric"], bins=bins)[0]
    assert_array_equal(total_counts, np.full(10, 100))
    assert component_counts[0] > total_counts[0]
    assert component_counts[-1] > total_counts[-1]
    assert component_counts[4] < total_counts[4]
    assert component_counts[5] < total_counts[5]
