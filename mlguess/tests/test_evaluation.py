"""Known-answer tests for held-out tutorial diagnostics."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import t
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mlguess.evaluation import evaluate_nig, evaluate_classifier


def test_student_t_predictive_intervals_and_log_density():
    y = np.array([[1., 3.], [2., 5.], [4., 2.]])
    mu = y - .2
    nu = np.ones_like(y)*2
    alpha = np.ones_like(y)*3
    beta = np.ones_like(y)*4
    result = evaluate_nig(y, (mu, nu, alpha, beta), interval_levels=(.5, .95))
    distribution = t(df=6, loc=mu, scale=np.sqrt(2))
    assert_allclose(result['metrics']['nll'], -distribution.logpdf(y).mean(axis=0))
    assert_allclose(result['metrics']['mae'], mean_absolute_error(y, mu, multioutput='raw_values'))
    assert_allclose(result['metrics']['rmse'], np.sqrt(mean_squared_error(y, mu, multioutput='raw_values')))
    assert_allclose(result['intervals'][.95][0], distribution.ppf(.025))
    assert_allclose(result['intervals'][.95][1], distribution.ppf(.975))
    assert_allclose(result['pit'], distribution.cdf(y))
    assert_allclose(result['aleatoric'], 2)
    assert_allclose(result['epistemic'], 1)
    assert_allclose(result['total'], distribution.var())


@pytest.mark.parametrize('scaler', [StandardScaler(), MinMaxScaler(feature_range=(-1, 1))])
def test_scaled_nig_evaluation_keeps_physical_units_and_nll_jacobian(scaler):
    y = np.array([[10.], [15.], [25.], [32.]])
    scaler.fit(y)
    z = scaler.transform(y)
    parameters = (z-.2, np.ones_like(z)*2, np.ones_like(z)*3, np.ones_like(z)*4)
    transformed = evaluate_nig(z, parameters)
    physical = evaluate_nig(y, parameters, y_scaler=scaler)
    slope = (scaler.inverse_transform([[1.]])-scaler.inverse_transform([[0.]]))[0, 0]
    assert_allclose(physical['total'], transformed['total']*slope**2)
    assert_allclose(physical['pit'], transformed['pit'])
    assert_allclose(physical['metrics']['nll'], transformed['metrics']['nll']+np.log(abs(slope)))
    assert_allclose(physical['intervals'][.9][0], scaler.inverse_transform(transformed['intervals'][.9][0]))


def test_constant_target_r_squared_is_undefined():
    values = np.ones((3, 1))
    result = evaluate_nig(values, (values, values, values*2, values))
    assert np.isnan(result['metrics']['r2'].iloc[0])


def test_interval_near_one_retains_finite_symmetric_tails():
    level = np.nextafter(1., 0.)
    result = evaluate_nig([0.], ([0.], [1.], [2.], [1.]), interval_levels=[level])
    lower, upper = result['intervals'][level]
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert_allclose(lower, -upper)


@pytest.mark.parametrize('levels', [[], [0], [1], [.9, .9], [np.nan]])
def test_invalid_interval_levels(levels):
    with pytest.raises(ValueError, match='interval_levels'):
        evaluate_nig([0.], ([0.], [1.], [2.], [1.]), interval_levels=levels)


def test_classifier_known_calibration_and_brier_convention():
    p = np.array([[.8, .2], [.6, .4], [.2, .8], [.4, .6]])
    y = np.array([0, 1, 1, 0])
    result = evaluate_classifier(y, p, n_bins=2)
    assert result['metrics']['accuracy'] == .5
    assert result['metrics']['ece'] == pytest.approx(.2)
    assert result['metrics']['brier'] == pytest.approx(.4)
    assert result['metrics']['log_loss'] == pytest.approx(log_loss(y, p))
    assert result['reliability']['count'].tolist() == [0, 4]
    assert_allclose(result['confusion'], [[1, 1], [1, 1]])


def test_classifier_perfect_and_impossible_predictions():
    y = np.array([0, 1])
    result = evaluate_classifier(y, np.eye(2))
    assert result['metrics'] == dict(accuracy=1., log_loss=0., brier=0., ece=0.)
    assert result['reliability']['count'].iloc[-1] == 2
    assert np.isinf(evaluate_classifier(y, np.eye(2)[::-1])['metrics']['log_loss'])


@pytest.mark.parametrize('y,p', [([0., 1.], [[.5,.5],[.5,.5]]),
                                ([0,2], [[.5,.5],[.5,.5]]),
                                ([0], [[1.,1.]]), ([0], [[np.nan,1.]])])
def test_invalid_classifier_inputs(y,p):
    with pytest.raises(ValueError):
        evaluate_classifier(y,p)
