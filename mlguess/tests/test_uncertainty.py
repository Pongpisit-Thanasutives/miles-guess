"""Independent distribution moments and finite-mixture identities."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import t
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer

from mlguess.uncertainty import (
    ensemble_uncertainties, nig_uncertainties, dirichlet_uncertainties,
    inverse_scale_moments,
)


def test_nig_mixture_against_student_t_second_moment():
    mu = np.array([[[0., 3.]], [[2., -1.]]])
    nu = np.array([[[2., 3.]], [[4., 1.]]])
    alpha = np.array([[[3., 4.]], [[5., 2.]]])
    beta = np.array([[[2., 6.]], [[4., 3.]]])
    _, a, e, total = nig_uncertainties(mu, nu, alpha, beta)
    distribution = t(df=2*alpha, loc=mu, scale=np.sqrt(beta*(1+nu)/(nu*alpha)))
    assert_allclose(total, distribution.var())
    result = ensemble_uncertainties(mu, a, e)
    independent_total = np.mean(distribution.var() + distribution.mean()**2, axis=0) - distribution.mean().mean(axis=0)**2
    assert_allclose(result['total'], independent_total)
    assert_allclose(result['epistemic'], e.mean(axis=0) + mu.var(axis=0))


def test_categorical_finite_mixture_identity():
    p = np.array([[[.2, .8]], [[.8, .2]]])
    result = ensemble_uncertainties(p, p*(1-p))
    assert_allclose(result['aleatoric'], .16)
    assert_allclose(result['epistemic'], .09)
    assert_allclose(result['total'], .25)


def test_dirichlet_concentration_survives_equal_mean_mixture():
    alpha = np.array([[[1., 1.]], [[10., 10.]]])
    p, a, e, _ = dirichlet_uncertainties(alpha)
    result = ensemble_uncertainties(p, a, e)
    assert_allclose(result['between_epistemic'], 0)
    assert_allclose(result['within_epistemic'], 1/21)
    assert_allclose(result['aleatoric'], 17/84)
    assert_allclose(result['total'], .25)


def test_weighted_mixture_matches_explicit_discrete_distribution():
    means = np.array([[-3.], [2.], [8.]])
    weights = np.array([1., 2., 1.])
    result = ensemble_uncertainties(means, np.zeros_like(means), weights=weights)
    assert_allclose(result['mean'], 2.25)
    assert_allclose(result['total'], np.dot(weights/4, means[:, 0]**2) - 2.25**2)
    one = ensemble_uncertainties(means[:1], np.ones_like(means[:1]), np.ones_like(means[:1]))
    assert_allclose(one['between_epistemic'], 0)
    assert_allclose(one['total'], 2)


def test_axes_permutation_preserves_samples_and_targets():
    rng = np.random.default_rng(21)
    means = rng.normal(size=(3, 5, 2))
    a, e = rng.random(size=(2, 3, 5, 2))
    expected = ensemble_uncertainties(means, a, e)
    actual = ensemble_uncertainties(means.transpose(1, 2, 0), a.transpose(1, 2, 0), e.transpose(1, 2, 0), axis=-1)
    for name in expected:
        assert actual[name].shape == (5, 2)
        assert_allclose(actual[name], expected[name])


@pytest.mark.parametrize('scaler', [StandardScaler(), StandardScaler(with_std=False), MinMaxScaler(feature_range=(-2, 3)), RobustScaler(), MaxAbsScaler()])
def test_inverse_scaling_matches_transformed_second_moments(scaler):
    data = np.array([[10., -4.], [12., 1.], [19., 5.], [31., 14.]])
    scaler.fit(data)
    means = np.array([[[.2, .3], [.4, .6]], [[.8, .1], [.7, .5]]])
    a = np.ones_like(means) * .2
    e = np.ones_like(means) * .1
    mu, aa, ee = inverse_scale_moments(means, a, e, scaler)
    z = np.zeros((1, 2))
    slopes = scaler.inverse_transform(z+1)-scaler.inverse_transform(z)
    assert_allclose(mu, scaler.inverse_transform(means.reshape(-1, 2)).reshape(means.shape))
    assert_allclose(aa, a*slopes**2)
    assert_allclose(ee, e*slopes**2)
    before = ensemble_uncertainties(means, a, e)
    after = ensemble_uncertainties(mu, aa, ee)
    assert_allclose(after['total'], before['total']*slopes**2)


def test_nonlinear_scaler_is_rejected():
    scaler = QuantileTransformer(n_quantiles=3).fit(np.arange(3).reshape(-1, 1))
    with pytest.raises(TypeError, match='scaling requires'):
        inverse_scale_moments([[0.]], [[1.]], [[1.]], scaler)


@pytest.mark.parametrize('kwargs', [
    {'aleatoric': None}, {'aleatoric': [[-1.], [0.]]},
    {'aleatoric': [[0.]]}, {'weights': [0, 0]}, {'weights': [-1, 2]},
    {'weights': [1, np.nan]}, {'means': [[np.inf], [0.]]}, {'axis': 3},
])
def test_invalid_mixture_inputs(kwargs):
    args = dict(means=[[0.], [1.]], aleatoric=[[1.], [1.]])
    args.update(kwargs)
    with pytest.raises(ValueError):
        ensemble_uncertainties(**args)


@pytest.mark.parametrize('params', [(0, 2, 1), (1, 1, 1), (1, 2, 0)])
def test_invalid_nig_moments(params):
    with pytest.raises(ValueError):
        nig_uncertainties(0., *params)


@pytest.mark.parametrize('alpha', [[0, 1], [-1, 2], [1], [np.nan, 1]])
def test_invalid_dirichlet_moments(alpha):
    with pytest.raises(ValueError):
        dirichlet_uncertainties(alpha)


def test_classifier_plotting_preserves_input_variance_units(monkeypatch):
    import pandas as pd
    from mlguess import classifier_uq
    df = pd.DataFrame(dict(true_label=[0, 1], pred_conf1=[.5, .2],
                           pred_conf2=[.2, .5], pred_conf3=[.2, .2],
                           pred_conf4=[.1, .1], aleatoric=[.16, .09],
                           epistemic=[.04, .01], entropy=[.5, .8]))
    original = df.copy(deep=True)
    plotted = []
    def capture(frame, xcols, ycols, **kwargs):
        assert all(name in frame for name in xcols + ycols)
        plotted.append(frame.copy())
    monkeypatch.setattr(classifier_uq, 'plot_uncertainties', capture)
    for name in ['classifier_attribution', 'classifier_skill_scores', 'classifier_discard_fraction']:
        monkeypatch.setattr(classifier_uq, name, lambda *args, **kwargs: None)
    classifier_uq.uq_results(df)
    classifier_uq.uq_results(df)
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(plotted[0], plotted[1])
    assert_allclose(plotted[0]['total'], np.sqrt(original.aleatoric + original.epistemic))


def test_overflowing_derived_moments_raise_instead_of_silent_infinity():
    with pytest.raises(ValueError, match="finite"):
        nig_uncertainties(0., 1e-300, 2., 1e300)
    with pytest.raises(ValueError, match="finite"):
        ensemble_uncertainties([[-1e200], [1e200]], [[0.], [0.]])
    with pytest.raises(ValueError, match="finite"):
        dirichlet_uncertainties([1e308, 1e308])
