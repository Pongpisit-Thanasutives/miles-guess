"""Analytical tests for legacy Keras moments and current/legacy MC aggregation."""

import os

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

os.environ.setdefault("KERAS_BACKEND", "torch")
keras = pytest.importorskip("keras")

from mlguess.keras.deprecated.models import GaussianRegressorDNN, EvidentialRegressorDNN, RegressorDNN
from mlguess.keras.models import RegressorDNN as CurrentRegressor
from mlguess.keras.monte_carlo import mc_raw_predictions, monte_carlo_ensemble, reset_weights


def constant_network(parameters, inputs=2):
    parameters = np.asarray(parameters, dtype=np.float32)
    return keras.Sequential([
        keras.layers.Input(shape=(inputs,)),
        keras.layers.Dense(len(parameters), kernel_initializer="zeros",
                           bias_initializer=keras.initializers.Constant(parameters)),
    ])


@pytest.mark.parametrize("model_cls,raw,expected_a,expected_e", [
    (GaussianRegressorDNN, [1., 2., 3., 4.], [9., 16.], None),
    (EvidentialRegressorDNN, [1., 2., 2., 4., 3., 5., 8., 12.], [4., 3.], [2., .75]),
])
@pytest.mark.parametrize("scaled", [False, True])
def test_legacy_parameter_units_and_variance_scale(model_cls, raw, expected_a, expected_e, scaled):
    wrapper = model_cls()
    wrapper.training_var = [9999., 7777.]  # Metadata must not multiply output variances.
    scaler = StandardScaler().fit([[8., 15.], [12., 25.]]) if scaled else None
    raw = np.asarray([raw])
    original = raw.copy()
    components = wrapper.calc_uncertainties(raw, scaler)
    np.testing.assert_allclose(components[0], [[12., 30.]] if scaled else [[1., 2.]])
    factor = np.array([4., 25.]) if scaled else 1.
    np.testing.assert_allclose(components[1], np.array([expected_a]) * factor)
    if expected_e is not None:
        np.testing.assert_allclose(components[2], np.array([expected_e]) * factor)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("model_cls,raw,parts", [
    (GaussianRegressorDNN, [.25, .5, 2., 3.], 2),
    (EvidentialRegressorDNN, [.25, .5, 2., 4., 3., 5., 8., 12.], 3),
])
def test_mc_decodes_parameter_width_and_inverse_transforms_once(model_cls, raw, parts):
    class CountingScaler(StandardScaler):
        calls = 0
        def inverse_transform(self, values, copy=None):
            self.calls += 1
            return super().inverse_transform(values, copy=copy)

    wrapper = model_cls(batch_size=1)
    wrapper.model = constant_network(raw)
    scaler = CountingScaler().fit([[8., 15.], [12., 25.]])
    samples = wrapper.predict_monte_carlo(np.zeros((2, 2), dtype=np.float32), 3, scaler=scaler)
    assert len(samples) == parts
    assert all(sample.shape == (3, 2, 2) for sample in samples)
    assert scaler.calls == 3
    expected = wrapper.calc_uncertainties(np.tile(raw, (2, 1)), scaler)
    for actual, component in zip(samples, expected):
        np.testing.assert_allclose(actual, np.broadcast_to(component, (3, 2, 2)), rtol=1e-6)


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("evidential", [False, True])
def test_mc_finite_mixture_includes_within_evidential_variance(monkeypatch, current, evidential):
    if evidential:
        raw = np.array([[[-1., 2., 3., 8.]], [[3., 4., 5., 12.]]])
        wrapper = CurrentRegressor(evidential=True, n_inputs=1) if current else EvidentialRegressorDNN()
        expected_a, expected_e = 3.5, 5.375  # E[A]=3.5; E[E]=1.375; Var(mu)=4.
    else:
        raw = np.array([[[-1., 2.]], [[3., 4.]]])
        wrapper = CurrentRegressor(uncertainty=True, n_inputs=1) if current else GaussianRegressorDNN()
        expected_a, expected_e = 10., 4.
    import mlguess.keras.monte_carlo as mc
    monkeypatch.setattr(mc, "mc_raw_predictions", lambda *args, **kwargs: raw)
    mu, ale, epi = monte_carlo_ensemble(wrapper, np.zeros((1, 1)), np.zeros((1, 1)), 2)
    np.testing.assert_allclose(mu, [[1.]])
    np.testing.assert_allclose(ale, [[expected_a]])
    np.testing.assert_allclose(epi, [[expected_e]])


def test_single_evidential_mc_member_preserves_within_variance(monkeypatch):
    import mlguess.keras.monte_carlo as mc
    monkeypatch.setattr(mc, "mc_raw_predictions", lambda *args, **kwargs: np.array([[[1., 2., 3., 8.]]]))
    _, ale, epi = monte_carlo_ensemble(EvidentialRegressorDNN(), np.zeros((1, 1)), np.zeros((1, 1)), 1)
    np.testing.assert_allclose(ale, [[4.]])
    np.testing.assert_allclose(epi, [[2.]])


def test_selective_dropout_preserves_batchnorm_and_disables_gaussian_noise():
    bn = keras.layers.BatchNormalization()
    dropout = keras.layers.Dropout(.5, seed=41)
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)), bn, keras.layers.GaussianNoise(1.), dropout,
        keras.layers.Dense(1, kernel_initializer="ones", bias_initializer="zeros"),
    ])
    x = np.ones((20, 3), dtype=np.float32)
    model(x, training=False)
    original_call = dropout.call
    mean_before = keras.ops.convert_to_numpy(bn.moving_mean).copy()
    variance_before = keras.ops.convert_to_numpy(bn.moving_variance).copy()
    draws = mc_raw_predictions(model, x, 5, batch_size=7)
    assert draws.shape == (5, 20, 1)
    assert np.var(draws, axis=0).max() > 0
    unit = 2. / np.sqrt(1. + bn.epsilon)
    np.testing.assert_allclose(draws / unit, np.rint(draws / unit), atol=1e-6)
    assert draws.min() >= 0 and draws.max() <= 6.01
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(bn.moving_mean), mean_before)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(bn.moving_variance), variance_before)
    assert dropout.call == original_call
    assert bn.trainable


def test_legacy_ensemble_uses_matching_input_and_target_scalers_and_restores_state(tmp_path):
    wrapper = GaussianRegressorDNN()
    wrapper.model = constant_network([0., 1.], inputs=1)
    layer = wrapper.model.layers[-1]
    files = []
    for index, bias in enumerate(([0., 1.], [5., 2.])):
        layer.set_weights([np.array([[1., 0.]], dtype=np.float32), np.array(bias, dtype=np.float32)])
        path = tmp_path / f"model_seed0_split{index}.weights.h5"
        wrapper.model.save_weights(path)
        np.savetxt(str(path).replace(".h5", "_training_var.txt"), np.array([1000. * (index + 1)]))
        files.append(str(path))
    wrapper.ensemble_member_files = files
    wrapper.training_var = [333.]
    before = wrapper.model.get_weights()
    input_scalers = [StandardScaler().fit([[8.], [12.]]), StandardScaler().fit([[16.], [24.]])]
    target_scalers = [StandardScaler().fit([[8.], [12.]]), MinMaxScaler().fit([[100.], [140.]])]
    x = np.array([[10.], [20.]])
    with pytest.raises(ValueError, match="member_scalers"):
        wrapper.predict_ensemble(x)
    with pytest.raises(ValueError, match="member_input_scalers"):
        wrapper.predict_ensemble(x, member_scalers=target_scalers)
    means, aleatoric = wrapper.predict_ensemble(x, member_scalers=target_scalers, member_input_scalers=input_scalers)
    np.testing.assert_allclose(means, [[[10.], [20.]], [[200.], [300.]]])
    np.testing.assert_allclose(aleatoric, [[[4.], [4.]], [[6400.], [6400.]]])
    assert wrapper.training_var == [333.]
    for old, new in zip(before, wrapper.model.get_weights()):
        np.testing.assert_array_equal(old, new)


def test_nonlinear_target_scaler_is_rejected():
    nonlinear = QuantileTransformer(n_quantiles=3).fit([[1.], [2.], [3.]])
    with pytest.raises(TypeError, match="scal"):
        GaussianRegressorDNN().calc_uncertainties([[1., 2.]], nonlinear)


def test_distribution_parameters_keep_single_target_shapes_and_correct_units():
    scaler = StandardScaler().fit([[8.], [12.]])
    gaussian = GaussianRegressorDNN()
    gaussian.model = constant_network([1., 3.])
    mu, sigma = gaussian.predict_dist_params(np.zeros((1, 2), dtype=np.float32), scaler=scaler)
    np.testing.assert_allclose(mu, [[12.]])
    np.testing.assert_allclose(sigma, [[6.]])
    evidential = EvidentialRegressorDNN()
    evidential.model = constant_network([1., 2., 3., 8.])
    mu, nu, alpha, beta = evidential.predict_dist_params(np.zeros((1, 2), dtype=np.float32), y_scaler=scaler)
    np.testing.assert_allclose(mu, [[12.]])
    np.testing.assert_allclose(nu, [[2.]])
    np.testing.assert_allclose(alpha, [[3.]])
    np.testing.assert_allclose(beta, [[32.]])


def test_reset_weights_works_with_current_keras_backend():
    model = constant_network([1., 2.])
    model.layers[-1].bias.assign([3., 4.])
    reset_weights(model)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(model.layers[-1].bias), [1., 2.])


@pytest.mark.parametrize("current", [False, True])
def test_deterministic_mc_does_not_claim_zero_observation_noise(monkeypatch, current):
    import mlguess.keras.monte_carlo as mc
    monkeypatch.setattr(mc, "mc_raw_predictions", lambda *args, **kwargs: np.array([[[-1.]], [[3.]]]))
    model = CurrentRegressor(n_inputs=1) if current else RegressorDNN()
    mu, ale, epi = monte_carlo_ensemble(model, np.zeros((1, 1)), np.zeros((1, 1)), 2)
    np.testing.assert_allclose(mu, [[1.]])
    assert ale is None
    np.testing.assert_allclose(epi, [[4.]])


@pytest.mark.parametrize("method", ["predict_monte_carlo", "predict_ensemble"])
@pytest.mark.parametrize("outputs", [1, 3])
def test_legacy_gaussian_rejects_incompatible_component_count(method, outputs):
    model = GaussianRegressorDNN()
    args = (np.ones((2, 1)), 2) if method == "predict_monte_carlo" else (np.ones((2, 1)),)
    with pytest.raises(ValueError, match="num_outputs must be 2"):
        getattr(model, method)(*args, num_outputs=outputs)


@pytest.mark.parametrize("raw", [
    [0., -1., 3., 2.],  # negative virtual observation count
    [0., 0., 3., 2.],
    [0., 1., 1., 2.],  # infinite predictive variance
    [0., 1., .5, -2.],  # two negative factors must not masquerade as a variance
    [0., 1., 3., 0.],
    [0., 1., 3., -2.],
])
def test_legacy_nig_rejects_invalid_parameters(raw):
    with pytest.raises(ValueError, match="NIG finite-variance"):
        EvidentialRegressorDNN().calc_uncertainties([raw])


def test_legacy_gaussian_rejects_negative_standard_deviation():
    with pytest.raises(ValueError, match="standard deviations"):
        GaussianRegressorDNN().calc_uncertainties([[0., -2.]])


def test_legacy_coupled_nig_validates_after_deriving_virtual_count():
    model = EvidentialRegressorDNN(loss="evidentialFix", coupling_coef=2.)
    mu, ale, epi = model.calc_uncertainties([[1., -100., 3., 8.]])
    np.testing.assert_allclose(mu, [[1.]])
    np.testing.assert_allclose(ale, [[4.]])
    np.testing.assert_allclose(epi, [[2.]])
    with pytest.raises(ValueError, match="NIG finite-variance"):
        model.calc_uncertainties([[1., 100., .5, -2.]])


@pytest.mark.parametrize("loss", ["categorical_crossentropy", "evidential"])
def test_legacy_categorical_dropout_exact_lotv_and_information(monkeypatch, loss):
    from mlguess.keras.deprecated.models import CategoricalDNN
    import mlguess.keras.deprecated.models as legacy
    from scipy.special import digamma

    evidential = loss == "evidential"
    raw = np.array([[[3., 1.]], [[1., 5.]]]) if evidential else np.array([[[.2, .8]], [[.6, .4]]])
    monkeypatch.setattr(legacy, "mc_raw_predictions", lambda *args, **kwargs: raw)
    wrapper = CategoricalDNN(loss=loss)
    mean, ale, epi, entropy, mutual_information = wrapper.predict_dropout(np.zeros((1, 2)), 2)
    if evidential:
        alpha = raw + 1
        strength = alpha.sum(axis=-1, keepdims=True)
        probabilities = alpha / strength
        within = probabilities * (1 - probabilities) / (strength + 1)
        expected_ale = np.mean(probabilities * (1 - probabilities) - within, axis=0)
        expected_epi = within.mean(axis=0) + probabilities.var(axis=0, ddof=0)
        expected_entropy = (digamma(strength + 1).squeeze(-1)
                            - np.sum(probabilities * digamma(alpha + 1), axis=-1)).mean(axis=0)
    else:
        probabilities = raw
        expected_ale = np.array([[.2, .2]])
        expected_epi = np.array([[.04, .04]])
        expected_entropy = -np.sum(raw * np.log(raw), axis=-1).mean(axis=0)
    expected_mean = probabilities.mean(axis=0)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(ale, expected_ale)
    np.testing.assert_allclose(epi, expected_epi)
    np.testing.assert_allclose(ale + epi, mean * (1 - mean))
    np.testing.assert_allclose(entropy, -np.sum(mean * np.log(mean), axis=-1))
    np.testing.assert_allclose(mutual_information, entropy - expected_entropy)


def test_legacy_categorical_dropout_handles_small_batches_and_batchnorm():
    from mlguess.keras.deprecated.models import CategoricalDNN

    wrapper = CategoricalDNN(batch_size=128)
    bn = keras.layers.BatchNormalization()
    wrapper.model = keras.Sequential([
        keras.layers.Input(shape=(2,)), bn, keras.layers.Dropout(.5, seed=23),
        keras.layers.Dense(2, activation="softmax"),
    ])
    x = np.ones((3, 2), dtype=np.float32)
    wrapper.model(x, training=False)
    moving_mean = keras.ops.convert_to_numpy(bn.moving_mean).copy()
    moving_variance = keras.ops.convert_to_numpy(bn.moving_variance).copy()
    mean, ale, epi, entropy, mutual_information = wrapper.predict_dropout(x, mc_forward_passes=3)
    assert mean.shape == ale.shape == epi.shape == (3, 2)
    assert entropy.shape == mutual_information.shape == (3,)
    np.testing.assert_allclose(ale + epi, mean * (1 - mean), atol=1e-7)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(bn.moving_mean), moving_mean)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(bn.moving_variance), moving_variance)


def test_legacy_categorical_trained_evidential_loss_object_is_recognized(monkeypatch):
    from mlguess.keras.deprecated.models import CategoricalDNN
    from mlguess.keras.deprecated.losses import DirichletEvidentialLoss
    import mlguess.keras.deprecated.models as legacy

    monkeypatch.setattr(legacy, "mc_raw_predictions", lambda *args, **kwargs: np.zeros((1, 1, 2)))
    wrapper = CategoricalDNN(loss=DirichletEvidentialLoss())
    mean, ale, epi, entropy, mutual_information = wrapper.predict_dropout(np.zeros((1, 2)), 1)
    np.testing.assert_allclose(mean, [[.5, .5]])
    np.testing.assert_allclose(ale, [[1 / 6, 1 / 6]])
    np.testing.assert_allclose(epi, [[1 / 12, 1 / 12]])
    np.testing.assert_allclose(entropy, [np.log(2)])
    np.testing.assert_allclose(mutual_information, [np.log(2) - .5])


def test_legacy_categorical_logits_are_decoded_before_moments(monkeypatch):
    from mlguess.keras.deprecated.models import CategoricalDNN
    import mlguess.keras.deprecated.models as legacy

    monkeypatch.setattr(legacy, "mc_raw_predictions", lambda *args, **kwargs: np.log(np.array([[[.2, .8]], [[.6, .4]]])))
    mean, ale, epi, _, _ = CategoricalDNN(output_activation="linear").predict_dropout(np.zeros((1, 2)), 2)
    np.testing.assert_allclose(mean, [[.4, .6]])
    np.testing.assert_allclose(ale, [[.2, .2]])
    np.testing.assert_allclose(epi, [[.04, .04]])
