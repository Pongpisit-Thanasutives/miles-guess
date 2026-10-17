"""Numerical uncertainty-unit and law-of-total-variance tests for Keras models."""
import math

import numpy as np
import pytest

keras = pytest.importorskip("keras")
from keras import ops
from scipy.special import digamma
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from mlguess.keras.models import CategoricalDNN, RegressorDNN


@pytest.mark.parametrize("targets", [1, 2])
@pytest.mark.parametrize("distribution", ["evidential", "gaussian"])
def test_fitted_regressor_moments_and_per_target_training_variance(targets, distribution):
    rng = np.random.default_rng(29)
    x = rng.normal(size=(12, 2)).astype("float32")
    y = rng.uniform(.2, .8, size=(12, targets)).astype("float32")
    model = RegressorDNN(n_inputs=2, n_output_tasks=targets, hidden_layers=0,
                         evidential=distribution == "evidential", uncertainty=distribution == "gaussian",
                         epochs=1, batch_size=4, verbose=0)
    fit_y = y[:, 0] if targets == 1 else y
    history = model.fit(x, fit_y, validation_data=(x, fit_y), verbose=0)
    assert np.isfinite(history.history["loss"]).all()
    assert np.isfinite(history.history["val_loss"]).all()
    assert np.asarray(model.training_var).shape == (targets,)
    np.testing.assert_allclose(model.training_var, np.var(y, axis=0))
    raw = model.predict(x[:5], return_uncertainties=False, verbose=0, batch_size=3)
    moments = model.predict_uncertainties(x[:5], verbose=0, batch_size=3)
    mean, aleatoric, epistemic = np.split(moments, 3, axis=-1)
    assert moments.shape == (5, 3 * targets)
    assert np.isfinite(moments).all()
    assert np.all(aleatoric > 0)
    if distribution == "evidential":
        mu, v, alpha, beta = np.split(raw, 4, axis=-1)
        np.testing.assert_allclose(aleatoric, beta / (alpha - 1))
        np.testing.assert_allclose(epistemic, beta / (v * (alpha - 1)), rtol=2e-7)
        np.testing.assert_allclose(model.predict(x[:5], verbose=0, batch_size=3), moments)
    else:
        mu, sigma = np.split(raw, 2, axis=-1)
        np.testing.assert_allclose(aleatoric, sigma ** 2)
        np.testing.assert_array_equal(epistemic, np.zeros_like(mu))
        # Preserve existing predict contract: Gaussian output is mean + std.
        np.testing.assert_allclose(model.predict(x[:5], verbose=0, batch_size=3), raw)
    np.testing.assert_allclose(mean, mu)


@pytest.mark.parametrize("factory", [StandardScaler, lambda: StandardScaler(with_std=False),
                                     lambda: MinMaxScaler(feature_range=(-2, 3)), RobustScaler, MaxAbsScaler])
@pytest.mark.parametrize("distribution", ["evidential", "gaussian"])
def test_regression_affine_inverse_scales_variances_with_squared_slope(factory, distribution):
    scaler = factory().fit(np.array([[-4., 20.], [0., 40.], [6., 80.], [10., 120.]]))
    model = RegressorDNN(n_inputs=2, n_output_tasks=2, hidden_layers=0,
                         evidential=distribution == "evidential", uncertainty=distribution == "gaussian",
                         training_var=[999., 888.])
    mu = np.array([[.25, .5], [-.1, .2]])
    if distribution == "evidential":
        v = np.array([[2., 4.], [4., 2.]])
        alpha = np.array([[3., 5.], [5., 3.]])
        beta = np.array([[6., 8.], [8., 6.]])
        raw = np.concatenate([mu, v, alpha, beta], axis=-1)
        aleatoric = beta / (alpha - 1)
        epistemic = aleatoric / v
    else:
        sigma = np.array([[2., 3.], [4., 5.]])
        raw = np.concatenate([mu, sigma], axis=-1)
        aleatoric, epistemic = sigma ** 2, np.zeros_like(sigma)
    mean_out, ale_out, epi_out = np.split(model.calc_uncertainties(raw, y_scaler=scaler), 3, axis=-1)
    inverse_slope = (scaler.inverse_transform(np.ones((1, 2))) -
                     scaler.inverse_transform(np.zeros((1, 2))))[0]
    np.testing.assert_allclose(mean_out, scaler.inverse_transform(mu))
    np.testing.assert_allclose(ale_out, aleatoric * inverse_slope ** 2)
    np.testing.assert_allclose(epi_out, epistemic * inverse_slope ** 2)
    # Changing metadata must never change distribution-derived moments.
    original = model.calc_uncertainties(raw)
    model.training_var = np.zeros(2)
    np.testing.assert_array_equal(model.calc_uncertainties(raw), original)


def test_unfitted_nig_moments_need_no_training_variance():
    model = RegressorDNN(n_inputs=1, n_output_tasks=1, evidential=True, hidden_layers=0)
    np.testing.assert_allclose(model.calc_uncertainties(np.array([[7., 2., 3., 6.]])), [[7., 3., 1.5]])


def _fixed_stochastic_outputs(monkeypatch, model, outputs, batch_size):
    """Keep Keras call plumbing but provide known pass outputs for exact tests."""
    samples = outputs.shape[1]
    model(np.zeros((1, model.n_inputs), dtype="float32"))
    calls = {"count": 0}
    batches = math.ceil(samples / batch_size)
    def fake_call(inputs):
        x = ops.convert_to_numpy(inputs)
        assert x.ndim == 2  # Old implementation incorrectly added a feature axis.
        run = calls["count"] // batches
        calls["count"] += 1
        return ops.convert_to_tensor(outputs[run, x[:, 0].astype(int)])
    monkeypatch.setattr(model, "call", fake_call)
    return np.arange(samples, dtype="float32")[:, None], calls


@pytest.mark.parametrize("batch_size", [2, 20])
@pytest.mark.parametrize("evidential", [False, True])
def test_categorical_dropout_exact_total_variance(monkeypatch, batch_size, evidential):
    model = CategoricalDNN(n_inputs=1, n_classes=2, hidden_layers=0, evidential=evidential)
    values = np.array([[3., 1.], [1., 3.]]) if evidential else np.array([[.8, .2], [.2, .8]])
    outputs = np.repeat(values[:, None, :], 5, axis=1).astype("float32")
    x, calls = _fixed_stochastic_outputs(monkeypatch, model, outputs, batch_size)
    probability, aleatoric, epistemic, entropy, mi = model.predict_dropout(
        x, mc_forward_passes=2, batch_size=batch_size)
    if evidential:
        alpha = outputs.astype(float) + 1
        concentration = alpha.sum(-1, keepdims=True)
        pass_probability = alpha / concentration
        within = pass_probability * (1 - pass_probability) / (concentration + 1)
        expected_entropy = (digamma(concentration + 1).squeeze(-1) -
                            np.sum(pass_probability * digamma(alpha + 1), axis=-1))
    else:
        pass_probability = outputs.astype(float)
        within = np.zeros_like(pass_probability)
        expected_entropy = -np.sum(pass_probability * np.log(pass_probability), axis=-1)
    expected_mean = pass_probability.mean(0)
    np.testing.assert_allclose(probability, expected_mean)
    np.testing.assert_allclose(aleatoric, (pass_probability * (1 - pass_probability) - within).mean(0))
    np.testing.assert_allclose(epistemic, within.mean(0) + pass_probability.var(0))
    np.testing.assert_allclose(aleatoric + epistemic, probability * (1 - probability), atol=1e-8)
    np.testing.assert_allclose(entropy, -np.sum(expected_mean * np.log(expected_mean), axis=-1))
    np.testing.assert_allclose(mi, entropy - expected_entropy.mean(0))
    assert calls["count"] == 2 * math.ceil(5 / batch_size)


def test_single_pass_dirichlet_has_nonzero_within_epistemic_uncertainty(monkeypatch):
    model = CategoricalDNN(n_inputs=1, n_classes=2, hidden_layers=0, evidential=True)
    x, _ = _fixed_stochastic_outputs(monkeypatch, model, np.zeros((1, 3, 2), dtype="float32"), 20)
    mean, ale, epi, entropy, mi = model.predict_dropout(x, mc_forward_passes=1, batch_size=20)
    np.testing.assert_allclose(mean, .5)
    np.testing.assert_allclose(ale, 1 / 6)
    np.testing.assert_allclose(epi, 1 / 12)
    np.testing.assert_allclose(entropy, np.log(2))
    np.testing.assert_allclose(mi, np.log(2) - .5)


def test_standard_linear_logits_are_converted_to_probabilities(monkeypatch):
    model = CategoricalDNN(n_inputs=1, n_classes=2, hidden_layers=0, output_activation="linear")
    logits = np.repeat(np.array([[[0., np.log(3.)]]], dtype="float32"), 3, axis=1)
    x, _ = _fixed_stochastic_outputs(monkeypatch, model, logits, 4)
    mean, ale, epi, _, mi = model.predict_dropout(x, mc_forward_passes=1, batch_size=4)
    np.testing.assert_allclose(mean, np.tile([.25, .75], (3, 1)), rtol=1e-7)
    np.testing.assert_allclose(ale, mean * (1 - mean))
    np.testing.assert_array_equal(epi, np.zeros_like(epi))
    np.testing.assert_allclose(mi, 0, atol=1e-14)


def test_actual_categorical_fit_and_small_batch_dropout():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(10, 2)).astype("float32")
    y = np.eye(2)[np.arange(10) % 2].astype("float32")
    model = CategoricalDNN(n_inputs=2, n_classes=2, hidden_layers=1, hidden_neurons=4,
                           evidential=True, use_dropout=True, dropout_alpha=.5, epochs=1, batch_size=5)
    history = model.fit(x, y, verbose=0)
    assert np.isfinite(history.history["loss"]).all()
    mean, ale, epi, entropy, mi = model.predict_dropout(x[:3], mc_forward_passes=4, batch_size=10)
    assert mean.shape == (3, 2)
    np.testing.assert_allclose(mean.sum(-1), 1, atol=1e-7)
    np.testing.assert_allclose(ale + epi, mean * (1 - mean), atol=1e-7)
    assert np.isfinite(entropy).all() and np.isfinite(mi).all()


def test_dropout_preserves_batch_norm_state_and_trainable_flag():
    model = CategoricalDNN(n_inputs=2, n_classes=2, hidden_layers=0)
    batch_norm = keras.layers.BatchNormalization()
    model.model_layers.insert(1, batch_norm)
    x = np.array([[1., 2.], [3., 5.], [7., 11.]], dtype="float32")
    model(x, training=False)
    batch_norm.moving_mean.assign([10., 20.])
    before = [ops.convert_to_numpy(weight).copy() for weight in batch_norm.non_trainable_weights]
    model.predict_dropout(x, mc_forward_passes=2, batch_size=2)
    after = [ops.convert_to_numpy(weight) for weight in batch_norm.non_trainable_weights]
    assert batch_norm.trainable is True
    for previous, current in zip(before, after):
        np.testing.assert_array_equal(previous, current)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"mc_forward_passes": 0}, {"batch_size": 1.5}])
def test_dropout_rejects_invalid_sampling_options(kwargs):
    model = CategoricalDNN(n_inputs=2, n_classes=2)
    with pytest.raises(ValueError):
        model.predict_dropout(np.ones((2, 2)), **kwargs)


def test_actual_dropout_produces_between_pass_variance():
    model = CategoricalDNN(n_inputs=2, n_classes=2, hidden_layers=1, hidden_neurons=4,
                           use_dropout=True, dropout_alpha=.5)
    x = np.full((3, 2), .1, dtype="float32")
    model(x)
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            layer.kernel.assign(np.full(layer.kernel.shape, .1, dtype="float32"))
            layer.bias.assign(np.zeros(layer.bias.shape, dtype="float32"))
    output = model.model_layers[-1]
    output.kernel.assign(np.tile([1., -1.], (4, 1)))
    _, _, epistemic, _, mi = model.predict_dropout(x, mc_forward_passes=20, batch_size=2)
    assert np.all(epistemic > 0)
    assert np.all(mi > 0)


@pytest.mark.parametrize("parameters", [
    [0., 1., .5, -1.],  # Two invalid negatives formerly produced positive variances.
    [0., 0., 2., 1.],
    [0., -1., 2., 1.],
    [0., 1., 1., 1.],
    [0., 1., 2., 0.],
    [np.inf, 1., 2., 1.],
    [0., 1., 2., np.nan],
])
def test_nig_uncertainties_reject_invalid_parameters(parameters):
    model = RegressorDNN(n_inputs=1, n_output_tasks=1, evidential=True, hidden_layers=0)
    with pytest.raises(ValueError):
        model.calc_uncertainties(np.array([parameters]))


@pytest.mark.parametrize("parameters", [[0., -1.], [0., np.nan], [0., np.inf], [np.inf, 1.]])
def test_gaussian_uncertainties_reject_invalid_parameters(parameters):
    model = RegressorDNN(n_inputs=1, n_output_tasks=1, uncertainty=True, hidden_layers=0)
    with pytest.raises(ValueError):
        model.calc_uncertainties(np.array([parameters]))


def test_gaussian_zero_sigma_is_valid_zero_variance():
    model = RegressorDNN(n_inputs=1, n_output_tasks=1, uncertainty=True, hidden_layers=0)
    np.testing.assert_array_equal(model.calc_uncertainties(np.array([[2., 0.]])), [[2., 0., 0.]])
