"""Independent numeric, fit, serialization, callback, and legacy Keras audit tests."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

keras = pytest.importorskip("keras")
from keras import ops
from scipy.special import digamma, gammaln
from scipy.stats import t

from mlguess.keras import callbacks, losses
from mlguess.keras.layers import DenseNormal, DenseNormalGamma
from mlguess.keras.models import CategoricalDNN, RegressorDNN
from mlguess.keras.deprecated import losses as old_losses
from mlguess.keras.deprecated import models as old_models
from mlguess.keras.seed import seed_everything


def array(value):
    return ops.convert_to_numpy(value)


def test_callback_builder_does_not_mutate_config_and_retains_lr(tmp_path):
    config = {"save_loc": str(tmp_path), "callbacks": {
        "ModelCheckpoint": {"filepath": "best.keras"},
        "CSVLogger": {"filename": "training.csv"},
        "EarlyStopping": {"patience": 2},
        "ReduceLROnPlateau": {"patience": 1},
        "LearningRateScheduler": {"drop": .5, "epochs_drop": 3, "init_lr": .004},
    }}
    original = copy.deepcopy(config)
    first = callbacks.get_callbacks(config, path_extend="first")
    second = callbacks.get_callbacks(config, path_extend="second")
    assert config == original
    assert len(first) == len(second) == 6
    scheduler = next(item for item in first if isinstance(item, keras.callbacks.LearningRateScheduler))
    assert scheduler.schedule(0) == pytest.approx(.004)
    assert callbacks.get_callbacks({}) == []
    with pytest.raises(ValueError):
        callbacks.step_decay(0, epochs_drop=0)


def test_lr_tracker_and_epoch_callback_update_given_objects():
    optimizer = keras.optimizers.Adam(learning_rate=.02)
    tracker = callbacks.LearningRateTracker()
    tracker.set_model(SimpleNamespace(optimizer=optimizer))
    logs = {}
    tracker.on_epoch_end(0, logs)
    assert logs["lr"] == pytest.approx(.02)
    epoch = keras.Variable(2., trainable=False)
    report = callbacks.ReportEpoch(epoch)
    report.on_epoch_end(0)
    assert float(array(epoch)) == 3
    restored = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(report))
    assert float(array(restored.epoch_var)) == 3


def test_callback_scientific_metrics_and_generic_model_prediction():
    y = np.eye(2)[[0, 1, 0, 1]]
    callback = callbacks.MetricsCallback(np.ones((4, 2)), y, n_bins=5)
    assert callback.mean_csi(y) == 1
    assert callback.mce(y.argmax(1), y) == 0
    assert callback.ece(y.argmax(1)[:, None], y) == 0
    assert callback.ave_acc(y.argmax(1), y.argmax(1)) == 1
    probability = np.array([[.6, .4], [.4, .6], [.9, .1], [.1, .9]])
    labels = np.array([0, 1, 0, 0])
    assert callback.ece(labels, probability) == pytest.approx(.4)
    assert callback.mce(labels, probability) == pytest.approx(.4)
    callback.set_model(SimpleNamespace(predict=lambda x, verbose=0: y))
    logs = {}
    callback.on_epoch_end(0, logs)
    assert logs["val_csi"] == 1 and logs["val_auc"] == 1
    callback.use_uncertainty = True
    callback.set_model(SimpleNamespace(predict=lambda x, verbose=0: 10 * y))
    callback.on_epoch_end(1, logs)
    assert logs["val_csi"] == 1
    with pytest.raises(ValueError):
        callbacks.MetricsCallback(np.ones((4, 2)), y, n_bins=0)


@pytest.mark.parametrize("layer_cls", [DenseNormalGamma, DenseNormal])
@pytest.mark.parametrize("spectral", [False, True])
def test_distribution_layers_preserve_config_and_nested_shapes(layer_cls, spectral):
    layer = layer_cls(2, eps=.002, spectral_normalization=spectral)
    rebuilt = keras.layers.deserialize(keras.layers.serialize(layer))
    assert rebuilt.eps == .002 and rebuilt.spectral_normalization is spectral
    assert rebuilt.compute_output_shape((None, 3, 5)) == (None, 3, layer.NUM_OUTPUT_PARAMS * 2)
    output = rebuilt(np.ones((4, 5), dtype="float32"))
    assert output.shape == (4, layer.NUM_OUTPUT_PARAMS * 2)
    assert np.isfinite(array(output)).all()
    with pytest.raises(ValueError):
        layer_cls(0)


def _nig_case():
    y = np.array([[.5, -1.], [2., 1.]], dtype="float32")
    mu = np.array([[0., .1], [1., .2]], dtype="float32")
    v = np.array([[2., 3.], [4., 2.]], dtype="float32")
    alpha = np.array([[3., 4.], [5., 2.]], dtype="float32")
    beta = np.array([[1., 2.], [3., 4.]], dtype="float32")
    raw = np.concatenate([mu, v, alpha, beta], axis=-1)
    return y, mu, v, alpha, beta, raw


@pytest.mark.parametrize("legacy", [False, True])
def test_nig_loss_numeric_parity_sample_weights_and_serialization(legacy):
    y, mu, v, alpha, beta, raw = _nig_case()
    coefficient = .37
    loss = old_losses.EvidentialRegressionLoss(coefficient) if legacy else losses.evidential_reg_loss(coefficient)
    scale = np.sqrt(beta * (1 + v) / (v * alpha))
    nll = -t.logpdf(y, df=2 * alpha, loc=mu, scale=scale)
    regularizer = np.abs(y - mu) * (2 * v + alpha)
    expected = np.mean(nll + coefficient * regularizer, axis=-1)
    np.testing.assert_allclose(array(loss(y, raw)), expected.mean(), rtol=3e-6)
    np.testing.assert_allclose(array(loss(y, raw, sample_weight=np.array([1., 0.]))), expected[0] / 2, rtol=3e-6)
    restored = keras.losses.deserialize(keras.losses.serialize(loss))
    np.testing.assert_allclose(array(restored(y, raw)), array(loss(y, raw)))
    if legacy:
        np.testing.assert_allclose(array(loss.nig_nll(*map(ops.convert_to_tensor, [y, mu, v, alpha, beta]), reduce=False)), nll, rtol=3e-6)
        np.testing.assert_allclose(array(loss.nig_reg(*map(ops.convert_to_tensor, [y, mu, v, alpha]), reduce=False)), regularizer)


@pytest.mark.parametrize("legacy", [False, True])
def test_categorical_loss_numeric_parity_numpy_class_weights_and_serialization(legacy):
    y = np.eye(2, dtype="float32")
    raw = np.array([[2., .5], [.2, 3.]], dtype="float32")
    alpha = np.maximum(raw, 0) + 1
    strength = alpha.sum(1, keepdims=True)
    mean = alpha / strength
    ahat = y + (1 - y) * alpha
    s = ahat.sum(1, keepdims=True)
    kl = gammaln(s) - gammaln(ahat).sum(1, keepdims=True) - gammaln(2.)
    kl += ((ahat - 1) * (digamma(ahat) - digamma(s))).sum(1, keepdims=True)
    expected = ((y - mean) ** 2 + alpha * (strength - alpha) / (strength ** 2 * (strength + 1))).sum(1) + .5 * kl[:, 0]
    report = callbacks.ReportEpoch(initial_epoch=1.)
    loss = (old_losses.DirichletEvidentialLoss(callback=report, annealing_coef=2.) if legacy else
            losses.evidential_cat_loss(2., report))
    np.testing.assert_allclose(array(loss(y, raw)), expected.mean(), rtol=3e-6)
    np.testing.assert_allclose(array(loss(y, raw, sample_weight=np.array([1., 0.]))), expected[0] / 2, rtol=3e-6)
    rebuilt = keras.losses.deserialize(keras.losses.serialize(loss))
    np.testing.assert_allclose(array(rebuilt(y, raw)), expected.mean(), rtol=3e-6)
    weighted = (old_losses.DirichletEvidentialLoss(callback=report, class_weights=np.array([1., 2.])) if legacy else
                losses.evidential_cat_loss(2., report, class_weights=np.array([1., 2.])))
    assert np.isfinite(array(weighted(y, raw)))


@pytest.mark.parametrize("module", [losses, old_losses])
def test_coupled_loss_numerics_config_and_gaussian_loss(module):
    y, mu, _, alpha, beta, raw = _nig_case()
    coupled = module.EvidentialRegressionCoupledLoss(r=2., coeff=.1)
    v = alpha - 1
    expected = np.mean(-t.logpdf(y, df=2 * alpha, loc=mu, scale=np.sqrt(beta * (1 + v) / (v * alpha))) +
                       .1 * np.abs(y - mu) * (v + 2 * alpha))
    np.testing.assert_allclose(array(coupled(y, raw)), expected, rtol=3e-6)
    restored = keras.losses.deserialize(keras.losses.serialize(coupled))
    np.testing.assert_allclose(array(restored(y, raw)), expected, rtol=3e-6)
    sigma = np.array([[1., 2.], [3., 4.]], dtype="float32")
    expected_gaussian = .5 * np.log(2 * np.pi) + np.log(sigma) + .5 * ((y - mu) / sigma) ** 2
    args = (ops.convert_to_tensor(y), ops.convert_to_tensor(np.concatenate([mu, sigma], -1)))
    np.testing.assert_allclose(array(module.gaussian_nll(*args, reduce=False)), expected_gaussian.mean(-1), rtol=2e-6)
    with pytest.raises(ValueError):
        module.EvidentialRegressionCoupledLoss(r=0.)


@pytest.mark.parametrize("evidential", [False, True])
def test_current_classifier_fit_overrides_callbacks_and_roundtrip(tmp_path, evidential):
    x = np.array([[0., 1.], [1., 0.], [0., 0.], [1., 1.]], dtype="float32")
    y = np.eye(2, dtype="float32")[[0, 1, 0, 1]]
    model = CategoricalDNN(n_inputs=2, n_classes=2, hidden_layers=0, evidential=evidential,
                           optimizer=keras.optimizers.Adam(.001), kernel_reg=keras.regularizers.L1(.01), epochs=3)
    seen = []
    callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: seen.append(epoch))
    history = model.fit(x, y, epochs=1, batch_size=2, callbacks=[callback], verbose=0)
    assert len(history.history["loss"]) == 1 and seen == [0]
    expected = model.predict(x, return_uncertainties=False, verbose=0)
    path = tmp_path / "classifier.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_allclose(restored.predict(x, return_uncertainties=False, verbose=0), expected, rtol=1e-6)
    assert isinstance(restored.model_layers[0].kernel_regularizer, keras.regularizers.L1)
    np.testing.assert_allclose(array(restored.loss(y, expected)), array(model.loss(y, expected))) if evidential else None


@pytest.mark.parametrize("family", [old_models.RegressorDNN, old_models.GaussianRegressorDNN, old_models.EvidentialRegressorDNN])
@pytest.mark.parametrize("targets", [1, 2])
def test_legacy_regressor_fit_metrics_and_safe_keras_save(tmp_path, family, targets):
    rng = np.random.default_rng(41)
    x = rng.uniform(.1, .9, size=(8, 2)).astype("float32")
    y = rng.uniform(.1, .9, size=(8, targets)).astype("float32")
    wrapper = family(hidden_layers=1, hidden_neurons=3, metrics="mae", kernel_reg="l1",
                     epochs=1, batch_size=4, save_path=str(tmp_path), model_name="regressor.keras")
    wrapper.build_neural_network(2, targets)
    assert isinstance(wrapper.model.layers[1].kernel_regularizer, keras.regularizers.L1)
    history = wrapper.fit(x, y[:, 0] if targets == 1 else y, validation_data=(x, y))
    assert np.isfinite(history.history["loss"]).all()
    raw = wrapper.model.predict(x, verbose=0)
    mu = raw[:, :targets]
    np.testing.assert_allclose(array(wrapper.mae(ops.convert_to_tensor(y), ops.convert_to_tensor(raw))), np.mean(np.abs(y-mu), -1), rtol=1e-6)
    np.testing.assert_allclose(array(wrapper.mse(ops.convert_to_tensor(y), ops.convert_to_tensor(raw))), np.mean((y-mu)**2, -1), rtol=1e-6)
    wrapper.save_model()
    assert (tmp_path / "regressor.keras").read_bytes()[:2] == b"PK"
    assert (tmp_path / "regressor_training_var.txt").is_file()
    restored = keras.models.load_model(tmp_path / "regressor.keras", compile=False)
    np.testing.assert_allclose(restored.predict(x, verbose=0), raw, rtol=1e-6)


@pytest.mark.parametrize("evidential", [False, True])
def test_legacy_classifier_can_actually_fit_and_ensemble_restores_weights(tmp_path, evidential):
    x = np.array([[0., 1.], [1., 0.], [0., 0.], [1., 1.]], dtype="float32")
    y = np.eye(2, dtype="float32")[[0, 1, 0, 1]]
    model = old_models.CategoricalDNN(hidden_layers=0, epochs=1, batch_size=2,
                                    loss="evidential" if evidential else "categorical_crossentropy")
    history = model.fit(x, y)
    assert np.isfinite(history.history["loss"]).all()
    model.save_model(tmp_path / "legacy.keras")
    first = tmp_path / "first.weights.h5"
    model.model.save_weights(first)
    original = model.model.get_weights()
    model.model.set_weights([value + .1 for value in original])
    second = tmp_path / "second.weights.h5"
    model.model.save_weights(second)
    model.model.set_weights(original)
    model.ensemble_member_files = [first, second]
    assert model.predict_ensemble(x).shape == (2, 4, 2)
    for actual, expected in zip(model.model.get_weights(), original):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(model.predict_proba(x), model.predict(x))
    if evidential:
        p, _, a, e = model.predict_uncertainty(x)
        np.testing.assert_allclose(array(a+e), array(p*(1-p)), rtol=1e-6)


def test_seed_and_locate_best_model(tmp_path):
    seed_everything(71)
    expected = np.random.random(3)
    seed_everything(71)
    np.testing.assert_array_equal(np.random.random(3), expected)
    model_path = tmp_path / "models"
    model_path.mkdir()
    (model_path / "training_log_4.csv").write_text("score\n0.3\n0.5\n")
    (model_path / "training_log_9.csv").write_text("score\n0.2\n0.4\n")
    assert old_models.locate_best_model(tmp_path, "score", "max") == 4
    assert old_models.locate_best_model(tmp_path, "score", "min") == 9
    with pytest.raises(ValueError):
        old_models.locate_best_model(tmp_path / "missing")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nig_layer_keeps_alpha_above_one_at_extreme_negative_logits(dtype):
    layer = DenseNormalGamma(1, dtype=dtype)
    layer(np.ones((1, 1), dtype=dtype))
    layer.dense.kernel.assign(np.zeros((1, 4)))
    layer.dense.bias.assign([0., -1000., -1000., -1000.])
    result = array(layer(np.ones((1, 1), dtype=dtype)))
    assert result[0, 2] > 1
    assert result[0, 1] > 0 and result[0, 3] > 0


def test_unweighted_loss_gradients_match_original_formulas():
    import torch
    y, mu, v, alpha, beta, raw = _nig_case()
    reference = torch.tensor(raw, requires_grad=True)
    predicted = torch.tensor(raw, requires_grad=True)
    target = torch.tensor(y)
    m, n, a, b = torch.chunk(reference, 4, dim=-1)
    two_b = 2 * b * (1 + n)
    nll = .5 * torch.log(torch.pi / n) - a * torch.log(two_b)
    nll += (a + .5) * torch.log(n * (target-m)**2 + two_b) + torch.lgamma(a) - torch.lgamma(a+.5)
    original = nll.mean() + .37 * (torch.abs(target-m) * (2*n+a)).mean()
    modern = losses.evidential_reg_loss(.37)(target, predicted)
    torch.testing.assert_close(torch.autograd.grad(modern, predicted)[0], torch.autograd.grad(original, reference)[0])

    target = torch.eye(2)
    ref = torch.tensor([[2., .5], [.2, 3.]], requires_grad=True)
    pred = ref.detach().clone().requires_grad_(True)
    a = torch.relu(ref) + 1
    s = a.sum(1, keepdim=True)
    ah = target + (1-target)*a
    sh = ah.sum(1, keepdim=True)
    kl = torch.lgamma(sh) - torch.lgamma(ah).sum(1, keepdim=True) - torch.lgamma(torch.tensor(2.))
    kl += ((ah-1)*(torch.digamma(ah)-torch.digamma(sh))).sum(1, keepdim=True)
    original = (((target-a/s)**2).sum(1, keepdim=True) + (a*(s-a)/(s*s*(s+1))).sum(1, keepdim=True) + .5*kl).mean()
    modern = losses.evidential_cat_loss(2., callbacks.ReportEpoch(initial_epoch=1.))(target, pred)
    torch.testing.assert_close(torch.autograd.grad(modern, pred)[0], torch.autograd.grad(original, ref)[0])


@pytest.mark.parametrize("family", [old_models.RegressorDNN, old_models.GaussianRegressorDNN, old_models.EvidentialRegressorDNN])
def test_legacy_regressor_configuration_loader_roundtrip(tmp_path, family):
    args = dict(hidden_layers=1, hidden_neurons=3, epochs=1, batch_size=2, verbose=0)
    wrapper = family(**args)
    wrapper.build_neural_network(2, 1)
    if family is old_models.GaussianRegressorDNN:
        destination = tmp_path / "single" / "models"
    elif family is old_models.EvidentialRegressorDNN:
        destination = tmp_path / "models"
    else:
        destination = tmp_path
    wrapper.save_path, wrapper.model_name, wrapper.training_var = str(destination), "best.h5", [1.]
    wrapper.save_model()
    config = {"save_loc": str(tmp_path), "model": {**args, "save_path": str(tmp_path)},
              "data": {"input_cols": ["x", "z"], "output_cols": ["y"]},
              "ensemble": {"n_models": 1, "n_splits": 1}}
    restored = family.load_model(config)
    x = np.ones((2, 2), dtype="float32")
    np.testing.assert_allclose(restored.model.predict(x, verbose=0), wrapper.model.predict(x, verbose=0))


@pytest.mark.parametrize("evidential", [False, True])
def test_legacy_classifier_configuration_loader_roundtrip(tmp_path, evidential):
    args = dict(hidden_layers=0, epochs=1, batch_size=2, loss="evidential" if evidential else "categorical_crossentropy")
    wrapper = old_models.CategoricalDNN(**args)
    wrapper.build_neural_network(2, 2)
    (tmp_path / "models").mkdir()
    wrapper.save_model(tmp_path / "models" / "best.h5")
    config = {"save_loc": str(tmp_path), "model": args, "input_features": ["x", "z"],
              "output_features": ["a", "b"], "ensemble": {"n_splits": 1}}
    restored = old_models.CategoricalDNN.load_model(config)
    x = np.ones((2, 2), dtype="float32")
    np.testing.assert_allclose(restored.predict(x), wrapper.predict(x))


def test_gaussian_linear_head_does_not_clip_negative_mean():
    layer = DenseNormal(1, eps=.01, output_activation="linear")
    layer(np.ones((1, 1), dtype="float32"))
    layer.dense.kernel.assign(np.zeros((1, 2)))
    layer.dense.bias.assign([-3., -2.])
    np.testing.assert_allclose(array(layer(np.ones((1, 1), dtype="float32"))), [[-3., .01]])


@pytest.mark.parametrize("family", [old_models.RegressorDNN, old_models.CategoricalDNN])
def test_legacy_build_from_sequential_accepts_optimizer_and_metrics(family):
    wrapper = family()
    network = keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(1)])
    optimizer = keras.optimizers.SGD(.003)
    wrapper.build_from_sequential(network, optimizer=optimizer, metrics=["mae"])
    assert wrapper.model.optimizer is optimizer
    result = wrapper.model.train_on_batch(np.ones((3, 2)), np.ones((3, 1)), return_dict=True)
    assert "mae" in result


def test_current_regressor_optimizer_state_restores_for_further_fit(tmp_path):
    x = np.ones((4, 2), dtype="float32")
    y = np.ones((4, 1), dtype="float32")
    model = RegressorDNN(n_inputs=2, hidden_layers=0, epochs=1, batch_size=2,
                         optimizer=keras.optimizers.SGD(.001), loss="mse")
    model.fit(x, y, verbose=0)
    iteration = int(array(model.optimizer.iterations))
    model.save(tmp_path / "resumable.keras")
    restored = keras.models.load_model(tmp_path / "resumable.keras")
    assert restored.optimizer_obj is restored.optimizer
    assert int(array(restored.optimizer.iterations)) == iteration
    restored.fit(x, y, epochs=1, batch_size=2, verbose=0)
    assert int(array(restored.optimizer.iterations)) == iteration + 2


def test_base_regressor_prediction_and_abstract_uncertainty_contracts():
    from sklearn.preprocessing import StandardScaler
    wrapper = old_models.RegressorDNN(hidden_layers=0)
    wrapper.build_neural_network(2, 2)
    x = np.array([[1., 2.], [3., 4.]], dtype="float32")
    raw = wrapper.model.predict(x, verbose=0)
    scaler = StandardScaler().fit([[-1., 2.], [3., 10.]])
    np.testing.assert_allclose(wrapper.predict(x, scaler=scaler), scaler.inverse_transform(raw))
    with pytest.raises(NotImplementedError):
        wrapper.calc_uncertainties(raw)
    with pytest.raises(NotImplementedError):
        wrapper.predict_uncertainty(x)
    from mlguess.keras.deprecated.callbacks import get_callbacks, step_decay, LearningRateTracker
    assert get_callbacks is callbacks.get_callbacks
    assert step_decay is callbacks.step_decay
    assert LearningRateTracker is callbacks.LearningRateTracker


@pytest.mark.parametrize("legacy", [False, True])
def test_evidential_classifier_accepts_documented_integer_labels(legacy):
    x = np.array([[0., 1.], [1., 0.], [0., 0.], [1., 1.]], dtype="float32")
    y = np.array([0, 1, 0, 1])
    if legacy:
        model = old_models.CategoricalDNN(hidden_layers=0, epochs=1, batch_size=2, loss="evidential")
    else:
        model = CategoricalDNN(n_inputs=2, n_classes=2, hidden_layers=0, epochs=1, batch_size=2, evidential=True)
    history = model.fit(x, y)
    assert np.isfinite(history.history["loss"]).all()


@pytest.mark.parametrize("module", [losses, old_losses])
def test_gaussian_loss_one_dimensional_target_does_not_broadcast_samples(module):
    y = ops.convert_to_tensor([1., 3.])
    pred = ops.convert_to_tensor([[0., 1.], [2., 2.]])
    expected = (.5*np.log(2*np.pi) + np.log([1., 2.]) + .5*(np.array([1., 1.])/np.array([1., 2.]))**2)
    np.testing.assert_allclose(array(module.gaussian_nll(y, pred, reduce=False)), expected, rtol=1e-6)


def test_legacy_evidential_prediction_and_ensemble_wrapper(tmp_path):
    wrapper = old_models.EvidentialRegressorDNN(hidden_layers=0)
    wrapper.build_neural_network(2, 1)
    x = np.ones((3, 2), dtype="float32")
    raw = wrapper.model.predict(x, verbose=0)
    expected = wrapper.calc_uncertainties(raw)
    direct = wrapper.predict_uncertainty(x, batch_size=2)
    for actual, component in zip(direct, expected):
        np.testing.assert_allclose(actual, component)
    first, second = tmp_path / "one.weights.h5", tmp_path / "two.weights.h5"
    wrapper.model.save_weights(first)
    wrapper.model.save_weights(second)
    wrapper.ensemble_member_files = [first, second]
    ensemble = wrapper.predict_ensemble(x, batch_size=2)
    assert len(ensemble) == 3
    for members, component in zip(ensemble, expected):
        np.testing.assert_allclose(members, np.stack([component, component]))


@pytest.mark.parametrize("module", [losses, old_losses])
@pytest.mark.parametrize("coupled", [False, True])
def test_nig_losses_one_dimensional_targets_are_per_observation(module, coupled):
    y = np.array([1., 4.], dtype="float32")
    mu = np.array([[0.], [2.]], dtype="float32")
    v = np.array([[2.], [3.]], dtype="float32")
    alpha = np.array([[3.], [4.]], dtype="float32")
    beta = np.array([[1.], [2.]], dtype="float32")
    raw = np.concatenate([mu, v, alpha, beta], axis=1)
    if coupled:
        loss = module.EvidentialRegressionCoupledLoss(r=2., coeff=.2)
        v = alpha - 1
        evidence = v + 2 * alpha
    else:
        loss = module.EvidentialRegressionLoss(.2)
        evidence = 2 * v + alpha
    expected = (-t.logpdf(y[:, None], df=2 * alpha, loc=mu,
                          scale=np.sqrt(beta * (1 + v) / (v * alpha)))
                + .2 * np.abs(y[:, None] - mu) * evidence)[:, 0]
    np.testing.assert_allclose(array(loss(y, raw)), expected.mean(), rtol=3e-6)
    np.testing.assert_allclose(array(loss.call(ops.convert_to_tensor(y), ops.convert_to_tensor(raw))),
                               expected, rtol=3e-6)
    if hasattr(loss, "nig_nll"):
        assert loss.nig_nll(*map(ops.convert_to_tensor, [y, mu, v, alpha, beta]), reduce=False).shape == (2, 1)
        assert loss.nig_reg(*map(ops.convert_to_tensor, [y, mu, v, alpha]), reduce=False).shape == (2, 1)


@pytest.mark.parametrize("legacy", [False, True])
def test_gaussian_model_compile_applies_sample_weights(legacy):
    x = np.array([[0., 1.], [1., 0.]], dtype="float32")
    y = np.array([[1.], [4.]], dtype="float32")
    if legacy:
        wrapper = old_models.GaussianRegressorDNN(hidden_layers=0)
        wrapper.build_neural_network(2, 1)
        model = wrapper.model
    else:
        model = RegressorDNN(n_inputs=2, hidden_layers=0, uncertainty=True)
        model.fit(x, y, epochs=0, verbose=0)
    raw = model.predict(x, verbose=0)
    expected = (.5 * np.log(2 * np.pi) + np.log(raw[:, 1])
                + .5 * ((y[:, 0] - raw[:, 0]) / raw[:, 1]) ** 2)
    actual = model.test_on_batch(x, y, sample_weight=np.array([1., 0.]), return_dict=True)["loss"]
    np.testing.assert_allclose(actual, expected[0] / 2, rtol=3e-6)
    restored = keras.losses.deserialize(keras.losses.serialize(model.loss))
    np.testing.assert_allclose(array(restored(ops.convert_to_tensor(y), ops.convert_to_tensor(raw))), expected,
                               rtol=3e-6)
