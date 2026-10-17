"""Broader function audit: concrete CPU correctness/runtime regressions."""

import copy
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm

from mlguess.torch.class_losses import EDLLoss, EDLDigammaLoss, edl_log_loss, kl_divergence
from mlguess.torch.models import DNN, CategoricalDNN, init_weights, seed_everything
from mlguess.torch.mc_dropout import as_model_input
from mlguess.torch.regression_losses import EvidentialRegressionLoss, EvidentialMarginalLikelihood
from mlguess.torch.trainer_classifier import Trainer as Classifier, one_hot_embedding
from mlguess.torch.trainer_regression import Trainer as Regressor
from mlguess.torch.training_utils import result_mean, best_metric_index


def config(tmp_path, **kwargs):
    settings = dict(mode="none", batches_per_epoch=0, valid_batches_per_epoch=0,
                    grad_accum_every=1, amp=False, use_scheduler=False,
                    start_epoch=0, epochs=1, stopping_patience=10)
    settings.update(kwargs)
    return {"save_loc": str(tmp_path / 'results'), "trainer": settings}


def no_metrics(*args, **kwargs):
    return {}


@pytest.mark.parametrize('weighted', [False, True])
def test_unequal_accumulation_matches_full_batch_mean(tmp_path, weighted):
    torch.manual_seed(20)
    x = torch.randn(5, 2, dtype=torch.float64)
    y = torch.tensor([0, 1, 0, 0, 1])
    model = nn.Linear(2, 2)
    reference = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 3.]) if weighted else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    loader = DataLoader(TensorDataset(x, y), batch_size=3)
    diagnostics = Classifier(model).train_one_epoch(
        0, config(tmp_path, grad_accum_every=2), loader,
        optimizer, criterion, None, None, no_metrics,
    )
    reference_optimizer.zero_grad()
    expected_loss = criterion(reference(x.float()), y)
    expected_loss.backward()
    reference_optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    assert result_mean(diagnostics, 'train_loss') == pytest.approx(expected_loss.item(), rel=1e-6)


def test_regression_training_float64_inputs_and_vector_targets(tmp_path):
    torch.manual_seed(12)
    model = DNN(2, 1, layer_size=[], lng=True)
    loader = DataLoader(TensorDataset(torch.randn(5, 2, dtype=torch.float64), torch.randn(5, dtype=torch.float64)), batch_size=3)
    trainer = Regressor(model)
    criterion = EvidentialRegressionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    trainer.train_one_epoch(0, config(tmp_path), loader, optimizer, criterion, None, None, no_metrics)
    result = trainer.predict(config(tmp_path), loader, criterion, no_metrics)
    assert result['mu'].shape == (5, 1)
    assert np.isfinite(result['total']).all()


def test_evaluation_loss_is_invariant_to_unequal_batch_partition(tmp_path):
    torch.manual_seed(10)
    model = nn.Linear(2, 2)
    x, y = torch.randn(5, 2, dtype=torch.float64), torch.tensor([0, 0, 0, 1, 1])
    data = TensorDataset(x, y)
    trainer = Classifier(model)
    criterion = nn.CrossEntropyLoss()
    first = trainer.predict(config(tmp_path), DataLoader(data, batch_size=3), criterion, no_metrics)
    second = trainer.predict(config(tmp_path), DataLoader(data, batch_size=5), criterion, no_metrics)
    np.testing.assert_allclose(first['metrics']['test_loss'], second['metrics']['test_loss'], rtol=1e-6)
    np.testing.assert_allclose(first['mu'].sum(axis=1), 1., rtol=1e-6)


def test_classifier_fit_uses_global_validation_metrics(tmp_path):
    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.randn(5, 2), torch.tensor([0, 1, 0, 1, 0])), batch_size=3)
    observed_sizes = []
    def metrics(y, probability, split):
        torch.testing.assert_close(probability.sum(1), torch.ones(len(y)))
        if split == 'valid':
            observed_sizes.append(len(y))
        return {f'{split}_sample_stat': float(len(y) ** 2)}
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    criterion = nn.CrossEntropyLoss()
    result = Classifier(model).fit(config(tmp_path), loader, loader, optimizer, criterion, criterion, None, None, metrics)
    assert observed_sizes == [5]
    assert result['valid_sample_stat'] == 25


@pytest.mark.parametrize('labels', [torch.tensor([-1]), torch.tensor([2]), torch.tensor([.5]), torch.tensor([float('nan')])])
def test_one_hot_rejects_negative_out_of_range_and_fractional_labels(labels):
    with pytest.raises(ValueError):
        one_hot_embedding(labels, 2)


@pytest.mark.parametrize('trainer_kind', ['classifier', 'regression'])
def test_empty_loaders_and_zero_epochs_raise_clear_errors(tmp_path, trainer_kind):
    is_regression = trainer_kind == 'regression'
    model = DNN(2, 1, layer_size=[], lng=True) if is_regression else nn.Linear(2, 2)
    trainer = Regressor(model) if is_regression else Classifier(model)
    labels = torch.empty(0, 1) if is_regression else torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.empty(0, 2), labels), batch_size=2)
    criterion = EvidentialRegressionLoss() if is_regression else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    with pytest.raises(ValueError, match='at least one batch'):
        trainer.train_one_epoch(0, config(tmp_path), loader, optimizer, criterion, None, None, no_metrics)
    with pytest.raises(ValueError, match='at least one batch'):
        trainer.predict(config(tmp_path), loader, criterion, no_metrics)
    with pytest.raises(ValueError, match='epochs must be positive'):
        trainer.fit(config(tmp_path, epochs=0), loader, loader, optimizer, criterion, criterion, None, None, no_metrics)


def test_wrapped_trainer_keeps_forward_wrapper_and_saves_plain_state(tmp_path):
    class Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0
        def forward(self, value):
            self.calls += 1
            return self.module(value)
    wrapper = Wrapper(nn.Linear(2, 2))
    trainer = Classifier(wrapper, module=True)
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2)
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=.01)
    criterion = nn.CrossEntropyLoss()
    trainer.fit(config(tmp_path), loader, loader, optimizer, criterion, criterion, None, None, no_metrics)
    assert wrapper.calls == 4
    saved = torch.load(tmp_path/'results'/'checkpoint.pt', weights_only=True)
    assert set(saved['model_state_dict']) == {'weight', 'bias'}


def test_spectral_weight_initializer_changes_trainable_parameter_after_forward():
    model = spectral_norm(nn.Linear(3, 2))
    model(torch.randn(2, 3))
    before = model.weight_orig.detach().clone()
    init_weights(model, init_type='normal', init_gain=.4, verbose=False)
    assert not torch.equal(model.weight_orig, before)
    assert torch.count_nonzero(model.weight_orig) > 0


def test_class_weights_list_array_tensor_agree_and_are_not_silently_ignored():
    output = torch.tensor([[1., 2.], [3., .5]], dtype=torch.float64)
    y = torch.eye(2, dtype=torch.float64)
    expected = edl_log_loss(output, y, 2, 2, 10, weights=torch.tensor([1., 4.]))
    for weights in ([1., 4.], np.array([1., 4.])):
        torch.testing.assert_close(edl_log_loss(output, y, 2, 2, 10, weights=weights), expected)
        module_value = EDLLoss(torch.log, 2, 10, weights=weights)(y, output + 1, 2).mean()
        torch.testing.assert_close(module_value, expected)
    assert not torch.isclose(expected, edl_log_loss(output, y, 2, 2, 10))


@pytest.mark.parametrize('epoch,steps', [(0, 0), (0, -1), (-1, 5)])
def test_invalid_annealing_parameters_rejected(epoch, steps):
    with pytest.raises(ValueError, match='annealing_step'):
        EDLDigammaLoss(2, steps)(torch.ones(2, 2), torch.eye(2), epoch)


def test_invalid_dirichlet_shapes_and_concentrations_rejected():
    with pytest.raises(ValueError, match='shape'):
        kl_divergence(torch.ones(2, 3), 1)
    with pytest.raises(ValueError, match='positive'):
        kl_divergence(torch.tensor([[0., 1.]]), 2)


def test_best_metric_selection_supports_maximum_and_skips_missing_values():
    assert best_metric_index([.2, float('nan'), .7], 'max') == 2
    assert best_metric_index([.2, .1, .7], 'min') == 1
    with pytest.raises(ValueError, match='finite observations'):
        best_metric_index([float('nan')])


def test_parameterless_input_conversion_follows_floating_buffer():
    model = nn.Module()
    model.register_buffer('constant', torch.tensor([1.], dtype=torch.float32))
    assert as_model_input(np.array([1.], dtype=np.float64), model).dtype == torch.float32


def test_seed_helper_reproduces_python_numpy_and_torch_rng():
    import random
    seed_everything(51)
    first = (random.random(), np.random.rand(), torch.rand(2))
    seed_everything(51)
    assert random.random() == first[0]
    assert np.random.rand() == first[1]
    torch.testing.assert_close(torch.rand(2), first[2])


def test_marginal_likelihood_pi_uses_input_float64_precision():
    from scipy.special import gammaln
    from mlguess.torch.regression_losses import tol
    arrays = [np.array([[v]], dtype=np.float64) for v in (.1, 1.3, 2.2, 1.7, .8)]
    gamma, nu, alpha, beta, target = arrays
    expected = (.5*np.log(np.pi/(nu+tol)) - alpha*np.log(2*beta*(1+nu)+tol)
                + (alpha+.5)*np.log(nu*(target-gamma)**2+2*beta*(1+nu)+tol)
                + gammaln(alpha+tol)-gammaln(alpha+.5+tol))
    observed = EvidentialMarginalLikelihood(reduction='none')(*[torch.from_numpy(a) for a in arrays])
    np.testing.assert_allclose(observed.numpy(), expected, rtol=0, atol=2e-14)


def test_categorical_uncertainty_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match='finite'):
        CategoricalDNN(1, 2, layer_size=[]).calc_uncertainty(torch.tensor([[float('inf'), 0.]]))


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_loss_reduction_preserved_across_unequal_batches(tmp_path, reduction):
    torch.manual_seed(29)
    model = nn.Linear(2, 2)
    reference = copy.deepcopy(model)
    x = torch.randn(5, 2)
    y = torch.tensor([0, 0, 1, 1, 0])
    data = TensorDataset(x, y)
    criterion = nn.CrossEntropyLoss(reduction=reduction)
    expected = criterion(reference(x), y)
    trainer = Classifier(model)
    prediction = trainer.predict(config(tmp_path), DataLoader(data, batch_size=3), criterion, no_metrics)
    assert prediction['metrics']['test_loss'] == pytest.approx([expected.item()], rel=1e-6)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    trainer.train_one_epoch(0, config(tmp_path, grad_accum_every=2), DataLoader(data, batch_size=3),
                            optimizer, criterion, None, None, no_metrics)
    expected.backward()
    reference_optimizer.step()
    for actual, ref in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, ref)


@pytest.mark.parametrize('mode', ['ddp', 'fsdp'])
def test_direct_trainers_reject_unvalidated_multi_rank_execution(tmp_path, monkeypatch, mode):
    from mlguess.torch.training_utils import distributed_enabled
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    with pytest.raises(NotImplementedError, match='Multi-rank'):
        distributed_enabled(config(tmp_path, mode=mode))
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    assert distributed_enabled(config(tmp_path, mode=mode))
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    with pytest.raises(RuntimeError, match='initialized process group'):
        distributed_enabled(config(tmp_path, mode=mode))


def test_fsdp_device_preparation_does_not_move_offloaded_module():
    from mlguess.torch.training_utils import prepare_model_device
    class FakeFSDP(nn.Module):
        def __init__(self):
            super().__init__()
            self._fsdp_wrapped_module = nn.Linear(2, 1)
            self.compute_device = torch.device('cuda:2')
        def to(self, *args, **kwargs):
            raise AssertionError('Initialized FSDP must manage its own placement')
    # API-routing test only: no CUDA tensors or FSDP collectives are executed.
    assert prepare_model_device(FakeFSDP()) == torch.device('cuda:2')
    assert prepare_model_device(FakeFSDP(), 'cuda:1') == torch.device('cuda:1')


@pytest.mark.parametrize('module_name', ['mlguess.torch.models', 'mlguess.torch.class_losses'])
@pytest.mark.parametrize('cuda_available', [False, True])
def test_legacy_device_helpers_follow_reported_availability(monkeypatch, module_name, cuda_available):
    import importlib
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    # Device selection only; this does not allocate or claim execution on CUDA.
    assert importlib.import_module(module_name).get_device() == torch.device('cuda' if cuda_available else 'cpu')


def test_alternative_evidence_transforms_are_finite_and_match_reference():
    from mlguess.torch.class_losses import exp_evidence, softplus_evidence
    values = np.array([-1000., -10., -1., 0., 1., 10., 1000.], dtype=np.float64)
    x = torch.from_numpy(values)
    exponential = exp_evidence(x)
    softplus = softplus_evidence(x)
    assert torch.isfinite(exponential).all() and torch.isfinite(softplus).all()
    np.testing.assert_allclose(exponential.numpy(), np.exp(np.clip(values, -10, 10)), rtol=2e-15)
    np.testing.assert_allclose(softplus.numpy(), np.logaddexp(0, values), rtol=2e-15)


def test_categorical_uncertainty_alias_accepts_features_and_restores_mode():
    model = CategoricalDNN(2, 3, layer_size=[])
    with torch.no_grad():
        model.fcn[0].weight.zero_()
        model.fcn[0].bias.copy_(torch.tensor([2., -3., 1.]))
    features = np.array([[1., 2.], [-1., 4.]], dtype=np.float64)
    model.train()
    prediction = model.predict_uncertainty(features)
    expected = model.predict(features, return_uncertainties=True)
    assert model.training
    assert len(prediction) == 4
    for actual, reference in zip(prediction, expected):
        torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(prediction[0], torch.tensor([[.5, 1/6, 1/3]]).repeat(2, 1))
    torch.testing.assert_close(prediction[2] + prediction[3], prediction[0] * (1-prediction[0]))


@pytest.mark.parametrize('module_name', ['mlguess.torch.trainer_classifier', 'mlguess.torch.trainer_regression'])
@pytest.mark.parametrize('available,initialized', [(False, False), (True, False), (True, True)])
def test_trainer_cleanup_only_destroys_active_group(monkeypatch, module_name, available, initialized):
    import importlib
    module = importlib.import_module(module_name)
    state = {'initialized': initialized, 'destroyed': 0}
    monkeypatch.setattr(module.dist, 'is_available', lambda: available)
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: state['initialized'])
    def destroy():
        state['destroyed'] += 1
        state['initialized'] = False
    monkeypatch.setattr(module.dist, 'destroy_process_group', destroy)
    module.cleanup()
    module.cleanup()
    assert state['destroyed'] == int(available and initialized)
