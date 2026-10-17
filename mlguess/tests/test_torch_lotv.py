"""Analytical distribution and hierarchy checks for Torch MC predictions."""

import numpy as np
import pytest
from scipy.special import digamma
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlguess.torch.mc_dropout import monte_carlo_dropout
from mlguess.torch.models import DNN, CategoricalDNN


class ScriptedMembers(nn.Module):
    """Cycle known member parameters so mixture moments have exact references."""

    def __init__(self, members):
        super().__init__()
        self.members = members
        self.index = 0

    def forward(self, x):
        values = self.members[self.index % len(self.members)]
        self.index += 1
        if isinstance(values, tuple):
            return tuple(torch.tensor(value, dtype=torch.float64).reshape(1, -1).expand(len(x), -1) for value in values)
        return torch.tensor(values, dtype=torch.float64).reshape(1, -1).expand(len(x), -1)


def run_members(members, kind, passes=None, **kwargs):
    model = ScriptedMembers(members)
    return monte_carlo_dropout(
        [torch.ones(3, 1)], len(members) if passes is None else passes,
        model, n_samples=3, output_kind=kind, **kwargs,
    )


@pytest.mark.parametrize("kind,members", [
    ("probabilities", [[.2, .8], [.8, .2]]),
    ("logits", [np.log([.2, .8]), np.log([.8, .2])]),
])
def test_categorical_finite_mixture_identity_uses_population_variance(kind, members):
    result = run_members(members, kind)
    np.testing.assert_allclose(result["mean"], .5)
    np.testing.assert_allclose(result["aleatoric"], .16)
    np.testing.assert_allclose(result["between_epistemic"], .09)
    np.testing.assert_allclose(result["within_epistemic"], 0)
    np.testing.assert_allclose(result["total"], .25)
    np.testing.assert_allclose(result["total"], result["mean"] * (1 - result["mean"]))
    np.testing.assert_allclose(result["mutual_info_within"], 0, atol=1e-15)
    np.testing.assert_allclose(result["mutual_info"], np.log(2) + .2 * np.log(.2) + .8 * np.log(.8))


@pytest.mark.parametrize("kind,members", [
    ("dirichlet", [[2., 8.], [8., 2.]]),
    ("evidence", [[1., 7.], [7., 1.]]),
])
def test_dirichlet_mixture_retains_within_and_between_uncertainty(kind, members):
    result = run_members(members, kind)
    np.testing.assert_allclose(result["aleatoric"], .16 * 10 / 11)
    np.testing.assert_allclose(result["within_epistemic"], .16 / 11)
    np.testing.assert_allclose(result["between_epistemic"], .09)
    np.testing.assert_allclose(result["epistemic"], .16 / 11 + .09)
    np.testing.assert_allclose(result["total"], .25)
    expected_entropy = digamma(11) - .2 * digamma(3) - .8 * digamma(9)
    np.testing.assert_allclose(result["mutual_info"], np.log(2) - expected_entropy)
    assert np.all(result["mutual_info_within"] > 0)
    np.testing.assert_allclose(
        result["mutual_info"], result["mutual_info_within"] + result["mutual_info_between"],
    )


def test_one_dirichlet_pass_has_within_uncertainty_but_no_between_disagreement():
    result = run_members([[2., 8.]], "dirichlet")
    np.testing.assert_allclose(result["between_epistemic"], 0)
    np.testing.assert_allclose(result["epistemic"], .16 / 11)
    assert np.all(result["mutual_info"] > 0)
    np.testing.assert_allclose(result["mutual_info_between"], 0, atol=1e-15)


@pytest.mark.parametrize("kind,members", [
    ("gaussian_std", [([0.], [2.]), ([2.], [2.])]),
    ("gaussian_variance", [([0.], [4.]), ([2.], [4.])]),
    ("gaussian_std", [[0., 2.], [2., 2.]]),
])
def test_gaussian_scale_interpretation_and_total_variance(kind, members):
    result = run_members(members, kind)
    np.testing.assert_allclose(result["mean"], 1)
    np.testing.assert_allclose(result["aleatoric"], 4)
    np.testing.assert_allclose(result["epistemic"], 1)
    np.testing.assert_allclose(result["total"], 5)
    assert result["aleatoric_available"] is True
    assert "entropy" not in result


def test_nig_hierarchical_mixture_and_affine_transform_once():
    members = [([0.], [2.], [3.], [4.]), ([2.], [2.], [3.], [4.])]
    result = run_members(members, "nig")
    np.testing.assert_allclose(result["aleatoric"], 2)
    np.testing.assert_allclose(result["within_epistemic"], 1)
    np.testing.assert_allclose(result["between_epistemic"], 1)
    np.testing.assert_allclose(result["epistemic"], 2)
    np.testing.assert_allclose(result["total"], 4)
    scaler = MinMaxScaler().fit([[10.], [20.]])
    scaled = run_members(members, "nig", y_scaler=scaler)
    np.testing.assert_allclose(scaled["mean"], 20)
    for key in ("aleatoric", "within_epistemic", "between_epistemic", "epistemic", "total"):
        np.testing.assert_allclose(scaled[key], result[key] * 100)


def test_multitarget_gaussian_parameters_do_not_broadcast_or_mix_targets():
    result = run_members([
        ([0., 10.], [2., 3.]), ([2., 14.], [2., 3.]),
    ], "gaussian_std")
    assert result["mean"].shape == (3, 2)
    np.testing.assert_allclose(result["mean"][0], [1., 12.])
    np.testing.assert_allclose(result["total"][0], [5., 13.])


def test_point_regression_reports_unknown_noise_not_bernoulli_variance():
    result = run_members([[-2.], [4.]], "regression")
    np.testing.assert_allclose(result["mean"], 1)
    np.testing.assert_allclose(result["epistemic"], 9)
    assert result["aleatoric"] is None
    assert result["total"] is None
    assert result["aleatoric_available"] is False
    assert "entropy" not in result


def test_binary_one_column_probability_entropy_is_bernoulli():
    result = run_members([[.2], [.8]], "probabilities")
    np.testing.assert_allclose(result["entropy"], np.log(2))
    np.testing.assert_allclose(result["total"], .25)


def test_dnn_default_declares_regression_and_explicit_logits_preserves_five_tuple():
    model = DNN(1, 1, layer_size=[], batch_norm=False)
    model.fcn = ScriptedMembers([[-2.], [4.]])
    mean, ale, epi, total = model.predict_dropout(np.ones((3, 1)), 2)
    np.testing.assert_allclose(mean, 1)
    np.testing.assert_allclose(epi, 9)
    assert ale is None and total is None
    model.fcn = ScriptedMembers([[.2, .8], [.8, .2]])
    output = model.predict_dropout(np.ones((3, 1)), 2, output_kind="probabilities")
    assert len(output) == 5
    np.testing.assert_allclose(output[1] + output[2], .25)


def test_categorical_dnn_default_mc_includes_dirichlet_uncertainty():
    model = CategoricalDNN(1, 2, layer_size=[], batch_norm=False)
    model.fcn = ScriptedMembers([[1., 7.], [7., 1.]])
    result = model.predict_dropout(np.ones((3, 1)), 2, return_dict=True)
    np.testing.assert_allclose(result["within_epistemic"], .16 / 11)
    np.testing.assert_allclose(result["total"], .25)


def test_nig_dnn_default_mc_accepts_parameter_tuples():
    model = DNN(1, 1, layer_size=[], batch_norm=False, lng=True)
    model.fcn = ScriptedMembers([([0.], [2.], [3.], [4.])])
    mean, ale, epi, total = model.predict_dropout(np.ones((3, 1)), 1)
    np.testing.assert_allclose(ale, 2)
    np.testing.assert_allclose(epi, 1)
    np.testing.assert_allclose(total, 3)


@pytest.mark.parametrize("kind,members,error", [
    ("probabilities", [[-1., 2.]], "probabilities"),
    ("probabilities", [[.2, .2]], "sum to one"),
    ("gaussian_std", [([0.], [-2.])], "nonnegative"),
    ("nig", [([0.], [1.], [1.], [1.])], "alpha>1"),
    ("dirichlet", [[0., 1.]], "alpha>0"),
])
def test_invalid_distribution_parameters_fail_explicitly(kind, members, error):
    with pytest.raises(ValueError, match=error):
        run_members(members, kind)


def test_generic_model_requires_distribution_declaration():
    with pytest.raises(ValueError, match="Specify output_kind"):
        monte_carlo_dropout([torch.ones(2, 1)], 2, ScriptedMembers([[0.]]))


def test_shuffled_loader_rejected_and_bad_parameters_restore_modes():
    loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2, shuffle=True)
    with pytest.raises(ValueError, match="shuffle=False"):
        monte_carlo_dropout(loader, 2, ScriptedMembers([[0.]]), output_kind="regression")
    model = DNN(1, 2, layer_size=[4], dr=.5)
    before = [module.training for module in model.modules()]
    with pytest.raises(ValueError, match="probabilities"):
        model.predict_dropout(np.ones((3, 1)), 2, output_kind="probabilities")
    assert before == [module.training for module in model.modules()]


def test_nonlinear_scaler_rejected_even_with_legacy_training_variance():
    model = DNN(1, 1, layer_size=[], lng=True)
    model.training_var = [1.]
    scaler = QuantileTransformer(n_quantiles=2).fit([[0.], [1.]])
    parameters = tuple(torch.tensor([[x]], dtype=torch.float32) for x in [0., 2., 3., 4.])
    with pytest.raises(TypeError):
        model.predict_uncertainty(parameters, y_scaler=scaler)


def test_single_nig_prediction_accepts_generator_and_concatenated_tensor():
    model = DNN(1, 1, layer_size=[], lng=True)
    parameters = tuple(torch.full((2, 1), value) for value in (0., 2., 3., 4.))
    expected = model.predict_uncertainty(parameters)
    from_generator = model.predict_uncertainty(value.detach() for value in parameters)
    from_tensor = model.predict_uncertainty(torch.cat(parameters, dim=-1))
    for reference, generated, concatenated in zip(expected, from_generator, from_tensor):
        torch.testing.assert_close(generated, reference)
        torch.testing.assert_close(concatenated, reference)


def test_single_nig_prediction_rejects_overflowing_derived_moments():
    model = DNN(1, 1, layer_size=[], lng=True)
    parameters = tuple(torch.tensor([[value]], dtype=torch.float32) for value in (0., 1e-38, 2., 1e38))
    with pytest.raises(ValueError, match="moments must be finite"):
        model.predict_uncertainty(parameters)
