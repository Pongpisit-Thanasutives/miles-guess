"""Independent analytical counterexamples and plotting smoke tests from the full audit."""
import builtins
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from mlguess import classifier_uq as cuq
from mlguess import regression_uq as ruq
from mlguess import pit, reliability, plotting
from mlguess.regression_metrics import calculate_skill_score, regression_metrics, rmse_crps_skill_scores
from mlguess.torch.metrics import MetricsCalculator
from mlguess.uncertainty import (
    affine_inverse_scale, dirichlet_uncertainties, ensemble_uncertainties,
    inverse_scale_moments, nig_uncertainties,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_regression_negative_targets_and_column_equivalence():
    truth = np.array([-2., -4., 2., 4.])
    pred = truth / 2
    variance = np.array([0., 1., 4., 9.])
    result = regression_metrics(truth, pred, variance)
    column = regression_metrics(truth[:, None], pred[:, None], variance[:, None])
    assert result.keys() == column.keys()
    for key in result:
        assert_allclose(result[key], column[key], equal_nan=True)
    assert result["val_mape"] == pytest.approx(50)
    # Gaussian CRPS = sigma[z(2Phi(z)-1)+2phi(z)-1/sqrt(pi)].
    expected = [abs(truth[0] - pred[0])]
    for y, mu, sigma in zip(truth[1:], pred[1:], np.sqrt(variance[1:])):
        z = (y - mu) / sigma
        expected.append(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))
    assert result["val_crps"] == pytest.approx(np.mean(expected))


def test_regression_deterministic_and_multioutput():
    y = np.array([[-2., 4.], [2., 7.]])
    result = regression_metrics(y, y + 1, np.zeros_like(y))
    assert result["val_crps"] == pytest.approx(1)
    assert result["val_mse"] == pytest.approx(1)
    assert np.isnan(result["val_r2_rmse_sigma"])
    assert result["val_pitd"] == pytest.approx(0)
    assert np.isnan(regression_metrics([0.], [1.])["val_r2"])
    assert np.isnan(rmse_crps_skill_scores([1, 2], [1, 2], [1, 1])["r2_crps"])


@pytest.mark.parametrize("truth,pred,variance", [
    ([1, 2], [1], None), ([1, np.nan], [1, 2], None),
    ([1, 2], [1, 2], [-1, 1]), ([1, 2], [1, 2], [[1, 1], [1, 1]]),
])
def test_regression_rejects_invalid_distributions(truth, pred, variance):
    with pytest.raises(ValueError):
        regression_metrics(truth, pred, variance)


def test_spread_summary_preserves_zero_scores_and_filters_paired_rows():
    y = [0, 2, np.nan, 9]
    p = [0, 0, 1, np.inf]
    sigma = [0, 0, 1, 1]
    result = calculate_skill_score(y, p, sigma, log=True)
    assert result["count"].sum() == 2
    assert result["crps"].item() == pytest.approx(1)
    assert result["rmse"].item() == pytest.approx(np.sqrt(2))
    assert_allclose(result, ruq.calculate_skill_score(y, p, sigma, log=True, filter_top_percentile=0))
    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_skill_score(np.ones((4, 2)), np.ones((4, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError, match="nonnegative"):
        calculate_skill_score([1], [1], [-1])


def test_brier_climatology_and_perfect_forecast():
    labels = np.array([0, 0, 0, 1])
    climatology = np.tile([.75, .25], (4, 1))
    assert cuq.brier_multi(labels, climatology) == pytest.approx(0)
    assert cuq.brier_multi(labels, np.eye(2)[labels]) == pytest.approx(1)
    assert cuq.brier_multi(labels, climatology, skill_score=False) == pytest.approx(.1875)
    assert cuq.brier_skill_score_multiclass(labels, climatology) == pytest.approx(0)
    assert cuq.brier_skill_score_multiclass(labels, climatology, skill_score=False) == pytest.approx(.1875)
    assert np.isnan(cuq.brier_multi([0], [[1., 0.]]))
    # An absent middle class is still scored in the non-skill macro average.
    assert cuq.brier_skill_score_multiclass([0, 2], [[0, 1, 0], [0, 1, 0]], skill_score=False) == pytest.approx(2 / 3)
    assert np.isnan(cuq.brier_skill_score_multiclass([0, 2], [[1, 0, 0], [0, 0, 1]]))


def test_classification_bins_preserve_endpoints_and_constant_values():
    labels = np.array([0, 1, 0, 1])
    probs = np.full((4, 2), .5)
    _, data = cuq.sort_arr(labels, probs, [0, 0, 1, 1], data_min=0, data_max=1)
    assert sum(data["count"]) == 4
    _, constant = cuq.sort_arr(labels, probs, np.zeros(4))
    assert sum(constant["count"]) == 4
    assert constant["bss"] == pytest.approx([0])


def test_attribution_uses_class_event_not_true_class_subset():
    # With a constant class probability of .75 and frequency .5, the
    # class-0 diagram must display (.75, .5), not (top confidence, 1).
    labels = np.array([0, 1, 0, 1])
    probs = np.tile([.75, .25], (4, 1))
    cuq.classifier_attribution(labels, probs, num_bins=4, legend_cols=["A", "B"])
    curve = plt.gcf().axes[2].lines[0]
    assert_allclose(curve.get_xdata(), [.75])
    assert_allclose(np.asarray(curve.get_ydata(), dtype=float), [.5])
    plt.gcf().canvas.draw()


def test_classifier_end_to_end_diagnostics_do_not_mutate_input(tmp_path):
    rng = np.random.default_rng(17)
    p = rng.dirichlet(np.ones(4), 80)
    y = np.tile(np.arange(4), 20)
    df = pd.DataFrame({f"pred_conf{k+1}": p[:, k] for k in range(4)})
    df["true_label"] = y
    df["pred_label"] = p.argmax(axis=1)
    df["aleatoric"] = rng.uniform(.01, .2, len(y))
    df["epistemic"] = rng.uniform(.01, .2, len(y))
    df["entropy"] = -np.sum(p * np.log(p), axis=1)
    original = df.copy(deep=True)
    cuq.uq_results(df, save_location=tmp_path)
    pd.testing.assert_frame_equal(df, original)
    assert len(list(tmp_path.glob("*.pdf"))) == 4
    for number in plt.get_fignums():
        plt.figure(number).canvas.draw()


def test_uncertainty_plot_filters_paired_values():
    ale = np.array([[1.], [np.nan], [3.], [4.]])
    epi = np.array([[2.], [2.], [np.nan], [5.]])
    ruq.plot_uncertainties(ale, epi, ["x"])
    plt.gcf().canvas.draw()
    cuq.plot_uncertainties(pd.DataFrame({"a": ale[:, 0], "e": epi[:, 0]}), ["a"], ["e"])
    plt.gcf().canvas.draw()


def test_coverage_and_rank_calibration_centers_and_spreads():
    frame = pd.DataFrame({"var": [1, 2, 3, 4], "error": [1, 2, 3, 4]})
    covered = ruq.compute_coverage(frame)
    assert_allclose(covered["var_cov"], [.75, .5, .25, 0])
    centers, means, ystd, xstd = ruq.calibration_curve(covered, bins=2)
    assert_allclose(centers, [.25, .75])
    assert_allclose(means, centers)
    assert_allclose(ystd, [0, 0])
    assert_allclose(xstd, [0, 0])
    # Empty bins are undefined, not a divide-by-zero exception.
    values = ruq.calibration_curve(frame.iloc[:1], bins=4)
    assert np.count_nonzero(np.isfinite(values[1])) == 1


def test_spread_skill_constant_and_right_endpoint():
    spread = np.ones(20)
    rmse, count, _ = ruq.compute_skill_score(np.ones(20), np.zeros(20), spread)
    assert count.sum() == 20
    assert_allclose(rmse[count > 0], [1])
    spread = np.r_[np.zeros(20), np.ones(20)]
    _, count, _ = ruq.compute_skill_score(np.ones(40), np.zeros(40), spread)
    assert count.sum() == 40
    for shape in ((20,), (20, 1)):
        a = np.ones(shape)
        ruq.plot_skill_score(a, a, a, np.zeros(shape), ["x"])
        plt.gcf().canvas.draw()


def test_regression_diagnostics_single_target_and_negative_predictions(tmp_path):
    rng = np.random.default_rng(7)
    mu = np.linspace(-10, -1, 30)[:, None]
    truth = mu + rng.normal(size=mu.shape)
    frame = pd.DataFrame({"x": truth[:, 0]})
    original = frame.copy()
    ale = np.linspace(.5, 2, 30)[:, None]
    epi = np.linspace(.1, .7, 30)[:, None]
    ruq.compute_results(frame, ["x"], mu, ale, epi,
                        ensemble_mu=np.stack([mu - .2, mu + .2]), ensemble_type="ensemble",
                        legend_cols=["X"], fn=tmp_path)
    pd.testing.assert_frame_equal(frame, original)
    assert len(list(tmp_path.glob("*.png"))) == 8
    for number in plt.get_fignums():
        plt.figure(number).canvas.draw()


def test_regression_attribution_keeps_maximum_bin_and_constant_predictions():
    for predictions in ([-3., -2., -1.], [-2., -2., -2.]):
        frame = pd.DataFrame({"x": [0., 1., 8.], "x_pred": predictions})
        ruq.regression_attributes(frame, ["x"], ["X"])
        x, y = plt.gca().lines[0].get_data()
        if len(set(predictions)) > 1:
            assert max(x) == pytest.approx(-1)
            assert max(y) == pytest.approx(8)
        else:
            assert_allclose(np.asarray(x, dtype=float), [-2])
            assert_allclose(np.asarray(y, dtype=float), [3])
        plt.gcf().canvas.draw()


def test_pit_cdfs_mixtures_ties_and_point_masses():
    assert_allclose(pit.probability_integral_transform_gaussian([0, 1], [[0, 1], [0, 1]]), [.5, norm.cdf(1)])
    mixture = np.array([[[0., 2.], [1., 1.]]])
    assert_allclose(pit.pit_gaussian_ensemble([1], mixture), [.5])
    # Preserve scipy percentileofscore's established rank treatment of ties.
    assert_allclose(pit.probability_integral_transform_ensemble([1], [[0, 1, 1, 2]]), [.625])
    assert_allclose(pit.probability_integral_transform_gaussian([-1, 0, 1], [[0, 0]] * 3), [0, 1, 1])
    counts, edges = pit.pit_histogram([1], mixture, pred_type="gaussian_ensemble", bins=4)
    assert counts.sum() == 1
    assert_allclose(edges, np.linspace(0, 1, 5))
    assert pit.pit_deviation_worst(4) == pytest.approx(np.sqrt(3) / 4)
    assert pit.pit_deviation_skill_score([0], [[0, 1]], pred_type="gaussian", bins=4) == pytest.approx(0)


@pytest.mark.parametrize("call", [
    lambda: pit.pit_histogram([0], [[0, 1]], pred_type="typo"),
    lambda: pit.probability_integral_transform_gaussian([0], [[0, -1]]),
    lambda: pit.probability_integral_transform_gaussian([], []),
    lambda: pit.probability_integral_transform_ensemble([0, 1], [[0, 1]]),
    lambda: pit.pit_histogram_values([np.nan]),
    lambda: pit.pit_histogram_values([.2], [0, .5]),
    lambda: pit.pit_deviation([0], [[0, 1]], "gaussian", bins=[0, .2, 1]),
    lambda: pit.pit_deviation_skill_score([0], [[0, 1]], "gaussian", bins=1),
])
def test_pit_rejects_invalid_definitions(call):
    with pytest.raises(ValueError):
        call()


def test_reliability_lists_and_manual_calibration():
    data = reliability.compute_calibration(["a", "a", "b"], ["a", "b", "b"], [0, .5, 1], num_bins=2)
    assert_allclose(data["counts"], [2, 1])
    assert data["expected_calibration_error"] == pytest.approx(1 / 6)
    for mode in ("alpha", "width"):
        fig = reliability.reliability_diagram([0, 1], [0, 0], [.25, .75], num_bins=2, draw_bin_importance=mode, return_fig=True)
        fig.canvas.draw()
    fig = reliability.reliability_diagrams({"one": {"true_labels": [0, 1], "pred_labels": [0, 1], "confidences": [.5, 1]}}, num_cols=1, return_fig=True)
    fig.canvas.draw()
    for confidence in ([np.nan], [-.1], [1.1]):
        with pytest.raises(ValueError):
            reliability.compute_calibration([0], [0], confidence)


def test_torch_metrics_match_manual_and_shared_calibration():
    calc = MetricsCalculator(n_bins=2)
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([[.9, .1], [.6, .4], [.2, .8], [.7, .3]])
    result = calc(np.eye(2)[labels], probabilities)
    assert result["train_ave_acc"] == pytest.approx(.75)
    assert result["train_csi"] == pytest.approx(1)
    expected = reliability.compute_calibration(labels, probabilities.argmax(1), probabilities.max(1), num_bins=2)
    assert calc.ece(labels, probabilities) == pytest.approx(expected["expected_calibration_error"])
    assert calc.mce(labels, probabilities) == pytest.approx(expected["max_calibration_error"])
    assert calc.mce(labels[:, None], probabilities) == pytest.approx(expected["max_calibration_error"])
    assert calc.ave_acc([0, 0, 1], [0, 1, 1]) == pytest.approx(.75)
    with pytest.raises(ValueError, match="probabilities"):
        calc.ece([0], [[2, -1]])
    with pytest.raises(ValueError, match="one-hot"):
        calc([[.5, .5]], [[.5, .5]])


def test_basic_plotting_does_not_require_cartopy_and_retains_absent_classes(tmp_path, monkeypatch):
    original_import = builtins.__import__
    def without_cartopy(name, *args, **kwargs):
        if name.startswith("cartopy"):
            raise ModuleNotFoundError("Cartopy deliberately unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_cartopy)
    importlib.reload(plotting)
    ax = plotting.plot_confusion_matrix([0, 0], [0, 1], ["A", "B", "C"], "test", normalize=True, filename=tmp_path / "cm.png")
    assert_allclose(ax.images[0].get_array(), [[.5, .5, 0], [0, 0, 0], [0, 0, 0]])
    frame = pd.DataFrame({"true_label": [0, 2, 0, 2], "pred_label": [0, 2, 1, 0], "pred_conf": [.8, .7, .5, .4], "evidential": [.1, .2, .3, .4]})
    original = frame.copy()
    plotting.coverage_figures(frame, ["A", "B", "C"], save_location=tmp_path / "cov.png")
    pd.testing.assert_frame_equal(frame, original)
    assert_allclose(plotting.compute_cov(frame, quan="evidential")["pred_conf_cov"], [.25, .5, .75, 1])
    plt.gcf().canvas.draw()


def test_shared_uncertainties_analytic_nested_mixture_and_scaling():
    # NIG predictive variance equals Student-t variance, including its nu term.
    mu, a, e, total = nig_uncertainties([0., 2.], [2., 4.], [3., 5.], [4., 8.])
    assert_allclose(a, [2, 2])
    assert_allclose(e, [1, .5])
    mix = ensemble_uncertainties(mu, a, e, weights=[1, 3])
    assert mix["mean"] == pytest.approx(1.5)
    assert mix["between_epistemic"] == pytest.approx(.75)
    assert mix["total"] == pytest.approx(3.375)
    probability, a, e, total = dirichlet_uncertainties([[2., 6.]])
    assert_allclose(probability, [[.25, .75]])
    assert_allclose(a + e, total)
    assert_allclose(total, [[.1875, .1875]])
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    data = np.array([[-2., 10.], [0., 20.], [6., 30.]])
    for scaler in (StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), StandardScaler(with_std=False)):
        scaler.fit(data)
        slope = affine_inverse_scale(scaler, 2)
        mean, ale, epi = inverse_scale_moments(np.zeros((3, 2)), np.ones((3, 2)), np.ones((3, 2)) * 2, scaler)
        assert_allclose(mean, scaler.inverse_transform(np.zeros((3, 2))))
        # Compare variance transformed from the actual +/-1 two-point support.
        points = scaler.inverse_transform(np.array([[-1., -1.], [1., 1.]]))
        assert_allclose(ale[0], points.var(axis=0))
        assert_allclose(epi[0], 2 * slope ** 2)
    with pytest.raises(ValueError, match="fitted feature"):
        affine_inverse_scale(StandardScaler(with_std=False).fit(data), 3)
    with pytest.raises(ValueError):
        ensemble_uncertainties([0, 1], [-1, 1])
    with pytest.raises(ValueError):
        nig_uncertainties(0, 1, 1, 1)
    with pytest.raises(ValueError):
        dirichlet_uncertainties([0, 1])
