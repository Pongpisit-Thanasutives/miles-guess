"""Independent tests for spatial separation, predictive math and source accounting."""
import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="Educational model checks require the educational or torch extra")
from scipy.stats import t

from pm25_edl.metrics import discard_curve, predictive_interval, predictive_pit, score_predictions
from pm25_edl.nn_data import Preprocessor, load_split, save_station_splits, validate_disjoint_stations
from pm25_edl.nn_models import build_model
from schema_fixture import station_fixture


def test_split_then_impute_training_only(tmp_path):
    train = station_fixture("train").rename(columns={"pm25_filled": "pm25"})
    raw = pd.concat([train, station_fixture("val"), station_fixture("test")], ignore_index=True)
    # Sentinel: the imputer must receive only the selected training stations, and
    # the exported validation/test labels must correspond exactly to the raw keys.
    seen = []
    def impute(frame):
        seen.extend(frame.stcode.unique())
        return frame.assign(pm25_filled=frame.pm25.fillna(22.0))
    save_station_splits(raw, tmp_path, train_imputer=impute)
    tr, vl, te = [load_split(tmp_path, s) for s in ("train", "val", "test")]
    validate_disjoint_stations(tr, vl, te)
    assert set(seen) == set(tr.stcode)
    assert "pm25_filled" not in vl and "pm25_filled" not in te
    assert "pm25_fill_n_neighbors" not in tr and tr.pm25_filled.notna().all()
    keys = ["stcode", "date"]
    raw["date"] = pd.to_datetime(raw.date)
    expected = raw.set_index(keys).pm25
    for frame in (vl, te):
        indexed = frame.set_index(keys).pm25
        np.testing.assert_allclose(indexed, expected.loc[indexed.index], equal_nan=True)
    with pytest.raises(ValueError):
        validate_disjoint_stations(tr, tr.iloc[:1])


@pytest.mark.parametrize("name", ["CNN1D", "DB", "TwoStageCNN1D", "TwoStageDB"])
def test_ordering_calendar_gaps_and_causality(name):
    tr, val = station_fixture("train"), station_fixture("val")
    prep = Preprocessor.fit(tr, name, "at-bb-aod")
    # Deliberately include a poisonous filled column. It must never become a val label.
    sp = prep.transform(val.assign(pm25_filled=99999.), "val")
    seen = []
    model = build_model(name, **prep.model_kwargs).eval()
    for b in sp.batches(32):
        for indices in b.row_ids.tolist():
            run = sp.frame.iloc[indices]
            assert run.stcode.nunique() == 1
            assert (run.date.diff().dropna() == pd.Timedelta(days=1)).all()
            seen.extend(indices)
        labels = val.pm25.to_numpy()[b.row_ids.numpy()]
        np.testing.assert_array_equal(b.y.isnan(), np.isnan(labels))
        with torch.no_grad():
            original = model(b.x, b.x_static)
            changed = b.x.clone()
            changed[:, -1, :] += 99  # last input cannot change earlier outputs
            modified = model(changed, b.x_static)
            for key in ("gamma", "nu", "alpha", "beta"):
                torch.testing.assert_close(getattr(original, key)[:, :-1], getattr(modified, key)[:, :-1])
    assert sorted(seen) == list(range(len(val)))


@pytest.mark.parametrize("name", ["TwoStageMLP", "TwoStageCNN1D", "TwoStageDB"])
def test_physical_source_conservation(name):
    tr = station_fixture("train")
    with pytest.raises(ValueError):
        Preprocessor.fit(tr, name, "baseline")
    prep = Preprocessor.fit(tr, name, "at-bb")
    split = prep.transform(tr, "train")
    b = next(iter(split.batches(4)))
    model = build_model(name, **prep.model_kwargs).eval()
    with torch.no_grad():
        out = model(b.x, b.x_static)
        assert torch.equal(out.pred_physical, out.s_corrected.sum(-1)) and (out.s_corrected >= 0).all()
        torch.testing.assert_close(out.gamma * prep.target_scale + prep.target_mean, out.pred_physical)


def test_student_t_density_pit_winkler_and_affine_units():
    frame = pd.DataFrame({"gamma": [2., 2., 2., 2.], "pm25": [2., -6., 10., np.nan],
                          "nu": [1.] * 4, "alpha": [2.] * 4, "beta": [1.] * 4,
                          "total_std": [1.] * 4})
    # nu=1, alpha=2, beta=1: df=4, scale=1, PDF at the origin is exactly 3/8.
    density = 3/8 * np.power(1 + np.square([0., -8., 8.]) / 4, -2.5)
    met = score_predictions(frame)
    assert met["n_obs"] == 3
    assert met["nll"] == pytest.approx(-np.log(density).mean())
    np.testing.assert_allclose(predictive_pit(frame), [0.5, t.cdf(-8., 4), t.cdf(8., 4)])
    lo, hi = predictive_interval(frame)
    w = hi[0] - lo[0]
    assert met["mean_winkler"] == pytest.approx(w + 2/3 * 20 * (10. - hi[0]))
    # Change physical units y' = 5 + 3*y: PIT unchanged; log-density shifts by -log(3).
    shifted = frame.assign(gamma=5 + 3*frame.gamma, pm25=5 + 3*frame.pm25, beta=9*frame.beta)
    converted = score_predictions(shifted)
    np.testing.assert_allclose(predictive_pit(frame), predictive_pit(shifted))
    assert converted["nll"] == pytest.approx(met["nll"] + math.log(3))
    for key in ("mae", "rmse", "mean_winkler"):
        assert converted[key] == pytest.approx(3 * met[key])
    with pytest.raises(ValueError):
        score_predictions(frame.rename(columns={"pm25": "pm25_filled"}))


def test_discard_orders_by_uncertainty_and_removes_missing_outcomes():
    df = pd.DataFrame(dict(pm25=[0, 1, 10, np.nan], gamma=0., total_std=[1., 2., 3., 99.]))
    result = discard_curve(df, fractions=[0., 1/3, 2/3])
    np.testing.assert_array_equal(result.n_retained, [3, 2, 1])
    np.testing.assert_allclose(result.rmse, [np.sqrt(101/3), np.sqrt(1/2), 0])


@pytest.mark.parametrize("name", ["MLP", "CNN1D", "DB", "TwoStageMLP", "TwoStageCNN1D", "TwoStageDB"])
def test_loss_gradients_and_exported_units(name):
    from pm25_edl.nn_training import evidential_loss, predict_frame
    train, test = station_fixture("train"), station_fixture("test")
    prep = Preprocessor.fit(train, name, "at-bb-aod")
    split = prep.transform(test, "test")
    model = build_model(name, **prep.model_kwargs)
    batch = next(b for b in split.batches(32) if b.y.isfinite().any())
    o = model(*batch.inputs)
    o.gamma.retain_grad()
    loss = evidential_loss(o, batch.y)
    loss.backward()
    assert torch.isfinite(loss)
    assert (o.gamma.grad[~batch.y.isfinite()] == 0).all()
    assert sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None) > 0
    exp = predict_frame(model, split)
    assert exp.stcode.tolist() == test.stcode.tolist() and exp.date.tolist() == pd.to_datetime(test.date).tolist()
    np.testing.assert_allclose(exp.pm25, test.pm25, equal_nan=True)
    np.testing.assert_allclose(exp.aleatoric_std**2, exp.beta/(exp.alpha-1))
    np.testing.assert_allclose(exp.total_std**2, exp.aleatoric_std**2 + exp.epistemic_std**2)
    if name.startswith("TwoStage"):
        cols = ["s_corrected_cmaq_pm25_" + x for x in ("bb", "anthro", "othr")]
        np.testing.assert_array_equal(exp[cols].to_numpy().sum(axis=1), exp.pred)
    assert score_predictions(exp)["n_obs"] == test.pm25.notna().sum()


def test_hpt_runs_without_test_csv_and_rejects_mismatched_artifacts(tmp_path):
    pytest.importorskip("optuna", reason="Study test exercises the optional tuning dependency")
    from pm25_edl.tuning import load_best, run_study
    split_dir, run_dir = tmp_path/"splits", tmp_path/"runs"
    split_dir.mkdir()
    for split in ("train", "val"):
        station_fixture(split).to_csv(split_dir/f"{split}.csv", index=False)
    # No test.csv exists. The study must still train, score and persist.
    study = run_study(split_dir, run_dir, model_name="MLP", feature_option="baseline",
                      n_trials=1, epochs=1, patience=1, seed=7)
    read_best = lambda: load_best(run_dir, model_name="MLP", feature_option="baseline", split_dir=split_dir)
    best = read_best()
    assert len(study.trials) == 1 and set(("r2", "rmse", "mae", "nll", "mean_winkler")) <= best["validation_scores"].keys()
    assert best["validation_scores"]["n_obs"] == station_fixture("val").pm25.notna().sum()
    assert best["validation_scores"]["nll"] == pytest.approx(study.best_value)
    assert (run_dir/"trials.csv").is_file()
    first = study.trials[0]
    before = (first.number, first.params, first.value, first.user_attrs)
    resumed = run_study(split_dir, run_dir, model_name="MLP", feature_option="baseline",
                        n_trials=2, epochs=1, patience=1, seed=7)
    assert len(resumed.trials) == 2
    preserved = resumed.trials[0]
    assert (preserved.number, preserved.params, preserved.value, preserved.user_attrs) == before
    assert all("train_nll" in trial.user_attrs for trial in resumed.trials)
    assert len(pd.read_csv(run_dir/"trials.csv")) == 2
    with (split_dir/"val.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError):
        read_best()


@pytest.mark.parametrize("name", ["MLP", "CNN1D", "DB", "TwoStageMLP", "TwoStageCNN1D", "TwoStageDB"])
@pytest.mark.parametrize("option", ["baseline", "at-bb", "at-bb-aod"])
def test_declared_input_ablation_and_output_shapes(name, option):
    """A true baseline must not leak decomposed sources into a single-stage fit."""
    train = station_fixture("train")
    if name.startswith("TwoStage") and option == "baseline":
        with pytest.raises(ValueError):
            Preprocessor.fit(train, name, option)
        return
    prep = Preprocessor.fit(train, name, option)
    cmaq = {c for c in prep.features if c.startswith("cmaq_")}
    expected = {"cmaq_pm25"} if option == "baseline" else {"cmaq_pm25_bb","cmaq_pm25_anthro","cmaq_pm25_othr"}
    assert cmaq == expected and ("aod" in prep.features) == (option == "at-bb-aod")
    assert len([c for c in prep.features if c.startswith("wrf_")]) == 4
    assert len(prep.features) == len(expected) + int(option == "at-bb-aod") + 4 + 7
    b = next(prep.transform(train, "train").batches(3))
    model = build_model(name, **prep.model_kwargs)
    assert model(*b.inputs).gamma.shape == b.y.shape


def test_cosine_switch_is_exclusive_to_two_stage_loss():
    from types import SimpleNamespace
    from pm25_edl.nn_training import evidential_loss

    def output(two_stage):
        g = torch.tensor([1., 2., 3.], requires_grad=True)
        d = torch.tensor([[1., 2., 4.], [2., 1., 3.], [3., 2., 1.]], requires_grad=True)
        return SimpleNamespace(gamma=g, nu=torch.ones(3), alpha=torch.full((3,), 2.),
                               beta=torch.ones(3), s_delta=d if two_stage else None), d
    labels = torch.tensor([1.5, float("nan"), 4.])
    on, d_on = output(True)
    off, d_off = output(True)
    positive = evidential_loss(on, labels, lambda_sim=.05, use_cosine_similarity=True)
    disabled = evidential_loss(off, labels, lambda_sim=.05, use_cosine_similarity=False)
    assert positive > disabled
    positive.backward()
    disabled.backward()
    assert d_on.grad is not None and d_on.grad.abs().sum() > 0
    assert (d_on.grad[1] == 0).all()  # Unobserved targets do not enter the penalty.
    assert d_off.grad is None  # Disabling the term removes its gradient path.
    for flag in (False, True):
        single, _ = output(False)
        with_weight = evidential_loss(single, labels, lambda_sim=999., use_cosine_similarity=flag)
        without_weight = evidential_loss(single, labels, lambda_sim=0., use_cosine_similarity=flag)
        torch.testing.assert_close(with_weight, without_weight)


def test_training_scores_explicitly_use_filled_targets():
    from pm25_edl.metrics import score_training_predictions
    frame = pd.DataFrame(dict(pm25_filled=[0., 2., 4.], pm25=[100., np.nan, 100.],
                              gamma=0., nu=1., alpha=2., beta=1.))
    original = frame.copy(deep=True)
    training = score_training_predictions(frame)
    observed = score_predictions(frame)
    assert training["n_obs"] == 3 and observed["n_obs"] == 2
    assert training["mae"] == 2 and observed["mae"] == 100
    assert training["rmse"] == pytest.approx(np.sqrt(20/3))
    assert training["nll"] == pytest.approx(-t.logpdf([0., 2., 4.], df=4).mean())
    reference = frame.drop(columns="pm25").rename(columns={"pm25_filled": "pm25"})
    assert training == score_predictions(reference)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError):
        score_training_predictions(frame.drop(columns="pm25_filled"))


@pytest.mark.parametrize("name,enabled", [("MLP", True), ("MLP", False),
                                          ("TwoStageMLP", True), ("TwoStageMLP", False)])
def test_hpt_search_loss_weights_and_training_metrics(tmp_path, name, enabled):
    pytest.importorskip("optuna")
    from pm25_edl.tuning import run_study, load_best
    split_dir, run_dir = tmp_path/"splits", tmp_path/"runs"
    split_dir.mkdir()
    for role in ("train", "val"):
        station_fixture(role).to_csv(split_dir/f"{role}.csv", index=False)
    study = run_study(split_dir, run_dir, model_name=name, n_trials=1, epochs=1,
                      patience=1, use_cosine_similarity=enabled, seed=11)
    trial = study.best_trial
    ew = trial.distributions["evidential_weight"]
    assert (ew.low, ew.high, ew.log) == (1e-4, 1e-1, True)
    assert trial.user_attrs["fit_kwargs"]["evidence_weight"] == trial.params["evidential_weight"]
    if name.startswith("TwoStage") and enabled:
        sim = trial.distributions["lambda_sim"]
        assert (sim.low, sim.high, sim.step, sim.log) == (.02, .05, .01, False)
        assert trial.user_attrs["fit_kwargs"]["lambda_sim"] == trial.params["lambda_sim"]
    else:
        assert "lambda_sim" not in trial.params
        assert trial.user_attrs["fit_kwargs"]["lambda_sim"] == 0
    metrics = ("r2", "rmse", "mae", "nll", "mean_winkler", "n_obs")
    assert all("train_" + key in trial.user_attrs for key in metrics)
    assert trial.user_attrs["train_n_obs"] == 21 and trial.user_attrs["n_obs"] == 13
    table = pd.read_csv(run_dir/"trials.csv")
    assert all("train_" + key in table for key in metrics)
    best = load_best(run_dir, model_name=name, feature_option="at-bb-aod", split_dir=split_dir)
    assert best["training_scores"]["n_obs"] == 21 and best["validation_scores"]["n_obs"] == 13
    assert best["fit_kwargs"]["use_cosine_similarity"] == (enabled and name.startswith("TwoStage"))


def test_study_resumes_25_to_50_without_replaying_trials(tmp_path, monkeypatch):
    """Exercise real SQLite resume with a cheap objective; fitting has separate tests."""
    pytest.importorskip("optuna")
    import json
    from pm25_edl import tuning
    split_dir, run_dir = tmp_path/"splits", tmp_path/"runs"
    split_dir.mkdir()
    for role in ("train", "val"):
        station_fixture(role).to_csv(split_dir/f"{role}.csv", index=False)
    called = []
    def fake_objective(*args):
        def objective(trial):
            called.append(trial.number)
            trial.suggest_float("evidential_weight", 1e-4, 1e-1, log=True)
            score = 1. / (trial.number + 1)
            for key in (*tuning.SCORES, "n_obs"):
                trial.set_user_attr(key, 13 if key == "n_obs" else score)
                trial.set_user_attr("train_" + key, 21 if key == "n_obs" else score)
            for key, value in dict(model_kwargs={}, fit_kwargs={}, best_epoch=1,
                                   train_loss=[score], val_nll=[score]).items():
                trial.set_user_attr(key, value)
            return score
        return objective
    monkeypatch.setattr(tuning, "make_objective", fake_objective)
    options = dict(model_name="MLP", epochs=1, patience=1, seed=9)
    first = tuning.run_study(split_dir, run_dir, n_trials=25, **options)
    assert called == list(range(25))
    scaler = run_dir/"preprocessor.json"
    unchanged = (scaler.read_bytes(), scaler.stat().st_mtime_ns)
    old = [(t.number, t.params, t.value, t.datetime_start, t.user_attrs) for t in first.trials]
    called.clear()
    resumed = tuning.run_study(split_dir, run_dir, n_trials=50, **options)
    assert called == list(range(25, 50))
    assert [(t.number, t.params, t.value, t.datetime_start, t.user_attrs) for t in resumed.trials[:25]] == old
    assert len(resumed.trials) == 50 and len(pd.read_csv(run_dir/"trials.csv")) == 50
    assert (scaler.read_bytes(), scaler.stat().st_mtime_ns) == unchanged
    assert json.loads((run_dir/"best.json").read_text())["best_trial"] == 49
    called.clear()
    tuning.run_study(split_dir, run_dir, n_trials=50, **options)
    assert called == []
    # A lower requested total cannot erase or rerun the existing study.
    with pytest.raises(ValueError):
        tuning.run_study(split_dir, run_dir, n_trials=25, **options)
    assert called == []
    # A CSV summary alone is not sufficient to reconstruct Optuna's history.
    summary = (run_dir/"trials.csv").read_bytes()
    (run_dir/"study.sqlite3").rename(run_dir/"study.saved.sqlite3")
    with pytest.raises(FileNotFoundError, match="study.sqlite3 is missing"):
        tuning.run_study(split_dir, run_dir, n_trials=51, **options)
    assert called == [] and (run_dir/"trials.csv").read_bytes() == summary


def test_study_exports_completed_history_when_next_trial_fails(tmp_path, monkeypatch):
    """An interrupted extension must retain the previously completed scores."""
    pytest.importorskip("optuna")
    import json
    from pm25_edl import tuning
    split_dir, run_dir = tmp_path/"splits", tmp_path/"runs"
    split_dir.mkdir()
    for role in ("train", "val"):
        station_fixture(role).to_csv(split_dir/f"{role}.csv", index=False)
    def fail_after_one(*args):
        def objective(trial):
            if trial.number == 1:
                raise RuntimeError("intentional test interruption")
            for key in (*tuning.SCORES, "n_obs"):
                trial.set_user_attr(key, 13 if key == "n_obs" else 1.)
                trial.set_user_attr("train_" + key, 21 if key == "n_obs" else 1.)
            for key, value in dict(model_kwargs={}, fit_kwargs={}, best_epoch=1,
                                   train_loss=[1.], val_nll=[1.]).items():
                trial.set_user_attr(key, value)
            return 1.
        return objective
    monkeypatch.setattr(tuning, "make_objective", fail_after_one)
    with pytest.raises(RuntimeError, match="intentional test interruption"):
        tuning.run_study(split_dir, run_dir, model_name="MLP", n_trials=2, epochs=1, patience=1)
    table = pd.read_csv(run_dir/"trials.csv")
    assert len(table) == 2 and table.state.tolist() == ["COMPLETE", "FAIL"]
    assert json.loads((run_dir/"best.json").read_text())["best_trial"] == 0
