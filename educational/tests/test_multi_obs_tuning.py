"""Integration checks for the independent three-objective Optuna entry point."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="Educational model checks require PyTorch")
optuna = pytest.importorskip("optuna", reason="Multi-objective studies require Optuna")

from pm25_edl import multi_obs_tuning as multi, tuning
from pm25_edl.metrics import score_predictions, score_training_predictions
from schema_fixture import station_fixture


def _splits(tmp_path, validation=None):
    directory = tmp_path / "splits"
    directory.mkdir()
    station_fixture("train").to_csv(directory / "train.csv", index=False)
    if validation is None:
        validation = station_fixture("val")
    validation.to_csv(directory / "val.csv", index=False)
    # Deliberately no test.csv: tuning must not require test observations.
    return directory


def _record(trial, values, fixed_fit):
    r2, nll, winkler = values
    # The searched width exponent and resolved model width must remain distinct.
    exponent = trial.suggest_int("h1", 4, 7)
    metrics = dict(r2=r2, rmse=1., mae=.5, nll=nll, mean_winkler=winkler, n_obs=13)
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
        trial.set_user_attr("train_" + key, 21 if key == "n_obs" else value)
    trial.set_user_attr("train_r2", None)  # Undefined training R2 is allowed.
    for key, value in dict(model_kwargs={"h1": 2**exponent}, fit_kwargs=fixed_fit,
                           best_epoch=1, train_loss=[1.], val_nll=[nll]).items():
        trial.set_user_attr(key, value)


def _fake_factory(called, *, vectors=None, fail_numbers=()):
    """Stub only expensive fitting; use the real adapter, SQLite and sampler."""
    def factory(model_name, prep, train, val, fixed_fit, interval_level):
        def objective(trial):
            called.append(trial.number)
            if trial.number in fail_numbers:
                raise RuntimeError("intentional fitting interruption")
            x = trial.suggest_float("x", 0., 1.)
            values = ((1 - (x - .5)**2, (x - .2)**2, (x - .8)**2)
                      if vectors is None else vectors[trial.number])
            _record(trial, values, fixed_fit)
            return values[1]  # The original objective returns only NLL.
        return objective
    return factory


def _options(**kwargs):
    return dict(model_name="MLP", epochs=1, patience=1, seed=17, **kwargs)


def _load_kwargs(splits, name="MLP"):
    return dict(model_name=name, feature_option="at-bb-aod", split_dir=splits)


@pytest.mark.parametrize("name", ["MLP", "TwoStageDB"])
def test_real_multi_objective_training_and_contracts(tmp_path, monkeypatch, name):
    validation = station_fixture("val").assign(pm25_filled=999999.)
    splits = _splits(tmp_path, validation)
    run = tmp_path / "run"
    predictions = []
    original_predict = tuning.predict_frame
    def capture(*args, **kwargs):
        frame = original_predict(*args, **kwargs)
        predictions.append(frame)
        return frame
    monkeypatch.setattr(tuning, "predict_frame", capture)
    opts = dict(model_name=name, epochs=1, patience=1, seed=19)
    study = multi.run_study(splits, run, n_trials=1, **opts)
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert [d.name for d in study.directions] == ["MAXIMIZE", "MINIMIZE", "MINIMIZE"]
    trial = study.trials[0]
    val_frame = next(f for f in predictions if "pm25" in f)
    train_frame = next(f for f in predictions if "pm25_filled" in f)
    scores = score_predictions(val_frame)
    train_scores = score_training_predictions(train_frame)
    np.testing.assert_allclose(trial.values, [scores[k] for k in ("r2", "nll", "mean_winkler")])
    assert scores["n_obs"] == 13 and train_scores["n_obs"] == 21
    for key in (*tuning.SCORES, "n_obs"):
        assert trial.user_attrs[key] == pytest.approx(scores[key])
        assert trial.user_attrs["train_" + key] == pytest.approx(train_scores[key])
    ew = trial.distributions["evidential_weight"]
    assert (ew.low, ew.high, ew.log) == (1e-4, 1e-1, True)
    if name.startswith("TwoStage"):
        sim = trial.distributions["lambda_sim"]
        assert (sim.low, sim.high, sim.step) == (.02, .05, .01)
        for frame in predictions:
            sources = [c for c in frame if c.startswith("s_corrected_")]
            np.testing.assert_array_equal(frame[sources].sum(axis=1), frame["pred"])
    else:
        assert "lambda_sim" not in trial.params
    best = multi.load_best(run, **_load_kwargs(splits, name))
    assert best["validation_scores"] == scores
    assert best["fit_kwargs"] == trial.user_attrs["fit_kwargs"]
    assert best["model_kwargs"] == trial.user_attrs["model_kwargs"]
    table = pd.read_csv(run / "trials.csv")
    for key in ("r2", "nll", "mean_winkler"):
        assert table.loc[0, "objective_" + key] == pytest.approx(scores[key])
    # Exercise numerical training again after reopening an actual saved study.
    if name == "MLP":
        old = (trial.params, trial.values, trial.datetime_start, trial.user_attrs)
        resumed = multi.run_study(splits, run, n_trials=2, **opts)
        prior = resumed.trials[0]
        assert (prior.params, prior.values, prior.datetime_start, prior.user_attrs) == old
        assert len(resumed.trials) == 2


def test_pareto_exports_selection_and_unambiguous_parameters(tmp_path, monkeypatch):
    splits = _splits(tmp_path)
    run = tmp_path / "run"
    vectors = [(.2, 1., 10.), (.4, -2., 9.), (.8, 0., 8.),
               (.3, 1., 12.), (.6, -2., 9.)]
    monkeypatch.setattr(tuning, "make_objective", _fake_factory([], vectors=vectors))
    study = multi.run_study(splits, run, n_trials=5, **_options())
    assert {t.number for t in study.best_trials} == {2, 4}
    front = multi.load_pareto(run, **_load_kwargs(splits))
    assert {t["best_trial"] for t in front} == {2, 4}
    best = multi.load_best(run, **_load_kwargs(splits))
    assert best["best_trial"] == 4  # Smaller NLL; negative NLL is valid.
    assert best["selection_rule"]
    chosen = multi.load_best(run, trial_number=2, **_load_kwargs(splits))
    assert chosen["best_trial"] == 2
    with pytest.raises(ValueError):
        multi.load_best(run, trial_number=1, **_load_kwargs(splits))
    table = pd.read_csv(run / "trials.csv")
    assert set(table.loc[table.is_pareto, "trial"]) == {2, 4}
    assert set(pd.read_csv(run / "pareto_front.csv").trial) == {2, 4}
    for t in study.trials:
        assert table.loc[t.number, "param_h1"] == t.params["h1"]
        assert table.loc[t.number, "h1"] == 2 ** t.params["h1"]
    exported = json.loads((run / "pareto_front.json").read_text(),
                          parse_constant=lambda s: pytest.fail(f"Invalid JSON constant: {s}"))
    assert exported["objective_names"] == ["r2", "nll", "mean_winkler"]
    assert exported["directions"] == ["maximize", "minimize", "minimize"]
    for record in exported["trials"]:
        assert record["values"] == list(vectors[record["best_trial"]])
        assert record["training_scores"]["r2"] is None
        assert record["params"] == study.trials[record["best_trial"]].params


def test_tpe_resume_preserves_history_and_adaptive_sequence(tmp_path, monkeypatch):
    splits = _splits(tmp_path)
    called = []
    monkeypatch.setattr(tuning, "make_objective", _fake_factory(called))
    full = multi.run_study(splits, tmp_path / "full", n_trials=15, **_options())
    called.clear()
    run = tmp_path / "resumed"
    first = multi.run_study(splits, run, n_trials=5, **_options())
    old = [(t.params, t.values, t.datetime_start, t.user_attrs) for t in first.trials]
    scaler = run / "preprocessor.json"
    old_scaler = (scaler.read_bytes(), scaler.stat().st_mtime_ns)
    called.clear()
    resumed = multi.run_study(splits, run, n_trials=15, **_options())
    assert called == list(range(5, 15))
    assert [(t.params, t.values, t.datetime_start, t.user_attrs) for t in resumed.trials[:5]] == old
    assert [(t.params, t.values) for t in resumed.trials] == [(t.params, t.values) for t in full.trials]
    assert (scaler.read_bytes(), scaler.stat().st_mtime_ns) == old_scaler
    called.clear()
    multi.run_study(splits, run, n_trials=15, **_options())
    assert not called
    with pytest.raises(ValueError):
        multi.run_study(splits, run, n_trials=5, **_options())
    assert not called


def test_failure_history_and_missing_database_protection(tmp_path, monkeypatch):
    splits = _splits(tmp_path)
    run = tmp_path / "run"
    monkeypatch.setattr(tuning, "make_objective", _fake_factory([], fail_numbers=(1,)))
    with pytest.raises(RuntimeError, match="intentional fitting interruption"):
        multi.run_study(splits, run, n_trials=2, **_options())
    assert pd.read_csv(run / "trials.csv").state.tolist() == ["COMPLETE", "FAIL"]
    assert multi.load_best(run, **_load_kwargs(splits))["best_trial"] == 0
    resumed = multi.run_study(splits, run, n_trials=2, **_options())
    assert [t.state.name for t in resumed.trials] == ["COMPLETE", "FAIL", "COMPLETE"]
    assert len(resumed.best_trials) >= 1
    summary = (run / "trials.csv").read_bytes()
    (run / "study.sqlite3").rename(run / "saved.sqlite3")
    with pytest.raises(FileNotFoundError):
        multi.run_study(splits, run, n_trials=3, **_options())
    assert (run / "trials.csv").read_bytes() == summary


def test_interruption_before_first_trial_exports_empty_history(tmp_path, monkeypatch):
    splits = _splits(tmp_path)
    run = tmp_path / "run"
    monkeypatch.setattr(tuning, "make_objective", _fake_factory([]))
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("interrupted before allocating a trial")
    with monkeypatch.context() as patch:
        patch.setattr(optuna.study.Study, "optimize", interrupt)
        with pytest.raises(KeyboardInterrupt):
            multi.run_study(splits, run, n_trials=1, **_options())
    assert pd.read_csv(run / "trials.csv").empty
    assert pd.read_csv(run / "pareto_front.csv").empty
    assert json.loads((run / "pareto_front.json").read_text())["trials"] == []
    assert not (run / "best.json").exists()
    resumed = multi.run_study(splits, run, n_trials=1, **_options())
    assert len(resumed.trials) == 1 and resumed.trials[0].number == 0


@pytest.mark.parametrize("values", [(None, 1., 2.), (float("nan"), 1., 2.),
                                    (.5, float("inf"), 2.), (.5, 1., float("inf"))])
def test_nonfinite_objectives_fail_without_fake_best(tmp_path, monkeypatch, values):
    splits = _splits(tmp_path)
    run = tmp_path / "run"
    monkeypatch.setattr(tuning, "make_objective", _fake_factory([], vectors=[values]))
    with pytest.raises((FloatingPointError, ValueError)):
        multi.run_study(splits, run, n_trials=1, **_options())
    assert pd.read_csv(run / "trials.csv").state.tolist() == ["FAIL"]
    assert json.loads((run / "pareto_front.json").read_text())["trials"] == []
    assert not (run / "best.json").exists()


@pytest.mark.parametrize("case", ["constant", "missing", "one_observation"])
def test_validation_r2_preflight_uses_observations_only(tmp_path, monkeypatch, case):
    validation = station_fixture("val")
    validation["pm25"] = 3. if case == "constant" else np.nan
    if case == "one_observation":
        validation.loc[0, "pm25"] = 4.
    validation["pm25_filled"] = np.arange(len(validation))
    splits = _splits(tmp_path, validation)
    run = tmp_path / "run"
    called = []
    monkeypatch.setattr(tuning, "make_objective", _fake_factory(called))
    with pytest.raises(ValueError):
        multi.run_study(splits, run, n_trials=1, **_options())
    assert not called and not run.exists()


def test_manifest_and_loader_mismatch_guards(tmp_path, monkeypatch):
    splits = _splits(tmp_path)
    run = tmp_path / "run"
    monkeypatch.setattr(tuning, "make_objective", _fake_factory([]))
    study = multi.run_study(splits, run, n_trials=1, **_options())
    with pytest.raises(ValueError):
        multi.run_study(splits, run, n_trials=2, interval_level=.8, **_options())
    with pytest.raises(ValueError):
        multi.load_best(run, model_name="CNN1D", feature_option="at-bb-aod", split_dir=splits)
    study.set_user_attr("manifest_sha256", "wrong manifest fingerprint")
    with pytest.raises(ValueError):
        multi.run_study(splits, run, n_trials=2, **_options())


def test_single_objective_directory_is_not_reused(tmp_path, monkeypatch):
    splits = _splits(tmp_path)
    run = tmp_path / "run"
    monkeypatch.setattr(tuning, "make_objective", _fake_factory([]))
    tuning.run_study(splits, run, n_trials=1, **_options())
    original = (run / "best.json").read_bytes()
    with pytest.raises(ValueError):
        multi.run_study(splits, run, n_trials=2, **_options())
    assert (run / "best.json").read_bytes() == original
