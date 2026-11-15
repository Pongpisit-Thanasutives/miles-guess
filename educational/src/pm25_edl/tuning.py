"""Local Optuna studies on saved station splits. No test file is read here."""
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path

import pandas as pd

from .metrics import score_predictions, score_training_predictions
from .nn_data import Preprocessor, load_split, validate_disjoint_stations
from .nn_models import build_model, suggest_chain
from .nn_training import fit_model, predict_frame, seed_everything

MODELS = ("MLP", "CNN1D", "DB", "TwoStageMLP", "TwoStageCNN1D", "TwoStageDB")
SCORES = ("r2", "rmse", "mae", "nll", "mean_winkler")


def _digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _split_hashes(split_dir):
    return {name: _digest(Path(split_dir) / name) for name in ("train.csv", "val.csv")}


def _code_hashes():
    modules = ("nn_models.py", "nn_data.py", "nn_training.py", "metrics.py", "tuning.py")
    return {f: _digest(Path(__file__).with_name(f)) for f in modules}


def _versions():
    # Core dependencies for training/refitting: Optuna is only needed to search.
    return {p: version(p) for p in ("torch", "numpy", "scipy", "pandas", "scikit-learn")}


def _save_json(path, record):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def suggest_parameters(trial, model_name):
    """Return actual widths (not exponents) and learning rate for one candidate."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")
    def chain(names):
        return dict(zip(names, suggest_chain(trial, names, low=4, high=7)))

    def drop(name):
        return trial.suggest_float(name, 0.0, 0.3, step=0.1)

    if model_name.endswith("DB"):
        model = {**chain(("hts1", "hts2", "hts3")), **chain(("hstt1", "hstt2")),
                 "drop_rate_ts": drop("drop_rate_ts"), "drop_rate_stt": drop("drop_rate_stt")}
        if model_name == "TwoStageDB":
            model["hout"] = 32  # Fusion width; the two branches have separate searches.
    else:
        model = {**chain(("h1", "h2", "hout")), "drop_rate": drop("drop_rate")}
    if model_name.startswith("TwoStage"):
        model.update(hh1=32, hh2=16, hsk=16, hdrop_rate=0.1, threshold=0.0)
    return model, trial.suggest_float("lr", 1e-4, 1e-2, log=True)


def make_objective(model_name, prep, train, val, fixed_fit, interval_level):
    """Close over data transformed once, with train-only scaling shared by all trials."""
    def objective(trial):
        params, lr = suggest_parameters(trial, model_name)
        # Keep the training API's evidence_weight spelling; the Optuna parameter
        # follows the requested evidential_weight name.
        evidence_weight = trial.suggest_float("evidential_weight", 1e-4, 1e-1, log=True)
        lambda_sim = (trial.suggest_float("lambda_sim", 0.02, 0.05, step=0.01)
                      if model_name.startswith("TwoStage") and fixed_fit["use_cosine_similarity"]
                      else 0.0)
        fit_kwargs = dict(fixed_fit, lr=lr, evidence_weight=evidence_weight, lambda_sim=lambda_sim)
        seed_everything(fit_kwargs["seed"])  # Must precede parameter initialization.
        model = build_model(model_name, **prep.model_kwargs, **params)
        history = fit_model(model, train, val, **fit_kwargs)
        # fit_model restores the best validation epoch; no third fit or test scoring.
        predicted = predict_frame(model, val, batch_size=fit_kwargs["batch_size"],
                                  device=fit_kwargs["device"])
        scores = score_predictions(predicted, interval_level=interval_level)
        train_predicted = predict_frame(model, train, batch_size=fit_kwargs["batch_size"],
                                        device=fit_kwargs["device"])
        train_scores = score_training_predictions(train_predicted, interval_level=interval_level)
        if not all(math.isfinite(scores[k]) for k in ("rmse", "mae", "nll", "mean_winkler")):
            raise FloatingPointError("Non-finite validation score; inspect data and model.")
        if not all(math.isfinite(train_scores[k]) for k in ("rmse", "mae", "nll", "mean_winkler")):
            raise FloatingPointError("Non-finite training score; inspect data and model.")
        for key, val_score in scores.items():
            trial.set_user_attr(key, val_score if math.isfinite(val_score) else None)
        for key, train_score in train_scores.items():
            trial.set_user_attr("train_" + key, train_score if math.isfinite(train_score) else None)
        for key, value in {"model_kwargs": params, "fit_kwargs": fit_kwargs,
                           "best_epoch": history.best_epoch, "train_loss": history.train_loss,
                           "val_nll": history.val_nll}.items():
            trial.set_user_attr(key, value)
        return scores["nll"]
    return objective


def load_best(run_dir, *, model_name, feature_option, split_dir):
    """Read the best *settings* for a fresh fit; reject a mismatched split or scaler."""
    directory = Path(run_dir)
    best = json.loads((directory / "best.json").read_text(encoding="utf-8"))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if best["manifest_sha256"] != _digest(manifest_path):
        raise ValueError("Study manifest does not match the exported best settings.")
    if manifest["code_sha256"] != _code_hashes() or manifest["versions"] != _versions():
        raise ValueError("Training code or core package versions changed since tuning. "
                         "Restore the original setup or tune in a new run directory.")
    if (best["model_name"], best["feature_option"]) != (model_name, feature_option):
        raise ValueError("Best settings belong to a different model or feature option.")
    if best["split_sha256"] != _split_hashes(split_dir):
        raise ValueError("train.csv or val.csv has changed since tuning. Use the original split.")
    if best["preprocessor_sha256"] != _digest(directory / "preprocessor.json"):
        raise ValueError("The saved preprocessor no longer matches this study.")
    return best


def run_study(split_dir, run_dir, *, model_name, feature_option="at-bb-aod",
              n_trials=25, epochs=100, patience=10, batch_size=64, seed=88,
              interval_level=0.90, use_cosine_similarity=True, device="cpu"):
    """Run or resume a single-worker validation study; n_trials is the desired total.

    SQLite is the trial history; increasing n_trials from 25 to 50 adds 25
    completed trials rather than restarting. Export best.json and trials.csv
    after every trial, including when training raises. Use a separate run
    directory for changed data, software or training/search policy.
    Evidence weight is searched for every model; source similarity weight is
    searched only for two-stage models when use_cosine_similarity=True.
    """
    import optuna  # Only needed for the tuning stage.

    if n_trials < 1 or batch_size < 1 or epochs < 1 or patience < 1 or not 0 < interval_level < 1:
        raise ValueError("Positive trial/training counts and 0 < interval_level < 1 are required")
    if model_name not in MODELS:
        raise ValueError(f"model_name must be one of {MODELS}")
    root = Path(run_dir).resolve()
    tr, vl = load_split(split_dir, "train"), load_split(split_dir, "val")
    validate_disjoint_stations(tr, vl)
    fixed_fit = dict(epochs=epochs, patience=patience, batch_size=batch_size, seed=seed,
                     device=device, optimizer_name="Adam", weight_decay=0.0,
                     use_cosine_similarity=bool(use_cosine_similarity and model_name.startswith("TwoStage")))
    manifest = dict(model_name=model_name, feature_option=feature_option,
                    split_sha256=_split_hashes(split_dir), fixed_fit=fixed_fit,
                    interval_level=interval_level, objective="validation Student-t NLL",
                    versions=_versions(), optuna_version=version("optuna"), code_sha256=_code_hashes())
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    scaler_path = root / "preprocessor.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.pop("preprocessor_sha256") != _digest(scaler_path) or saved != manifest:
            raise ValueError("Incompatible study or scaler. Use the original setup or a new directory.")
        prep = Preprocessor.load(scaler_path)  # Resume using the immutable train-fit scaler.
    else:
        if any(root.iterdir()):
            raise ValueError("No manifest in this nonempty directory. Choose an empty directory.")
        prep = Preprocessor.fit(tr, model_name, feature_option)
        tmp = scaler_path.with_suffix(".tmp")
        prep.save(tmp)
        tmp.replace(scaler_path)
        _save_json(manifest_path, dict(manifest, preprocessor_sha256=_digest(scaler_path)))
    train, val = prep.transform(tr, split="train"), prep.transform(vl, split="val")
    history_path = root / "study.sqlite3"
    db = "sqlite:///" + history_path.as_posix()
    name = f"{model_name}/{feature_option}"
    if history_path.exists():
        # The database, not the CSV summary, owns trial numbers and states.
        study = optuna.load_study(study_name=name, storage=db)
    else:
        if (root / "trials.csv").exists() or (root / "best.json").exists():
            raise FileNotFoundError("Saved trial summaries exist, but study.sqlite3 is missing. "
                                    "Restore the database to resume; do not restart in this directory.")
        study = optuna.create_study(study_name=name, storage=db, direction="minimize")
    objective = make_objective(model_name, prep, train, val, fixed_fit, interval_level)

    def save_history(study):
        rows = []
        for t in study.trials:
            a = t.user_attrs
            rows.append({"trial": t.number, "state": t.state.name, "best_epoch": a.get("best_epoch"),
                         "n_obs": a.get("n_obs"), **{k: a.get(k) for k in SCORES},
                         **{"train_" + k: a.get("train_" + k) for k in (*SCORES, "n_obs")},
                         "interval_level": interval_level, **t.params, **a.get("model_kwargs", {})})
        tmp = (root / "trials.csv").with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(root / "trials.csv")
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            return
        best = study.best_trial
        attrs = best.user_attrs
        _save_json(root / "best.json", dict(model_name=model_name, feature_option=feature_option,
                   split_sha256=manifest["split_sha256"], preprocessor_sha256=_digest(scaler_path),
                   manifest_sha256=_digest(manifest_path),
                   best_trial=best.number, best_epoch=attrs["best_epoch"], model_kwargs=attrs["model_kwargs"],
                   fit_kwargs=attrs["fit_kwargs"], interval_level=interval_level,
                   training_scores={k: attrs["train_" + k] for k in (*SCORES, "n_obs")},
                   validation_scores={k: attrs[k] for k in (*SCORES, "n_obs")}))

    trials = study.get_trials(deepcopy=False)
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    if completed > n_trials:
        raise ValueError(f"{completed} trials already completed; requested only {n_trials}.")
    if trials:
        save_history(study)
    for _ in range(n_trials - completed):
        # Index each random stream by trial number so restarting does not replay the
        # first candidate. A stopped trial remains visible in SQLite; it has no scores.
        next_number = len(study.get_trials(deepcopy=False))
        sampler = optuna.samplers.RandomSampler(seed=(seed + next_number) % (2**32))
        study = optuna.load_study(study_name=name, storage=db, sampler=sampler)
        try:
            study.optimize(objective, n_trials=1, n_jobs=1, show_progress_bar=False)
        finally:
            save_history(study)

    return study


if __name__ == "__main__":
    from .verification import check_model_contract
    check_model_contract("TwoStageCNN1D")
