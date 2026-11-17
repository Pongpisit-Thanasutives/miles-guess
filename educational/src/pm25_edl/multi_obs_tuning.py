"""Three-objective Optuna tuning on saved station splits; never read test.csv.

Objectives, in order: maximize observed validation R2, minimize predictive
Student-t NLL, minimize mean Winkler score. Reuse tuning.py's complete fit,
search space and scoring path, including filled-target training diagnostics.
Early stopping within each trial still uses validation NLL.

There is no unique multi-objective best trial. Save the entire Pareto front;
best.json is explicitly its lowest-NLL representative (ties: trial number).
load_best(..., trial_number=...) can instead load any saved Pareto member.
Use a new run directory, separate from single-objective studies.

Example (from the educational directory, with your saved CSV splits)::

    from pathlib import Path
    from pm25_edl.multi_obs_tuning import run_study, load_best, load_pareto
    from pm25_edl.nn_data import Preprocessor
    from pm25_edl.nn_models import build_model

    splits = Path("data/splits")
    run = Path("runs_multi/TwoStageDB/at-bb-aod")
    study = run_study(splits, run, model_name="TwoStageDB", n_trials=25)
    context = dict(model_name="TwoStageDB", feature_option="at-bb-aod",
                   split_dir=splits)
    front = load_pareto(run, **context)
    best = load_best(run, **context)  # Clearly labeled lowest-NLL representative.
    # Or choose a Pareto member explicitly:
    # best = load_best(run, **context, trial_number=front[0]["best_trial"])
    prep = Preprocessor.load(run / "preprocessor.json")
    model = build_model("TwoStageDB", **prep.model_kwargs, **best["model_kwargs"])
    # Then use fit_model(..., **best["fit_kwargs"]) as in the usage notebooks.
    # Rerun run_study with n_trials=50 to extend to 50 completed trials.

Exports: study.sqlite3 (authoritative history), manifest.json,
preprocessor.json, trials.csv, pareto_front.csv, pareto_front.json, best.json.
Every Pareto JSON record uses the original best-settings schema, augmented
with raw Optuna params and three values in OBJECTIVE_NAMES order. CSV
param_* columns preserve search exponents alongside resolved model widths.
Trial user attributes retain validation/training metrics and fit histories.
Single-objective notebook imports remain unchanged; use this module's
load_best for new multi-objective runs. Inspect trade-offs on validation only;
reserve test observations for evaluation after selecting a model.

Optuna references:
https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
"""
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import tuning as _single
from .nn_data import Preprocessor, load_split, validate_disjoint_stations

MODELS = _single.MODELS
SCORES = _single.SCORES
OBJECTIVE_NAMES = ("r2", "nll", "mean_winkler")
DIRECTIONS = ("maximize", "minimize", "minimize")
SAMPLER_SETTINGS = dict(name="TPESampler", multivariate=False,
                        constant_liar=False, n_startup_trials=10,
                        seed_policy="(seed + trial_number) % 2**32")
SELECTION_RULE = "Minimum validation NLL on the Pareto front; ties by trial number."


def _code_hashes():
    # Include the reused tuner as well as this adapter in study provenance.
    return {**_single._code_hashes(), Path(__file__).name: _single._digest(__file__)}


def _make_sampler(seed, trial_number):
    import optuna

    # Independent TPE supports the original dynamic/conditional search space.
    # Re-create it on *every* trial, not just on resume. It learns from the
    # SQLite history; trial-indexed seeds avoid needing a sampler pickle.
    return optuna.samplers.TPESampler(
        seed=(seed + trial_number) % (2**32), multivariate=False,
        constant_liar=False, n_startup_trials=10)


def make_objective(model_name, prep, train, val, fixed_fit, interval_level):
    """Return three validation scores from the unchanged training/scoring path."""
    single_objective = _single.make_objective(
        model_name, prep, train, val, fixed_fit, interval_level)

    def objective(trial):
        single_objective(trial)  # Records score_predictions and train scores.
        values = tuple(trial.user_attrs[key] for key in OBJECTIVE_NAMES)
        if any(value is None or not math.isfinite(value) for value in values):
            raise FloatingPointError(
                "All three validation objectives must be finite. R2 requires "
                "at least two nonconstant observed validation targets.")
        return values

    return objective


def _save_csv(path, frame):
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _save_history(study, root, manifest):
    """Refresh all summaries after each trial, including failed trials."""
    front = sorted(study.best_trials, key=lambda t: t.number)
    front_numbers = {t.number for t in front}
    rows = []
    for trial in study.trials:
        attrs = trial.user_attrs
        values = trial.values or (None,) * len(OBJECTIVE_NAMES)
        rows.append({
            "trial": trial.number, "state": trial.state.name,
            "is_pareto": trial.number in front_numbers,
            "best_epoch": attrs.get("best_epoch"), "n_obs": attrs.get("n_obs"),
            **{key: attrs.get(key) for key in SCORES},
            **{"train_" + key: attrs.get("train_" + key) for key in (*SCORES, "n_obs")},
            **{"objective_" + key: value for key, value in zip(OBJECTIVE_NAMES, values)},
            "interval_level": manifest["interval_level"],
            **trial.params, **attrs.get("model_kwargs", {}),
            # Widths in model_kwargs are actual sizes; params contain exponents.
            **{"param_" + key: value for key, value in trial.params.items()},
        })
    columns = ["trial", "state", "is_pareto", "best_epoch", "n_obs", *SCORES,
               *("train_" + key for key in (*SCORES, "n_obs")),
               *("objective_" + key for key in OBJECTIVE_NAMES), "interval_level"]
    table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    _save_csv(root / "trials.csv", table)
    _save_csv(root / "pareto_front.csv", table.loc[table["is_pareto"].astype(bool)].copy())

    context = dict(
        model_name=manifest["model_name"], feature_option=manifest["feature_option"],
        split_sha256=manifest["split_sha256"],
        preprocessor_sha256=_single._digest(root / "preprocessor.json"),
        manifest_sha256=_single._digest(root / "manifest.json"),
        interval_level=manifest["interval_level"],
        objective_names=list(OBJECTIVE_NAMES), directions=list(DIRECTIONS))
    records = []
    for trial in front:
        attrs = trial.user_attrs
        records.append(dict(
            **context, best_trial=trial.number, values=list(trial.values),
            params=trial.params, best_epoch=attrs["best_epoch"],
            model_kwargs=attrs["model_kwargs"], fit_kwargs=attrs["fit_kwargs"],
            training_scores={key: attrs["train_" + key] for key in (*SCORES, "n_obs")},
            validation_scores={key: attrs[key] for key in (*SCORES, "n_obs")}))
    _single._save_json(root / "pareto_front.json", dict(**context, trials=records))
    if records:
        representative = min(records, key=lambda r: (r["validation_scores"]["nll"],
                                                       r["best_trial"]))
        _single._save_json(root / "best.json", dict(
            **representative, selection_rule=SELECTION_RULE))
    else:
        (root / "best.json").unlink(missing_ok=True)


def _validate_export(root, exported, *, model_name, feature_option, split_dir):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if exported["manifest_sha256"] != _single._digest(manifest_path):
        raise ValueError("Study manifest does not match the exported settings.")
    if (manifest["code_sha256"] != _code_hashes()
            or manifest["versions"] != _single._versions()):
        raise ValueError("Training code or core package versions changed since tuning. "
                         "Restore the original setup or tune in a new run directory.")
    for record in (manifest, exported):
        if (record["model_name"], record["feature_option"]) != (model_name, feature_option):
            raise ValueError("Settings belong to a different model or feature option.")
        if (record.get("objective_names") != list(OBJECTIVE_NAMES)
                or record.get("directions") != list(DIRECTIONS)):
            raise ValueError("Expected this three-objective study and objective order.")
        if record["split_sha256"] != _single._split_hashes(split_dir):
            raise ValueError("train.csv or val.csv has changed since tuning. Use the original split.")
        if record["preprocessor_sha256"] != _single._digest(root / "preprocessor.json"):
            raise ValueError("The saved preprocessor no longer matches this study.")


def load_pareto(run_dir, *, model_name, feature_option, split_dir):
    """Load Pareto settings for refitting, with the same provenance checks as tuning.py."""
    root = Path(run_dir)
    exported = json.loads((root / "pareto_front.json").read_text(encoding="utf-8"))
    context = dict(model_name=model_name, feature_option=feature_option, split_dir=split_dir)
    _validate_export(root, exported, **context)
    records = exported["trials"]
    for record in records:
        _validate_export(root, record, **context)
    return records


def load_best(run_dir, *, model_name, feature_option, split_dir, trial_number=None):
    """Load the lowest-NLL representative, or explicitly choose a Pareto trial.

    Returns the original best-settings fields (model_kwargs, fit_kwargs, etc.).
    This selects settings for a fresh fit; it does not load trained weights.
    Use this loader for multi-objective runs, not tuning.load_best.
    """
    front = load_pareto(run_dir, model_name=model_name, feature_option=feature_option,
                        split_dir=split_dir)
    if trial_number is not None:
        for record in front:
            if record["best_trial"] == trial_number:
                return dict(record, selection_rule="Explicitly selected Pareto trial.")
        raise ValueError(f"Trial {trial_number} is not in the saved Pareto front.")
    if not front:
        raise ValueError("No completed Pareto trial is available.")
    best = json.loads((Path(run_dir) / "best.json").read_text(encoding="utf-8"))
    representative = min(front, key=lambda r: (r["validation_scores"]["nll"], r["best_trial"]))
    if best != dict(representative, selection_rule=SELECTION_RULE):
        raise ValueError("best.json does not match the Pareto summary. Rerun run_study to refresh exports.")
    return best


def run_study(split_dir, run_dir, *, model_name, feature_option="at-bb-aod",
              n_trials=25, epochs=100, patience=10, batch_size=64, seed=88,
              interval_level=0.90, use_cosine_similarity=True, device="cpu"):
    """Run/resume a single-worker study; n_trials is the desired completed total.

    TPE learns from all saved completed trials and uses a deterministic seed
    per trial. An uninterrupted run and a resumed run use the same seeding
    policy; this differs from keeping one long-lived sampler RNG. Failed or
    interrupted trials stay in history and do not count toward n_trials.
    No test file is loaded, and training metrics never determine Pareto rank.
    """
    import optuna

    if n_trials < 1 or batch_size < 1 or epochs < 1 or patience < 1 or not 0 < interval_level < 1:
        raise ValueError("Positive trial/training counts and 0 < interval_level < 1 are required")
    if model_name not in MODELS:
        raise ValueError(f"model_name must be one of {MODELS}")
    root = Path(run_dir).resolve()
    tr, vl = load_split(split_dir, "train"), load_split(split_dir, "val")
    validate_disjoint_stations(tr, vl)
    if "pm25" not in vl:
        raise ValueError("Validation requires observed pm25, not pm25_filled.")
    observed = vl.loc[np.isfinite(vl["pm25"]), "pm25"]
    if len(observed) < 2 or observed.nunique() < 2:
        raise ValueError("R2 requires at least two nonconstant observed validation targets.")
    fixed_fit = dict(epochs=epochs, patience=patience, batch_size=batch_size, seed=seed,
                     device=device, optimizer_name="Adam", weight_decay=0.0,
                     use_cosine_similarity=bool(use_cosine_similarity and model_name.startswith("TwoStage")))
    manifest = dict(
        model_name=model_name, feature_option=feature_option,
        split_sha256=_single._split_hashes(split_dir), fixed_fit=fixed_fit,
        interval_level=interval_level, objective="validation R2, Student-t NLL, mean Winkler",
        objective_names=list(OBJECTIVE_NAMES), directions=list(DIRECTIONS),
        sampler=dict(SAMPLER_SETTINGS), versions=_single._versions(),
        optuna_version=version("optuna"), code_sha256=_code_hashes())
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    scaler_path = root / "preprocessor.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.pop("preprocessor_sha256") != _single._digest(scaler_path) or saved != manifest:
            raise ValueError("Incompatible study or scaler. Use the original setup or a new directory.")
        prep = Preprocessor.load(scaler_path)
    else:
        if any(root.iterdir()):
            raise ValueError("No manifest in this nonempty directory. Choose an empty directory.")
        prep = Preprocessor.fit(tr, model_name, feature_option)
        tmp = scaler_path.with_suffix(".tmp")
        prep.save(tmp)
        tmp.replace(scaler_path)
        _single._save_json(manifest_path, dict(
            manifest, preprocessor_sha256=_single._digest(scaler_path)))
    train, val = prep.transform(tr, split="train"), prep.transform(vl, split="val")
    history_path = root / "study.sqlite3"
    db = "sqlite:///" + history_path.as_posix()
    name = f"{model_name}/{feature_option}/multi-objective"
    if history_path.exists():
        study = optuna.load_study(study_name=name, storage=db, sampler=_make_sampler(seed, 0))
        if ([d.name.lower() for d in study.directions] != list(DIRECTIONS)
                or study.user_attrs.get("manifest_sha256") != _single._digest(manifest_path)):
            raise ValueError("SQLite study does not match this manifest or objective directions.")
    else:
        if any((root / file).exists() for file in ("trials.csv", "best.json",
                                                  "pareto_front.json", "pareto_front.csv")):
            raise FileNotFoundError("Saved summaries exist, but study.sqlite3 is missing. "
                                    "Restore the database to resume in this directory.")
        study = optuna.create_study(study_name=name, storage=db,
                                    directions=list(DIRECTIONS), sampler=_make_sampler(seed, 0))
        study.set_user_attr("manifest_sha256", _single._digest(manifest_path))
        study.set_user_attr("objective_names", list(OBJECTIVE_NAMES))
    objective = make_objective(model_name, prep, train, val, fixed_fit, interval_level)
    trials = study.get_trials(deepcopy=False)
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    if completed > n_trials:
        raise ValueError(f"{completed} trials already completed; requested only {n_trials}.")
    if trials:
        _save_history(study, root, manifest)
    for _ in range(n_trials - completed):
        next_number = len(study.get_trials(deepcopy=False))
        study = optuna.load_study(study_name=name, storage=db,
                                  sampler=_make_sampler(seed, next_number))
        try:
            study.optimize(objective, n_trials=1, n_jobs=1, show_progress_bar=False)
        finally:
            _save_history(study, root, manifest)
    return study


if __name__ == "__main__":
    from .verification import check_model_contract
    check_model_contract("TwoStageDB")
