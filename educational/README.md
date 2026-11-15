# PM2.5 evidential regression: educational edition

Train six PyTorch models on **your daily PM2.5 station data**, tune on unseen validation stations, and evaluate a frozen model on unseen test stations. This `educational/` directory adapts the supplied deterministic backbones and evidential output pattern. Install it independently as `pm25-edl-edu` to use the `pm25_edl` API, or install it with the main MILES-GUESS package. Both routes use the same source files. No station observations or fitted research models were supplied. The notebooks read your CSVs. Their last cells use tiny fixtures solely to verify the implementation.

**Start here:** [file-by-file guide](FILES.md) · [model, scores and scientific interpretation](METHODS.md) · [test evidence and limits](VALIDATION.md).

## Install on Python 3.11 (CPU)

Extract the MILES-GUESS archive and open a Conda terminal inside **`educational/`**, containing this README, `pyproject.toml` and `src/pm25_edl/`. The educational source archive can also be extracted and used on its own. Create a fresh environment:

```shell
conda create --no-default-packages --override-channels -c conda-forge -n pm25-edl python=3.11 pip -y
conda activate pm25-edl
```

Run **one** command for your platform. The pinned versions include PyTorch 2.14.1, NumPy 2.4.6, Optuna 5.0.0, Matplotlib 3.11.2 and JupyterLab 4.6.6. A CPU build of PyTorch is explicitly selected on Windows and Linux; macOS uses its native wheel.

| Platform | Editable install |
|---|---|
| macOS 14+ (native Apple Silicon, not Intel/Rosetta) | `python -m pip install -r requirements/optuna-cpu/macos-py311.txt -e ".[optuna,notebooks]"` |
| Windows x86-64 | `python -m pip install -r requirements/optuna-cpu/windows-py311.txt -e ".[optuna,notebooks]"` |
| Linux x86-64, compatible glibc | `python -m pip install -r requirements/optuna-cpu/linux-py311.txt -e ".[optuna,notebooks]"` |

Run `python -m pip check`, then `python -m jupyter lab` from this directory. Open the notebooks under `notebooks/hpt/` and `notebooks/usage/`. Select this environment's kernel; in a notebook, `import sys; print(sys.executable)` shows which interpreter you are using. No ECHO, Weights & Biases or remote account is used.

For an existing compatible CPU PyTorch environment, run `python -m pip install -e ".[optuna]"` here for the educational library and Optuna, or use `.[optuna,notebooks]` to include Jupyter tools. Plain `python -m pip install -e .` installs the educational runtime without Optuna or notebook extras. These commands install only the `pm25_edl` implementation; they do not install `mlguess`. The pinned profiles above include the full CPU notebook stack.

The combined installation remains available from the **main project directory** with `python -m pip install -e ".[educational]"`. It provides both `mlguess` and `pm25_edl`; use the main README's platform commands for that route.

## Choose or switch installation routes

Use one distribution per environment because `miles-guess` and `pm25-edl-edu`
both provide `pm25_edl` and the `pm25-split` command. A fresh environment needs
no uninstallation. To switch an existing environment from the main package
to educational-only, run `python -m pip uninstall miles-guess` before the
educational install above. To switch back, run `python -m pip uninstall pm25-edl-edu` before installing
the main package. An existing standalone
installation can be updated directly with this directory's installer.

Copy your old `data/` and `runs/` into this directory, or keep them where they
are and set `PM25_DATA_DIR` and `PM25_RUN_DIR` to their absolute paths.
Models use the same `pm25_edl` imports; existing study provenance checks still
apply. This installer-only change preserves the scientific code and study
fingerprints from py311.12; earlier fixed-weight studies still need a new
run directory. The notebooks locate this directory whether launched here,
from the main project or from a notebook folder.

## Stage 1 — prepare or reuse saved data

**If you already have your three files**, put `train.csv`, `val.csv`, and `test.csv` in `data/splits/` relative to this directory, or set `PM25_DATA_DIR` to their directory. Do not repartition or re-impute trusted pre-split CSVs. Audit them once:

```shell
pm25-split audit data/splits
```

The audit prints row and station counts, observed-target counts, dates, file hashes and the raw CMAQ source-sum discrepancy (among comparable rows). It flags differences outside absolute tolerance 0.0001 plus relative tolerance 0.00001 without changing inputs. Check mismatches before comparing feature options. The splits must have disjoint station IDs. This checks numeric values and station membership; it cannot certify that an externally imputed or preprocessed file was created without leakage.

**If starting from raw observations**, split stations by region first and save exactly the three files. Validation size is a fraction of stations remaining after reserving test stations. Single-station regions remain in training in stratified draws. Use `--test-region NAME` for a geographical test holdout. To reproduce your original imputation, pass your importable function; it receives only the training frame:

```shell
pm25-split create raw_station_days.csv data/splits --seed 51 --test-size 0.2 --val-size 0.2 --train-imputer fill_missing_val:fill_missing_values
```

The referenced imputer was not attached; no replacement method is assumed. Omit the argument if training observations are complete. When missing training labels cannot be filled, the operation stops. Validation/test labels are never filled. Use a new output directory rather than overwriting a saved split.

| Input group | Columns |
|---|---|
| Identifiers | `date` (one calendar day), `stcode` (string; leading zeros retained), `region5` |
| CMAQ / AOD | `cmaq_pm25`, `cmaq_pm25_bb`, `cmaq_pm25_anthro`, `cmaq_pm25_othr`, `aod` |
| WRF (always used) | `wrf_pbl`, `wrf_temp`, `wrf_wspd`, `wrf_rh` |
| Static (always used) | `population`, `elevation`, `builtup`, `rice`, `forest`, `minor_road_length`, `major_road_length` |
| Training only | `pm25_filled` is the label; `pm25_fill_n_neighbors` is ignored if present |
| Validation / test only | `pm25` contains observations or NaN. Predictions are saved for every input row; scores use only finite observed values. |

Inputs used by a model must be finite; duplicate station-day keys are errors. Train target statistics use the actual supplied filled training labels. Feature/target scalers are fitted on training data only. To prevent leakage, verify that upstream predictor construction (including “static” variables and any imputed predictors) used only information available at prediction time.

## Stages 2 and 3 — tune, then evaluate

1. **Choose the model and feature option in advance.** Start with [CNN1D tuning](notebooks/hpt/cnn1d_hpt.ipynb). The first settings cell defaults to `at-bb-aod`, matching your reference. Use the same settings in its [usage notebook](notebooks/usage/cnn1d_usage.ipynb). [FILES.md](FILES.md) links all six pairs. You may also set `PM25_FEATURE_OPTION`, `PM25_DATA_DIR`, and `PM25_RUN_DIR` (default `runs/` relative to this directory) before starting Jupyter.
2. **Run the tuning notebook.** It uses `train.csv` and `val.csv` only. It saves an Optuna SQLite study, a CSV with all five training and validation scores for each completed trial, the selected settings and scaler under `runs/<ModelName>/<feature-option>/` relative to this directory. The supplied notebook actually used a **single saved spatial holdout**, not five folds. This edition preserves that design. The default is 25 random trials, up to 100 epochs, patience 10, seed 88. The search also samples `evidential_weight` logarithmically from 0.0001 to 0.1. Only two-stage models with `USE_COSINE_SIMILARITY=True` sample `lambda_sim` from 0.02 to 0.05 in steps of 0.01; set the flag False to disable that penalty. It is always off for single-stage models. For a dry run reduce `PM25_N_TRIALS` and `PM25_MAX_EPOCHS`; use a separate run directory from your research study.
3. **Run the matching usage notebook in a fresh kernel.** It loads the saved choice (no Optuna search), fits with training data and validation early stopping, saves weights/preprocessing, reloads the frozen model, and then opens `test.csv`. Set `REFIT=False` to reload an existing fit. Training predictions and scores against `pm25_filled` are saved as `train_predictions.csv` and `train_scores.csv` in the run's `evaluation/` directory, alongside the observed-only test predictions, metrics and three diagnostic figures. Training scores are calculated at the restored model in evaluation mode. Test scores and plots are for reporting: do not use them to choose the architecture, feature set or tuning budget.

**Continue a saved study:** keep the same data, code and run directory, increase `N_TRIALS` (or `PM25_N_TRIALS`) from 25 to 50, and rerun the HPT notebook. The existing `study.sqlite3` is reused; only the missing 25 completed trials are added. `trials.csv` is a score export, not a substitute for the database. Older studies from before this changed search space require a new run directory and remain intact.

| CMAQ feature option | Single stage | Two stage |
|---|---|---|
| `baseline` | Total CMAQ only, plus WRF/static | Not defined for corrections to *individual* source priors |
| `at-bb` | Three individual CMAQ sources, plus WRF/static | Same; each head corrects its own source |
| `at-bb-aod` | `at-bb` + AOD | Same |

Keep the same station split when comparing architectures or feature sets. For instance, compare all six on `at-bb` or all six on `at-bb-aod`; use the three single-stage `baseline` runs as true total-only references. A fair comparison among feature options may also require restricting all to identical dates/stations with available predictors.

## Verify the installation

For the educational-only installation, run from this directory:

```shell
python -m pm25_edl.verification
python -m pip install -e ".[optuna,test]"
python -m pytest -q tests
```

For an end-to-end smoke run of every notebook with explicitly generated schema fixtures and a small training budget, see [VALIDATION.md](VALIDATION.md). Research notebooks do not silently create or substitute data.
