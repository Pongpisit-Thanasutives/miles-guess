# File-by-file summary

Start with the [installation and workflow](README.md). Choose a model’s HPT notebook first, then its usage notebook.

| File | Purpose or key change |
|---|---|
| [src/pm25_edl/nn_models.py](src/pm25_edl/nn_models.py) | Replaces deterministic scalar outputs with NIG heads; preserves six backbones and exact two-stage physical source accounting. |
| [src/pm25_edl/nn_data.py](src/pm25_edl/nn_data.py) | Station splits, train-only preprocessing, contiguous daily segments, length batches, original row positions and portable scaler JSON. |
| [src/pm25_edl/nn_training.py](src/pm25_edl/nn_training.py) | Shared masked EDL training, optional two-stage cosine penalty, validation-NLL early stopping, train/val role checks and physical-unit exports. |
| [src/pm25_edl/tuning.py](src/pm25_edl/tuning.py) | Local Optuna search including evidence/source-penalty weights, train/validation scores, best settings and resumable SQLite with provenance checks. |
| [src/pm25_edl/metrics.py](src/pm25_edl/metrics.py) | Physical Student-t NLL, interval score, PIT, point scores, prediction intervals and discard curve, with separate training/observed-target scoring APIs. |
| [src/pm25_edl/plots.py](src/pm25_edl/plots.py) | PIT, discard curve and station/date timeline; the two SD components have separate panels. |
| [src/pm25_edl/split_cli.py](src/pm25_edl/split_cli.py) | Stage 1 CLI for raw splitting with a train-only imputer, or read-only auditing of existing splits. |
| [src/pm25_edl/verification.py](src/pm25_edl/verification.py) | Small explicit 21/20-column fixtures and end-of-notebook contracts for all six models. |
| [src/pm25_edl/__init__.py](src/pm25_edl/__init__.py) | Stable educational import namespace and shared component version for both installers. |
| [notebooks/hpt/mlp_hpt.ipynb](notebooks/hpt/mlp_hpt.ipynb) | Tune the MLP using saved training/validation stations; record all five training and validation metrics per completed trial. |
| [notebooks/usage/mlp_usage.ipynb](notebooks/usage/mlp_usage.ipynb) | Fit/reload the selected MLP; export separate training/test scores and predictions plus test uncertainty figures. |
| [notebooks/hpt/cnn1d_hpt.ipynb](notebooks/hpt/cnn1d_hpt.ipynb) | Tune the causal CNN1D using saved training/validation stations; record all five training and validation metrics per completed trial. |
| [notebooks/usage/cnn1d_usage.ipynb](notebooks/usage/cnn1d_usage.ipynb) | Fit/reload the selected causal CNN1D; export separate training/test scores and predictions plus test uncertainty figures. |
| [notebooks/hpt/db_hpt.ipynb](notebooks/hpt/db_hpt.ipynb) | Tune the dual-branch model using saved training/validation stations; record all five training and validation metrics per completed trial. |
| [notebooks/usage/db_usage.ipynb](notebooks/usage/db_usage.ipynb) | Fit/reload the selected dual-branch model; export separate training/test scores and predictions plus test uncertainty figures. |
| [notebooks/hpt/two_stage_mlp_hpt.ipynb](notebooks/hpt/two_stage_mlp_hpt.ipynb) | Tune the two-stage MLP using saved training/validation stations; record all five training and validation metrics per completed trial. |
| [notebooks/usage/two_stage_mlp_usage.ipynb](notebooks/usage/two_stage_mlp_usage.ipynb) | Fit/reload the selected two-stage MLP; export separate training/test scores and predictions plus test uncertainty figures and check all three corrected sources. |
| [notebooks/hpt/two_stage_cnn1d_hpt.ipynb](notebooks/hpt/two_stage_cnn1d_hpt.ipynb) | Tune the two-stage CNN1D using saved training/validation stations; record all five training and validation metrics per completed trial. |
| [notebooks/usage/two_stage_cnn1d_usage.ipynb](notebooks/usage/two_stage_cnn1d_usage.ipynb) | Fit/reload the selected two-stage CNN1D; export separate training/test scores and predictions plus test uncertainty figures and check all three corrected sources. |
| [notebooks/hpt/two_stage_db_hpt.ipynb](notebooks/hpt/two_stage_db_hpt.ipynb) | Tune the two-stage dual-branch model using saved training/validation stations; record all five training and validation metrics per completed trial. |
| [notebooks/usage/two_stage_db_usage.ipynb](notebooks/usage/two_stage_db_usage.ipynb) | Fit/reload the selected two-stage dual-branch model; export separate training/test scores and predictions plus test uncertainty figures and check all three corrected sources. |
| [tests/test_scientific_contracts.py](tests/test_scientific_contracts.py) | Independent numerical and integration checks; includes split/target leakage, causal segment behavior, scoring, HPT persistence and source accounting. |
| [tests/schema_fixture.py](tests/schema_fixture.py) | Small explicit mock split files used only for tests. |
| [tests/run_notebooks.py](tests/run_notebooks.py) | Run all 12 notebooks end-to-end in isolated QA data paths with bounded budgets. |
| [pyproject.toml](pyproject.toml) | Independent pm25-edl-edu installer; installs pm25_edl and pm25-split with optional optuna, notebooks and test extras. |
| [MANIFEST.in](MANIFEST.in) | Includes educational source, notebooks, guides, tests and local CPU profiles in the source archive. |
| [LICENSE](LICENSE) | Copy of the parent project's license included with the independent distribution. |
| Main project's `pyproject.toml` (optional alternative) | Combined miles-guess installation still supplies both mlguess and pm25_edl through its educational extra. |
| [requirements/optuna-cpu/macos-py311.txt](requirements/optuna-cpu/macos-py311.txt) | Local pinned CPU notebook profile for native macOS 14+ Apple Silicon. |
| [requirements/optuna-cpu/windows-py311.txt](requirements/optuna-cpu/windows-py311.txt) | Local pinned CPU notebook profile for Windows x86-64. |
| [requirements/optuna-cpu/linux-py311.txt](requirements/optuna-cpu/linux-py311.txt) | Local pinned CPU notebook profile for Linux x86-64. |
| [README.md](README.md) | Install, audit data, select features and run each stage. |
| [METHODS.md](METHODS.md) | Mathematics, departures from the deterministic reference, plotting interpretation, publication caveats and citations. |
| [VALIDATION.md](VALIDATION.md) | Current and historical verification, main-release report link and commands to repeat the tests. |
| [validation-standalone-installer.json](validation-standalone-installer.json) | Independent editable/source/wheel installation and notebook smoke checks. |
| [validation-hpt-update.json](validation-hpt-update.json) | Current loss-control, search-space, training-score, resume and notebook test evidence. |
| [validation.json](validation.json) | Machine-readable notebook/fixture checks and SHA-256 fingerprints of the four supplied reference files. |
| [FILES.md](FILES.md) | This file-by-file summary. |
