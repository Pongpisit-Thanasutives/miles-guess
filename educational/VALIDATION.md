# What was verified

## Standalone installer: 0.1.1 / main release py311.13 (2026-11-15)

A fresh standalone installation passed **43 educational tests** and `pip check`
on Linux CPU / Python 3.11.18 with the supplied pinned profile. `mlguess` was
not installed. The independent source archive installed successfully and
its TwoStageDB HPT/usage pair ran all 14 code cells, produced 4 inline and
3 saved figures, and passed training/test score and source-sum checks.
The check used one tuning trial and at most two epochs on the small schema
fixture, not research observations. The `pm25-split` command and an existing
py311.12 study also worked from the standalone installation.

All runtime and notebook files are unchanged from py311.12. The main-package
installer remains available; use one installation route per environment.
See [validation-standalone-installer.json](validation-standalone-installer.json)
for current evidence. Native macOS/Windows/GPU execution and external
Jupyter kernel transport were not tested.

## Loss-weight and score update: main release py311.12 (2026-11-15)

All **43 educational tests** and the combined **478-test suite** passed on
Python 3.11.18 / PyTorch 2.14.1 CPU / NumPy 2.4.6 / Optuna 5.0.0; the combined
suite skipped 6 optional-dependency, socket or GPU checks. All **12 notebooks**
ran with one tuning trial and at most two epochs per model on the small
schema fixture: 84 code cells, 24 inline figures and 18 saved test figures.
Training and test exports were checked separately against their correct
labels, including exact two-stage source sums after CSV round-trip parsing.

New tests verify exact loss-weight search ranges, optional/exclusive cosine
loss, filled-target training scores, per-trial history exports and compatible
study resumption. Actual model training was checked for 1 to 2 trials; the
25 to 50 persistence check uses a lightweight objective to verify unchanged
trial records and the scaler. No real-data skill or native Windows/macOS/GPU
execution is claimed. See [validation-hpt-update.json](validation-hpt-update.json)
for evidence and limits. The main bundle also includes a consolidated report in its root `VALIDATION.md`.

## Earlier verification

**Historical checks (2026-11-10): pm25-edl-edu 0.1.0 before integration.** These results describe the earlier educational algorithms; checks for subsequent releases are recorded above and in the local JSON reports. The tests use an explicitly constructed station-schema fixture, not PM2.5 measurements. They verify software behavior, not forecast skill, calibration, accuracy of source apportionment or reproduction of a published result.

| Check | Outcome |
|---|---|
| Environment | Linux x86-64 CPU; CPython 3.11.17, PyTorch 2.14.1+cpu, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, pandas 2.3.3, Matplotlib 3.11.2, Optuna 5.0.0. A fresh editable Linux install from the supplied profile passed pip check, all six model contracts and the station-split audit. All three platform profiles resolved for Python 3.11 with binary wheels; no native macOS or Windows execution was performed. |
| Automated tests | 35 tests passed. Checks cover finite masked losses and gradients, station ordering/gaps, train-only scaling, split isolation, causal outputs, absolute source accounting, Student-t probabilities/scores and the model/feature combinations. |
| Backbone fidelity | Corresponding shared backbones/head compared with matched weights from the uploaded deterministic reference. MLP, CNN1D and correction-head outputs/gradients matched; largest DB differences were 4.47×10⁻⁸ for outputs and 2.98×10⁻⁷ for input gradients (pointwise Linear vs 1×1 convolution). |
| Evidence head | With identical projection parameters, the single-stage head matched the MILES-GUESS LinearNormalGamma reference to 1.20×10⁻⁷. Extreme-logit checks kept ν, β positive and α > 1. The three evidence outputs for a two-stage model accompany the exact source-sum location. |
| Likelihood and masking | Training and physical-unit prediction NLL were checked against independent Student-t implementations. Imputed-label distractors, missing observed labels, all-missing station targets and unequal-length segments were exercised. Undefined R² is left undefined. |
| Notebook execution | All six HPT and six usage notebooks ran end-to-end: 84 code cells, 24 inline figures, including 18 nonblank saved test diagnostic PNGs. One trial and two epochs per model for this smoke test. The researchers' default trial/epoch budgets were neither changed nor run for this test. |
| Rows and source sums | For every model all 21 fixture test rows were exported, and only the 13 with observed targets were scored. Each two-stage CSV passed exact source accounting on a round-trip decimal parse; max absolute difference **0**. |
| Persistence | A local Optuna study resumed without modifying its scaler. A changed manifest was rejected. Reloading the frozen TwoStageDB fit without retraining gave byte-identical weights, test predictions and metrics. |
| Limits | Each notebook executed in a fresh, separate IPython in-process kernel because this host disallows Jupyter socket binding. Browser/kernel transport, interactive GUI backends and GPU execution were not tested. No real station CSVs or training imputer accompanied the four references. |

## Re-run on your machine

After either installation route in the README, run these commands from the directory containing this file:

```shell
python -m pip install pytest
python -m pytest -q tests
python -m pm25_edl.verification
python -m pm25_edl.split_cli --self-test
python tests/run_notebooks.py
```

The notebook runner needs the standalone `optuna,notebooks` extras or the main `educational` extra. Use `--work /path/to/empty/qa_directory` to retain its mock splits, executed notebooks and run artifacts. The runner refuses to use a nonempty output directory. For hosts without Jupyter socket access add `--in-process`. Every research notebook's last cell calls a schema-shaped contract check. The ordinary notebooks do not read the QA dataset unless you explicitly point them at it.

[validation.json](validation.json) provides compact machine-readable evidence; there are no fixture weights, sample predictions or artificial performance scores packaged as research outputs.
