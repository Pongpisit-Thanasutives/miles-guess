# Executable CPU tutorials

Install the package with the `torch,notebooks` extras using the Python 3.11
instructions in the main README. Start Jupyter from this repository. Each
notebook is self-contained and can be run from its own directory or the
repository root. Saved outputs show one seeded run, not a benchmark guarantee.

| Notebook | What it demonstrates |
| --- | --- |
| `regression_example.ipynb` | NIG regression; untouched test MAE, RMSE, R², Student-t NLL, predictive intervals, coverage, PIT, and aleatoric/epistemic variances. |
| `classifier_example.ipynb` | Evidential classification; untouched test accuracy, log loss, multiclass Brier score, ECE, confusion matrix, reliability, and uncertainty diagnostics. |
| `regression_example_torch.ipynb`, `classifier_example_torch.ipynb` | Identical copies of the corresponding CPU tutorials, retained for existing links. |
| `evidential_learning_your_data.ipynb` | Start here for your own CSV: explicit feature/target columns, regression or classification, random/group/time splits, train-only preprocessing, validation checkpoint selection, test results, saved models and preprocessing, and prediction after reload. Supports multiple regression targets. |
| `dna_example_torch.ipynb` | A runnable sequence-classification pipeline with held-out test diagnostics. Uses synthetic DNA-like sequences and base-frequency features; it is a software example, not a biological model or a reproduction of the original transformer experiment. |

The custom-data guide defaults to synthetic regression, so it runs before you
provide a CSV. Change only the configuration cell to begin with your data. Its
classification and real-CSV, multi-target branches have also been executed.
Group and chronological splitting are tested for disjoint membership, including
repeated timestamps. Group/time split fractions refer to groups/time blocks and
need not produce exactly those fractions of rows.

All tutorials keep test outcomes out of optimizer updates, preprocessing fits,
checkpoint selection, and hyperparameter selection. If you tune settings after
inspecting test performance, reserve a new final test set. Real repeated measures
and time series normally require group or temporal splitting chosen for the
scientific prediction task. Useful uncertainty must be assessed on held-out data;
an evidential architecture alone does not guarantee calibration or detection of
distribution shift.

Regression intervals are central **Student-t predictive intervals for future
observations**, not confidence intervals for the mean. Marginal variances do not
model cross-target covariance. Classification's DST uncertainty mass and its
aleatoric/epistemic variance components are distinct quantities.

Run every tutorial cell in order using Jupyter's Restart Kernel and Run All, or:

```bash
python examples/execute_notebooks.py notebooks/regression_example.ipynb notebooks/classifier_example.ipynb notebooks/evidential_learning_your_data.ipynb notebooks/dna_example_torch.ipynb
python examples/validate_custom_notebook.py
```

The execution helper uses IPython in process and saves rendered plots and cell
outputs; it is also useful on hosts that prohibit Jupyter kernel sockets.

## Historical research material

The original introductory and DNA notebooks are retained in `legacy/` so the
authors' original workflows remain available. They are historical reference
material and have not been made self-contained or rerun against private data.

`paper_figures_regression.ipynb`, `paper_figures_categorical.ipynb`,
`evidential_figure_1.ipynb`, `sounding-figure.ipynb`, `shap.ipynb`, and
`subset_reference_datasets.ipynb` depend on original datasets, saved models,
external services, or site-specific paths. They are research reproduction and
data-preparation notebooks, not the supported introductory tutorials. Genuine
uncertainty calculation corrections are documented in `SCIENTIFIC_CHANGES.md`;
these original experiments have **not** been rerun or their results replaced
with synthetic results. Use the current tutorials' evaluation workflow when
adapting them to accessible datasets.
