# Authors' examples with held-out evaluation

The original examples use the authors' datasets and model settings. They now
add explicit test-set metrics and visualisations, with compatibility and genuine
calculation errors corrected. The Keras and PyTorch examples are **different
original workflows**, not aliases. Their original snapshots remain in `legacy/`.

The explanations use the paper's mathematical notation and equation numbers.
For NIG regression, the predicted location is $\gamma_n$ [Eq. (24)], while $\mu_n$
is the uncertain latent mean. Aleatoric and epistemic variances are
$\mathbb{E}[\sigma_n^2]$ [Eq. (25)] and $\operatorname{Var}(\mu_n)$ [Eq. (26)]. In-notebook notation tables
map existing code names such as `mu`, `mean`, `aleatoric`, and `epistemic` to
these symbols. The Gaussian-model and Dirichlet-classification sections use
their own corresponding paper notation. Calculations and training results
are preserved; the reference snapshots in `legacy/` are unchanged.

| Notebook | Original data and configuration | Added evaluation |
| --- | --- | --- |
| `regression_example_torch.ipynb` | Bundled Cabauw CSV; `evidential_regression_torch.yml`; five 1057-unit layers; original year-based split, batch 128, Adam 0.001, one pass for deterministic and evidential models. | Physical-unit errors, predictive Student-t NLL/intervals/coverage/PIT, and aleatoric/epistemic plots. |
| `classifier_example_torch.ipynb` | Original mPING precipitation configuration and four 212-unit layers; 100 epochs with at most 10 batches of 1024. Full data path is retained; explicitly select the bundled authors' sample when the full dataset is unavailable. | Test accuracy/log loss/Brier/ECE, confusion matrices, reliability and uncertainty plots for standard and evidential classifiers. |
| `regression_example.ipynb` | Bundled Cabauw CSV; original surface-layer deterministic/Gaussian/evidential configurations, day-based splits, original 1/1/5-epoch demonstrations, two-member Gaussian ensemble, and MC-dropout settings. | Comparable held-out metrics and plots, valid Gaussian/Student-t diagnostics, and corrected ensemble/MC variance handling. |
| `classifier_example.ipynb` | Original bundled precipitation sample, original deterministic/evidential Keras configurations, quantile preprocessing, balanced training, class weights, and 1/5 epochs. The deterministic architecture retains four 6024-unit layers. | Held-out test metrics and plots for standard, dropout, and evidential predictions. |
| `dna_example_torch.ipynb` | Original `bert-base-uncased` demonstration, single sequence, 10-class and one-task heads. The original provides no train/test dataset. | Optional evaluation of a supplied trained checkpoint and real labeled test CSV; no replacement dataset or invented test scores. |
| `evidential_learning_your_data.ipynb` | Separate adaptation guide. Defaults to the authors' Cabauw sample and original PyTorch regression model/settings; accepts your regression/classification CSV and adapts input/output dimensions. | Split guidance, train-only preprocessing, held-out metrics/plots, saved preprocessing/models, and inference after reload. |

## Install and run

Follow the Python 3.11 / PyTorch CPU Conda setup in the main README. The weather
notebooks require the `data` extra; Keras notebooks also require `keras`. The
original balanced Keras classifier needs `imbalanced-learn`:

```bash
python -m pip install -c requirements/validated-py311.txt -e ".[torch,keras,data,notebooks]"
python -m pip install imbalanced-learn
python -m jupyter lab
```

Set `KERAS_BACKEND=torch` before importing Keras (bash/zsh:
`export KERAS_BACKEND=torch`; PowerShell: `$env:KERAS_BACKEND="torch"`).
The DNA notebook documents its separate optional `transformers` and pretrained
model/checkpoint requirements. Its guarded setup can be read without downloading
weights.

Start Jupyter from this unpacked repository or its `notebooks` directory. The
notebooks locate repository-relative CSV/config paths automatically. Use
Restart Kernel and Run All. An alternative execution method is:

```bash
python examples/execute_notebooks.py notebooks/regression_example_torch.ipynb
```

The PyTorch classifier's default YAML path refers to the authors' full mPING
file on their filesystem. To explicitly use their bundled real sample:

```bash
# bash/zsh; run from the repository root
MILES_PTYPE_DATA="$PWD/data/ptype_sample.parquet" python examples/execute_notebooks.py notebooks/classifier_example_torch.ipynb
```

```powershell
# PowerShell; run from the repository root
$env:MILES_PTYPE_DATA = (Resolve-Path data/ptype_sample.parquet).Path
python examples/execute_notebooks.py notebooks/classifier_example_torch.ipynb
```

The notebook records the selected path. There is no automatic substitution of
generated data. The bundled Keras classifier already uses this sample in the
original notebook. The large original Keras architecture requires several GB
of memory; it has not been replaced by a smaller network.

## Interpretation and scope

The PyTorch regression notebook additionally provides Figure 10-style
aleatoric, epistemic, and total Student-t CDF histograms. Only the total panel
is a predictive PIT for observed targets. The first two compare the observed
residual with component marginal distributions; they need not be uniform.
All panels use the complete test set and nine equal-width relative-frequency
bins. The notebook derives the scales, preserves the original training run,
and exports PNG/SVG figures plus per-row CDF values and histogram tables.

The regression notebooks also include a diurnal composite in the style of
Figure 11 of [Schreck et al. (2024)](https://doi.org/10.1175/AIES-D-23-0093.1).
All held-out dates are grouped into 10-minute clock-time bins. Blue triangles
show the mean observations, orange squares the mean predictions, green circles
aleatoric SD, and red pentagons epistemic SD. Following the authors' code, the
SD curves are `sqrt(mean(component_variance))` in each bin. They are neither
standard errors of the plotted means nor complete temporal-mixture SDs.
Empty bins remain gaps; CSV exports include the number of observations in every
bin. PNG and SVG exports accompany the figure. The custom-data guide uses this
view only for regression data with actual clock timestamps, and explains when
diurnal grouping is appropriate. `mlguess.diurnal.plot_diurnal_uncertainty`
can make one panel per available target without changing the fitted model.
The Keras example keeps the full linear plot and adds a labelled symlog
companion because its original short training run has extreme epistemic SD;
both views use identical values.

Temporal prediction and uncertainty plots use real timestamps with readable
date ticks. A display window is chosen by calendar time, independently of model
errors; metrics still cover the complete held-out set. Interval bars and
separate markers avoid joining unrelated observations across gaps or series.
For your data, supply the plotting-time column when available. Without time,
use a meaningful variable such as the predicted mean, rather than a row index
or rank. Training progress uses epochs/batches; calibration, confusion matrices,
and physical-variable diagnostics retain their meaningful axes.

Scalers fit training data only. New test-set summaries do not select epochs,
settings, thresholds, or models in the restored originals. The custom-data guide
separately illustrates validation-based selection. Report data version, split,
seed, training settings, and package revision when using these examples in a
paper. The short original demonstration budgets can yield imperfect predictive
skill and calibration; the displayed results are not tuned on the test set.

Regression intervals describe future observations, not confidence intervals for
the mean. Gaussian models use Gaussian predictive distributions; NIG models use
Student-t predictions. A finite ensemble is a mixture, not a single distribution
obtained by averaging its parameters. Classification DST uncertainty mass and
variance components are distinct quantities.

`AUTHORS_NOTEBOOK_UPDATES.md` records what was preserved and why individual
corrections were necessary. `VALIDATION.md` distinguishes actual author-data runs
from unit-test fixtures and unavailable external resources.

The paper-figure, sounding, SHAP, and dataset-preparation notebooks retain their
original research role. Full experiments requiring external weather data and
saved checkpoints have not been rerun or replaced with generated-data results.
