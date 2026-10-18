# Restoring the authors' examples

Revision `2025.1.0+py311.8` aligns notebook mathematical notation with the
paper. Regression explanations use gamma_n for the predicted location, mu_n
for the uncertain latent mean, E[sigma_n²] for aleatoric variance [Eq. (25)],
and Var(mu_n) for epistemic variance [Eq. (26)]. In-notebook keys map existing
code/API names to those symbols. Figure 10 Student-t scales and Figure 11
clock-bin averages use the same notation and explicit equation references.
Classification explanations distinguish latent class probabilities, predicted
Dirichlet means, component variances, and DST mass. Gaussian ensemble means
retain their Gaussian notation. Historical snapshots remain unchanged.

Revision `2025.1.0+py311.7` adds a three-panel Figure 10-style diagnostic to
`regression_example_torch.ipynb`: aleatoric reference, epistemic reference,
and total predictive PIT. All use the NIG Student-t marginals. The first two
evaluate observed residuals against component reference distributions; only
total has uniformity as its predictive-calibration target. Nine equal-width
bins show relative frequency. PNG, SVG, per-observation CDF values with source
timestamps, and bin counts are exported to `outputs/authors_regression_torch`.
All pre-existing notebook cell sources and the authors' dataset/model/training
choices are retained. The figure presents the existing friction-velocity model.

Revision `2025.1.0+py311.6` adds a Figure 11-style diurnal composite to both
regression examples and the custom-data guide. Blue triangles show observed
means, orange squares predicted means, green circles aleatoric SD, and red
pentagons epistemic SD. All held-out dates contribute to 10-minute clock-time
bins; empty bins break the curves. The uncertainty curves use
`sqrt(mean(component_variance))`, matching the authors' source implementation.
Physical units, the source clock, and per-bin counts are retained. The bundled
examples have one target, friction velocity; the helper supports additional
targets when genuine predictions are available. Model settings and budgets
remain those of the original examples, so these are new figures of those runs,
not reproduced scores from the paper's tuned experiments.

The subsequent `2025.1.0+py311.5` update replaces ordinal prediction axes with
actual timestamps. Cabauw interval panels display a stated calendar window,
using separate interval bars and observation/mean markers so missing dates do
not imply continuous measurements. Full-test metrics still use every test row.
The custom-data and DNA guides accept an explicit plotting-time column; without
timestamps their interval panels use actual predicted means on the x-axis.
Epochs, calibration probabilities, physical variables, and class labels retain
their natural axes. Research plotting helpers also stop treating batches as
time or spacing timestamp strings as equally spaced categories.

Revision `2025.1.0+py311.4` corrects the scope of the preceding notebook update.
The authors' original datasets, architectures and notebook training settings
are restored. New code focuses on held-out evaluation, clear visualisations,
portable execution, and genuine calculation/API corrections. The previous
replacement experiments and their generated datasets are not the examples in
this revision.

Base: ai2es/miles-guess commit
`69775bef5d65b50e53692d860092e0295b4e7d23`. Original notebook snapshots are retained
in `notebooks/legacy/`. `validation/authors-data-provenance.json` records exact
file hashes, original settings, and dataset/split details.

## Preserved data and settings

Both bundled data files are byte-identical to upstream. All model dictionaries
in the original YAML files are unchanged. Notebook-specific overrides are also
preserved; these sometimes differ from production settings in the YAML.

| Workflow | Data / split | Preserved training choices |
| --- | --- | --- |
| PyTorch regression | Authors' 72,197-row Cabauw sample; original year grouping and both split RNG seeds 1000; split index 0. Train/validation/test: 55,007 / 6,598 / 10,592. | Five 1057-unit layers, dropout 0.263, batch 128, Adam 0.001, one pass each for deterministic L1 and evidential Lipschitz-MSE models. Original regularizer coefficient 5.806527626063491. Robust inputs and MinMax targets. |
| PyTorch classification | Original full mPING path remains the default. For recorded execution, the bundled authors' precipitation sample is explicitly selected. Original quality filters, case-study days and cutoff are retained; 7,225 / 908 / 4,031 rows. | Four 212-unit layers, dropout 0.1167, batch 1024, Adam 0.001, 100 epochs with at most 10 batches, evidential annealing step 10. |
| Keras regression | Authors' Cabauw sample and original day groups; 57,431 / 7,578 / 7,188 rows. | Original surface-layer YAML architectures and losses; notebook overrides of 1 deterministic, 1 Gaussian, and 5 evidential epochs; two Gaussian ensemble members; original MC counts. |
| Keras classification | Authors' precipitation sample, original quality/split rules, shared quantile preprocessing from the original notebook. | Original four-layer, 6024-unit deterministic model, balanced-class sampling and class weights, one epoch; original evidential architecture/settings, five epochs. |
| DNA | Original literal sequence `AGCTAGCTAGCT` and `bert-base-uncased`; no original train/test dataset exists. | Original CLS pooling, 10-class head, one-task NIG head, example target 1.45456457, variance metadata, and loss settings. |

The full mPING dataset is not silently replaced. The PyTorch classifier records
an explicit `MILES_PTYPE_DATA` override when the bundled authors' real sample is
used. That run is a sample-data evaluation, not a reproduction on the full
external dataset. The Keras original already uses the bundled sample.

## Evaluation extensions and required corrections

- Evaluate the held-out **test** split and label it clearly. Fit preprocessing
  only on training observations. Preserve the original training budgets without
  tuning against the test results.
- Restore physical target units before comparing means with observations.
  Transform variances by the squared inverse affine scale; do not confuse
  standard deviation with variance.
- Report point errors and probabilistic diagnostics appropriate to each model:
  Gaussian predictions for Gaussian heads, Student-t predictions for NIG heads,
  and actual mixture densities/quantiles for ensembles. Deterministic dropout
  spread alone does not identify observation noise.
- Use each ensemble member's own fitted input and target scaler. This corrects
  the original undefined-path/reload/scaler problems without changing its
  architectures, data splits or number of members.
- Train the standard Torch classifier with logits for `CrossEntropyLoss`.
  Its original notebook applied softmax first, giving the loss the wrong input.
  Apply softmax when computing probability-based test metrics.
- Pass the evidential loss's device argument by keyword. The original positional
  call can place the device object in the class-weight argument.
- Adapt stale Keras imports/constructor metadata to the maintained APIs. Use the
  legacy deterministic classifier where needed to retain the original balanced
  generator and weights; retain the current evidential classifier to avoid
  changing its regularization and annealing convention.
- Restore the original BERT demonstration rather than the replacement
  base-frequency model. Correct its inference-state handling, NIG numerical
  stability, parameter unpacking and target scaling. Its optional evaluation
  requires a trained checkpoint and real labeled test data; neither is invented.

All library-level scientific corrections from earlier revisions remain.
`SCIENTIFIC_CHANGES.md` describes those corrections and their analytical tests.
Large uncertainty, poor scores or calibration errors from the original short
training demonstrations are displayed honestly. Settings were not changed to
make the new plots look better.

## Separate guide for your datasets

`evidential_learning_your_data.ipynb` remains a separate adaptation guide. Its
default now uses the authors' Cabauw sample, original year split, preprocessing,
PyTorch regression architecture, loss coefficient, Adam rate and one-pass budget.
For your CSV it adapts input/output dimensions while retaining the starting
architecture. Classification starts from the authors' Torch classifier
configuration and requires your labeled CSV. The guide explains validation-only
selection, group/time splits, test diagnostics, saving/reloading and new-row
prediction. Small generated fixtures are used only in automated tests of CSV
branches; they are not notebook example data or reported authors' results.

## Validation scope

Actual Cabauw and bundled precipitation runs, plot reviews, notebook source
comparisons, and numerical tests are recorded in `VALIDATION.md` and
`validation/authors-*`. DNA's unavailable pretrained weights, trained checkpoints
and real test dataset are explicitly distinguished from mocked interface tests.
The external paper-figure/SHAP/sounding experiments retain their original data
requirements and are not claimed to have been rerun.
