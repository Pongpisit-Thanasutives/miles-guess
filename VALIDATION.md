# Validation report

Date: **2026-10-18**. Local revision: **2025.1.0+py311.8**.
Upstream base: `69775bef5d65b50e53692d860092e0295b4e7d23`.

This revision restores the authors' examples and extends their held-out
assessment. It does not replace their datasets or architectures with generated
examples. `AUTHORS_NOTEBOOK_UPDATES.md` documents preservation and the specific
compatibility/calculation errors corrected.

## Paper-notation update (py311.8)

Eleven active notebooks now use the paper's mathematical symbols, equation
references, and code-to-symbol keys. All notebook documents pass format
validation, and 250 mathematical expressions in changed Markdown pass delimiter
and brace checks. All 83 code cells in the six main tutorials, including their
saved outputs and execution metadata, are identical to py311.7. Runtime library
Python files, datasets, configurations, and legacy snapshots are byte-identical.
The two notebook copies used by the documentation match their updated sources.

The only research code edits replace SHAP output-label strings in two cells.
Those labels now describe the predicted location and component variances.
All four expressions were parsed and rendered with Matplotlib. Ten stored SHAP
figures carrying obsolete labels were cleared, along with their execution
counts; no new attribution results are claimed. Other saved research outputs
are preserved. The external Captum workflow and its checkpoints were not run.

**59 existing notebook tests passed.** Two test lookup helpers now ignore only
the tagged Markdown notation guides when resolving historical cell positions;
the scientific assertions and fixtures are unchanged. No training was rerun
for these exposition edits. Detailed source/output comparisons and validation
records are in `validation/notation-*`; the numerical runtime and platform
validation from prior revisions retains its documented scope.

## Figure 10-style Student-t update (py311.7)

The PyTorch regression notebook adds three Student-t CDF panels: aleatoric
reference, epistemic reference, and total predictive PIT. Independent numerical
integration over the Normal/inverse-gamma hierarchy and closed-form Student-t
values validate the component calculations. Tests also cover physical-unit
affine transformations, total-PIT preservation, a vanishing epistemic component,
and representable scales for very small positive variances. A deterministic
predictive-quantile fixture shows that uniform total PIT does not imply uniform
component-reference CDF histograms.

**63 focused tests passed**, including nine new component-CDF cases. The full
PyTorch regression notebook executed **25 code cells and 5 figures**, retaining
the original Cabauw data and one-pass training settings. All pre-existing
notebook cell sources, complete-test metrics, and saved per-observation means,
variances, and intervals match the previous delivery. The new figure uses all
10,592 held-out observations, nine equal-width relative-frequency bins, and a
shared vertical axis. The rendered output was inspected for labels, scale,
legend placement, and clipping. PNG/SVG figures, timestamped CDF values, bin
counts, and execution/test records are under `validation/figure10-*`.

Only the total panel is a predictive PIT for observed targets. The component
panels are explicitly labelled reference diagnostics. The published Figure 10
CDF has not been conclusively traced; this update specifies Student-t marginals
from the NIG hierarchy, consistent with the paper's predictive Equation (15).
The previously documented full-suite and native-platform scope is unchanged.

## Figure 11-style update (py311.6)

Figure 11 was inspected directly in the supplied paper (PDF page 16), and its
aggregation was compared with the authors' original plotting code. New plots
group every test observation by 10-minute time of day, displaying mean truth,
mean predictions, and square root of mean aleatoric/epistemic variances.
The additive `mlguess.diurnal` helper leaves existing scientific implementations
unchanged. Analytical tests distinguish the authors' statistic from mean SD
and from the variance of a temporal mixture, and cover multiple targets,
timestamp alignment, empty bins, units, and invalid inputs.

Execution, numerical and visual-validation records are under
`validation/figure11-*`. Figures and bin-count tables are exported in PNG, SVG,
and CSV formats. The current tutorials genuinely predict friction velocity;
no sensible/latent-heat model or results were invented to fill three panels.

**99 focused tests passed**, including 45 analytical/plotting cases for the new
helper. The three affected notebooks executed **46 code cells and 15 figures**.
Their existing test metrics match the previous delivery; the 37 pre-existing
core Python files, two datasets, all 13 configuration files, and every existing
notebook cell source are unchanged. Only a new section is appended to each of
the three regression notebooks. The custom CSV branches also execute and skip
the diurnal view when clock timestamps are unavailable or the task is
classification. Full-suite and hardware validation retain their earlier scope.

The original short Keras run produces very large epistemic SD. Its new linear
figure preserves that complete range. A separately labelled symlog companion
uses the same values, with a linear region below 0.1 m/s, so the remaining curves
can also be read. Neither view clips, smooths, or changes the predictions.

## Time-axis update (py311.5)

The current update uses actual timestamps for temporal prediction/uncertainty
panels and meaningful predicted-value axes where timestamps are unavailable.
Calendar windows and disconnected interval bars preserve irregular spacing and
avoid implying measurements across gaps. Training epochs/batches and existing
calibration, physical-variable and class axes remain appropriate.

The three affected author-data notebooks were re-executed: **43 code cells
and 11 figures**. The guarded DNA notebook also executed its 11 cells without
external model downloads. **54 focused tests passed**, covering timestamp/value
alignment, date validation, timezone handling, and the preserved original
notebook settings. Both custom CSV branches passed. All six author-regression
metric tables match the previous delivery. Records are in
`validation/time-axis-*`, including `time-axis-focused-tests.log`.
The 37 core library Python files, both datasets, and all 13 configuration files
are byte-identical to py311.4. Naive source timestamps are not assigned an
invented timezone; explicit timezone-aware inputs are handled separately.
External paper-figure and SHAP plotting helpers are checked with isolated
plotting fixtures; the full external experiments remain unexecuted.

## Previous complete validation (py311.4)

Executed on **Linux x86-64, Python 3.11.17, PyTorch 2.14.1+cpu,
NumPy 2.4.6, Keras 3.15.1 with its Torch backend**. The complete test suite
passed **453 tests, with 4 skips**: three require CUDA and one requires Gloo
socket creation prohibited by this host. Logs and JUnit results are
`validation/authors-full-suite.log` and `.xml`. Upstream Keras/NumPy array
conversion deprecation warnings are recorded, not suppressed.

New tests compare notebook settings with the authors' originals, exercise
Gaussian-mixture densities and quantiles, compare the custom guide's actual
split indices against the original notebook, verify dataset hashes, and test
BERT-facing interfaces with explicitly mocked models. The original core tests
continue to pass. All 37 core library Python files are byte-identical to the
previous corrected delivery; only the local version marker changes. Source
comparison is recorded in `validation/authors-core-preservation.json`.

The prior function-level audit remains in `validation/function-audit.csv` and
`validation/function-audit-summary.json`: 355 named functions plus three lambdas
reviewed, with 352 named function bodies exercised. Its coverage measurements
belong to py311.3; this revision did not recompute coverage or claim newly tested
hardware/backend branches. The core implementations were preserved.

## Original author-data executions (py311.4)

Both bundled datasets are byte-identical to upstream. The seven relevant YAML
model dictionaries also match upstream. All original notebook training-budget
overrides are retained. No architecture was reduced to make CPU execution easier.

| Notebook | Execution on original observations | Code cells / figures |
| --- | --- | --- |
| `regression_example_torch.ipynb` | Cabauw; 55,007 train / 6,598 validation / 10,592 test. Original five 1057-unit layers, one pass per model. | 23 / 3 |
| `classifier_example_torch.ipynb` | Explicitly selected authors' bundled precipitation sample; 7,225 / 908 / 4,031. Original four 212-unit layers, 100 epochs, cap of 10 batches. | 20 / 4 |
| `regression_example.ipynb` | Cabauw; 57,431 / 7,578 / 7,188. Original Keras model configurations and 1/1/5-epoch overrides, ensemble and MC settings. | 10 / 6 |
| `classifier_example.ipynb` | Authors' bundled precipitation sample and original split. Preserved the 6024-unit, four-layer deterministic classifier, balanced training/weights, and original evidential model; 1/5 epochs. | 5 / 3 |
| `evidential_learning_your_data.ipynb` | Default changed to the authors' Cabauw data, original Torch model, preprocessing, year split and one-pass budget. | 10 / 2 |

At py311.4, these five notebooks contained **68 executed code cells and 18 figures**, with
saved outputs and no error outputs. The figures were visually inspected.
Keras documentation copies match their corresponding restored notebooks; the
Keras and PyTorch examples are distinct, not aliases.

The PyTorch precipitation notebook still defaults to the authors' full mPING
path in YAML. Execution here explicitly used `MILES_PTYPE_DATA` to select their
bundled real sample. That substitution is recorded and is not a full-dataset
reproduction claim. The original Keras classifier already selects that sample.
Both workflows retain the authors' filters, held-out dates, features and class
ordering. Exact paths, hashes, settings and split counts are in
`validation/authors-data-provenance.json` and `validation/authors-torch-notebooks.json`.

Test outcomes did not select training settings or tune the new plots. Poor
short-budget performance, minority-class errors, and imperfect calibration
remain visible. Physical-unit metrics, Student-t predictive intervals, actual
Gaussian-mixture evaluation, and per-member scalers are verified. The separate
custom-data guide illustrates validation-only checkpoint selection, saving and
reload; it does not change the originals' training selection rules.

Notebook records:

- `validation/authors-torch-notebooks.json` and `.log`
- `validation/authors-keras-notebooks-summary.json` and `authors-keras-notebooks.log`
- `validation/authors-custom-guide.json` and `.log`
- `validation/authors-notebook-review.json` (independent source/fidelity review)
- `validation/authors_notebook_figures/` (standalone PNGs)
- `validation/authors_tutorial_results/` and `authors_keras_metrics/` (CSV results)

The custom guide's CSV regression/classification branches were also executed
using temporary generated **test fixtures**. These validate input handling,
multiple targets, singleton batches, and save/reload. They are not the default
example data or reported authors' results. Group/time partition tests include
repeated timestamps; an additional test verifies the default Cabauw split
exactly against the authors' archived code.

## DNA and unavailable external resources

The original DNA notebook contains one illustrative sequence and BERT model
heads, not an experimental dataset, training procedure or held-out split. Those
original settings are restored. An optional section now evaluates a supplied
trained checkpoint on a real labeled test CSV with metrics and plots.

Its **11 guarded code cells** execute successfully without the external model
assets. **13 mock-interface tests** exercise numerical heads, inference state,
loss calls, target scaling, checkpoint handling and both evaluation/plot paths.
Pretrained BERT inference/training and actual DNA test-set metrics were **not**
executed: cached weights, trained checkpoints and real labeled test data were
not supplied. No dataset or scores are invented. See
`validation/authors-dna-notebook.json` and `.log`.

All notebooks were run in-order with in-process IPython because this host blocks
Jupyter kernel sockets. Native Jupyter kernel startup was not validated here.
External paper-figure, sounding and SHAP experiments were not rerun; their
original data/checkpoint requirements remain.

## Packaging and compatibility

The recommended Python 3.11 / PyTorch 2.14.1 CPU / NumPy 2.4.6 setup is unchanged.
The prior Windows x86-64 and macOS 14+ Apple Silicon binary-wheel resolution
records remain applicable to the unchanged core requirements. They are
compatibility checks, not native execution. **Native Windows/macOS, CUDA/AMP,
FSDP, TensorFlow/JAX and live external services remain unexecuted.** Multi-rank
Trainer/application execution remains explicitly unsupported.

Build, dependency, lint, notebook-format and installed-wheel smoke checks are
recorded under `validation/authors-*`. Old `audit-*` and `lotv-*` files describe
previous core audits. Superseded synthetic-notebook execution logs are marked
historical under `validation/previous-py311.3/`; their plots/results have been
removed from the current delivery.

## Reproduce

Follow the platform-specific Conda and pip commands in `README.md` and the
additional notebook dependencies/explicit precipitation path instructions in
`notebooks/README.md`. Then:

```bash
python -m pip check
python -m pytest -q
python examples/execute_notebooks.py notebooks/regression_example_torch.ipynb
python examples/execute_notebooks.py notebooks/regression_example.ipynb notebooks/classifier_example.ipynb
python examples/execute_notebooks.py notebooks/evidential_learning_your_data.ipynb
python examples/validate_custom_notebook.py
python -m build
```

Set `KERAS_BACKEND=torch` before importing Keras. The full suite requires the
optional Keras/data dependencies; PyTorch-only installations skip those paths.
Passing these tests supports the stated use cases and corrections, not universal
absence of errors, empirical calibration on new data, or reproduction of every
result in the authors' paper.
