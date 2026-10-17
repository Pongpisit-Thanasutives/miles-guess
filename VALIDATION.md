# Validation report

Date: 2026-10-17. Local version: **2025.1.0+py311.3**.
Upstream base: `69775bef5d65b50e53692d860092e0295b4e7d23`.
This is a corrected local derivative, not an upstream release.

## Executed environment and results

Execution used **Linux x86-64, Python 3.11.17, CPU**, with PyTorch
**2.14.1+cpu**, NumPy **2.4.6**, SciPy 1.17.1, pandas 2.3.3,
scikit-learn 1.9.1, Matplotlib 3.11.2, and optional Keras 3.15.1 using its
PyTorch backend. The exact environment is in
`validation/environment-audit-py311.txt`; recommended direct constraints are
`requirements/validated-py311.txt`.

The complete combined suite passed **426 tests, with 4 skips**. Three skips
require GPUs; the fourth attempts a real CPU Gloo process group, whose socket
creation is prohibited by this host. This is an environmental skip, not a
successful distributed runtime test. Logs and JUnit results are
`validation/audit-py311-torch214.log` and `.xml`.

The optional Keras paths emit upstream NumPy `__array__(copy=...)`
deprecation warnings from Keras' Torch backend. They are recorded in the log,
not globally suppressed. The supported CPU PyTorch workflow does not need
Keras or TensorFlow. A separately installed native-only environment passed **296 tests, with 6
skips**: one optional Keras module, one optional bridgescaler test, the Gloo
socket restriction, and three GPU tests. Four older Keras-only test files are
not collected without Keras. This environment has no Keras, TensorFlow, JAX,
or notebook runtime packages. The final native-only run emits no warnings.
These tests overlap the full suite and are not additional distinct tests.
Results are recorded in `validation/audit-native*`.

Coverage over `mlguess` and `applications`, excluding the test code itself:

| Measure | Executed / measured | Coverage |
| --- | --- | --- |
| Statements | 4,509 / 5,125 | **87.98%** |
| Branch arcs | 1,234 / 1,748 | **70.59%** |
| Combined statements and branch arcs | 5,743 / 6,873 | **83.56%** |

Coverage is evidence of execution, not proof that every behavior or input is
correct. `validation/function-audit.csv` lists the individual functions,
source review, observed function-body execution, missing lines/branches, and
reviewer limitations. It does not count importing a function definition as
calling that function. Lambdas have separate, conservative invocation status.
`validation/function-audit-summary.json` summarizes the inventory: **355 named
functions/methods/nested functions and 3 lambdas across 41 modules**. Every entry
received source review; **352 of the 355 named functions** have observed own-body
execution. Three named function bodies were not observed, and lambda invocation
is conservatively marked indeterminate from line coverage. All gaps are listed.

## What was checked

The audit combines source review with independently calculated expected
values and runtime tests. It covers:

- NIG Student-t likelihood/moments, Dirichlet moments/KL, original-equation
  loss and gradient comparisons, finite-mixture law of total variance,
  MC dropout state, Gaussian variance conversion, and affine inverse scaling.
- Brier/CSI/ECE/MCE, negative-target MAPE, missing-pair alignment, constant
  and endpoint bins, PIT/CRPS, and rendered classification/regression plots.
- Unequal minibatches, weighted/sum-reduced losses, optimizer updates,
  validation aggregation, min/max checkpoint selection, resume behavior,
  checkpoint format/device routing, and actual save/reload prediction equality.
- Current and legacy Keras fit/loss/layer/callback paths, one-dimensional and
  multiple targets, sample weights, serialization, and optimizer restoration.
- Dataset/preprocessor options, train-only scaling, class axes, small training
  sets, application CLI overrides, HPO test isolation, prediction row order,
  and PBS command construction using dry runs/mocks only.

Per-area logs and ledgers are `validation/audit-{torch,torch-checkpoint,keras,
metrics,applications,evaluation}.json`. Some are overlapping subsets of the
combined suite; do not add their test totals together. Corrections and exact
counterexamples are documented in `SCIENTIFIC_CHANGES.md`. Genuine numerical,
shape, weighting, and metric errors were fixed while retaining the authors'
model families and valid scientific equations.

## Executed notebooks

All six current tutorial filenames have saved executed outputs and pass
notebook-format validation. Four are distinct tutorials; the `_torch` regression
and classification filenames are exact aliases. Documentation copies are also
refreshed. The distinct notebooks executed **32 code cells and 13 figures**:

| Tutorial | Code cells | Figures |
| --- | --- | --- |
| `regression_example.ipynb` | 6 | 4 |
| `classifier_example.ipynb` | 6 | 5 |
| `evidential_learning_your_data.ipynb` | 10 | 2 |
| `dna_example_torch.ipynb` | 10 | 2 |

Plots were visually reviewed. Metrics, predictions, intervals, calibration,
and confusion matrices use held-out test data after validation-only checkpoint
selection. The custom-data guide's real-CSV classification and multi-target
regression branches were separately executed, including a 109-row dataset
that leaves a final training minibatch of one row. The guide saves/reloads
models, fitted scalers, class mappings, split assignments, and experiment
settings. Group/time splits have independent disjointness tests, including
repeated timestamps.

The guide explicitly disables BatchNorm to support singleton minibatches.
Its classification example initializes the output bias positively to prevent
an all-negative ReLU-evidence head from producing zero training gradients;
the library's evidence transform is unchanged. Results are shown honestly,
including imperfect calibration. Synthetic test accuracy is not evidence of
performance on a user's scientific dataset.

All code cells were executed in order using **in-process IPython** because
this host prohibits native Jupyter kernel sockets. Ordinary Jupyter kernel
execution was not verified here. `examples/execute_notebooks.py` reproduces
this method; `examples/validate_custom_notebook.py` exercises the CSV branches.
Execution logs and PNGs are under `validation/notebooks-*` and
`validation/notebook_figures/`.

Historical external-data research notebooks and original archived examples
were not rerun. Their required datasets/checkpoints are unavailable here.
Affected scientific cells have focused synthetic tests; this does not reproduce
the paper's experiments. See `notebooks/README.md` for the precise scope.

## Compatibility and remaining runtime gaps

Fresh **binary-only dependency resolution** succeeded with the recommended
constraints for Python 3.11 on Windows x86-64 and macOS 14+ Apple Silicon.
Records are `validation/audit-resolve-{windows,macos}.log` and
`validation/audit-resolved-{windows,macos}.txt`. This verifies compatible wheel
availability, not native runtime behavior. Intel/Rosetta macOS is not a target
of these modern official PyTorch wheels.

The CI workflow configures Linux, Windows, and macOS ARM64 test jobs. It was
not pushed or executed remotely for this delivery. **Native macOS/Windows,
CUDA/AMP, GPU FSDP, TensorFlow/JAX, real cluster submissions, live tracking/tuning
services, and production geospatial data remain unexecuted.** Mocked device and
wrapper tests are labeled as such. Multi-rank trainers/applications now reject
execution explicitly because global metrics and coordinated model selection
are not implemented reliably. Single-device CPU and CUDA interfaces remain;
only CPU execution was verified.

The recommended upgrade is PyTorch **2.14.1** while keeping Python **3.11** and
NumPy **2.4.6**. The older declared minimum PyTorch 2.5.1 was exercised in the
previous revision, but this revision's full audit uses 2.14.1. Older `lotv-*`
and `environment-lotv-*` files are retained historical records, not current
validation of the old environment.

## Reproduce

Follow the platform-specific installation commands in `README.md`, then:

```bash
python -m pip check
python -m pytest -q
python examples/cpu_quickstart.py
python examples/uncertainty_quickstart.py
python examples/execute_notebooks.py notebooks/regression_example.ipynb notebooks/classifier_example.ipynb notebooks/evidential_learning_your_data.ipynb notebooks/dna_example_torch.ipynb
python examples/validate_custom_notebook.py
python -m build
```

For optional Keras tests install `.[torch,keras,data,test]` and set
`KERAS_BACKEND=torch` before importing Keras. For coverage also install
`pytest-cov`, then run `python -m pytest -q --cov --cov-report=json:validation/audit-coverage.json`
and `python examples/build_function_audit.py`. Optional uninstalled dependency
branches are skipped rather than silently treated as tested.

Build, dependency, lint, wheel-install, and outside-source CPU smoke evidence
is retained in the `validation/audit-*` records. Passing this audit supports the
stated use cases; it cannot certify that every function is error-free or that
uncertainty estimates will be calibrated on a new dataset.
