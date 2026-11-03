# Scientific preservation and verified corrections

Base: ai2es/miles-guess commit `69775bef5d65b50e53692d860092e0295b4e7d23`.
Local revision: `2025.1.0+py311.9`; this is not an upstream release.

The py311.9 extension adds the five PM2.5 station architectures described in
[PM25_REGRESSION.md](PM25_REGRESSION.md). The point variants follow the supplied
reference backbones. The new evidential variants apply the project's scalar
NIG regression to the total PM2.5 target; for a two-stage model the location
is constrained to the physical sum of its corrected source contributions.
The station notebooks, models and splits are separate from the original
Cabauw and precipitation experiments. The archive includes the historical
`UPSTREAM.patch` through py311.8 and `PM25_EXTENSION.patch` for this extension.

The py311.8 update aligns notebook explanations and display notation with the
paper, especially its Eqs. (19) and (24)–(26). Predicted NIG location gamma_n
is distinguished from the latent mean mu_n; code aliases remain documented
without changing public APIs. Classification notation follows Eqs. (3)–(7)
and (20)–(23). This is an exposition update: mathematical calculations and
training are unchanged. The SHAP research notebook's output labels now denote
the predicted location and variances actually returned by its model, rather
than incorrectly labelling those variance outputs as standard deviations.
No new attribution results or external paper experiments are claimed.

The py311.7 update adds Figure 10-style Student-t reference diagnostics to the
PyTorch regression notebook. `evaluate_nig` now returns `component_cdf`: the
aleatoric noise marginal, epistemic latent-mean marginal, and full predictive
CDF, each evaluated at the observed target with the appropriate location.
For each physical-unit variance V, the Student-t scale squared is
`V*(alpha-1)/alpha`, with `2*alpha` degrees of freedom. The total CDF reuses the
existing predictive PIT exactly. Independent Normal/inverse-gamma integration
and closed-form tests validate these calculations.

Only total is a genuine predictive PIT for observed Y. The component reference
values compare the observed residual, which contains both sources, against
one component's marginal distribution. They need not be uniform even for a
calibrated model. The components share a latent variance and are generally
marginally dependent; their variances, not their CDF values, add. The notebook
explains these distinctions and saves all 10,592 test-row values and nine-bin
relative-frequency tables. Existing notebook cells, data, training settings,
and predictive calculations are preserved. Figure 10's exact generation path
is unverified: the released code includes Gaussian PIT diagnostics, but that
does not establish which CDF produced the published evidential figure.

The py311.6 update adds diurnal-composite visualisation following Figure 11 of
Schreck et al. (2024), DOI `10.1175/AIES-D-23-0093.1`. Within each 10-minute
clock-time bin, truth and prediction curves are arithmetic means; each
uncertainty curve is the square root of the mean of its per-observation
variance. This matches the authors' plotting code. It is an RMS conditional
standard deviation, not a confidence interval for a bin mean or the complete
variance of a mixture across observations. No training or uncertainty
decomposition calculation changes. The new `mlguess.diurnal` module supports
one panel per available target and exports bin counts with the plotted values.

The py311.5 update changes visualisation axes and their supporting metadata.
Temporal observations use actual timestamps; interval panels without timestamps
use predicted values instead of observation ranks. It preserves the authors'
data, models, splits, training choices, and the corrected core library.

The model families, Normal-Inverse-Gamma (NIG) and Dirichlet parameterizations,
evidential regularizers, and Lipschitz-modified MSE equations are retained.
`mlguess/torch/regression_losses.py` is **not byte-identical** to upstream:
its only computational change constructs π with `nu.new_tensor(np.pi)` so
that the constant follows the input precision and device. Its equations and
existing numerical stabilizers are unchanged. An independent SciPy calculation
matches the stabilized float64 marginal likelihood within `2e-14`. The legacy
unused `LipschitzMSELoss.tol` argument is now documented accurately; changing it
does not change those stabilizers. The modified MSE's detached, batch-dependent
coefficient is retained.

This record distinguishes corrected implementations from changes to the
research method. It does not establish that every original result or every
possible input is correct. Function-level evidence and unexecuted branches
are recorded in `validation/audit-*.json`; test totals in overlapping ledgers
must not be added together.

## Predictive distributions and uncertainty

The earlier modernization and law-of-total-variance corrections are retained:

| Area | Correction and evidence |
| --- | --- |
| Finite ensembles and MC dropout | Combine average conditional aleatoric variance, average within-member epistemic variance, and variance of member means. Equal-weight finite mixtures use `ddof=0`, including one member. Two Bernoulli probabilities `.2,.8` give A=`.16`, E=`.09`, total=`.25`; the old sample-variance combination gave `.34`. This describes the finite mixture, not an unbiased estimate of an unknown population variance. |
| Output semantics | Decode logits, probabilities, evidence, NIG parameters, and Gaussian parameters explicitly. Point-regression predictions alone leave aleatoric and full total variance unknown (`None`), rather than applying Bernoulli formulas to unrestricted real values. Gaussian standard deviations are squared. |
| Evidential components | NIG moments remain A=`beta/(alpha-1)` and E=`A/nu`; Dirichlet class-indicator moments sum to `p(1-p)`. Mixtures retain within-member epistemic variance and, where returned, within-Dirichlet mutual information. Analytical moment, Student-t, digamma, and mixture tests cover these paths. |
| Physical units | Inverse-transform means once and multiply variances by the squared inverse affine slope. Standard, MinMax, Robust, and MaxAbs scalers are tested, including offsets, multiple targets, and fitted feature counts. Training-target variance is metadata, not a substitute for an arbitrary scaler slope; unsupported nonlinear/custom transformations raise. |
| Valid distribution outputs | Stable softplus avoids exponential overflow without changing its mathematical function. Keras NIG layers retain `alpha>1` even at extreme negative logits, with dtype-aware floors. Invalid concentrations, moments, or nonfinite evidence fail explicitly. |
| Keras Gaussian head | Only sigma is clamped positive. A linear head with raw mean `-3` and sigma `-2`, epsilon `.01`, now returns `[-3,.01]`; previously the negative mean was also clipped. The default output activation is unchanged. |
| Inference state | Ordinary prediction uses evaluation mode. MC inference activates dropout while keeping BatchNorm and additive-noise updates disabled, then restores the caller's state. Tests check stochastic output and unchanged running statistics. |

The new `mlguess.evaluation.evaluate_nig` evaluates a **single model's actual
predictive Student-t distribution**, including intervals, PIT, and log density
in physical target units. It must not receive averaged ensemble NIG parameters;
a mixture is generally not a single Student-t. `test_evaluation.py` checks
Student-t quantiles/densities, affine density scaling, and interval coverage.
Gaussian spread/CRPS functions retain their explicitly Gaussian diagnostic
interpretation; a Gaussian with matching moments is not asserted to be an
exact NIG predictive distribution.

## Metrics and diagnostic figures

| Verified defect | Correction and independent check |
| --- | --- |
| Variance supplied as Gaussian CRPS sigma | Use standard deviation `sqrt(total)`. Closed-form Gaussian tests also check the exact zero-spread limit, absolute error. |
| Negative-target MAPE denominator | Divide absolute error by absolute target magnitude, retaining the existing near-zero floor. Targets `[-2,-4]` and predictions `[-1,-2]` give `50%`, rather than `1.5e10%`. |
| Inconsistent multiclass Brier skill reference | Use per-class climatological frequencies and the same class normalization for forecast and reference. Labels `[0,0,0,1]` with forecasts `[.75,.25]` now give skill `0`, formerly `.625`. Zero-reference skill is undefined (`NaN`). |
| Wrong event in class reliability plots | Use each class probability and that class's indicator over all observations. Labels `[0,1,0,1]` with constant class-0 probability `.75` plot `(.75,.5)`, rather than top-label accuracy within a true-class subset. |
| CSI and ECE/MCE | CSI thresholds each class's own probability. ECE is the sample-weighted sum of absolute bin-mean gaps; MCE is their maximum. Boolean masks, label shapes, and probability validation prevent broadcasting or silently dropping invalid forecasts. Torch and Keras known-answer tests include perfect predictions. |
| Dropped or mispaired diagnostic samples | Include upper endpoints and constant-value bins; filter paired observations together; handle one-output arrays without introducing a third dimension. For aleatoric `[1,NaN,3,4]` and epistemic `[2,2,NaN,5]`, retain `(1,2),(4,5)`, not fabricated pairs. |
| Rank-calibration and regression attribution | Use actual histogram centers and within-bin conditional spread; retain the last occupied prediction bin and negative/constant predictions. The climatological reference is the sample mean. Identical rankings in two bins give centers `[.25,.75]` and zero conditional spread. |
| PIT and reliability boundaries | PIT histograms cover `[0,1]`; PIT deviation requires equally spaced bins across that domain. Reliability includes zero confidence. Gaussian mixtures average member CDFs; zero sigma uses a right-continuous point-mass CDF. The continuous-PIT uniformity result does not apply automatically to point masses. Ensemble ties retain SciPy's established average-rank convention. |

Raw Brier conventions are explicit: legacy `brier_multi(..., skill_score=False)`
uses the mean over classes; `evaluate_classifier` uses the sum over classes
before averaging observations. Their raw scores differ by the number of
classes; correctly normalized skill is unaffected. Undefined metrics remain
`NaN` rather than synthetic penalties. The existing spread-skill R² against
bin left edges is retained; it is not a universal calibration guarantee.

`test_audit_metrics.py` contains analytical counterexamples and executes the
classifier and regression diagnostic pipelines on synthetic data, including
saved figures. Plotting copies protect caller variance columns from repeated
square-root conversions. Basic plots no longer require optional Cartopy.

## Loss, training, and data semantics

| Area | Correction and evidence |
| --- | --- |
| Keras probabilistic loss reduction | NIG, Dirichlet, and coupled losses expose per-observation objectives; Gaussian model compilation uses a per-observation adapter while preserving the public `gaussian_nll(reduce=True)` default. Thus Keras `sample_weight` weights observations. Unweighted NIG/Dirichlet values and gradients match the original equations; Gaussian/coupled values also match independent calculations. Under the default Keras reduction, weights `[1,0]` on two samples give the first loss divided by two, not division by the sum of weights. |
| Loss serialization and class weights | Registered Keras loss classes retain coefficients and annealing state; nested factories previously collided under the same serialized name. NumPy/list class weights no longer fail truth-value checks or get silently ignored where supported. Existing experimental class-weight formulas are retained; operational tests do not establish their statistical validity. Unsupported Torch evidential-MSE class weighting raises explicitly. |
| Target/parameter shapes | Single-target vectors no longer broadcast against `(N,1)` predictions into `(N,N)` residuals. Legacy Keras metrics decode deterministic/Gaussian/NIG outputs by model family, not merely by widths two or four, which also occur in multi-target deterministic models. |
| Gradient accumulation and loss reporting | Mean losses use observation counts; weighted cross-entropy uses its target-class-weight denominator; sum-reduced losses remain sums. Unequal microbatches of sizes three and two match a five-sample SGD update in tested mean, sum, and weighted cross-entropy cases. The preserved batch-dependent modified-MSE coefficient means accumulation is not generally identical to concatenating batches for that objective. |
| Validation and model selection | Classifiers pass predictive probabilities to metrics. `fit`/`predict` compute nonlinear validation metrics over all selected samples, rather than averaging minibatch AUC/calibration values. Training minibatch metrics remain diagnostics. Selection honors min/max direction and finite values; resumed early-stopping offsets are corrected. |
| Schedules, initialization, and reload | Schedulers follow optimizer updates and preserve configured batch limits; explicit cosine stepping agrees with its incremental recurrence. Spectral initialization updates the trainable original weight. Keras archive/metadata separation, optimizer state, loss configuration, and wrapper settings survive tested save/reload paths. |
| Managed preprocessing and held-out data | Scalers fit only training data, constructor options are honored, and the declared class axis includes classes absent from training. Held-out precipitation dates use complete calendar days. Tests use validation outliers, missing training classes, and mocked input tables to check partition and scaling behavior. |
| Applications and exports | Preserve prediction row order and every target column; retain small training sets; record effective CLI settings. Hyperparameter tuning returns validation results before post-fit test evaluation. External custom datasets still own their split and preprocessing correctness. |

Independent references include SciPy Student-t and Dirichlet calculations,
original-equation gradient comparisons, larger-batch optimizer updates,
classical contingency/calibration calculations, train-only scaling checks,
and checkpoint round trips. Relevant tests are in `test_scientific_regression.py`,
`test_audit_keras.py`, `test_audit_torch.py`, `test_audit_torch_checkpoint.py`,
`test_audit_applications.py`, and the uncertainty/evaluation suites.

## Notebook corrections and validation limits

Retained notebook corrections remove the extra square from categorical
`p(1-p)`, define the ensemble mean, keep the same class event across sounding
ensemble members, and use a consistent full-Dirichlet marginal and scaled-density
Jacobian in the evidential figure. Corrected cells have synthetic execution
and rendering tests. Downstream outputs of affected paper cells were cleared;
external-data paper experiments and their scientific conclusions were not
reproduced. The authors' original regression, precipitation, and BERT example workflows
are restored in py311.4, with their datasets and model configurations retained.
Held-out evaluations and plots extend those workflows. Original snapshots are
in `notebooks/legacy/`; exact preservation and required API/calculation changes
are listed in `AUTHORS_NOTEBOOK_UPDATES.md`. Current execution records use the
`validation/authors-*` prefix. A correct variance decomposition alone does
not establish empirical calibration or identify a uniquely correct aleatoric/
epistemic split.

This revision explicitly rejects **multi-rank Trainer/application execution**.
Global weighted gradient synchronization, calibration/selection metrics,
sampler-padding deduplication, and coordinated early stopping have not been
validated. Retained distributed wrappers and mocked FSDP API-routing tests do
not establish working distributed training. Actual Gloo initialization was
blocked by the host's socket restrictions.

Runtime tests were conducted on Linux CPU with Python 3.11, NumPy 2, PyTorch,
and Keras using its PyTorch backend. Native macOS/Windows execution, CUDA/AMP,
accelerator FSDP, TensorFlow/JAX backends, real PBS submissions, production
geospatial data, and live tuning/tracking services were not executed here.
Per-function ledgers distinguish direct tests, mocked interfaces, source review,
and unexecuted branches. Passing tests and function coverage are evidence for
the stated cases, not a claim that the library is universally error-free.
