# Uncertainty calculations and migration notes

This local revision, `2025.1.0+py311.2`, corrects implementation errors found
after the first Python 3.11 modernization. Use this revision for new analyses.
The previous revision's compatibility tests did not cover all ensemble and
MC-dropout uncertainty paths. Numerical results from those affected paths
must be recomputed; changing a plot label does not correct stored estimates.

The reference is Schreck et al., *Evidential Deep Learning: Enhancing Predictive
Uncertainty Estimation for Earth System Science Applications*,
[DOI 10.1175/AIES-D-23-0093.1](https://doi.org/10.1175/AIES-D-23-0093.1),
[arXiv v2, Section 2.6 and Appendix SII](https://arxiv.org/html/2309.13207v2).
The NIG/Dirichlet model families, likelihoods and regularization are retained.

## Variances and ensembles

All `aleatoric`, `epistemic` and `total` outputs described here are **variances**.
For one NIG prediction `(mu, nu, alpha, beta)`,

```text
aleatoric = beta / (alpha - 1)
epistemic = beta / (nu * (alpha - 1))
total = aleatoric + epistemic
```

For a finite mixture of trained models or dropout passes, each evaluated at
the same input and in the same target units:

```text
mean = mean(member_mean)
aleatoric = mean(member_aleatoric)
within_epistemic = mean(member_epistemic)
between_epistemic = var(member_mean, ddof=0)
epistemic = within_epistemic + between_epistemic
total = aleatoric + epistemic
```

This hierarchical mixture follows from the law of total variance. It keeps
within-member evidential uncertainty as well as differences between members;
it is not a new training objective. Do not average NIG or Dirichlet parameters
to construct the mixture. A one-member mixture has zero between-member variance.
`ddof=0` describes the finite mixture being evaluated; `ddof=1` estimates a
different population quantity and does not satisfy the same finite-mixture
identity. Unequal weights are supported by the shared helper.

```python
import numpy as np
from mlguess.uncertainty import ensemble_uncertainties

# models: trained PyTorch DNN(..., lng=True) instances
# X: identical input rows for every model; scaler: fitted target scaler or None
member_moments = [model.predict(X, y_scaler=scaler) for model in models]
means = np.stack([v[0].detach().cpu().numpy() for v in member_moments])
aleatoric = np.stack([v[1].detach().cpu().numpy() for v in member_moments])
epistemic = np.stack([v[2].detach().cpu().numpy() for v in member_moments])
result = ensemble_uncertainties(means, aleatoric, epistemic, axis=0)
# Arrays are (members, samples, targets); result arrays are (samples, targets).
assert np.allclose(result['total'], result['aleatoric'] + result['epistemic'])
```

If members were trained with different input/target scalers, transform the
common physical inputs with each member's input scaler and inverse-transform
that member's moments with its own target scaler **before** combining them.
The array helper cannot detect misaligned rows, classes or units.

## PyTorch MC dropout

```python
result = model.predict_dropout(
    X, mc_forward_passes=100, batch_size=128, return_dict=True
)
```

`DNN` declares its default output type from its architecture: `lng=True`
means NIG; `softmax=True` means probabilities; otherwise it means point
regression. `CategoricalDNN` declares evidential classification. For a custom
network or ordinary classifier emitting logits, specify `output_kind`:
`regression`, `logits`, `probabilities`, `evidence`, `dirichlet`, `nig`,
`gaussian_std`, or `gaussian_variance`. The standalone
`mlguess.torch.mc_dropout.monte_carlo_dropout` accepts the same declarations.
Gaussian and NIG packed arrays use contiguous parameter blocks, with target
as the final dimension (mean block then standard deviation/variance block,
or mean, nu, alpha, beta blocks respectively).

For point regression, only the mean and between-pass variance are estimated:
`aleatoric` and `total` are `None`. There is no learned observation-noise
distribution. Treating those missing variances as zero would be an additional
point-mass assumption, not an estimate of full predictive uncertainty.

Classification variances refer to **each fixed class indicator**. For ordinary
probabilities, member aleatoric variance is `p*(1-p)`. For Dirichlet members
with `p=alpha/S`, member epistemic variance is `p*(1-p)/(S+1)` and member
aleatoric variance is the remainder. Both contributions are retained across
dropout passes. Per-class total variance equals `mean_p*(1-mean_p)`; the
decomposition, not this total alone, separates the two contributions. These
are marginal variances, not the full class covariance matrix.

Classification `entropy` is categorical predictive entropy in nats.
`mutual_info` for evidential dropout includes within-Dirichlet and between-pass
information; Torch dictionary outputs also expose `mutual_info_within` and
`mutual_info_between`. These information quantities are not variances.
The Dirichlet/Dempster-Shafer diagnostic `K/S` is also not a variance.

With `return_dict=False`, classification keeps the five-value tuple
`(mean, aleatoric, epistemic, entropy, mutual_info)`; regression uses
`(mean, aleatoric, epistemic, total)`. This fixes the old default-regression
path that treated unrestricted values as probabilities. Dropout inference
preserves model modes and freezes BatchNorm updates.

## Scaling and Keras

An affine inverse transform `y = a*z + b` transforms the mean once and
multiplies every variance by `a**2`. StandardScaler, MinMaxScaler,
RobustScaler and MaxAbsScaler are supported. Custom/nonlinear scalers are
rejected by the moment-scaling helper; training-target variance is not a
general substitute for the squared inverse slope. Perform an appropriate
distribution transformation separately if nonlinear scaling is required.

The Keras Gaussian head returns `(mu, sigma)` blocks, with **standard
deviation** `sigma`, consistent with its original likelihood. Obtain moments
with `RegressorDNN.predict_uncertainties(X, y_scaler=scaler)` or
`calc_uncertainties(raw_outputs, y_scaler=scaler)`: these return concatenated
`(mu, aleatoric_variance, epistemic_variance)` blocks. A Gaussian member has
`aleatoric=sigma**2`, `epistemic=0`. Ordinary Gaussian `predict` retains its
raw-output compatibility contract. Evidential moments no longer multiply by
the unrelated training-target variance; that multiplication previously gave
zero uncertainty for single-target Keras training.

Legacy ensemble APIs still return member arrays unless explicitly aggregated.
Use `ensemble_uncertainties` to combine them. The standalone Keras MC helper
now aggregates Gaussian and NIG moments, including within-member epistemic
variance, without updating BatchNorm statistics. Legacy split ensembles need
their per-member preprocessing; unavailable preprocessing is not reconstructed
from stored training variances.

## Reproducibility for research

Run `python examples/uncertainty_quickstart.py` and `python -m pytest -q` after
installing the corrected source. `VALIDATION.md` records the actual environments
and test results. Archive this exact source revision/patch, dependency versions,
seeds, data splits, fitted preprocessing and checkpoints with your experiment.
The corrected notebook cells have stale outputs cleared, but paper figures
requiring external data and trained checkpoints have not been regenerated.

These tests check formulas, implementation and small end-to-end examples;
they do not establish calibration or reproduce the paper's experimental
results. Assess calibration and predictive skill on held-out data for the
intended application. If citing the package, cite the original paper and
identify this local corrected revision alongside your archived code.
