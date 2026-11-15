# What is being predicted?

These educational methods are available independently or inside the main MILES-GUESS package. The equations are preserved; the loss controls, tuning and training-score additions are described below.

The response is **total PM2.5 for one station-day**, in µg/m³. All six models parameterize a predictive distribution for this one response. For a two-stage model there are also three point estimates: corrected CMAQ contributions for biomass burning (BB), anthropogenic sources (anthro), and other sources (othr). They do not have separate training labels or predictive distributions.

## Backbones and outputs

| Model | Inputs and computation |
|---|---|
| MLP | All selected predictors for a station-day; three dense ReLU/dropout layers. |
| CNN1D | All selected predictors, including the static columns, at each date; three left-padded causal convolutions of widths 7, 5 and 3 (13 consecutive days of receptive field). |
| DB | Dynamic CMAQ/optional AOD/WRF features enter the causal CNN. The seven spatial features enter a parallel two-layer MLP once per station segment. The model combines the two branch outputs per date. |
| TwoStageMLP / TwoStageCNN1D / TwoStageDB | The corresponding shared representation feeds three separate correction heads, each also given its own standardized CMAQ source. A separate evidential head estimates uncertainty for the total. The scalar location is constrained to the sum of corrections. |

The supplied backbone and head structures, chronological grouping, and input feature roles are maintained. The **output** follows the spectrally normalized `LinearNormalGamma` pattern of MILES-GUESS’s `DNN(lng=True)`. Single-stage models return four NIG parameters; two-stage models learn three positive evidence parameters and obtain the fourth (location) by source accounting. All evidence heads enforce positive ν, β and α > 1. Static variables for a DB must be constant within each station. A missing calendar **row** starts a new causal segment. A present row with a missing **label** remains valid model context; it is masked only for scoring. Segment predictions map back to original row positions.

**Source accounting.** For source k, let zₖ be its standardized CMAQ value, δₖ the learned residual and (mₖ,sₖ) its train-fitted mean and scale. The corrected physical value is cₖ = max(0, mₖ + sₖ(zₖ + δₖ)) with the default zero floor. The total mean is γ = c_BB + c_anthro + c_othr in µg/m³. Clamping standardized concentrations at zero would instead impose a floor at each source's training mean. The code sums *after* inverse scaling; it uses this very sum as the total, not a separate total-prediction head. The DataFrame recomputes the sum in its export precision. When checking a CSV after decimal serialization, read with round-trip precision or use a numerical tolerance.

Only the total is observed. Many component triples have the same sum. The original CMAQ inputs, model structure, nonnegativity and source-diversity penalty constrain the decomposition, but do not independently validate it. Do not interpret these three outputs as measured source attribution or assign each a share of the total variance without an additional model and data.

## NIG notation and target scaling

In the notation of Schreck et al., let μ be the **latent mean** and σ² the **latent conditional variance**. The network produces (γ,ν,α,β). For a given input, σ² follows Inverse-Gamma(α,β) and μ | σ² follows Normal(γ,σ²/ν). An observation has distribution Normal(μ,σ²) conditional on these latent variables. Integrating them gives a Student-t predictive distribution:

```text
degrees of freedom = 2α
location           = γ
scale²             = β(1 + 1/ν)/α
aleatoric variance = E[σ²]   = β/(α − 1)
epistemic variance = Var[μ]  = β/[ν(α − 1)]
total variance     = E[σ²] + Var[μ]
```

These are the **single-model** terms in the law of total variance, not variances across multiple fitted models. The Student-t **scale** is not its standard deviation. The package uses the Student-t CDF/density/quantiles for PIT, NLL and central prediction intervals. On inverse-transforming a standardized target z to y = m + s·z, γ becomes m + sγ, β becomes s²β, ν and α stay fixed, and each SD scales by s. Exported parameters and metrics use physical units. For target densities, NLL in physical units equals standardized NLL + log(s).

The component SDs are estimates under the chosen evidential model. Calibration of the **total** predictive distribution does not establish that its aleatoric and epistemic components have been individually recovered. Also, the Student-t has support on the real line; intervals can extend below zero even when the two-stage predictive mean is nonnegative. The evaluation does not silently truncate or clip these intervals, which would change the distribution.

## Training and tuning choices

Training uses standardized `pm25_filled` and the mean predictive NLL plus the Amini evidence penalty λ·|y−γ|(2ν+α), with the low-level training default λ = 0.01. When enabled, two-stage models add the reference absolute pairwise cosine similarity penalty on the source residual vectors (low-level default weight 0.05). Set `use_cosine_similarity=False` to disable it; single-stage models never include it. **This is a stated choice of EDL objective**: it replaces deterministic MSE and does not add the optional Lipschitz auxiliary MSE used in one MILES-GUESS regression example. Reported NLL excludes both penalties. Imputed training labels are treated as training values; this procedure does not propagate uncertainty in the imputer itself.

The Optuna search follows the supplied CNN1D notebook: random search, width exponents 4–7 constrained to a non-increasing chain, dropout 0.0–0.3 in steps of 0.1, log-uniform learning rate from 0.0001 to 0.01. It also samples `evidential_weight` logarithmically from 0.0001 to 0.1, passing the value to the existing `evidence_weight` training argument. For enabled two-stage source penalties, `lambda_sim` is sampled from 0.02 to 0.05 in increments of 0.01; otherwise the effective source-penalty weight is zero. For DB, the temporal and static width chains are searched separately; for two-stage models the head widths are fixed and recorded explicitly. Freshly initialized Adam models use the same seed and split across trials. Equal-length station segments are batched without padding; the maximum batch size is now explicit, rather than one potentially very large update per length group. Early stopping **and selection** minimize observed validation predictive NLL in physical units. This intentionally replaces the deterministic reference’s validation MSE. All five scores are retained separately for training (`pm25_filled`, including imputed values) and observed validation targets at each trial's restored best checkpoint in evaluation mode; the five metrics are not combined into one selection rule. The trial sequence is seeded per trial so the local study can resume without replaying the first candidate. A matching saved `study.sqlite3` is loaded and `n_trials` specifies the desired completed total: 25 → 50 adds 25 completed trials. The CSV export is refreshed as trials finish. A previous study with different code, data or training settings is rejected rather than combined with the new search.

The final usage notebooks likewise export training scores separately from test scores. Training metrics assess fit to `pm25_filled`; test metrics assess only the finite observed `pm25` values. The frozen model and prediction-unit conversions are the same for both.

## Reading the evaluation

| Output | Interpretation |
|---|---|
| R², RMSE, MAE | Point accuracy of γ on observed station-days. Higher R² and lower RMSE/MAE are better. R² is undefined for constant observations. |
| Mean NLL | Average negative log predictive density at observed PM2.5. Compare on the same rows, in the same units; smaller is better. |
| Mean Winkler | For a central interval [L,U] of coverage 1−q, score = (U−L) + (2/q)·max(L−y,0) + (2/q)·max(y−U,0). The default is 90% (q=0.10). It penalizes unnecessarily wide intervals and missed observations; smaller is better. |
| PIT histogram | Predictive Student-t CDF evaluated at each observation; marginal uniformity is the reference. Not a test of component-specific calibration. |
| Discard fraction | Remove station-days with the largest total predictive SD first, then plot RMSE over retained observations. Falling error suggests the model ranks harder observations as more uncertain. Tied cutoffs are handled without dependence on row order. |
| Station timeline | Observations, predicted mean and separate mean ± one component SD bands against actual dates at one selected station. They illustrate variance components, **not** 90% prediction intervals. Calendar gaps are visibly broken. |

For a manuscript, report observed counts and station coverage, the split rule/seed, feature option, trial search space/budget, chosen settings, target units and interval level. Pooled scores here weight each observed station-day equally; assess station/region differences and sensitivity to seeds when scientifically relevant. These notebooks evaluate spatial generalization to held-out stations. Causal convolutions alone do not make this a future-time forecasting experiment; forecasting would also require a time-based evaluation and predictors available at the forecast origin.

## References

- Schreck et al., [*Evidential Deep Learning: Enhancing Predictive Uncertainty Estimation for Earth System Science Applications*](https://doi.org/10.1175/AIES-D-23-0093.1), particularly sections 2.5–2.7; [MILES-GUESS source](https://github.com/ai2es/miles-guess). The companion local MILES-GUESS release used for checking the output head was `2025.1.0+py311.10`.
- Amini et al., [*Deep Evidential Regression*](https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html), NeurIPS 2020.
- Gneiting & Raftery, [*Strictly Proper Scoring Rules, Prediction, and Estimation*](https://doi.org/10.1198/016214506000001437), JASA 2007, interval score.
- The station architectures, schema, split strategy and reference search were provided in `nn_models.py`, `nn_data.py`, `nn_training.py`, and `cnn1d_hpt.ipynb`. No observations, station locations, authorship metadata or imputation implementation are assumed from these files.
