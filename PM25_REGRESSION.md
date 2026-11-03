# PM2.5 regression with PyTorch

Use these notebooks with a daily station CSV containing the CMAQ, AOD, weather and land-use fields in the supplied reference code. The station observations and the original imputation helper were not supplied. The Cabauw friction-velocity data and the precipitation-type examples have different predictors and targets.

| Notebook | Inputs and prediction | Configuration |
| --- | --- | --- |
| [CNN1D](notebooks/cnn1d_regression_torch.ipynb) | Causal temporal convolutions; one PM2.5 estimate per day | [YAML](config/cnn1d_regression_torch.yml) |
| [DB](notebooks/db_regression_torch.ipynb) | A temporal CNN plus an MLP for static station descriptors; one estimate per day | [YAML](config/db_regression_torch.yml) |
| [TwoStageMLP](notebooks/two_stage_mlp_regression_torch.ipynb) | A tabular MLP; one corrected concentration per source and their total | [YAML](config/two_stage_mlp_regression_torch.yml) |
| [TwoStageCNN1D](notebooks/two_stage_cnn1d_regression_torch.ipynb) | Causal temporal convolutions followed by separate source correction heads | [YAML](config/two_stage_cnn1d_regression_torch.yml) |
| [TwoStageDB](notebooks/two_stage_db_regression_torch.ipynb) | Temporal and static branches followed by separate source correction heads | [YAML](config/two_stage_db_regression_torch.yml) |

## Run a notebook

1. In the Python 3.11 environment with CPU PyTorch installed, unpack the project, enter its root and run `python -m pip install -e ".[torch,notebooks]"`.
2. Put the station CSV at `data/pm25_station_daily.csv`, set `data.csv_path` in the chosen YAML, or set the environment variable `MILES_GUESS_PM25_CSV` to an absolute path.
3. Run `python -m jupyter lab`, open the matching notebook and run all cells. The example widths, learning rates and epoch limits are starting values; the references do not specify tuned values for your data. Each run uses training stations to fit scalers, validation stations to select the epoch, and held-out test stations to report results.

Each input row identifies a station (`stcode`), calendar day (`date`) and region (`region5`). The target is `pm25` in µg/m³. The models use:

- Three additive CMAQ source estimates, in order: `cmaq_pm25_bb`, `cmaq_pm25_anthro`, `cmaq_pm25_othr`, each in µg/m³.
- Daily environmental predictors: `aod`, `wrf_pbl`, `wrf_temp`, `wrf_wspd`, `wrf_rh`. AOD is not an additive source.
- Station descriptors: `population`, `elevation`, `builtup`, `rice`, `forest`, `minor_road_length`, `major_road_length`. For DB models these must be constant within a station. Latitude and longitude, when present, remain metadata.

The reference split holds out entire stations within regions; it assesses prediction at new stations. Sequence models process each station in date order. A missing target retains its day in the input sequence and is masked in the loss and metrics. A missing calendar day starts a new sequence, so a convolution never treats nonconsecutive days as neighbors. Equal-length sequences batch without padding.

Predictors must be finite; if you use an external predictor imputer, fit it on training data only. To use the reference's separate training-label imputer, pass it as `train_imputer` to `load_pm25_data`. If it creates a column such as `pm25_filled`, pass `training_target_col="pm25_filled"` to `prepare_pm25`; validation and test scores still use the original `pm25` observations. By default no labels are imputed.

## Interpret the outputs

All YAMLs enable evidential regression by default. The Normal–Inverse-Gamma (NIG) head predicts uncertainty for **total PM2.5**: the predictive mean is the paper's γ; aleatoric variance is β/(α−1) and epistemic variance is β/[ν(α−1)]. The notebook computes the predictive interval from the corresponding Student-t distribution. The two-stage models constrain γ to the sum of corrected sources in physical units; no separate per-source uncertainty distributions are assumed.

The notebooks write `test_predictions.csv`, `test_metrics.csv`, a model checkpoint, fitted preprocessing and plots to the directory configured in `runtime.output_dir`. The test predictions include rows with missing labels; metrics use only measured targets. The time plot shows one held-out station, while the table covers all held-out stations.

Each two-stage record includes `pred`, the vector `s_corrected`, one numeric `corrected_<source>` column per source, and `delta_<source>` (corrected minus original concentration, after clipping). The vector is a JSON array in the CSV. Contributions and totals are in µg/m³; variance columns are in (µg/m³)². The notebook checks the source sum on every row. Agreement with the measured total alone does not establish accuracy of individual source estimates.

For point regression as in the attached models, set `model.kwargs.evidential: false`. The trainer then uses mean squared error; two-stage models retain the configured cosine penalty on the proposed source corrections. Set `training.device` for your installed PyTorch backend; the supplied defaults run on CPU.

## Predict for a new station

Reuse the fitted preprocessing and weights from the **same** run. From the project root, this example loads the TwoStageDB run and accepts an unlabeled CSV. Supply past daily predictor rows for each station if their context should contribute to later estimates.

```python
from pathlib import Path
import pandas as pd
import yaml
from mlguess.torch.nn_data import PM25Preprocessor
from mlguess.torch.nn_training import build_model, predict_frame

cfg = yaml.safe_load(Path("config/two_stage_db_regression_torch.yml").read_text())
run = Path(cfg["runtime"]["output_dir"])
processor = PM25Preprocessor.load(run / "preprocessor.json")
model = build_model(cfg["model"]["name"], cfg["model"]["kwargs"], processor)
model.load_weights(run / "best_state.pt")
new_rows = pd.read_csv("new_station_days.csv", dtype={"stcode": str})
predictions = predict_frame(model, processor.transform(new_rows), processor)
print(predictions[["stcode", "date", "pred", "s_corrected"]].head())
```
