# PyTorch in MILES-GUESS

Use the [installation guide](install.rst) or the [top-level README](../../README.md)
for Python 3.11 CPU setup. The audited environment uses PyTorch 2.14.1 and
NumPy 2.4.6. Windows/Linux use the official CPU wheel index; native Apple
Silicon macOS 14+ uses the ordinary PyPI wheel. Native platform CI is configured,
but Windows/macOS runs were not executed for this delivery.

The [notebook guide](../../notebooks/README.md) is the entry point for synthetic
regression/classification examples and adapting the models to your own data.
For a short script, run `python examples/cpu_quickstart.py` from the project root.

## CPU application scripts

The supplied weather configurations require their original datasets and updated
local paths. Use one process and explicit CPU settings:

```yaml
trainer:
  device: cpu
  mode: none
  amp: false
  thread_workers: 0
  valid_thread_workers: 0
```

Train an evidential regressor and export predictions for each data split:

```bash
python applications/train_regressor_torch.py -c config/evidential_regression_torch.yml --mode none --device cpu
```

Reload its checkpoint and export predictions:

```bash
python applications/predict_regressor_torch.py -c /path/to/run/model.yml --mode none --device cpu
```

Train a classifier and export predictions:

```bash
python applications/train_classifier_torch.py -c config/evidential_classifier_torch.yml --mode none --device cpu
```

All scripts accept `-c/--config`, `-m/--mode`, `--device`, and `-l` (integer,
default `0`, optional PBS submission). Run a script with `--help` for its options.
ECHO/Optuna and W&B are optional and are imported only when requested. CLI mode
and device overrides are recorded in `model.yml` for training and `prediction.yml`
for prediction. Prediction preserves the training configuration file.

Regression CSVs retain every target's mean and variance components. Classification
CSVs contain normalized class probabilities, with evidential uncertainty columns
when `trainer.uncertainty: true`. Export rows follow the original split order.
Hyperparameter trials return validation results without post-fit test evaluation.

The regression scripts load a custom `Dataset` class configured through
`data.dataset_path` and `data.dataset_name`. That dataset owns its train/validation/
test splitting and train-only scaler fitting. It supplies `train_data`,
`valid_data`, and `test_data` DataFrames for export; `y_scaler` and `training_var`
metadata are optional. Use an importable Python module filename. With spawned
workers on macOS/Windows, put program entry points under
`if __name__ == "__main__":`.

## Training configuration

This fragment shows CPU defaults; add your dataset settings, `save_loc`, and
loss configuration to make a complete application configuration:

```yaml
trainer:
  device: cpu
  mode: none
  training_metric: valid_loss
  train_batch_size: 64
  valid_batch_size: 64
  batches_per_epoch: 0
  valid_batches_per_epoch: 0
  learning_rate: 0.001
  weight_decay: 0.0001
  start_epoch: 0
  epochs: 100
  amp: false
  grad_accum_every: 1
  grad_max_norm: 1.0
  thread_workers: 0
  valid_thread_workers: 0
  stopping_patience: 5
  load_weights: false
  load_optimizer: false
  use_scheduler: false

model:
  input_size: 10
  output_size: 2
  layer_size: [64, 64]
  dr: [0.1, 0.1]
  batch_norm: false
  lng: true
```

`batches_per_epoch` and `valid_batches_per_epoch` are optional caps; zero uses
the full loader. `grad_accum_every` sets gradient accumulation. CPU training
uses float32; CUDA AMP is a separate optional path. Worker counts refer to
DataLoader processes. A training dataset smaller than the batch size still
produces one batch.

`load_weights` restores a checkpoint, while `load_optimizer` also restores its
optimizer state. `start_epoch` controls the resumed epoch number. Checkpoints
use explicit device mapping and safe tensor/state loading.

Schedulers are optional. For example:

```yaml
trainer:
  use_scheduler: true
  scheduler:
    scheduler_type: plateau
    mode: min
    factor: 0.1
    patience: 2
    cooldown: 2
    min_lr: 1.0e-7
    threshold: 1.0e-4
```

The loader also supports `lambda` and `cosine-annealing`; see
`mlguess/torch/scheduler.py` for their parameters. Modern PyTorch's plateau
scheduler no longer accepts the old `verbose` option.

## Model output configuration

`input_size` and `output_size` accept counts or feature-name lists. `layer_size`
specifies hidden widths; `dr` specifies dropout rates. `lng: true` selects the
Normal–Inverse-Gamma regression head. The bundled regression applications use
evidential regression losses; use the notebooks/library for point and Gaussian
regression training examples.

For the classifier application, use `lng: false`, `softmax: false`, and set
`output_size` to the number of classes. `trainer.uncertainty: true` selects the
evidential classification loss and uncertainty calculation; `false` selects
ordinary cross-entropy training on logits. The trainer converts outputs to
probabilities for metrics and exported predictions. `output_activation` is a
Keras setting and is not a PyTorch model constructor argument.

The [uncertainty guide](../../UNCERTAINTY.md) documents ensemble and MC-dropout
APIs, variance units, within/between epistemic components, finite-mixture
normalization, and output-kind selection. Run
`python examples/uncertainty_quickstart.py` for a self-contained CPU example.

## Optional distributed and PBS integrations

The library retains DDP and FSDP wrappers, but multi-rank Trainer execution
raises explicitly because global validation and model selection are unsupported.
The bundled applications currently
reject `world_size > 1` before training because rank-local predictions cannot
be attached safely to full DataFrames. Multi-rank application CSV export is
unsupported. GPU, FSDP, and real multi-node execution were not exercised in
this CPU audit.

PBS helpers are cluster-specific script generators for Casper/Derecho. They
require local cluster commands and credentials. Inspect a script without
submitting a job by calling `launch_pbs_jobs(..., launch=False)` or
`launch_distributed_jobs(..., launch=False)`. The application flag `-l 1`
submits a job; it is not a local CPU setup step. See `config/pbs_example.yaml`
for configuration, and the validation report for the mocked dry-run coverage.
