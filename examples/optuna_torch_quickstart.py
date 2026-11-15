"""Tune a Cabauw evidential DNN's learning rate on validation years (CPU).

Run from the unpacked source: python examples/optuna_torch_quickstart.py
Uses the regression notebook's real data, model, year split, scalers and loss.
The default four training batches per trial demonstrate the API, not paper performance.
"""
import argparse
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import torch
import yaml

from mlguess.torch.models import DNN
from mlguess.torch.regression_losses import LipschitzMSELoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--train-batches", type=int, default=4,
                        help="Batches of 128 per trial, up to one full pass")
    parser.add_argument("--plot", type=Path, help="Optional new PNG for the optimization history")
    args = parser.parse_args()
    if args.trials < 1 or args.train_batches < 1:
        parser.error("--trials and --train-batches must be positive")
    if args.plot and (args.plot.exists() or args.plot.suffix.lower() != ".png"):
        parser.error("--plot must name a new .png file")

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config/evidential_regression_torch.yml").read_text())
    df = pd.read_csv(root / "data/sample_cabauw_surface_layer.csv")
    years = pd.to_datetime(df["Time"]).dt.year
    seed = cfg["seed"]
    outer = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=seed)
    dev, _ = next(outer.split(df, groups=years))  # Keep the test years untouched.
    inner = GroupShuffleSplit(n_splits=1, train_size=0.885, random_state=seed)
    train, valid = next(inner.split(df.iloc[dev], groups=years.iloc[dev]))
    train, valid = df.iloc[dev[train]], df.iloc[dev[valid]]
    inputs, targets = cfg["data"]["input_cols"], cfg["data"]["output_cols"]
    xs, ys = RobustScaler(), MinMaxScaler()
    x = torch.tensor(xs.fit_transform(train[inputs]), dtype=torch.float32)
    y = torch.tensor(ys.fit_transform(train[targets]), dtype=torch.float32)
    x_valid = torch.tensor(xs.transform(valid[inputs]), dtype=torch.float32)
    truth = valid[targets].to_numpy()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    batches = indices.split(128)[:args.train_batches]

    def objective(trial):
        # Match the initialization, dropout seed and batch order across trials.
        torch.manual_seed(seed)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        model = DNN(**cfg["model"]).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = LipschitzMSELoss(**cfg["train_loss"])
        model.train()
        for ix in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(*model(x[ix]), y[ix])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            gamma = torch.cat([model(block)[0] for block in x_valid.split(512)])
        return float(np.abs(ys.inverse_transform(gamma.numpy()) - truth).mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=args.trials, n_jobs=1)
    print(f"Best learning rate: {study.best_params['learning_rate']:.6g}; "
          f"validation MAE: {study.best_value:.5f} m/s ({len(valid)} observations)")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        ax = optuna.visualization.matplotlib.plot_optimization_history(
            study, target_name="Validation MAE (m/s)")
        with args.plot.open("xb") as out:
            ax.figure.savefig(out, format="png", bbox_inches="tight")
        plt.close(ax.figure)


if __name__ == "__main__":
    main()
