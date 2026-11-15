"""Small schema-shaped contracts. Fixtures are never used by the research pipeline."""
import numpy as np
import pandas as pd
import torch

from .nn_data import MODEL_NAMES, SOURCES, STATIC, WRF, Preprocessor
from .nn_models import build_model
from .nn_training import evidential_loss, predict_frame
from .metrics import score_predictions


def mock_station_frame(station, role):
    """Five days with real schema names; numbers are solely for software checks."""
    if role not in ("train", "val", "test"):
        raise ValueError("Choose train, val or test.")
    t = np.arange(5, dtype=float)
    d = {"date": pd.date_range("2021-03-01", periods=5), "stcode": station, "region5": "fixture"}
    for i, s in enumerate(SOURCES):
        d[s] = 2 + i + (i + 1) * t
    d["cmaq_pm25"] = sum(d[s] for s in SOURCES)
    d["aod"] = .1 + .1 * t
    d.update({s: 2 * i + t for i, s in enumerate(WRF)})
    d.update({s: 1. + i for i, s in enumerate(STATIC)})
    if role == "train":
        d["pm25_filled"] = 4 + 2 * t
        d["pm25_fill_n_neighbors"] = np.arange(5)
    else:
        d["pm25"] = [4., np.nan, 9., np.nan, 12.]
    return pd.DataFrame(d)


def check_model_contract(model_name):
    """Check forward, finite masked loss/backward, predictions, and conservation."""
    train = mock_station_frame("training", "train")
    val = mock_station_frame("unseen", "val")
    assert train.shape == (5, 21) and val.shape == (5, 20)
    prep = Preprocessor.fit(train, model_name, "at-bb-aod")
    # This deliberate distractor must never be scored as an observed target.
    split = prep.transform(val.assign(pm25_filled=-1.e6), "val")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(88)
        model = build_model(model_name, **prep.model_kwargs)
        model.eval()
        batch = next(split.batches(batch_size=8))
        out = model(*batch.inputs)
        assert out.gamma.shape == batch.y.shape
        assert (out.nu > 0).all() and (out.alpha > 1).all() and (out.beta > 0).all()
        assert torch.isfinite(batch.y).sum().item() == 3
        loss = evidential_loss(out, batch.y)
        loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        result = predict_frame(model, split)
    scores = score_predictions(result)
    assert len(result) == 5 and scores["n_obs"] == 3 and np.isfinite(scores["nll"])
    np.testing.assert_allclose(result.total_std**2, result.aleatoric_std**2 + result.epistemic_std**2)
    if model_name.startswith("TwoStage"):
        assert torch.equal(out.pred_physical, out.s_corrected.sum(-1))
        cols = ["s_corrected_" + s for s in SOURCES]
        np.testing.assert_array_equal(result.pred.to_numpy(), result[cols].to_numpy().sum(1))
    return {"model": model_name, "rows": len(result), "observed": scores["n_obs"], "status": "passed"}


if __name__ == "__main__":
    for name in MODEL_NAMES:
        print(check_model_contract(name))
