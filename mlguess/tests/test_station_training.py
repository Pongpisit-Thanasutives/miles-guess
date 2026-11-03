"""Real model updates, held-out source accounting, and unlabeled inference."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from mlguess.torch.models import seed_everything
from mlguess.torch.nn_data import prepare_pm25
from mlguess.torch.nn_training import build_model, fit_model, predict_frame, regression_metrics


@pytest.fixture(scope="module", autouse=True)
def small_cpu_models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def station_split(stations):
    rows = []
    for i, station in enumerate(stations):
        for day in range(3 + i):
            a, b = 8 + i + 2 * day, 3 + i - 0.3 * day
            rows.append(dict(station=station, time=pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                             s1=a, s2=b, wind=day % 2, height=35.0 + i, y=a + b - 1.0))
    frame = pd.DataFrame(rows).iloc[::-1].copy()
    frame.iloc[1, frame.columns.get_loc("y")] = np.nan
    return frame


def make_case(name, evidential):
    seed_everything(321)
    p = prepare_pm25(*(station_split(pair) for pair in (("A", "B"), ("C", "D"), ("E", "F"))),
                     model_name=name, source_names=("s1", "s2"), ts_cols=("s1", "s2", "wind"),
                     static_cols=("height",), target_col="y", group_col="station", date_col="time")
    if name.endswith("DB"):
        kwargs = dict(hts1=4, hts2=4, hts3=4, hstt1=4, hstt2=4, drop_rate_ts=0.1, drop_rate_stt=0.1)
        if name.startswith("TwoStage"):
            kwargs["hout"] = 4
    else:
        kwargs = dict(h1=4, h2=4, hout=4, drop_rate=0.1)
    if name.startswith("TwoStage"):
        kwargs.update(hh1=4, hh2=3, hdrop_rate=0.1, hsk=2)
    return p, build_model(name, dict(**kwargs, evidential=evidential), p)


@pytest.mark.parametrize("name", ["CNN1D", "DB", "TwoStageMLP", "TwoStageCNN1D", "TwoStageDB"])
@pytest.mark.parametrize("evidential", [True, False])
def test_fit_with_missing_labels_and_collect_aligned_predictions(name, evidential, tmp_path):
    p, model = make_case(name, evidential)
    checkpoint = tmp_path / "model.pt"
    before = [x.detach().clone() for x in model.parameters()]
    history = fit_model(model, p.train, p.validation, epochs=2, batch_size=1,
                        patience=1, checkpoint_path=checkpoint)
    assert np.isfinite(history.train_loss + history.val_loss).all()
    assert history.best_epoch == np.argmin(history.val_loss) + 1
    assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))
    pred = predict_frame(model, p.test, p, batch_size=1)
    assert np.all(pred[["station", "time"]].values == p.test.frame[["station", "time"]].values)
    assert regression_metrics(pred, "y")["n_observed"] == pred.y.notna().sum()
    # Removing test labels does not remove rows, influence mean predictions, or
    # change the source vectors. Loading the selected portable state reproduces the fit.
    _, restored = make_case(name, evidential)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    unknown = p.transform(p.test.frame.drop(columns="y"))
    actual = predict_frame(restored, unknown, p)
    np.testing.assert_allclose(actual.pred, pred.pred, rtol=2e-6, atol=2e-6)
    if model.two_stage:
        sources = pred[["corrected_s1", "corrected_s2"]].to_numpy()
        np.testing.assert_allclose(sources.sum(axis=1), pred.pred, atol=2e-5)
        np.testing.assert_allclose(np.stack(pred.s_corrected), sources)
        np.testing.assert_allclose(sources - pred[["s1", "s2"]].to_numpy(),
                                   pred[["delta_s1", "delta_s2"]], atol=2e-6)
    if model.evidential:
        np.testing.assert_allclose(pred.aleatoric_variance, pred.beta / (pred.alpha - 1))
        np.testing.assert_allclose(pred.epistemic_variance, pred.aleatoric_variance / pred.nu)
        np.testing.assert_allclose(pred.total_variance, pred.aleatoric_variance + pred.epistemic_variance)
        for key, tensors in p.test.tensors.items():
            inputs = tensors[:-1]
            with torch.no_grad():
                raw = model(*inputs)
            raw_params = raw[0] if model.two_stage else raw
            rows = p.test.groups[key]["row_positions"].reshape(-1)
            rise = p.target_scaler.scale_[0]
            np.testing.assert_allclose(pred.beta.to_numpy()[rows],
                                       raw_params[3].numpy().reshape(-1) * rise**2, rtol=1e-6)
    else:
        assert "beta" not in pred and "total_variance" not in pred


def test_position_collision_is_not_silently_merged():
    p, model = make_case("CNN1D", True)
    key = next(iter(p.test.groups))
    indices = p.test.groups[key]["row_positions"].ravel()
    indices[1] = indices[0]
    with pytest.raises(ValueError, match="row positions"):
        predict_frame(model, p.test, p)
