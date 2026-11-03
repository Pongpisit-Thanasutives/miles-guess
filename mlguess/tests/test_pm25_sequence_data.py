"""Station-level splits, calendar alignment, train-only scaling, and unlabeled inference."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from mlguess.torch.nn_data import (
    CMAQ_FEATURES, FEATURES, STATIC_FEATURES, TS_FEATURES, inverse_sources,
    load_pm25_data, prepare_3d_input, prepare_pm25, source_sum_scaler_params,
    split_pm25_stations,
)


def observations(station, days=(1, 2, 3), offset=0.0):
    dates = pd.to_datetime([f"2020-01-{i:02d}" for i in days])
    rows = []
    for t, date in enumerate(dates):
        row = {"stcode": station, "region5": "north", "date": date, "pm25": offset + 2 * t + 1.0}
        row.update({col: offset + 0.5 * i + t for i, col in enumerate(TS_FEATURES)})
        row.update({col: offset + 0.8 * i for i, col in enumerate(STATIC_FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_sequence_keys_survive_shuffling_missing_labels_and_actual_gaps():
    a = observations("a", [1, 2, 3, 5, 6])
    a.loc[1, "pm25"] = np.nan
    df = pd.concat([a, observations("b", [1, 2], 30)], ignore_index=True).sample(frac=1, random_state=6)
    groups = prepare_3d_input(df, TS_FEATURES, STATIC_FEATURES)
    assert set(groups) == {2, 3}
    assert groups[2]["X_ts"].shape == (2, 2, len(TS_FEATURES))
    assert groups[3]["X_static"].shape == (1, len(STATIC_FEATURES))
    assert np.isnan(groups[3]["y"][0, 1])
    positions = []
    for T, group in groups.items():
        for i, station in enumerate(group["stations"]):
            raw = df.iloc[group["row_positions"][i]]
            assert raw.stcode.eq(station).all()
            assert (raw.date.diff().dropna() == pd.Timedelta(days=1)).all()
            np.testing.assert_equal(raw.date.to_numpy(), group["dates"][i])
            np.testing.assert_equal(raw[TS_FEATURES].to_numpy(), group["X_ts"][i])
            positions.extend(group["row_positions"][i])
    assert sorted(positions) == list(range(len(df)))


@pytest.mark.parametrize("model_name", ["CNN1D", "DB", "TwoStageMLP", "TwoStageCNN1D", "TwoStageDB"])
def test_train_scalers_and_unlabeled_transform(model_name):
    tr = observations("train")
    va = observations("validation", offset=5000)
    te = observations("test", offset=10000).drop(columns=["pm25"])
    tr.loc[1, "pm25"] = np.nan
    bundle = prepare_pm25(tr, va, te, model_name)
    np.testing.assert_allclose(bundle.feature_scaler.mean_, tr[list(bundle.ts_cols)].mean())
    assert bundle.target_scaler.mean_[0] == tr["pm25"].mean()
    n_db = model_name.endswith("DB")
    assert (bundle.static_scaler is not None) == n_db
    assert list(bundle.ts_cols) == (TS_FEATURES if n_db else FEATURES)
    np.testing.assert_array_equal(bundle.source_names, CMAQ_FEATURES)
    inp = next(iter(bundle.train.tensors.values()))
    assert len(inp) == (3 if n_db else 2)
    assert inp[0].ndim == (2 if model_name == "TwoStageMLP" else 3)
    assert np.isnan(inp[-1].numpy()).sum() == 1
    assert bundle.test.frame["pm25"].isna().all()
    unlabelled = bundle.transform(te)
    assert next(iter(unlabelled.tensors.values()))[-1].isnan().all()
    assert np.allclose(source_sum_scaler_params(bundle.feature_scaler, bundle.target_scaler)["source_mean"],
                       tr[CMAQ_FEATURES].mean())
    raw_sources = inverse_sources(inp[0].numpy()[..., :3], bundle.feature_scaler, len(bundle.ts_cols))
    np.testing.assert_allclose(raw_sources.reshape(-1, 3), tr[CMAQ_FEATURES].values, atol=1e-6)


def test_source_order_static_invariance_and_station_overlap_are_checked():
    tr, va, te = [observations(st) for st in "abc"]
    with pytest.raises(ValueError, match="first input features"):
        prepare_pm25(tr, va, te, "TwoStageDB", ts_cols=TS_FEATURES[::-1])
    tr.loc[0, "elevation"] = -22
    with pytest.raises(ValueError, match="Static features vary"):
        prepare_pm25(tr, va, te, "DB")
    with pytest.raises(ValueError, match="station-disjoint"):
        prepare_pm25(tr, tr.copy(), te, "CNN1D")
    with pytest.raises(ValueError, match="unique"):
        prepare_3d_input(pd.concat([te, te.iloc[:1]]), TS_FEATURES)


def test_station_stratification_and_train_only_callback(tmp_path):
    frames = [observations(f"S{i:02d}") for i in range(14)]
    for i, frame in enumerate(frames):
        frame["region5"] = "singleton" if i == 0 else "north" if i < 7 else "south"
    data = pd.concat(frames, ignore_index=True)
    data.loc[data.date.dt.day == 2, "pm25"] = np.nan
    path = tmp_path / "pm25.csv"
    data.to_csv(path, index=False)
    seen = set()

    def fill_training_only(frame):
        seen.update(frame.stcode.unique())
        frame["pm25_filled"] = frame.pm25.fillna(0.0)
        return frame

    tr, va, te = load_pm25_data(path, split_val=True, train_imputer=fill_training_only)
    assert seen == set(tr.stcode)
    assert "S00" in seen
    assert set(va.stcode).isdisjoint(seen | set(te.stcode))
    assert seen.isdisjoint(te.stcode)
    assert len(tr) + len(va) + len(te) == len(data)
    assert "pm25_filled" not in va and "pm25_filled" not in te
    result = prepare_pm25(tr, va, te, "CNN1D", training_target_col="pm25_filled")
    assert not next(iter(result.train.tensors.values()))[-1].isnan().any()
    assert next(iter(result.validation.tensors.values()))[-1].isnan().any()
    one_region_per_station = pd.concat([data, data.iloc[:1].assign(region5="different")], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one region"):
        split_pm25_stations(one_region_per_station)


def test_checkpoint_companion_restores_unlabeled_predictions_and_sources(tmp_path):
    import json
    import torch
    from mlguess.torch.nn_data import PM25Preprocessor
    from mlguess.torch.nn_training import build_model, predict_frame

    train, valid, test = [observations(name, offset=i) for i, name in enumerate(("train-A", "val-B", "test-C"))]
    prepared = prepare_pm25(train, valid, test, "TwoStageDB")
    unseen = observations("unseen", (1, 2, 5), offset=7).drop(columns="pm25")
    model_kwargs = dict(hts1=3, hts2=3, hts3=3, hstt1=3, hstt2=3, drop_rate_ts=0,
                        drop_rate_stt=0, hh1=3, hh2=3, hout=3, hdrop_rate=0, hsk=2)
    model = build_model("TwoStageDB", model_kwargs, prepared)
    split = prepared.transform(unseen)
    expected = predict_frame(model, split, prepared)
    path = tmp_path / "preprocessor.json"
    prepared.save(path)
    torch.save(model.state_dict(), tmp_path / "best_state.pt")
    metadata = json.loads(path.read_text())
    assert set(metadata) == {"format", "schema", "scalers"}
    assert not any(name in path.read_text() for name in ("train-A", "val-B", "test-C"))
    restored = PM25Preprocessor.load(path)
    reload_split = restored.transform(unseen)
    assert set(reload_split.tensors) == {1, 2}
    assert not hasattr(restored, "train")
    for key, batch in split.tensors.items():
        for actual, original in zip(reload_split.tensors[key], batch):
            torch.testing.assert_close(actual, original, equal_nan=True, rtol=0, atol=0)
    reloaded_model = build_model("TwoStageDB", model_kwargs, restored)
    reloaded_model.load_state_dict(torch.load(tmp_path / "best_state.pt", weights_only=True))
    actual = predict_frame(reloaded_model, reload_split, restored)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.pm25.isna().all()
    np.testing.assert_allclose(actual.pred, np.stack(actual.s_corrected).sum(axis=1), atol=1e-5)
