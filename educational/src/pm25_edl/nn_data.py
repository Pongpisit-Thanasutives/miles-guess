"""PM2.5 station splitting and preprocessing. Run with python -m pm25_edl.nn_data."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SOURCES = ("cmaq_pm25_bb", "cmaq_pm25_anthro", "cmaq_pm25_othr")
WRF = ("wrf_pbl", "wrf_temp", "wrf_wspd", "wrf_rh")
STATIC = ("population", "elevation", "builtup", "rice", "forest",
          "minor_road_length", "major_road_length")
FEATURE_OPTIONS = {
    "baseline": ("cmaq_pm25",),
    "at-bb": SOURCES,
    "at-bb-aod": (*SOURCES, "aod"),
}
MODEL_NAMES = ("MLP", "CNN1D", "DB", "TwoStageMLP", "TwoStageCNN1D", "TwoStageDB")
IDS = ["date", "stcode", "region5"]
RAW_FEATURES = ["cmaq_pm25", *SOURCES, "aod", *WRF, *STATIC]
TARGET = {"train": "pm25_filled", "val": "pm25", "test": "pm25", "predict": None}


def normalize_model_name(name):
    key = name.replace("_", "").replace("-", "").casefold()
    for model in MODEL_NAMES:
        if model.casefold() == key:
            return model
    raise ValueError(f"Unknown model {name!r}. Choose one of {MODEL_NAMES}.")


def _require_columns(frame, names):
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _validate_frame(frame):
    """Keep row positions stable and reject ambiguous station/date keys."""
    _require_columns(frame, IDS)
    if frame.empty or frame[IDS].isna().any().any():
        raise ValueError("Need nonempty data with a date, station and region on each row.")
    frame = frame.reset_index(drop=True).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].dt.tz is not None or not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("date must be a calendar day, without a time or timezone.")
    frame["stcode"] = frame["stcode"].astype(str)
    frame["region5"] = frame["region5"].astype(str)
    if frame.duplicated(["stcode", "date"]).any():
        raise ValueError("Duplicate (stcode, date): one row per station per day is required.")
    if frame.groupby("stcode", sort=False)["region5"].nunique().gt(1).any():
        raise ValueError("A station cannot belong to more than one region.")
    return frame.drop(columns=["pm25_fill_n_neighbors"], errors="ignore")


def load_split(directory, split):
    """Read just train.csv, val.csv or test.csv; preserve station codes like '001'."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    frame = pd.read_csv(Path(directory) / f"{split}.csv", dtype={"stcode": str, "region5": str})
    frame = _validate_frame(frame)
    _require_columns(frame, [TARGET[split]])
    return frame


def validate_disjoint_stations(*frames):
    """Audit a spatial holdout; checking train/val never requires opening test.csv."""
    seen = set()
    for frame in frames:
        stations = set(frame["stcode"].astype(str))
        common = seen & stations
        if common:
            raise ValueError(f"Station overlap in spatial holdout: {sorted(common)[:5]}")
        seen.update(stations)


def _draw_holdout_stations(stations, fraction, seed):
    """Largest-remainder region quotas. Singletons remain in training."""
    if not 0 < fraction < 1:
        raise ValueError("Holdout fractions must be strictly between 0 and 1.")
    counts = stations["region5"].value_counts().sort_index()
    counts = counts[counts >= 2]
    if counts.empty:
        raise ValueError("Spatial split impossible: no region has two remaining stations.")
    # Clamp the requested count to the feasible range under one per side per region.
    target = int(np.clip(round(counts.sum() * fraction), len(counts), (counts - 1).sum()))
    quota = counts * target / counts.sum()
    seats = np.floor(quota).astype(int).clip(lower=1, upper=counts - 1)
    while seats.sum() != target:
        if seats.sum() < target:
            region = (quota - seats)[seats < counts - 1].idxmax()
            seats.loc[region] += 1
        else:
            region = (seats - quota)[seats > 1].idxmax()
            seats.loc[region] -= 1
    held = set()
    for region, n in seats.items():
        codes = sorted(stations.loc[stations["region5"] == region, "stcode"])
        _, sample = train_test_split(codes, test_size=int(n), random_state=seed)
        held.update(sample)
    return held


def save_station_splits(raw_df, directory, *, test_size=.2, val_size=.2, seed=51,
                        test_regions=None, train_imputer=None):
    """Create train/val/test CSVs once, from raw observations (pm25).

    test_size applies to splittable stations, val_size to the remaining stations.
    Alternatively, hold out complete named test_regions. Singleton regions stay in
    training unless explicitly held out. train_imputer, when needed, accepts ONLY
    training rows and must return the same keys with a finite pm25_filled column.
    The user's external imputer is not bundled. Already-split files need no imputer.
    """
    raw = _validate_frame(raw_df)
    _require_columns(raw, RAW_FEATURES + ["pm25"])
    # Never trust a filled target computed before splitting; retain ancillary
    # fields (e.g., lat/lon) for the user's train-only imputer.
    raw = raw.drop(columns=["pm25_filled"], errors="ignore")
    stations = raw[["stcode", "region5"]].drop_duplicates()
    if test_regions is None:
        test = _draw_holdout_stations(stations, test_size, seed)
    else:
        regions = {test_regions} if isinstance(test_regions, str) else set(test_regions)
        if not regions or regions - set(stations["region5"]):
            raise ValueError("Specify nonempty, known test_regions.")
        test = set(stations.loc[stations["region5"].isin(regions), "stcode"])
    val = _draw_holdout_stations(stations[~stations["stcode"].isin(test)], val_size, seed)
    train = raw[~raw["stcode"].isin(test | val)].copy()
    if train_imputer is not None:
        filled = _validate_frame(train_imputer(train.copy()))
        _require_columns(filled, ["pm25_filled"])
        join_keys = ["stcode", "date"]
        values = filled.set_index(join_keys)["pm25_filled"]
        keys = pd.MultiIndex.from_frame(train[join_keys])
        if len(values) != len(keys) or not keys.isin(values.index).all():
            raise ValueError("Training imputer changed the station/date records.")
        train["pm25_filled"] = values.reindex(keys).to_numpy()
    else:
        train["pm25_filled"] = train["pm25"]
    if not np.isfinite(train["pm25_filled"].to_numpy(dtype=float)).all():
        raise ValueError("Training contains unfilled values. Supply your train-only "
                         "train_imputer or use your pre-split train.csv with pm25_filled.")
    observed = train["pm25"].notna()
    if not np.allclose(train.loc[observed, "pm25"], train.loc[observed, "pm25_filled"]):
        raise ValueError("Training imputation must preserve observed pm25 values.")
    columns = IDS + RAW_FEATURES
    splits = {"train": train[columns + ["pm25_filled"]],
              "val": raw.loc[raw["stcode"].isin(val), columns + ["pm25"]],
              "test": raw.loc[raw["stcode"].isin(test), columns + ["pm25"]]}
    validate_disjoint_stations(*splits.values())
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    if any((dest / f"{role}.csv").exists() for role in splits):
        raise FileExistsError("Existing split files: choose a new output directory.")
    for role, df in splits.items():
        df.to_csv(dest / f"{role}.csv", index=False)
    return {role: dest / f"{role}.csv" for role in splits}


@dataclass
class Batch:
    """A tabular mini-batch or same-length station segments; y is standardized."""
    x: torch.Tensor
    x_static: torch.Tensor | None
    y: torch.Tensor
    row_ids: torch.Tensor

    @property
    def inputs(self):
        return (self.x,) if self.x_static is None else (self.x, self.x_static)

    def take(self, idx):
        return Batch(self.x[idx], None if self.x_static is None else self.x_static[idx],
                     self.y[idx], self.row_ids[idx])


@dataclass
class PreparedSplit:
    frame: pd.DataFrame
    role: str
    target_col: str | None
    groups: list[Batch]

    def batches(self, batch_size=32, shuffle=False, generator=None):
        """Yield minibatches; never shuffle dates within a station segment."""
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        order = (torch.randperm(len(self.groups), generator=generator).tolist()
                 if shuffle else range(len(self.groups)))
        for i in order:
            group = self.groups[i]
            n = len(group.x)
            indices = torch.randperm(n, generator=generator) if shuffle else None
            for lo in range(0, n, batch_size):
                yield group.take(slice(lo, lo + batch_size) if indices is None else indices[lo:lo + batch_size])


def _numeric(frame, columns):
    _require_columns(frame, columns)
    arr = frame[list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"Inputs {list(columns)} must be finite (impute features separately).")
    return arr


@dataclass
class Preprocessor:
    """Fixed feature order + affine scaling fitted on training observations only."""
    model_name: str
    feature_option: str
    feature_mean: list
    feature_scale: list
    target_mean: float
    target_scale: float

    @property
    def features(self):
        return (*FEATURE_OPTIONS[self.feature_option], *WRF, *STATIC)

    @property
    def is_db(self):
        return self.model_name.endswith("DB")

    @property
    def source_mean(self):
        return tuple(self.feature_mean[:3]) if self.feature_option != "baseline" else None

    @property
    def source_scale(self):
        return tuple(self.feature_scale[:3]) if self.feature_option != "baseline" else None

    @property
    def model_kwargs(self):
        ns = len(STATIC) if self.is_db else 0
        kwargs = dict(n_features=len(self.features)-ns, n_static=ns,
                      target_mean=self.target_mean, target_scale=self.target_scale,
                      feature_option=self.feature_option)
        if self.model_name.startswith("TwoStage"):
            kwargs.update(source_mean=self.source_mean, source_scale=self.source_scale)
        return kwargs

    @classmethod
    def fit(cls, train_df, model_name, feature_option):
        model = normalize_model_name(model_name)
        if feature_option not in FEATURE_OPTIONS:
            raise ValueError(f"Choose feature_option from {tuple(FEATURE_OPTIONS)}.")
        if model.startswith("TwoStage") and feature_option == "baseline":
            raise ValueError("Two-stage source correction requires the three individual "
                             "CMAQ inputs; a total alone cannot identify its components. "
                             "Use at-bb or at-bb-aod.")
        frame = _validate_frame(train_df)
        features = (*FEATURE_OPTIONS[feature_option], *WRF, *STATIC)
        scaler = StandardScaler().fit(_numeric(frame, features))
        _require_columns(frame, ["pm25_filled"])
        y = frame["pm25_filled"].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("train.pm25_filled must contain finite training labels.")
        target = StandardScaler().fit(y[:, None])
        return cls(model, feature_option, scaler.mean_.tolist(), scaler.scale_.tolist(),
                   float(target.mean_[0]), float(target.scale_[0]))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def transform(self, df, split):
        """Use fixed scalers. Keep missing pm25 rows as sequence context, not labels."""
        if split not in TARGET:
            raise ValueError(f"Unknown split {split!r}.")
        frame = _validate_frame(df)
        cols = self.features
        raw = _numeric(frame, cols)
        scaled = (raw - self.feature_mean) / self.feature_scale
        if not np.isfinite(scaled.astype(np.float32)).all():
            raise ValueError("Standardized features overflow float32.")
        x = torch.from_numpy(scaled.astype(np.float32))
        target_col = TARGET[split]
        if target_col is None:
            values = np.full(len(frame), np.nan)
        else:
            _require_columns(frame, [target_col])
            values = frame[target_col].to_numpy(dtype=float)
            if np.isinf(values).any() or (split == "train" and np.isnan(values).any()):
                raise ValueError(f"{target_col} contains invalid labels for {split}.")
        y = torch.from_numpy(((values - self.target_mean) / self.target_scale).astype(np.float32))
        if torch.isinf(y).any():
            raise ValueError("Standardized labels overflow float32.")
        if self.model_name.endswith("MLP"):
            groups = [Batch(x, None, y, torch.arange(len(frame)))]
        else:
            # Keep true calendar spacing: a missing *row* starts a new segment;
            # a present row with a missing *observation* remains sequence context.
            ordered = frame.sort_values(["stcode", "date"])
            same_station = ordered["stcode"].eq(ordered["stcode"].shift())
            next_day = ordered["date"].diff().eq(pd.Timedelta(days=1))
            segment = (~(same_station & next_day)).cumsum()
            by_length = defaultdict(list)
            for _, run in ordered.groupby(segment, sort=False):
                by_length[len(run)].append(run.index.to_numpy())
            groups = []
            if self.is_db and frame.groupby("stcode")[list(STATIC)].nunique().gt(1).any().any():
                raise ValueError("DB static inputs vary within a station; check the seven static columns.")
            for t in sorted(by_length, reverse=True):
                ids = torch.from_numpy(np.stack(by_length[t]))
                groups.append(Batch(x[ids, :-len(STATIC)] if self.is_db else x[ids],
                                    x[ids[:, 0], -len(STATIC):] if self.is_db else None,
                                    y[ids], ids))
        return PreparedSplit(frame, split, target_col, groups)


def self_test():
    """Tiny schema fixture for alignment, affine transforms, gaps and observed masking."""
    frame = pd.DataFrame({"date": ["2024-02-04", "2024-02-01", "2024-02-02"],
                          "stcode": "001", "region5": "R", "pm25_filled": [8., 5., 7.]})
    for i, col in enumerate(RAW_FEATURES):
        frame[col] = float(i + 1) if col in STATIC else np.arange(3.) + i
    frame["cmaq_pm25"] = frame[list(SOURCES)].sum(axis=1)
    for model in MODEL_NAMES:
        prep = Preprocessor.fit(frame, model, "at-bb-aod")
        val = frame.assign(pm25=[np.nan, 4., 6.], pm25_filled=1e6)
        split = prep.transform(val, "val")
        finite = []
        for b in split.batches(2):
            finite.extend(b.row_ids[torch.isfinite(b.y)].tolist())
            if prep.is_db:
                assert b.x_static.shape[-1] == 7
        assert sorted(finite) == [1, 2]
        if not model.endswith("MLP"):
            assert sorted(g.x.shape[1] for g in split.groups) == [1, 2]
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "scalers.json"
            prep.save(p)
            assert Preprocessor.load(p) == prep
    # Start with observations only. The imputer callback must see training stations only.
    raw = pd.concat([frame.assign(stcode=f"{i:03}", pm25=[np.nan, 5., 7.])
                     for i in range(10)], ignore_index=True)
    seen = set()

    def fill_demo(training):
        # Test fixture only; use your own validated imputer on real data.
        seen.update(training.stcode)
        return training.assign(pm25_filled=training.pm25.fillna(training.pm25.mean()))

    with TemporaryDirectory() as tmp:
        save_station_splits(raw, tmp, train_imputer=fill_demo)
        train, val, test = [load_split(tmp, name) for name in ("train", "val", "test")]
        validate_disjoint_stations(train, val, test)
        assert seen == set(train.stcode) and train.pm25_filled.notna().all()
        assert "pm25_filled" not in test and val.pm25.isna().any()
        prep = Preprocessor.fit(train, "TwoStageDB", "at-bb")
        assert prep.features[:3] == SOURCES and prep.model_kwargs["n_features"] == 7
    print("Data checks passed.")


if __name__ == "__main__":
    self_test()
