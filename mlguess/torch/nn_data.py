"""Station-based PM2.5 data preparation for the PyTorch regression architectures.

Sequences are chronological, per station and per contiguous run. Equal-length
runs share a batch without padding. A row with an unobserved target still
occupies its timestep. Standardization is fitted on the training split only.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


CMAQ_BL_FEATURES = ["cmaq_pm25"]
CMAQ_FEATURES = ["cmaq_pm25_bb", "cmaq_pm25_anthro", "cmaq_pm25_othr"]
AOD_FEATURES = ["aod"]
WRF_FEATURES = ["wrf_pbl", "wrf_temp", "wrf_wspd", "wrf_rh"]
LU_FEATURES = ["population", "elevation", "builtup", "rice", "forest",
               "minor_road_length", "major_road_length"]
KEY_FEATURES = CMAQ_FEATURES + AOD_FEATURES
CONTROL_FEATURES = WRF_FEATURES + LU_FEATURES
FEATURES = KEY_FEATURES + CONTROL_FEATURES
FEATURES_BL = CMAQ_BL_FEATURES + CONTROL_FEATURES
FEATURES_AB = CMAQ_FEATURES + CONTROL_FEATURES
TS_FEATURES = CMAQ_FEATURES + AOD_FEATURES + WRF_FEATURES
STATIC_FEATURES = LU_FEATURES
N_SOURCES, N_FKEY, N_FCONTROL = len(CMAQ_FEATURES), len(KEY_FEATURES), len(CONTROL_FEATURES)
N_FTS, N_FSTATIC = len(TS_FEATURES), len(STATIC_FEATURES)

_MODEL_NAMES = {n.replace("_", ""): n for n in (
    "cnn1d", "db", "two_stage_mlp", "two_stage_cnn1d", "two_stage_db"
)}


def _required(df, names):
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _finite_features(df, names):
    _required(df, names)
    values = df[list(names)].to_numpy(dtype=np.float64, na_value=np.nan)
    invalid = ~np.isfinite(values).all(axis=0)
    if invalid.any():
        bad = [str(n) for n, flag in zip(names, invalid) if flag]
        raise ValueError(f"Nonfinite predictor(s): {bad}. Supply finite predictors or "
                         "apply a predictor imputer fitted on training data only.")
    return values


def _station_map(frame, group_col, region_col):
    _required(frame, [group_col, region_col])
    stations = frame[[group_col, region_col]].drop_duplicates().reset_index(drop=True)
    if stations.isna().any().any():
        raise ValueError("Station and region identifiers may not be missing.")
    if stations[group_col].duplicated().any():
        raise ValueError("Each station must belong to exactly one region.")
    return stations


def stratified_random_sampling(df, test_size, seed=51, min_per_side=1, *,
                               group_col="stcode", region_col="region5"):
    """Split stations using regional largest-remainder quotas; singletons stay in train.

    Subject to the requested minimum per side in splittable regions. The test
    fraction applies to eligible stations only, as in the reference pipeline.
    """
    if not 0 < test_size < 1 or not isinstance(min_per_side, int) or min_per_side < 1:
        raise ValueError("Require 0 < test_size < 1 and a positive integer min_per_side.")
    stations = _station_map(df, group_col, region_col)
    sizes = stations[region_col].value_counts()
    eligible = sizes[sizes >= 2 * min_per_side]
    if eligible.empty:
        raise ValueError("No region has enough distinct stations for this holdout. "
                         "Supply explicit disjoint splits or select a region holdout.")
    total = int(eligible.sum())
    target = int(np.clip(round(total * test_size), min_per_side * len(eligible),
                         total - min_per_side * len(eligible)))
    quotas = eligible / total * target
    alloc = {r: min(max(int(np.floor(quotas[r])), min_per_side), n - min_per_side)
             for r, n in eligible.items()}
    # Choose the largest unmet quota (ties follow the original region order).
    while sum(alloc.values()) < target:
        r = max((r for r in alloc if alloc[r] < eligible[r] - min_per_side),
                key=lambda r: quotas[r] - alloc[r])
        alloc[r] += 1
    while sum(alloc.values()) > target:
        r = max((r for r in alloc if alloc[r] > min_per_side),
                key=lambda r: alloc[r] - quotas[r])
        alloc[r] -= 1
    test_ids = []
    for region, count in alloc.items():
        ids = stations.loc[stations[region_col] == region, group_col]
        _, held = train_test_split(ids, test_size=count, random_state=seed)
        test_ids.extend(held.tolist())
    mask = stations[group_col].isin(test_ids)
    return stations.loc[~mask].reset_index(drop=True), stations.loc[mask].reset_index(drop=True)


def split_pm25_stations(frame, *, split_val=True, val_size=0.2, test_size=0.2,
                        seed=51, group_col="stcode", region_col="region5"):
    """Split all rows of a station together. Validation fraction is of remaining stations."""
    train_st, test_st = stratified_random_sampling(frame, test_size, seed,
                                                   group_col=group_col, region_col=region_col)
    parts = [train_st, test_st]
    if split_val:
        train_st, val_st = stratified_random_sampling(train_st, val_size, seed,
                                                     group_col=group_col, region_col=region_col)
        parts = [train_st, val_st, test_st]
    return tuple(frame.loc[frame[group_col].isin(st[group_col])].copy() for st in parts)


def _read_pm25(path, features):
    frame = pd.read_csv(path, dtype={"stcode": "string"})
    _required(frame, [*features, "region5", "stcode", "date", "pm25"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("Missing dates cannot be assigned to station sequences.")
    frame["year"] = frame["date"].dt.year
    return frame


def _apply_train_imputer(parts, train_imputer):
    if train_imputer is None:
        return parts
    train, *held = parts
    updated = train_imputer(train.copy())
    if not isinstance(updated, pd.DataFrame):
        raise ValueError("train_imputer must return a DataFrame.")
    keys = ["stcode", "date"]
    if not train[keys].reset_index(drop=True).equals(updated[keys].reset_index(drop=True)):
        raise ValueError("train_imputer must retain the training station/date rows in order.")
    return (updated, *held)


def load_pm25_data(pm25_dir, features=FEATURES, split_val=False, val_size=0.2,
                   test_size=0.2, seed=51, *, train_imputer=None):
    """Read a PM2.5 CSV and split by station. A supplied imputer sees only train.

    The attachment's external fill_missing_val module is not part of the
    project. Without a callback, unobserved labels stay NaN and training uses
    an observed-label mask. Validation/test labels are never imputed here.
    """
    frame = _read_pm25(pm25_dir, features)
    parts = split_pm25_stations(frame, split_val=split_val, val_size=val_size,
                                test_size=test_size, seed=seed)
    return _apply_train_imputer(parts, train_imputer)


def load_pm25_data_region_holdout(pm25_dir, features=FEATURES, region="Central",
                                  *, train_imputer=None):
    """Return train and test frames holding out whole regions and all their stations."""
    frame = _read_pm25(pm25_dir, features)
    _station_map(frame, "stcode", "region5")
    held = {region} if isinstance(region, str) else set(region)
    unknown = held - set(frame["region5"])
    if unknown:
        raise ValueError(f"Unknown held-out region(s): {unknown}")
    mask = frame.region5.isin(held)
    if mask.all() or not mask.any():
        raise ValueError("Region holdout must leave both train and test rows.")
    return _apply_train_imputer((frame.loc[~mask].copy(), frame.loc[mask].copy()), train_imputer)


def transform_target_preserve_nan(series, scaler):
    """Standardize observed labels only; missing labels remain at their original positions."""
    values = pd.Series(series).to_numpy(dtype=float, na_value=np.nan).reshape(-1, 1)
    out = np.full(values.shape, np.nan)
    valid = np.isfinite(values[:, 0])
    if valid.any():
        out[valid] = scaler.transform(values[valid])
    return out[:, 0]


class MLPDataset(Dataset):
    """Tabular inputs [N,F] and standardized targets [N,1], with missing labels retained."""

    def __init__(self, X, y, feature_scaler, target_scaler):
        self.X = torch.as_tensor(feature_scaler.transform(X), dtype=torch.float32)
        y_std = transform_target_preserve_nan(np.asarray(y).reshape(-1), target_scaler)
        self.y = torch.as_tensor(y_std[:, None], dtype=torch.float32)
        if len(self.X) != len(self.y):
            raise ValueError("Expected one target per input row.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def _canonical_frame(frame, group_col, date_col):
    _required(frame, [group_col, date_col])
    if frame.empty or frame[group_col].isna().any():
        raise ValueError("A split must contain observations with station identifiers.")
    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    if frame[date_col].isna().any() or frame.duplicated([group_col, date_col]).any():
        raise ValueError("Each station must have unique, nonmissing timestamps.")
    return frame


def prepare_3d_input(df, ts_cols, static_cols=None, target_col="pm25",
                     group_col="stcode", date_col="date", verbose=False, *,
                     cadence="1D"):
    """Bucket per-station sequences by actual length without padding.

    X or X_ts is [N,T,F]; a DB group's X_static is [N,F_static]. y, dates and
    row_positions are [N,T]; row_positions index the input frame by position.
    Missing *labels* do not break a sequence. Missing *timesteps* start a new
    causal run. Set cadence=None only if consecutive rows, regardless of time
    separation, should count as adjacent steps in your application.
    """
    df = _canonical_frame(df, group_col, date_col).reset_index(drop=True)
    ts_cols = tuple(ts_cols)
    static_cols = tuple(static_cols or ())
    _required(df, [target_col, *ts_cols, *static_cols])
    if not ts_cols or len(set(ts_cols + static_cols)) != len(ts_cols + static_cols):
        raise ValueError("Supply distinct temporal and static feature names.")
    _finite_features(df, ts_cols + static_cols)
    step = None if cadence is None else pd.Timedelta(cadence)
    if step is not None and step <= pd.Timedelta(0):
        raise ValueError("cadence must be a positive time interval or None.")
    feature_key = "X_ts" if static_cols else "X"
    groups = defaultdict(lambda: defaultdict(list))
    for station, part in df.sort_values([group_col, date_col]).groupby(group_col, sort=False, observed=True):
        if static_cols and (part[list(static_cols)].nunique(dropna=False) != 1).any():
            raise ValueError(f"Static features vary within station {station!r}.")
        boundaries = part[date_col].diff().ne(step).cumsum() if step is not None else np.zeros(len(part))
        for _, run in part.groupby(boundaries, sort=False):
            group = groups[len(run)]
            group["stations"].append(station)
            group["dates"].append(run[date_col].to_numpy())
            group["row_positions"].append(run.index.to_numpy(dtype=np.int64))
            group["y"].append(run[target_col].to_numpy(dtype=float, na_value=np.nan))
            group[feature_key].append(run[list(ts_cols)].to_numpy(dtype=float))
            if static_cols:
                group["X_static"].append(run[list(static_cols)].iloc[0].to_numpy(dtype=float))
    result = {T: {key: (vals if key == "stations" else np.stack(vals)) for key, vals in group.items()}
              for T, group in groups.items()}
    if verbose:
        for T, group in result.items():
            print(f"T={T}: {len(group['stations'])} sequence(s), {feature_key}={group[feature_key].shape}")
    return result


def groups_to_tensors(groups, mode="db", reverse_sort=True):
    """Convert grouped arrays to (inputs..., targets), keeping the station axis first."""
    mode = mode.lower()
    if mode not in {"db", "cnn1d", "mlp"}:
        raise ValueError("mode must be db, cnn1d or mlp.")
    keys = ("X_ts", "X_static", "y") if mode == "db" else ("X", "y")
    lengths = sorted(groups, reverse=True) if reverse_sort else list(groups)
    return {T: tuple(torch.as_tensor(groups[T][k], dtype=torch.float32) for k in keys) for T in lengths}


def source_sum_scaler_params(source_scaler, target_scaler, n_sources=3):
    """Return training affine constants for summing corrected sources in physical units."""
    if not 1 <= n_sources <= len(source_scaler.scale_) or len(target_scaler.scale_) != 1:
        raise ValueError("Expect the first n_sources input features and one target.")
    return dict(source_mean=np.asarray(source_scaler.mean_[:n_sources]).copy(),
                source_scale=np.asarray(source_scaler.scale_[:n_sources]).copy(),
                target_mean=float(target_scaler.mean_[0]), target_scale=float(target_scaler.scale_[0]))


def inverse_sources(arr_std, scaler, n_total_features):
    """Undo feature standardization for the first K source columns of any [..., K] array."""
    arr = np.asarray(arr_std)
    k = arr.shape[-1]
    if scaler.n_features_in_ != n_total_features or not 1 <= k <= n_total_features:
        raise ValueError("Array/scaler feature widths are incompatible.")
    return arr * scaler.scale_[:k] + scaler.mean_[:k]


@dataclass
class PreparedSplit:
    """Original chronological rows, aligned NumPy groups, and CPU tensors for one split."""

    frame: pd.DataFrame
    groups: dict
    tensors: dict


@dataclass
class PM25Preprocessor:
    """Fitted training transforms. transform() is also suitable for unlabeled inference."""

    model_name: str
    ts_cols: tuple
    static_cols: tuple
    source_names: tuple
    target_col: str
    group_col: str
    date_col: str
    cadence: object
    feature_scaler: StandardScaler
    static_scaler: StandardScaler | None
    target_scaler: StandardScaler

    def save(self, path):
        """Write the input schema and fitted training statistics to JSON, without row data."""
        def stats(scaler):
            if scaler is None:
                return None
            return {k: np.asarray(getattr(scaler, k)).tolist()
                    for k in ("mean_", "scale_", "var_", "n_samples_seen_")}

        schema = {k: getattr(self, k) for k in (
            "model_name", "ts_cols", "static_cols", "source_names", "target_col", "group_col", "date_col")}
        schema["cadence"] = None if self.cadence is None else str(pd.Timedelta(self.cadence))
        payload = {"format": "miles-guess-pm25-v1", "schema": schema,
                   "scalers": {k: stats(getattr(self, k)) for k in
                               ("feature_scaler", "static_scaler", "target_scaler")}}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @staticmethod
    def load(path):
        """Restore inference transforms; no original observations or refitting are needed."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "miles-guess-pm25-v1":
            raise ValueError("Unsupported PM2.5 preprocessor format.")
        schema = data["schema"]
        for k in ("ts_cols", "static_cols", "source_names"):
            schema[k] = tuple(schema[k])

        def restore(stats, width):
            if not width and stats is None:
                return None
            if not width or stats is None:
                raise ValueError("Scaler and schema disagree about the input width.")
            scaler = StandardScaler()
            for attr in ("mean_", "scale_", "var_"):
                value = np.asarray(stats[attr], dtype=float)
                if value.shape != (width,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid fitted scaler statistic: {attr}")
                setattr(scaler, attr, value)
            if np.any(scaler.scale_ <= 0) or np.any(scaler.var_ < 0):
                raise ValueError("Invalid fitted scaler scale or variance.")
            scaler.n_features_in_ = width
            scaler.n_samples_seen_ = int(stats["n_samples_seen_"])
            return scaler

        widths = {"feature_scaler": len(schema["ts_cols"]), "static_scaler": len(schema["static_cols"]),
                  "target_scaler": 1}
        return PM25Preprocessor(**schema, **{k: restore(data["scalers"][k], n) for k, n in widths.items()})

    def transform(self, frame, *, training_target_col=None):
        frame = _canonical_frame(frame, self.group_col, self.date_col)
        frame = frame.sort_values([self.group_col, self.date_col]).reset_index(drop=True)
        if self.target_col not in frame:
            frame[self.target_col] = np.nan
        label = self.target_col if training_target_col is None else training_target_col
        _required(frame, [label])
        X = self.feature_scaler.transform(_finite_features(frame, self.ts_cols))
        y = transform_target_preserve_nan(frame[label], self.target_scaler)
        if self.model_name == "two_stage_mlp":
            groups = {1: {"X": X, "y": y[:, None], "stations": frame[self.group_col].tolist(),
                          "dates": frame[self.date_col].to_numpy(), "row_positions": np.arange(len(frame))}}
            mode = "mlp"
        else:
            scaled = frame.copy()
            scaled[list(self.ts_cols)] = X
            scaled[self.target_col] = y
            if self.static_cols:
                scaled[list(self.static_cols)] = self.static_scaler.transform(_finite_features(frame, self.static_cols))
            groups = prepare_3d_input(scaled, self.ts_cols, self.static_cols, self.target_col,
                                      self.group_col, self.date_col, cadence=self.cadence)
            mode = "db" if self.static_cols else "cnn1d"
        return PreparedSplit(frame, groups, groups_to_tensors(groups, mode))


@dataclass
class PreparedPM25(PM25Preprocessor):
    train: PreparedSplit
    validation: PreparedSplit
    test: PreparedSplit


def prepare_pm25(train, validation, test, model_name, *, ts_cols=None, static_cols=None,
                 source_names=None, target_col="pm25", group_col="stcode", date_col="date",
                 cadence="1D", training_target_col=None):
    """Fit train-only scalers and prepare all three disjoint station splits.

    CNNs and the tabular model receive all predictors. DBs receive temporal
    and static predictors as separate inputs. In every two-stage architecture,
    the source columns must be the first temporal/input features, in source_names
    order. If an external training-only imputer supplies a separate label column,
    name it in training_target_col; held-out labels always use target_col.
    """
    key = str(model_name).replace("_", "").lower()
    if key not in _MODEL_NAMES:
        raise ValueError(f"Unknown model_name: {model_name!r}")
    name = _MODEL_NAMES[key]
    dual = name in {"db", "two_stage_db"}
    if ts_cols is None:
        ts_cols = TS_FEATURES if dual else FEATURES
    ts_cols = tuple(ts_cols)
    static_cols = tuple((STATIC_FEATURES if dual else ()) if static_cols is None else static_cols)
    if not dual:
        # Single-branch models consume static predictors as ordinary repeated channels.
        ts_cols += static_cols
        static_cols = ()
    features = ts_cols + static_cols
    if not ts_cols or (dual and not static_cols) or len(set(features)) != len(features):
        raise ValueError("Expected distinct feature names and nonempty inputs for every branch.")
    if set(features).intersection({target_col, training_target_col, group_col, date_col}):
        raise ValueError("Targets, station IDs and timestamps cannot be predictors.")
    sources = tuple(CMAQ_FEATURES if source_names is None else source_names)
    if name.startswith("two_stage"):
        if not sources or tuple(ts_cols[:len(sources)]) != sources:
            raise ValueError("The first input features must be the declared source columns, in order.")
    elif not set(sources).issubset(ts_cols):
        sources = ()
    for df in (train, validation, test):
        _required(df, [group_col, date_col])
    sets = [set(df[group_col]) for df in (train, validation, test)]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Expected station-disjoint train, validation and test splits.")
    label = target_col if training_target_col is None else training_target_col
    _required(train, [label])
    _required(validation, [target_col])
    y = train[label].to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(y)
    if not finite.any():
        raise ValueError("Training needs at least one finite target.")
    processor = PM25Preprocessor(name, ts_cols, static_cols, sources, target_col,
                                 group_col, date_col, cadence,
                                 StandardScaler().fit(_finite_features(train, ts_cols)),
                                 StandardScaler().fit(_finite_features(train, static_cols)) if dual else None,
                                 StandardScaler().fit(y[finite, None]))
    return PreparedPM25(**vars(processor),
                        train=processor.transform(train, training_target_col=training_target_col),
                        validation=processor.transform(validation), test=processor.transform(test))


def collect_predictions(model, split, prepared, *, batch_size=128, device=None):
    """Collect physical-unit predictions and, for two-stage models, corrected source columns."""
    from .nn_training import predict_frame
    return predict_frame(model, split, prepared, batch_size=batch_size, device=device)
