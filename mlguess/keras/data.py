"""Precipitation-type splitting and preprocessing without backend dependencies."""

import logging
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from mlguess.preprocessing import load_preprocessor

logger = logging.getLogger(__name__)


def _calendar_day(value):
    """Normalize dates, timestamps, and ISO strings to their calendar-day key."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Observation and split dates must not be missing")
    return timestamp.date().isoformat()


def load_ptype_uq(conf, data_split=0, verbose=0, drop_mixed=False):
    """Reserve requested test days, then split remaining whole days into train/val.

    Times on the same date remain one group, including ISO timestamps using
    ``T`` as the date/time separator. No held-out rows influence preprocessing.
    ``verbose`` is retained for API compatibility; logging controls verbosity.
    """
    if isinstance(data_split, bool) or not isinstance(data_split, Integral):
        raise ValueError("data_split must be an integer split index")
    n_splits = conf["ensemble"]["n_splits"]
    if not 0 <= data_split < n_splits:
        raise ValueError("data_split must be between zero and ensemble.n_splits - 1")
    df = pd.read_parquet(conf["data_path"])
    if drop_mixed:
        columns = ["ra_percent", "sn_percent", "pl_percent", "fzra_percent"]
        df = df[df[columns].eq(1.0).any(axis=1)].copy()
    qc_value = str(conf["qc"])
    dg = df[(df[f"wetbulb{qc_value}_filter"] == 0.0) & (df["usa"] == 1.0)].copy()
    if dg.empty:
        raise ValueError("No observations remain after precipitation-type quality filters")
    dg["day"] = dg["datetime"].map(_calendar_day)
    dg["id"] = np.arange(len(dg))
    test_days = {_calendar_day(day) for case in conf["case_studies"].values() for day in case}
    cutoff = _calendar_day(conf["test_cutoff"])
    test_condition = dg["day"].isin(test_days) | (dg["day"] >= cutoff)
    trainable = dg[~test_condition].copy()
    test_data = dg[test_condition].copy()
    if trainable["day"].nunique() < 2:
        raise ValueError("At least two non-test calendar days are required for train/validation splitting")
    splitter = GroupShuffleSplit(n_splits=n_splits, random_state=conf["seed"], train_size=conf["train_size1"])
    train_index, valid_index = list(splitter.split(trainable, groups=trainable["day"]))[data_split]
    data = {"train": trainable.iloc[train_index].copy(), "val": trainable.iloc[valid_index].copy(), "test": test_data}
    logger.info("Train, validation, and test fractions: %s", ", ".join(str(len(data[key]) / len(df)) for key in data))
    return data


def preprocess_data(data, input_features, output_features, scaler_type="standard", encoder_type="onehot", groups=None, seed=1000):
    """Fit feature transforms on train only and preserve the declared class axis.

    ``output_features`` specifies all class columns in their fixed order. A
    class absent from the training sample retains its column and numeric label
    in validation/test. This uses the declared schema, never held-out labels to
    fit a class vocabulary. Returned DataFrames preserve source row indices.
    Group scaling applies to standard/min-max/robust transforms; quantile
    scaling retains its per-feature definition.
    """
    if encoder_type not in {"label", "onehot"}:
        raise ValueError("encoder_type must be 'label' or 'onehot'")
    if not input_features or not output_features:
        raise ValueError("input_features and output_features must not be empty")
    if data["train"].empty:
        raise ValueError("Training data must not be empty")
    groups = [] if groups is None else list(groups)
    groupby = bool(groups) and "quantile" not in scaler_type
    if groupby:
        from bridgescaler.group import GroupMinMaxScaler, GroupRobustScaler, GroupStandardScaler
        choices = {
            "normalize": (GroupMinMaxScaler, {}),
            "symmetric": (GroupMinMaxScaler, {"feature_range": (-1, 1)}),
            "standard": (GroupStandardScaler, {}),
            "robust": (GroupRobustScaler, {}),
        }
        if scaler_type not in choices:
            raise ValueError(f"Unsupported grouped scaler: {scaler_type}")
        factory, options = choices[scaler_type]
        input_scaler = factory(**options)
    else:
        params = {"n_quantiles": min(1000, len(data["train"]))} if "quantile" in scaler_type else {}
        input_scaler = load_preprocessor(scaler_type, params, seed)
    train_values = data["train"][input_features]
    if groupby:
        train_scaled = input_scaler.fit_transform(train_values, groups=groups)
    else:
        train_scaled = input_scaler.fit_transform(train_values)
    labels = LabelEncoder().fit(np.arange(len(output_features)))
    scalers = {"input": input_scaler, "output_label": labels}
    if encoder_type == "onehot":
        encoder = OneHotEncoder(categories=[np.arange(len(output_features))], sparse_output=False)
        encoder.fit(np.arange(len(output_features)).reshape(-1, 1))
        scalers["output_onehot"] = encoder
    scaled_data = {}
    for split, frame in data.items():
        if split == "train":
            features = train_scaled
        elif frame.empty:
            features = np.empty((0, len(input_features)))
        else:
            features = input_scaler.transform(frame[input_features])
        scaled_data[f"{split}_x"] = pd.DataFrame(features, columns=input_features, index=frame.index)
        class_ids = np.argmax(frame[output_features].to_numpy(), axis=1)
        class_ids = labels.transform(class_ids)
        if encoder_type == "onehot":
            encoded = encoder.transform(class_ids.reshape(-1, 1)) if len(class_ids) else np.empty((0, len(output_features)))
        else:
            encoded = class_ids
        scaled_data[f"{split}_y"] = encoded
    return scaled_data, scalers
