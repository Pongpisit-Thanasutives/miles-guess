"""Stage 1: split raw observations by station or audit existing CSV splits."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nn_data import SOURCES, RAW_FEATURES, load_split, save_station_splits, validate_disjoint_stations


def audit_splits(directory):
    """Check split membership/schema and report what fraction has a usable target."""
    directory = Path(directory)
    roles = ("train", "val", "test")
    frames = [load_split(directory, role) for role in roles]
    validate_disjoint_stations(*frames)
    report = {}
    for role, frame in zip(roles, frames):
        missing = [c for c in RAW_FEATURES if c not in frame]
        if missing:
            raise ValueError(f"{role}: missing schema columns: {missing}")
        target = "pm25_filled" if role == "train" else "pm25"
        y = frame[target].to_numpy(dtype=float)
        if np.isinf(y).any() or (role == "train" and not np.isfinite(y).all()):
            raise ValueError(f"{role}: invalid values in {target}")
        if not np.isfinite(y).any():
            raise ValueError(f"{role}: no usable values in {target}")
        with (directory / f"{role}.csv").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report[role] = {"rows": len(y), "stations": frame.stcode.nunique(),
                        "date_min": str(frame.date.min().date()), "date_max": str(frame.date.max().date()),
                        "target": target, "finite_targets": int(np.isfinite(y).sum()), "sha256": digest}
        # Audit the user-specified CMAQ identity without altering any raw values.
        raw = frame[["cmaq_pm25", *SOURCES]].to_numpy(dtype=float)
        clean = raw[np.isfinite(raw).all(axis=1)]
        if len(clean):
            total, source_sum = clean[:, 0], clean[:, 1:].sum(axis=1)
            report[role]["cmaq_accounting"] = {
                "comparable_rows": len(clean),
                "max_abs_difference": float(abs(total - source_sum).max()),
                "outside_tolerance": int((~np.isclose(total, source_sum, rtol=1.e-5, atol=1.e-4)).sum())
            }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("audit", help="Check the three existing CSVs without changing them.")
    check.add_argument("directory", type=Path)
    split = sub.add_parser("create", help="Create train.csv, val.csv, test.csv from raw observations.")
    split.add_argument("csv", type=Path)
    split.add_argument("directory", type=Path)
    split.add_argument("--test-size", type=float, default=.2)
    split.add_argument("--val-size", type=float, default=.2, help="Fraction of the remaining stations.")
    split.add_argument("--seed", type=int, default=51)
    split.add_argument("--test-region", action="append", help="Hold out this entire region; repeat as needed.")
    split.add_argument("--train-imputer", metavar="MODULE:FUNCTION",
                       help="Your importable function: train dataframe -> dataframe with pm25_filled.")
    args = parser.parse_args(argv)
    if args.action == "create":
        imputer = None
        if args.train_imputer:
            module, separator, name = args.train_imputer.partition(":")
            if not separator or not module or not name:
                parser.error("--train-imputer must be MODULE:FUNCTION")
            imputer = getattr(importlib.import_module(module), name)
        raw = pd.read_csv(args.csv, dtype={"stcode": str, "region5": str})
        save_station_splits(raw, args.directory, test_size=args.test_size, val_size=args.val_size,
                            seed=args.seed, test_regions=args.test_region, train_imputer=imputer)
    print(json.dumps(audit_splits(args.directory), indent=2))


def self_test():
    from tempfile import TemporaryDirectory
    from .verification import mock_station_frame
    with TemporaryDirectory() as tmp:
        path = Path(tmp)
        for role, station in (("train", "A"), ("val", "B"), ("test", "C")):
            mock_station_frame(station, role).to_csv(path / f"{role}.csv", index=False)
        report = audit_splits(path)
        assert report["train"]["finite_targets"] == 5 and report["test"]["finite_targets"] == 3


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        main()
