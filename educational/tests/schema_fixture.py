"""Small deterministic QA fixture, not an experimental PM2.5 dataset."""
from pathlib import Path

import numpy as np
import pandas as pd


def station_fixture(split, *, shuffle=True):
    """Return 3 stations of unequal length, one date gap and missing observations."""
    stations = {"train": (0, 1, 2), "val": (3, 4, 5), "test": (6, 7, 8)}[split]
    rows = []
    static_cols = ["population", "elevation", "builtup", "rice", "forest",
                   "minor_road_length", "major_road_length"]
    for j, station in enumerate(stations):
        for k in range(6 + j):
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=k + int(k >= 3 and j == 0))
            sources = [5 + 0.2*station + 0.4*k, 7 + 0.25*station + k%3, 4 + 0.2*k]
            observation = sum(sources) * 0.92 + np.sin(k / 3) + j / 2
            row = dict(date=date.strftime("%Y-%m-%d"), stcode=f"S{station:03}", region5=f"region{j%2}",
                       cmaq_pm25=sum(sources), cmaq_pm25_bb=sources[0],
                       cmaq_pm25_anthro=sources[1], cmaq_pm25_othr=sources[2],
                       aod=0.11 + 0.07*k, wrf_pbl=11 + 0.5*k, wrf_temp=24.5 + 0.1*k,
                       wrf_wspd=4 + np.cos(k / 7), wrf_rh=62 + k)
            row.update({c: float((i + 1) * (station + 2)) for i, c in enumerate(static_cols)})
            if split == "train":
                row.update(pm25_filled=observation, pm25_fill_n_neighbors=4.0 if k == 2 else np.nan)
            else:
                row["pm25"] = np.nan if (j == 0 or k == 1) else observation
            rows.append(row)
    frame = pd.DataFrame(rows)
    assert frame.shape[1] == (21 if split == "train" else 20)
    return frame.sample(frac=1, random_state=77).reset_index(drop=True) if shuffle else frame


def write_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        path = directory / f"{split}.csv"
        if path.exists():
            raise FileExistsError(f"Refusing to replace a dataset: {path}")
        station_fixture(split).to_csv(path, index=False)
    return directory
