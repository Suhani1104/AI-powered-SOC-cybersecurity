"""
Step 3 — Feature Engineering

Converts each raw session into numeric features measuring deviation from the
entity's Digital Twin baseline (see baseline_profiler.py), plus a single
Similarity Index (0-100%, higher = more normal) used in the dashboard's
Digital Twin comparison table.

Input:  data/access_logs_full.csv (or _inference.csv) + data/entity_profiles.json
Output: data/features.csv

Run:    python src/feature_engineering.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from data_utils import load_logs
    from baseline_profiler import get_profile
    from config import (
        SIMILARITY_WEIGHTS, GEO_NORM_KM, HOUR_NORM_STD, DURATION_NORM_Z,
        AUTH_FAIL_WINDOW_MIN, RESOURCE_BREADTH_WINDOW_MIN,
        PATH_LOGS_FULL, PATH_PROFILES, PATH_FEATURES,
    )
except ImportError:
    from src.data_utils import load_logs
    from src.baseline_profiler import get_profile
    from src.config import (
        SIMILARITY_WEIGHTS, GEO_NORM_KM, HOUR_NORM_STD, DURATION_NORM_Z,
        AUTH_FAIL_WINDOW_MIN, RESOURCE_BREADTH_WINDOW_MIN,
        PATH_LOGS_FULL, PATH_PROFILES, PATH_FEATURES,
    )



def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points."""
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [lat1, lon1, lat2, lon2]):
        return 0.0
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _compute_auth_fail_streak(df: pd.DataFrame) -> pd.Series:
    """
    For each row: count of auth_success==False events for this
    (entity_id, source_ip) pair within the trailing AUTH_FAIL_WINDOW_MIN
    minutes (inclusive of the current row). Manual sliding window per group
    for correctness and clarity at this dataset size.
    """
    result = pd.Series(0, index=df.index, dtype=int)
    window = pd.Timedelta(minutes=AUTH_FAIL_WINDOW_MIN)

    for _, group in df.groupby(["entity_id", "source_ip"]):
        group = group.sort_values("timestamp")
        timestamps = group["timestamp"].tolist()
        fails = (~group["auth_success"].astype(bool)).tolist()
        idxs = group.index.tolist()

        fail_window = []  # timestamps of failures currently inside the trailing window
        for i, ts in enumerate(timestamps):
            # drop failures that have aged out of the window
            fail_window = [t for t in fail_window if ts - t <= window]
            if fails[i]:
                fail_window.append(ts)
            result.loc[idxs[i]] = len(fail_window)

    return result


def _compute_resource_breadth(df: pd.DataFrame) -> pd.Series:
    """
    For each row: count of DISTINCT resources accessed by this entity_id
    within the trailing RESOURCE_BREADTH_WINDOW_MIN minutes (inclusive).
    """
    result = pd.Series(0, index=df.index, dtype=int)
    window = pd.Timedelta(minutes=RESOURCE_BREADTH_WINDOW_MIN)

    for _, group in df.groupby("entity_id"):
        group = group.sort_values("timestamp")
        timestamps = group["timestamp"].tolist()
        resources = group["resource_accessed"].tolist()
        idxs = group.index.tolist()

        buffer = []  # list of (timestamp, resource) inside the trailing window
        for i, ts in enumerate(timestamps):
            buffer = [(t, r) for (t, r) in buffer if ts - t <= window]
            buffer.append((ts, resources[i]))
            result.loc[idxs[i]] = len({r for (_, r) in buffer})

    return result


def compute_features(logs_df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    df = logs_df.copy()

    # --- per-row deviation features that only need the entity's own profile ---
    hour_devs, geo_dists = [], []
    is_new_resource, is_new_device, is_new_auth = [], [], []
    duration_z, is_cold_start = [], []

    for _, row in df.iterrows():
        profile = get_profile(row["entity_id"], profiles, row["entity_type"])

        hour_dev = abs(row["timestamp"].hour - profile["hour_mean"]) / max(profile["hour_std"], 1)
        geo_dist = haversine_km(
            row["geo_lat"], row["geo_lon"],
            profile["home_geo"]["lat"], profile["home_geo"]["lon"],
        )
        new_resource = int(row["resource_accessed"] not in profile["typical_resources"])
        new_device = int(row["device_mac"] != profile["typical_device_mac"])
        new_auth = int(row["auth_method"] != profile["typical_auth_method"])
        avg_dur = max(profile["avg_session_duration"], 1)
        dur_z = (row["session_duration_sec"] - avg_dur) / max(avg_dur * 0.5, 1)

        hour_devs.append(hour_dev)
        geo_dists.append(geo_dist)
        is_new_resource.append(new_resource)
        is_new_device.append(new_device)
        is_new_auth.append(new_auth)
        duration_z.append(dur_z)
        is_cold_start.append(bool(profile["is_cold_start"]))

    df["hour_deviation"] = hour_devs
    df["geo_distance"] = geo_dists
    df["is_new_resource"] = is_new_resource
    df["is_new_device"] = is_new_device
    df["is_new_auth_method"] = is_new_auth
    df["duration_zscore"] = duration_z
    df["is_cold_start"] = is_cold_start

    # --- trailing-window features (need the whole df, computed per group) ---
    df["auth_fail_streak"] = _compute_auth_fail_streak(df)
    df["resource_breadth_1h"] = _compute_resource_breadth(df)

    # --- Similarity Index (0-100%, higher = more normal) ---------------------
    w = SIMILARITY_WEIGHTS
    penalty = (
        w["hour"] * np.minimum(df["hour_deviation"] / HOUR_NORM_STD, 1)
        + w["geo"] * np.minimum(df["geo_distance"] / GEO_NORM_KM, 1)
        + w["new_resource"] * df["is_new_resource"]
        + w["new_device"] * df["is_new_device"]
        + w["duration"] * np.minimum(df["duration_zscore"].abs() / DURATION_NORM_Z, 1)
    )
    df["similarity_index"] = 100 * (1 - np.minimum(1, penalty))

    return df


def main():
    logs_df = load_logs(PATH_LOGS_FULL)
    with open(PATH_PROFILES) as f:
        profiles = json.load(f)

    features_df = compute_features(logs_df, profiles)
    features_df.to_csv(PATH_FEATURES, index=False)

    print(f"Features written to {PATH_FEATURES}")
    print(f"  Rows: {len(features_df)}")
    print(f"  NaN check: {features_df.isna().sum().sum()} total NaNs (should be 0)")
    print(f"  similarity_index range: {features_df['similarity_index'].min():.1f} - {features_df['similarity_index'].max():.1f}")
    print(f"  similarity_index by label (mean):")
    print(features_df.groupby("label")["similarity_index"].mean().sort_values())


if __name__ == "__main__":
    main()