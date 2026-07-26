"""
Step 2 — Baseline Profiler (THE ADAPTIVE DIGITAL TWIN)

Builds a "normal behavior" profile per entity_id from labeled-normal sessions
only (never fits on attack data). Handles the cold-start problem via cohort
(entity_type-level) fallback profiles for entities with too little history.

Input:  data/access_logs_full.csv
Output: data/entity_profiles.json

Run:    python src/baseline_profiler.py
"""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add src directory to sys.path if not present
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from data_utils import load_logs
    from config import COLD_START_MIN_SESSIONS, PATH_LOGS_FULL, PATH_PROFILES
except ImportError:
    from src.data_utils import load_logs
    from src.config import COLD_START_MIN_SESSIONS, PATH_LOGS_FULL, PATH_PROFILES


INPUT_PATH = PATH_LOGS_FULL
OUTPUT_PATH = PATH_PROFILES


def _mode(series: pd.Series):
    """Most frequent value in a series; falls back to first value on ties."""
    counts = Counter(series.dropna())
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _top_resources_90pct(series: pd.Series) -> list:
    """Smallest set of resources covering >=90% of this entity's accesses."""
    counts = series.value_counts()
    if counts.empty:
        return []
    total = counts.sum()
    cumulative = 0
    selected = []
    for resource, count in counts.items():
        selected.append(resource)
        cumulative += count
        if cumulative / total >= 0.90:
            break
    return selected


def _profile_from_group(entity_id: str, entity_type: str, group: pd.DataFrame) -> dict:
    hours = group["timestamp"].dt.hour
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "n_sessions_seen": int(len(group)),
        "hour_mean": float(hours.mean()),
        "hour_std": float(hours.std()) if len(group) > 1 else 1.0,
        "home_geo": {
            "city": _mode(group["geo_city"]) if "geo_city" in group.columns else "Unknown",
            "country": _mode(group["geo_country"]) if "geo_country" in group.columns else "Unknown",
            "lat": round(float(group["geo_lat"].mean()), 1),
            "lon": round(float(group["geo_lon"].mean()), 1),
        },
        "typical_resources": _top_resources_90pct(group["resource_accessed"]),
        "typical_auth_method": _mode(group["auth_method"]),
        "typical_device_mac": _mode(group["device_mac"]),
        "typical_device_os": _mode(group["device_os"]),
        "avg_session_duration": float(group["session_duration_sec"].mean()),
        "is_cold_start": False,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def build_profiles(logs_df: pd.DataFrame) -> dict:
    """
    Builds the full profile dict: one entry per entity_id that has enough
    history to be "warm", plus one cohort fallback entry per entity_type
    for cold-start entities. Only fits on label == 'normal' rows so attack
    sessions never poison the baseline.
    """
    normal_df = logs_df[logs_df["label"] == "normal"].copy()
    profiles = {}

    # --- per-entity profiles -------------------------------------------------
    for entity_id, group in normal_df.groupby("entity_id"):
        entity_type = group["entity_type"].iloc[0]
        profile = _profile_from_group(entity_id, entity_type, group)
        if profile["n_sessions_seen"] < COLD_START_MIN_SESSIONS:
            profile["is_cold_start"] = True
        profiles[entity_id] = profile

    # --- cohort fallback profiles (one per entity_type) -----------------------
    for entity_type, group in normal_df.groupby("entity_type"):
        cohort_key = f"__cohort_{entity_type}__"
        cohort_profile = _profile_from_group(cohort_key, entity_type, group)
        cohort_profile["is_cold_start"] = True  # cohort profiles are always the cold-start fallback
        profiles[cohort_key] = cohort_profile

    return profiles


def get_profile(entity_id: str, profiles: dict, entity_type: str) -> dict:
    """
    Returns the entity's own profile if it exists and is warm (has enough
    history). Otherwise returns the cohort fallback profile for its
    entity_type, tagged is_cold_start=True. This is the single access point
    every downstream module (feature engineering, dashboard) should use
    instead of reading entity_profiles.json directly.
    """
    profile = profiles.get(entity_id)
    if profile is not None and not profile.get("is_cold_start", False):
        return profile

    cohort_key = f"__cohort_{entity_type}__"
    cohort_profile = profiles.get(cohort_key)
    if cohort_profile is not None:
        result = dict(cohort_profile)
        result["is_cold_start"] = True
        result["entity_id"] = entity_id  # report under the real entity_id, not the cohort key
        return result

    # last-resort fallback if even the cohort is missing (shouldn't happen
    # once build_profiles has run on a non-trivial dataset)
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "n_sessions_seen": 0,
        "hour_mean": 12.0,
        "hour_std": 6.0,
        "home_geo": {"city": "Unknown", "country": "Unknown", "lat": 0.0, "lon": 0.0},
        "typical_resources": [],
        "typical_auth_method": None,
        "typical_device_mac": None,
        "typical_device_os": None,
        "avg_session_duration": 300.0,
        "is_cold_start": True,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def main():
    logs_df = load_logs(INPUT_PATH)
    profiles = build_profiles(logs_df)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(profiles, f, indent=2, default=str)

    n_entities = sum(1 for k in profiles if not k.startswith("__cohort_"))
    n_cohorts = sum(1 for k in profiles if k.startswith("__cohort_"))
    n_cold = sum(1 for v in profiles.values() if v["is_cold_start"] and not v["entity_id"].startswith("__cohort_"))

    print(f"Profiles written to {OUTPUT_PATH}")
    print(f"  Entity profiles: {n_entities}")
    print(f"  Cohort fallback profiles: {n_cohorts}")
    print(f"  Cold-start entities (< {COLD_START_MIN_SESSIONS} sessions): {n_cold}")


if __name__ == "__main__":
    main()