"""
Step 5 — Attack Classification (Rule Engine)

For sessions flagged by the detection model (anomaly_score > THRESHOLD_FLAG),
assigns an attack_type label using a fixed rule order (first match wins),
per spec Section 7. Only classifies flagged rows — everything below the
threshold stays unclassified (attack_type = None).

Input:  data/features.csv (must already have anomaly_score from Step 4)
Output: data/features.csv (overwritten, with attack_type added)

Run:    python src/classifier.py
"""
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from config import (
        THRESHOLD_FLAG, IMPOSSIBLE_TRAVEL_KM, IMPOSSIBLE_TRAVEL_WINDOW_MIN,
        BRUTE_FORCE_FAIL_STREAK, LATERAL_MOVEMENT_BREADTH, LOW_AND_SLOW_HOURS,
        LOW_AND_SLOW_MAX_DURATION_SEC, INSIDER_DRIFT_MIN_SIMILARITY, PATH_FEATURES,
    )
except ImportError:
    from src.config import (
        THRESHOLD_FLAG, IMPOSSIBLE_TRAVEL_KM, IMPOSSIBLE_TRAVEL_WINDOW_MIN,
        BRUTE_FORCE_FAIL_STREAK, LATERAL_MOVEMENT_BREADTH, LOW_AND_SLOW_HOURS,
        LOW_AND_SLOW_MAX_DURATION_SEC, INSIDER_DRIFT_MIN_SIMILARITY, PATH_FEATURES,
    )



def _time_since_last_session_same_entity(df: pd.DataFrame) -> pd.Series:
    """
    Minutes since this entity's previous row (any row, chronologically).
    Used for the impossible_travel rule. First occurrence per entity gets
    a large sentinel value (999999) so it never triggers impossible_travel.
    """
    df_sorted = df.sort_values(["entity_id", "timestamp"])
    diffs = df_sorted.groupby("entity_id")["timestamp"].diff().dt.total_seconds() / 60.0
    diffs = diffs.fillna(999999)
    return diffs.reindex(df.index)


def classify_row(row) -> str:
    time_since = row.get("time_since_last_min", 999999)  # sentinel for live-scored single rows
    if row["geo_distance"] > IMPOSSIBLE_TRAVEL_KM and time_since < IMPOSSIBLE_TRAVEL_WINDOW_MIN:
        return "impossible_travel"
    if row["auth_fail_streak"] >= BRUTE_FORCE_FAIL_STREAK:
        return "brute_force"
    if row["is_new_device"] == 1:
        return "device_spoofing"
    if row["resource_breadth_1h"] >= LATERAL_MOVEMENT_BREADTH and row["is_new_resource"] == 1:
        return "lateral_movement"
    if (row["timestamp"].hour in LOW_AND_SLOW_HOURS
            and row["session_duration_sec"] < LOW_AND_SLOW_MAX_DURATION_SEC
            and row["is_new_resource"] == 0):
        return "low_and_slow_exfiltration"
    if row["is_new_resource"] == 1 and row["similarity_index"] > INSIDER_DRIFT_MIN_SIMILARITY:
        return "insider_drift"
    return "unclassified_anomaly"


def classify(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()

    if "anomaly_score" not in df.columns:
        raise RuntimeError(
            "anomaly_score column not found — run src/detection_model.py "
            "(Step 4, both --mode train and --mode score) before classifier.py."
        )

    df["time_since_last_min"] = _time_since_last_session_same_entity(df)

    flagged_mask = df["anomaly_score"] > THRESHOLD_FLAG
    df["attack_type"] = None
    df.loc[flagged_mask, "attack_type"] = df.loc[flagged_mask].apply(classify_row, axis=1)

    df = df.drop(columns=["time_since_last_min"])
    return df


def main():
    features_df = pd.read_csv(PATH_FEATURES, parse_dates=["timestamp"])
    df = classify(features_df)
    df.to_csv(PATH_FEATURES, index=False)

    n_flagged = df["attack_type"].notna().sum()
    print(f"Classified {n_flagged} flagged sessions, written to {PATH_FEATURES}")
    print(f"  attack_type distribution:")
    print(df["attack_type"].value_counts(dropna=False))
    print()
    print(f"  Cross-check against ground truth label (for sanity, not final accuracy):")
    print(pd.crosstab(df["label"], df["attack_type"], dropna=False))


if __name__ == "__main__":
    main()