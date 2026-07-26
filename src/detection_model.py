"""
Step 4 — Detection Model

Trains an Isolation Forest ONLY on label=='normal' rows (per spec — never
fits on attack data, since real intrusion is a tiny minority). Produces a
normalized anomaly_score (0-1) per session, dampened for cold-start entities,
plus a cumulative session-level "Cyber Health Meter" risk score.

Input:  data/features.csv
Output: data/features.csv (overwritten, with anomaly_score + cumulative_risk_score added)
        models/isolation_forest.pkl

Run:    python src/detection_model.py --mode train
        python src/detection_model.py --mode score
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from config import ISOLATION_FOREST_PARAMS, COLD_START_DAMPENING, PATH_FEATURES, PATH_MODEL
except ImportError:
    from src.config import ISOLATION_FOREST_PARAMS, COLD_START_DAMPENING, PATH_FEATURES, PATH_MODEL


# Columns fed into the model — purely the engineered deviation features,
# never raw identifiers like entity_id/source_ip.
FEATURE_COLUMNS = [
    "hour_deviation",
    "geo_distance",
    "is_new_resource",
    "is_new_device",
    "is_new_auth_method",
    "duration_zscore",
    "auth_fail_streak",
    "resource_breadth_1h",
]


def train(features_df: pd.DataFrame) -> IsolationForest:
    normal_df = features_df[features_df["label"] == "normal"]
    X_train = normal_df[FEATURE_COLUMNS].values

    model = IsolationForest(**ISOLATION_FOREST_PARAMS)
    model.fit(X_train)

    with open(PATH_MODEL, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained on {len(X_train)} normal sessions, saved to {PATH_MODEL}")
    return model


def score(features_df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    df = features_df.copy()
    X = df[FEATURE_COLUMNS].values

    raw_score = model.decision_function(X)  # higher raw_score = more normal in sklearn's convention
    # normalize to 0-1, flip so higher = more anomalous
    normalized = 1 - (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
    df["anomaly_score"] = normalized

    # cold-start dampening
    df.loc[df["is_cold_start"], "anomaly_score"] *= COLD_START_DAMPENING

    # cumulative session risk score ("Cyber Health Meter")
    df = df.sort_values(["session_id", "timestamp"]) if "session_id" in df.columns else df.sort_values("timestamp")
    # group sequential actions by entity_id (proxy for "session" since session_id in
    # this dataset is currently 1 row = 1 session; group by entity_id + calendar day
    # to approximate a real session's action sequence)
    df["session_group"] = df["entity_id"] + "_" + df["timestamp"].dt.date.astype(str)

    cumulative = []
    for _, group in df.groupby("session_group"):
        group = group.sort_values("timestamp")
        cum_risk = None
        for score_val in group["anomaly_score"]:
            cum_risk = score_val if cum_risk is None else 1 - (1 - score_val) * (1 - cum_risk)
            cumulative.append(cum_risk)
    # reattach in the same row order as the groupby iteration produced them
    df = df.sort_values(["session_group", "timestamp"]).reset_index(drop=True)
    df["cumulative_risk_score"] = cumulative

    df.to_csv(PATH_FEATURES, index=False)
    print(f"Scored {len(df)} rows, written to {PATH_FEATURES}")
    print(f"  anomaly_score range: {df['anomaly_score'].min():.3f} - {df['anomaly_score'].max():.3f}")
    print(f"  mean anomaly_score by label:")
    print(df.groupby("label")["anomaly_score"].mean().sort_values())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "score"], required=True)
    args = parser.parse_args()

    features_df = pd.read_csv(PATH_FEATURES, parse_dates=["timestamp"])

    if args.mode == "train":
        train(features_df)
    else:
        with open(PATH_MODEL, "rb") as f:
            model = pickle.load(f)
        score(features_df, model)


if __name__ == "__main__":
    main()