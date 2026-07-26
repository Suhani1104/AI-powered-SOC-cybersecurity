"""
Step 4b — Predictive Attack Progression

Given the attack type observed for a flagged session, predicts the
attacker's likely NEXT action using a fixed kill-chain lookup table (not a
trained model — honest choice given no real attack-chain history exists to
train a sequence model on; see spec Section 6b and the report's limitations
section).

Input:  data/features.csv (must already have attack_type from Step 5 —
        NOTE: run this AFTER classifier.py, see run order below)
Output: data/features.csv (overwritten, with predicted_next_action +
        prediction_confidence + prediction_alternatives added)

Run:    python src/attack_predictor.py
"""
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from config import ATTACK_PROGRESSION_MAP, PATH_FEATURES
except ImportError:
    from src.config import ATTACK_PROGRESSION_MAP, PATH_FEATURES



def predict_next_action(current_attack_type: str) -> dict:
    """
    Returns the predicted next attacker action given the current
    classified attack_type, per the fixed kill-chain map in config.py.
    """
    if pd.isna(current_attack_type):
        return {"predicted_next": None, "confidence": 0.0, "alternatives": []}

    candidates = ATTACK_PROGRESSION_MAP.get(current_attack_type, [])
    if not candidates:
        return {"predicted_next": None, "confidence": 0.0, "alternatives": []}

    return {
        "predicted_next": candidates[0],
        "confidence": 0.75 if len(candidates) == 1 else 0.60,
        "alternatives": candidates[1:],
    }


def apply_predictions(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()

    if "attack_type" not in df.columns:
        raise RuntimeError(
            "attack_type column not found — run src/classifier.py (Step 5) "
            "before running attack_predictor.py (Step 4b)."
        )

    predicted_next, confidences, alternatives = [], [], []
    for attack_type in df["attack_type"]:
        result = predict_next_action(attack_type)
        predicted_next.append(result["predicted_next"])
        confidences.append(result["confidence"])
        alternatives.append("|".join(result["alternatives"]) if result["alternatives"] else "")

    df["predicted_next_action"] = predicted_next
    df["prediction_confidence"] = confidences
    df["prediction_alternatives"] = alternatives

    return df


def main():
    features_df = pd.read_csv(PATH_FEATURES, parse_dates=["timestamp"])
    df = apply_predictions(features_df)
    df.to_csv(PATH_FEATURES, index=False)

    n_predicted = df["predicted_next_action"].notna().sum()
    print(f"Predictions written to {PATH_FEATURES}")
    print(f"  Rows with a predicted next action: {n_predicted}")
    print(f"  Prediction distribution:")
    print(df["predicted_next_action"].value_counts(dropna=False))


if __name__ == "__main__":
    main()