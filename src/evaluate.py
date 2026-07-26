"""
evaluate.py — computes real detection/classification metrics for the report.
Run AFTER classifier.py has produced attack_type in features.csv.

Input:  data/features.csv, data/ground_truth_labels.csv
Output: outputs/evaluation_metrics.txt

Run:    python src/evaluate.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from config import THRESHOLD_FLAG, PATH_FEATURES, PATH_GROUND_TRUTH, PATH_EVAL_METRICS
except ImportError:
    from src.config import THRESHOLD_FLAG, PATH_FEATURES, PATH_GROUND_TRUTH, PATH_EVAL_METRICS



def main():
    features_df = pd.read_csv(PATH_FEATURES, parse_dates=["timestamp"])
    ground_truth = pd.read_csv(PATH_GROUND_TRUTH)  # columns: row_id, label

    df = features_df.reset_index().rename(columns={"index": "row_id"})
    df = df.merge(ground_truth, on="row_id", suffixes=("", "_truth"))

    y_true = (df["label_truth"] != "normal").astype(int)
    y_pred = (df["anomaly_score"] > THRESHOLD_FLAG).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # False positive rate at top-1% alert budget
    top1pct_n = max(1, int(len(df) * 0.01))
    top1pct = df.sort_values("anomaly_score", ascending=False).head(top1pct_n)
    fp_rate_top1 = (top1pct["label_truth"] == "normal").mean()

    # Per-attack-type classification accuracy (only rows the model actually flagged)
    flagged = df[df["attack_type"].notna()]
    type_accuracy = (flagged["attack_type"] == flagged["label_truth"]).mean() if len(flagged) else 0.0

    os.makedirs("outputs", exist_ok=True)
    with open(PATH_EVAL_METRICS, "w") as f:
        f.write("=== Detection Accuracy (Imbalanced Labels) ===\n")
        f.write(f"Threshold: anomaly_score > {THRESHOLD_FLAG}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
        f.write(f"Confusion Matrix [[TN,FP],[FN,TP]]:\n{cm}\n\n")

        f.write("=== False Positive Rate at Top-1% Alert Budget ===\n")
        f.write(f"Top {top1pct_n} sessions by anomaly_score\n")
        f.write(f"FP Rate: {fp_rate_top1:.4f}\n\n")

        f.write("=== Attack-Type Classification Accuracy ===\n")
        f.write(f"Accuracy on flagged rows: {type_accuracy:.4f}\n")
        f.write(f"Flagged rows: {len(flagged)}\n")

    print(f"Metrics written to {PATH_EVAL_METRICS}")
    print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} FP@top1%={fp_rate_top1:.4f}")


if __name__ == "__main__":
    main()