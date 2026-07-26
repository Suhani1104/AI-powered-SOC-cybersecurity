# Adaptive Digital Twin — Behavioral Security Copilot

AI-powered behavioral anomaly detection for cybersecurity. Instead of firing isolated alerts from static rules, this system builds a live **digital twin** of each user, device, and service account's normal behavior — then learns, detects deviation, predicts the attacker's next move, explains the alert in plain English, and recommends a response.

**Flow:** `Learning → Deviation → Prediction → Explanation → Recommendation`

---

## What it does

- Learns a behavioral baseline ("digital twin") per entity from access logs — login hours, geo location, typical resources, device fingerprint, auth method
- Detects deviations using an Isolation Forest trained only on normal behavior, producing an anomaly score and a cumulative session-level risk score
- Classifies the deviation into an attack type: brute force, impossible travel, credential stuffing, lateral movement, device spoofing, low-and-slow exfiltration, or insider drift (edge case, used for false-positive tuning)
- Predicts the attacker's likely next action using a kill-chain progression map
- Generates a plain-English incident brief and a recommended SOC action via the Google Gemini API
- Handles cold-start entities (no history yet) via cohort-level fallback baselines
- Handles concept drift (legitimate behavior evolving over time) via exponential moving average updates to each entity's baseline
- Live Streamlit dashboard with an attack simulator to demo detection, prediction, and explanation in real time

---

## Tech stack

Python · pandas · numpy · scikit-learn (Isolation Forest) · Streamlit · Plotly · Google Gemini API · Faker

---

## Project structure

```
project/
├── data/                          # generated at runtime 
├── models/                        # trained model artifact 
├── outputs/                       # evaluation metrics 
├── src/
│   ├── config.py                  # all thresholds/weights/paths, centralized
│   ├── data_utils.py              # shared log-loading helper
│   ├── generate_data.py           # Step 1 — synthetic log + attack generator
│   ├── baseline_profiler.py       # Step 2 — digital twin baseline builder
│   ├── feature_engineering.py     # Step 3 — deviation features + Similarity Index
│   ├── detection_model.py         # Step 4 — Isolation Forest + cumulative risk score
│   ├── classifier.py               # Step 5 — rule-based attack-type classification
│   ├── attack_predictor.py         # Step 4b — kill-chain next-action prediction
│   ├── explainability.py           # Step 6 — reason strings + Gemini incident narrative
│   ├── drift_handler.py            # Step 7 — EMA baseline drift updates
│   └── evaluate.py                 # metrics export (precision/recall/F1/FP-rate)
├── dashboard/
│   └── app.py                      # Step 8 — Streamlit dashboard
├── requirements.txt
├── .env                             # GEMINI_API_KEY 
└── .gitignore
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root :
```
GEMINI_API_KEY=your_key_here
```

---

## Run order

Run every command from the project root.

| Step | Command | Output |
|---|---|---|
| 1 | `python src/generate_data.py` | `data/access_logs_full.csv`, `access_logs_inference.csv`, `ground_truth_labels.csv` |
| 2 | `python src/baseline_profiler.py` | `data/entity_profiles.json` |
| 3 | `python src/feature_engineering.py` | `data/features.csv` |
| 4a | `python src/detection_model.py --mode train` | `models/isolation_forest.pkl` |
| 4b | `python src/detection_model.py --mode score` | `features.csv` updated — anomaly_score, cumulative_risk_score |
| 5 | `python src/classifier.py` | `features.csv` updated — attack_type |
| 6 | `python src/attack_predictor.py` | `features.csv` updated — predicted_next_action, prediction_confidence |
| 7 | `python src/explainability.py` | `features.csv` updated — explanation; `data/explanations_cache.json` |
| 8 | `python src/drift_handler.py` | `entity_profiles.json` updated (baseline drift demo) |
| 9 | `python src/evaluate.py` | `outputs/evaluation_metrics.txt` |
| 10 | `streamlit run dashboard/app.py` | Live dashboard |

---

## Evaluation metrics

Run `python src/evaluate.py` after Step 5 to generate `outputs/evaluation_metrics.txt`, containing:
- Precision, Recall, F1-score at the alert threshold
- Confusion matrix
- False positive rate at the top 1% alert budget
- Attack-type classification accuracy

---

## Known limitations

- Trained and evaluated on synthetic data; real-world adversarial evasion patterns are not represented
- Attack progression prediction is a fixed kill-chain lookup table, not a trained sequence model — this was a deliberate, honest tradeoff given the lack of real historical attack-chain data to train on
- LLM narrative calls add latency; results are cached per session in the dashboard to avoid repeat API calls
- One-click SOC response buttons (Force Password Reset, Revoke Session Token, Isolate Endpoint) are simulated for demo purposes and do not trigger real actions

---

