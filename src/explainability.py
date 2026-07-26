"""
Step 6 — Explainability Layer + AI Copilot (Powered by Google Gemini API)

Generates a rule-based reason string for every flagged session (no API
needed), plus an LLM incident narrative + recommended action that
explicitly references the Step 4b prediction. Automatically loads
GEMINI_API_KEY from .env file if present.

Input:  data/features.csv (must have attack_type + predicted_next_action)
Output: data/features.csv (overwritten, with explanation added)
        data/explanations_cache.json (LLM narrative cache, batch mode only)

Run:    python src/explainability.py
"""
import os
import sys
import json
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src directory to sys.path if not present
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from config import PATH_FEATURES, PATH_EXPLANATIONS_CACHE
except ImportError:
    from src.config import PATH_FEATURES, PATH_EXPLANATIONS_CACHE

# Try importing Google Gemini API libraries
HAS_GEMINI = False
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Global flag to avoid repeated failed API retries during batch runs
_API_FAILED = False

PROMPT_TEMPLATE = """You are a SOC analyst assistant. Given structured alert data, write a 3-sentence
incident summary in plain English, then recommend ONE action from this exact list:
[Force Password Reset, Revoke Session Token, Isolate Endpoint, Monitor Only].
Respond ONLY in this JSON format: {{"summary": "...", "recommended_action": "..."}}

entity_id={entity_id}, attack_type={attack_type}, similarity_index={similarity_index}%,
reason={reason}, geo_current={geo_city}/{geo_country}, geo_baseline={home_lat}/{home_lon},
device_current={device_os}, auth_method={auth_method}, cumulative_risk={cumulative_risk_score},
predicted_next_action={predicted_next_action}, prediction_confidence={prediction_confidence}"""


def build_reason(row) -> str:
    reasons = []
    if row.get("geo_distance", 0) > 1000:
        reasons.append("geo-velocity")
    if row.get("is_new_device", 0) == 1:
        reasons.append("new device fingerprint")
    if row.get("auth_fail_streak", 0) >= 5:
        reasons.append("repeated auth failures")
    if row.get("is_new_resource", 0) == 1:
        reasons.append("unusual resource access")
    return "flagged due to " + " + ".join(reasons) if reasons else "flagged due to statistical deviation"


def build_prompt(alert_payload: dict) -> str:
    return PROMPT_TEMPLATE.format(**alert_payload)


def _get_fallback_narrative(alert_payload: dict) -> dict:
    attack_type = alert_payload.get("attack_type") or "anomalous_session"
    entity_id = alert_payload.get("entity_id", "unknown")
    reason = alert_payload.get("reason", "statistical deviation")
    next_action = alert_payload.get("predicted_next_action") or "none"
    similarity = alert_payload.get("similarity_index", 0)

    rec_map = {
        "brute_force": "Force Password Reset",
        "credential_stuffing": "Force Password Reset",
        "impossible_travel": "Revoke Session Token",
        "device_spoofing": "Revoke Session Token",
        "lateral_movement": "Isolate Endpoint",
        "low_and_slow_exfiltration": "Isolate Endpoint",
        "insider_drift": "Monitor Only"
    }
    recommended_action = rec_map.get(str(attack_type), "Monitor Only")

    summary = (
        f"Entity {entity_id} triggered a high-risk security alert ({attack_type}). "
        f"Session behavior showed {reason} with a similarity index of {similarity}%. "
        f"Predicted next attacker action: {next_action}."
    )
    return {"summary": summary, "recommended_action": recommended_action}


def get_llm_narrative(session_id: str, alert_payload: dict) -> dict:
    """
    Fetches LLM narrative via Google Gemini API if GEMINI_API_KEY is active and valid,
    otherwise returns an automated rule-based narrative.
    """
    global _API_FAILED
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if HAS_GEMINI and api_key and api_key.strip() and not api_key.startswith("your_") and not _API_FAILED:
        try:
            prompt = build_prompt(alert_payload)
            import google.generativeai as ggenai
            ggenai.configure(api_key=api_key)
            model = ggenai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            text = response.text.strip()

            if text:
                if "```" in text:
                    lines = text.splitlines()
                    text = "\n".join([l for l in lines if not l.startswith("```")]).strip()
                return json.loads(text)
        except Exception as e:
            _API_FAILED = True
            print(f"Gemini API notice: {e}. Switching to rule-based narrative engine.")

    return _get_fallback_narrative(alert_payload)


def build_alert_payload(row) -> dict:
    return {
        "entity_id": row["entity_id"],
        "attack_type": row["attack_type"],
        "similarity_index": round(row["similarity_index"], 1),
        "reason": build_reason(row),
        "geo_city": row["geo_city"],
        "geo_country": row["geo_country"],
        "home_lat": "n/a",
        "home_lon": "n/a",
        "device_os": row["device_os"],
        "auth_method": row["auth_method"],
        "cumulative_risk_score": round(row.get("cumulative_risk_score", 0), 2),
        "predicted_next_action": row.get("predicted_next_action", "none"),
        "prediction_confidence": row.get("prediction_confidence", 0),
    }


def main():
    features_df = pd.read_csv(PATH_FEATURES, parse_dates=["timestamp"])

    if "attack_type" not in features_df.columns:
        raise RuntimeError("Run classifier.py (Step 5) before explainability.py.")

    flagged = features_df[features_df["attack_type"].notna()].copy()
    flagged["explanation"] = flagged.apply(build_reason, axis=1)

    if "explanation" in features_df.columns:
        features_df = features_df.drop(columns=["explanation"])

    features_df = features_df.merge(
        flagged[["explanation"]], left_index=True, right_index=True, how="left"
    )
    features_df.to_csv(PATH_FEATURES, index=False)

    cache = {}
    if "session_id" in flagged.columns:
        for session_id, row in flagged.set_index("session_id").iterrows():
            payload = build_alert_payload(row)
            cache[str(session_id)] = get_llm_narrative(str(session_id), payload)

    os.makedirs(os.path.dirname(PATH_EXPLANATIONS_CACHE), exist_ok=True)
    with open(PATH_EXPLANATIONS_CACHE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Explanations written to {PATH_FEATURES}")
    print(f"  Flagged rows with reason string: {flagged['explanation'].notna().sum()}")
    print(f"  LLM narratives cached: {len(cache)} -> {PATH_EXPLANATIONS_CACHE}")


if __name__ == "__main__":
    main()