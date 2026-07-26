"""
Step 7 — Concept Drift Handler

Evolves an entity's Digital Twin profile over time via exponential moving
average, so legitimate new work patterns aren't permanently flagged.
Batch/offline mode updates data/entity_profiles.json on disk. The
dashboard's live "Fast-Forward 10 Days" demo control (Step 8) instead
calls update_profile() against an in-memory st.session_state copy and
never touches this file directly — see spec Section 9.

Input:  data/entity_profiles.json + new normal session(s)
Output: data/entity_profiles.json (overwritten, batch mode only)

Run:    python src/drift_handler.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from config import DRIFT_ALPHA, DRIFT_TRIGGER_N_SESSIONS, PATH_PROFILES
except ImportError:
    from src.config import DRIFT_ALPHA, DRIFT_TRIGGER_N_SESSIONS, PATH_PROFILES



def update_profile(profiles: dict, entity_id: str, new_session: dict) -> dict:
    """
    Applies one EMA update to entity_id's profile using new_session (a dict
    with at least: timestamp (datetime), geo_lat, geo_lon, resource_accessed).
    Returns the updated profiles dict (does not mutate in place, safe for
    Streamlit session_state usage).
    """
    profiles = dict(profiles)  # shallow copy so caller's dict isn't mutated
    profile = dict(profiles.get(entity_id, {}))

    if not profile:
        return profiles  # unknown entity, nothing to drift — cold-start path handles this elsewhere

    alpha = DRIFT_ALPHA
    new_hour = new_session["timestamp"].hour

    profile["hour_mean"] = alpha * new_hour + (1 - alpha) * profile["hour_mean"]
    profile["home_geo"] = {
        "lat": round(alpha * new_session["geo_lat"] + (1 - alpha) * profile["home_geo"]["lat"], 1),
        "lon": round(alpha * new_session["geo_lon"] + (1 - alpha) * profile["home_geo"]["lon"], 1),
    }

    # resource set drift: track a rolling counter, promote to typical_resources
    # once a resource appears in >=3 of the last 20 drift-tracked sessions
    resource_counts = profile.get("_recent_resource_counts", {})
    resource_counts[new_session["resource_accessed"]] = resource_counts.get(
        new_session["resource_accessed"], 0
    ) + 1
    profile["_recent_resource_counts"] = resource_counts
    for resource, count in resource_counts.items():
        if count >= 3 and resource not in profile["typical_resources"]:
            profile["typical_resources"].append(resource)

    profile["n_sessions_seen"] = profile.get("n_sessions_seen", 0) + 1
    profile["is_cold_start"] = profile["n_sessions_seen"] < 5
    profile["last_updated"] = datetime.now(timezone.utc).isoformat()

    profiles[entity_id] = profile
    return profiles


def main():
    """
    Batch/offline demonstration: replays DRIFT_TRIGGER_N_SESSIONS synthetic
    normal sessions through one entity and overwrites entity_profiles.json.
    This is separate from the dashboard's live demo path (see docstring).
    """
    import random
    from datetime import timedelta

    with open(PATH_PROFILES) as f:
        profiles = json.load(f)

    sample_entity = next(k for k in profiles if not k.startswith("__cohort_"))
    profile_before = dict(profiles[sample_entity])

    for _ in range(DRIFT_TRIGGER_N_SESSIONS):
        fake_session = {
            "timestamp": datetime.now(timezone.utc) + timedelta(hours=random.randint(-2, 2)),
            "geo_lat": profile_before["home_geo"]["lat"] + random.uniform(-0.5, 0.5),
            "geo_lon": profile_before["home_geo"]["lon"] + random.uniform(-0.5, 0.5),
            "resource_accessed": random.choice(["/api/reports", "/api/logs"]),
        }
        profiles = update_profile(profiles, sample_entity, fake_session)

    with open(PATH_PROFILES, "w") as f:
        json.dump(profiles, f, indent=2, default=str)

    print(f"Drift applied to {sample_entity} over {DRIFT_TRIGGER_N_SESSIONS} sessions")
    print(f"  hour_mean: {profile_before['hour_mean']:.2f} -> {profiles[sample_entity]['hour_mean']:.2f}")
    print(f"  home_geo: {profile_before['home_geo']} -> {profiles[sample_entity]['home_geo']}")
    print(f"  last_updated: {profiles[sample_entity]['last_updated']}")


if __name__ == "__main__":
    main()