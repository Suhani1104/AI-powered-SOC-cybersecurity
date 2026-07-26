"""
Central configuration. Every threshold, weight, and magic number used across
the pipeline lives here — no module hardcodes its own constants. Tune here
to adjust the whole system (and the live demo) in one place.
"""

# --- Step 2: Baseline Profiler --------------------------------------------
COLD_START_MIN_SESSIONS = 5

# --- Step 3: Feature Engineering / Similarity Index ------------------------
SIMILARITY_WEIGHTS = {
    "hour": 0.25,
    "geo": 0.30,
    "new_resource": 0.15,
    "new_device": 0.15,
    "duration": 0.15,
}
GEO_NORM_KM = 5000          # distance (km) that normalizes to "fully anomalous" geo deviation
HOUR_NORM_STD = 3           # hour_deviation normalizes to 1.0 at this many std-devs
DURATION_NORM_Z = 3         # duration_zscore normalizes to 1.0 at this many z-score units
AUTH_FAIL_WINDOW_MIN = 5    # trailing window for auth_fail_streak
RESOURCE_BREADTH_WINDOW_MIN = 60  # trailing window for resource_breadth_1h

# --- Step 4: Detection Model -----------------------------------------------
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.02,   # matches Step 1's ANOMALY_RATE
    "random_state": 42,
}
COLD_START_DAMPENING = 0.7   # multiplier applied to anomaly_score for cold-start entities

# --- Step 5: Classification --------------------------------------------------
THRESHOLD_FLAG = 0.6         # anomaly_score above this gets classified + surfaced as an alert
IMPOSSIBLE_TRAVEL_KM = 3000
IMPOSSIBLE_TRAVEL_WINDOW_MIN = 60
BRUTE_FORCE_FAIL_STREAK = 10
LATERAL_MOVEMENT_BREADTH = 4
LOW_AND_SLOW_HOURS = [0, 1, 2, 3, 4, 23]
LOW_AND_SLOW_MAX_DURATION_SEC = 150
INSIDER_DRIFT_MIN_SIMILARITY = 70

# --- Step 4b: Attack Progression Predictor ----------------------------------
ATTACK_PROGRESSION_MAP = {
    "brute_force": ["credential_stuffing", "lateral_movement"],
    "credential_stuffing": ["impossible_travel", "lateral_movement"],
    "impossible_travel": ["device_spoofing", "lateral_movement"],
    "device_spoofing": ["lateral_movement", "low_and_slow_exfiltration"],
    "lateral_movement": ["low_and_slow_exfiltration"],
    "low_and_slow_exfiltration": ["low_and_slow_exfiltration"],
    "insider_drift": [],
}

# --- Step 7: Concept Drift ---------------------------------------------------
DRIFT_ALPHA = 0.2
DRIFT_TRIGGER_N_SESSIONS = 50

# --- File paths (relative to project root) ----------------------------------
PATH_LOGS_FULL = "data/access_logs_full.csv"
PATH_LOGS_INFERENCE = "data/access_logs_inference.csv"
PATH_GROUND_TRUTH = "data/ground_truth_labels.csv"
PATH_PROFILES = "data/entity_profiles.json"
PATH_FEATURES = "data/features.csv"
PATH_MODEL = "models/isolation_forest.pkl"
PATH_EXPLANATIONS_CACHE = "data/explanations_cache.json"
PATH_EVAL_METRICS = "outputs/evaluation_metrics.txt"