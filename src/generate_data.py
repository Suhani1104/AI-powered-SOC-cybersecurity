"""
Synthetic Behavioral Access-Log Generator
AI-Powered Behavioral Anomaly Detection for Cybersecurity — Step 1: Data Generation

Generates per-entity "normal" behavioral baselines, then injects labeled attack
patterns (brute force, impossible travel, credential stuffing, lateral movement,
device spoofing, low-and-slow exfiltration, insider drift) at controlled rates.

Output:
- access_logs_full.csv — full dataset with ground-truth label (for training/eval)
- access_logs_inference.csv — same data with label stripped (simulates real deployment)
- ground_truth_labels.csv — ground truth labels indexed by row_id
- digital_twins_baseline.json — baseline entity profiles dictionary
"""

import os
import json
import random
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

# Default Config Constants
SEED = 42
N_USERS = 250
N_SERVICE_ACCOUNTS = 30
N_EDGE_DEVICES = 60

SIM_DAYS = 30
SIM_START = datetime(2026, 6, 1)
SIM_END = SIM_START + timedelta(days=SIM_DAYS)

ANOMALY_RATE = 0.02   # ~2% of total sessions will be anomalous (tune 0.5-3%)

AUTH_METHODS = ["password", "token", "certificate", "biometric"]
RESOURCE_POOL = [f"/api/{r}" for r in [
    "hr-records", "finance-ledger", "customer-db", "billing", "inventory",
    "admin-console", "device-config", "logs", "reports", "user-mgmt",
    "payments", "scada-panel", "camera-feed", "vpn-gateway", "file-share"
]]
OS_LIST = ["Windows11", "Ubuntu22.04", "macOS14", "iOS17", "Android14", "FirmwareV3"]

# Initialize RNG & Faker
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)


def random_geo():
    """Generate random geographical location metadata."""
    return {
        "city": fake.city(),
        "country": fake.country(),
        "lat": float(fake.latitude()),
        "lon": float(fake.longitude())
    }


def make_entities(n, entity_type, prefix):
    """Create entity universe with baselines."""
    entities = []
    for i in range(n):
        home_geo = random_geo()
        entities.append({
            "entity_id": f"{prefix}_{i:04d}",
            "entity_type": entity_type,
            "home_geo": home_geo,
            "home_hour_mean": np.random.uniform(7, 20) if entity_type == "user" else np.random.uniform(0, 24),
            "home_hour_std": np.random.uniform(1, 3),
            "typical_resources": random.sample(RESOURCE_POOL, k=random.randint(3, 7)),
            "typical_auth": random.choice(AUTH_METHODS),
            "device_os": random.choice(OS_LIST),
            "device_mac": fake.mac_address(),
        })
    return entities


# Instantiate entity universe deterministically at module level for import by dashboard/simulators
users = make_entities(N_USERS, "user", "usr")
service_accounts = make_entities(N_SERVICE_ACCOUNTS, "service_account", "svc")
edge_devices = make_entities(N_EDGE_DEVICES, "edge_device", "dev")
all_entities = users + service_accounts + edge_devices
entity_lookup = {e["entity_id"]: e for e in all_entities}


def random_timestamp():

    """Generate a random timestamp within simulation start and end."""
    delta = SIM_END - SIM_START
    return SIM_START + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def sample_normal_session(entity, ts=None):
    """Sample a benign session from entity baseline with realistic noise."""
    if ts is None:
        # bias timestamp toward the entity's typical hour
        day_offset = random.randint(0, SIM_DAYS - 1)
        hour = np.clip(np.random.normal(entity["home_hour_mean"], entity["home_hour_std"]), 0, 23)
        ts = SIM_START + timedelta(days=day_offset, hours=float(hour), minutes=random.randint(0, 59))

    geo = entity["home_geo"]
    resource = random.choice(entity["typical_resources"])
    n_cmds = random.randint(1, 5)
    command_sequence = [f"cmd_{random.randint(1, 20)}" for _ in range(n_cmds)]

    return {
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "timestamp": ts,
        "source_ip": fake.ipv4_public(),
        "geo_city": geo["city"],
        "geo_country": geo["country"],
        "geo_lat": geo["lat"],
        "geo_lon": geo["lon"],
        "resource_accessed": resource,
        "auth_method": entity["typical_auth"],
        "auth_success": True,
        "session_duration_sec": max(5, int(np.random.exponential(600))),
        "command_sequence": "|".join(command_sequence),
        "device_os": entity["device_os"],
        "device_mac": entity["device_mac"],
        "label": "normal",
    }


def random_far_geo(home_geo, min_km=3000):
    """Pick a geo far enough away to be implausible within a short time gap."""
    while True:
        g = random_geo()
        dist = ((g["lat"] - home_geo["lat"])**2 + (g["lon"] - home_geo["lon"])**2) ** 0.5
        if dist > 20:  # implies large real-world distance
            return g


def inject_brute_force(entity, n_attacks=1):
    """Inject brute force authentication attack sessions."""
    rows = []
    for _ in range(n_attacks):
        ts0 = random_timestamp()
        src_ip = fake.ipv4_public()
        n_attempts = random.randint(15, 60)
        for i in range(n_attempts):
            ts = ts0 + timedelta(seconds=i * random.randint(1, 5))
            row = sample_normal_session(entity, ts=ts)
            row["source_ip"] = src_ip
            row["auth_success"] = False
            row["session_duration_sec"] = random.randint(1, 5)
            row["label"] = "brute_force"
            rows.append(row)
    return rows


def inject_impossible_travel(entity, n_events=1):
    """Inject impossible travel sessions."""
    rows = []
    for _ in range(n_events):
        ts1 = random_timestamp()
        row1 = sample_normal_session(entity, ts=ts1)
        row1["label"] = "impossible_travel"
        far_geo = random_far_geo(entity["home_geo"])
        ts2 = ts1 + timedelta(minutes=random.randint(2, 20))  # implausibly short gap
        row2 = sample_normal_session(entity, ts=ts2)
        row2["geo_city"] = far_geo["city"]
        row2["geo_country"] = far_geo["country"]
        row2["geo_lat"] = far_geo["lat"]
        row2["geo_lon"] = far_geo["lon"]
        row2["source_ip"] = fake.ipv4_public()
        row2["label"] = "impossible_travel"
        rows.extend([row1, row2])
    return rows


def inject_credential_stuffing(pool_entities, n_campaigns=1):
    """Inject credential stuffing attack sessions across multiple accounts."""
    rows = []
    for _ in range(n_campaigns):
        ts0 = random_timestamp()
        src_ip = fake.ipv4_public()
        targets = random.sample(pool_entities, k=min(len(pool_entities), random.randint(20, 50)))
        for i, entity in enumerate(targets):
            ts = ts0 + timedelta(seconds=i * random.randint(1, 3))
            row = sample_normal_session(entity, ts=ts)
            row["source_ip"] = src_ip
            row["auth_success"] = random.random() > 0.85  # high failure rate
            row["session_duration_sec"] = random.randint(1, 5)
            row["label"] = "credential_stuffing"
            rows.append(row)
    return rows


def inject_lateral_movement(entity, n_events=1):
    """Inject lateral movement across atypical resources."""
    rows = []
    unused_resources = [r for r in RESOURCE_POOL if r not in entity["typical_resources"]]
    for _ in range(n_events):
        ts0 = random_timestamp()
        breadth = random.randint(4, min(8, len(unused_resources)))
        chosen = random.sample(unused_resources, k=breadth)
        for i, res in enumerate(chosen):
            ts = ts0 + timedelta(minutes=i * random.randint(1, 4))
            row = sample_normal_session(entity, ts=ts)
            row["resource_accessed"] = res
            row["command_sequence"] = "|".join([f"cmd_{random.randint(1, 20)}" for _ in range(random.randint(3, 8))])
            row["label"] = "lateral_movement"
            rows.append(row)
    return rows


def inject_device_spoofing(entity, n_events=1):
    """Inject device fingerprint spoofing sessions."""
    rows = []
    for _ in range(n_events):
        ts = random_timestamp()
        row = sample_normal_session(entity, ts=ts)
        spoof_os = random.choice([o for o in OS_LIST if o != entity["device_os"]])
        row["device_os"] = spoof_os
        row["device_mac"] = fake.mac_address()  # mismatched fingerprint
        row["label"] = "device_spoofing"
        rows.append(row)
    return rows


def inject_low_and_slow(entity, n_events=1, span_days=10):
    """Inject low-and-slow exfiltration over extended period."""
    rows = []
    for _ in range(n_events):
        day0 = random.randint(0, SIM_DAYS - span_days)
        n_touches = random.randint(6, 15)
        for i in range(n_touches):
            day = day0 + int(i * span_days / n_touches)
            off_hour = random.choice([0, 1, 2, 3, 4, 23])  # off-hours
            ts = SIM_START + timedelta(days=day, hours=off_hour, minutes=random.randint(0, 59))
            row = sample_normal_session(entity, ts=ts)
            row["resource_accessed"] = random.choice(entity["typical_resources"])
            row["session_duration_sec"] = random.randint(30, 120)  # short, quiet
            row["label"] = "low_and_slow_exfiltration"
            rows.append(row)
    return rows


def inject_insider_drift(entity, n_events=1):
    """Inject insider drift (ambiguous edge case of slow footprint expansion)."""
    rows = []
    unused_resources = [r for r in RESOURCE_POOL if r not in entity["typical_resources"]]
    for _ in range(n_events):
        ts0 = random_timestamp()
        n_new = random.randint(1, 2)
        chosen = random.sample(unused_resources, k=min(n_new, len(unused_resources)))
        for i, res in enumerate(chosen):
            ts = ts0 + timedelta(days=i * random.randint(1, 3))
            row = sample_normal_session(entity, ts=ts)
            row["resource_accessed"] = res
            row["label"] = "insider_drift"
            rows.append(row)
    return rows


def generate_data(output_dir="data"):
    """Main data generation workflow."""
    os.makedirs(output_dir, exist_ok=True)
    print("Generating entity universe...")

    print(f"Total entities created: {len(all_entities)}")

    # Save digital twins baseline JSON
    baseline_path = os.path.join(output_dir, "digital_twins_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(all_entities, f, indent=4)
    print(f"Saved baseline digital twins profile: {baseline_path}")

    # Generate normal sessions
    print("Generating normal sessions...")
    SESSIONS_PER_ENTITY_PER_DAY = 3
    normal_rows = []
    for entity in all_entities:
        n_sessions = int(SESSIONS_PER_ENTITY_PER_DAY * SIM_DAYS * np.random.uniform(0.5, 1.5))
        for _ in range(n_sessions):
            normal_rows.append(sample_normal_session(entity))

    df_normal = pd.DataFrame(normal_rows)
    print(f"Normal sessions generated: {len(df_normal)}")

    # Inject anomalies
    print("Injecting anomalous session patterns...")
    anomaly_rows = []
    n_normal = len(df_normal)
    target_anomaly_count = int(n_normal * ANOMALY_RATE / (1 - ANOMALY_RATE))

    pattern_weights = {
        "brute_force": 0.15,
        "impossible_travel": 0.15,
        "credential_stuffing": 0.15,
        "lateral_movement": 0.15,
        "device_spoofing": 0.15,
        "low_and_slow_exfiltration": 0.15,
        "insider_drift": 0.10,
    }

    for pattern, weight in pattern_weights.items():
        budget = int(target_anomaly_count * weight)
        generated = 0
        while generated < budget:
            entity = random.choice(all_entities)
            if pattern == "brute_force":
                rows = inject_brute_force(entity, n_attacks=1)
            elif pattern == "impossible_travel":
                rows = inject_impossible_travel(entity, n_events=1)
            elif pattern == "credential_stuffing":
                rows = inject_credential_stuffing(all_entities, n_campaigns=1)
            elif pattern == "lateral_movement":
                rows = inject_lateral_movement(entity, n_events=1)
            elif pattern == "device_spoofing":
                rows = inject_device_spoofing(entity, n_events=1)
            elif pattern == "low_and_slow_exfiltration":
                rows = inject_low_and_slow(entity, n_events=1)
            elif pattern == "insider_drift":
                rows = inject_insider_drift(entity, n_events=1)
            anomaly_rows.extend(rows)
            generated += len(rows)

    df_anomaly = pd.DataFrame(anomaly_rows)
    print(f"Anomalous sessions generated: {len(df_anomaly)}")

    # Combine & Sort
    df_full = pd.concat([df_normal, df_anomaly], ignore_index=True)
    df_full = df_full.sort_values("timestamp").reset_index(drop=True)
    df_full.insert(0, "session_id", [f"sess_{i:06d}" for i in range(len(df_full))])

    print(f"\nTotal sessions generated: {len(df_full)}")
    print(f"Overall Anomaly Rate: {(df_full['label'] != 'normal').mean():.2%}")
    print("\nBreakdown by Label:")
    print(df_full["label"].value_counts())

    # Export datasets
    full_path = os.path.join(output_dir, "access_logs_full.csv")
    inference_path = os.path.join(output_dir, "access_logs_inference.csv")
    labels_path = os.path.join(output_dir, "ground_truth_labels.csv")

    df_full.to_csv(full_path, index=False)
    df_inference = df_full.drop(columns=["label"])
    df_inference.to_csv(inference_path, index=False)
    df_full[["label"]].to_csv(labels_path, index=True, index_label="row_id")

    print(f"\nSuccessfully saved datasets to '{output_dir}':")
    print(f"  - {full_path}")
    print(f"  - {inference_path}")
    print(f"  - {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic behavioral access logs.")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save generated CSV/JSON files.")
    args = parser.parse_args()

    generate_data(output_dir=args.output_dir)
