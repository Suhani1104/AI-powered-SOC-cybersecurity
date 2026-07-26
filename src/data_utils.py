"""
Shared helpers used across pipeline modules.
"""
import pandas as pd


def load_logs(path: str) -> pd.DataFrame:
    """
    Loads an access-log CSV and guarantees a `session_id` column exists.
    The Step 1 generator notebook didn't originally emit session_id; this
    patches it in deterministically (row order is stable since Step 1 sorts
    by timestamp before export) so every downstream module has a stable key
    to group a session's sequential actions by.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "session_id" not in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.insert(0, "session_id", [f"sess_{i:06d}" for i in range(len(df))])
    return df