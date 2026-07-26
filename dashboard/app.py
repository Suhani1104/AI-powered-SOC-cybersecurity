"""
Adaptive Digital Twin — AI SOC Behavioral Security Copilot
Premium interactive dashboard with rich visualizations.

Run:  python -m streamlit run dashboard/app.py   (from project root)
"""

import sys, json, pickle, math
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR  = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.config import (
    PATH_FEATURES, PATH_PROFILES, PATH_MODEL, THRESHOLD_FLAG, COLD_START_DAMPENING,
)
from src.baseline_profiler import get_profile
from src.feature_engineering import haversine_km
from src.classifier import classify_row
from src.attack_predictor import predict_next_action
from src.explainability import build_reason, build_alert_payload, get_llm_narrative
from src.drift_handler import update_profile
from src.generate_data import (
    entity_lookup, sample_normal_session,
    inject_brute_force, inject_impossible_travel, inject_lateral_movement,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI SOC — Digital Twin Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Global ────────────────────────────────────── */
  html, body, [class*="css"] { font-family:'Inter',sans-serif; }
  .stApp { background:#080b12; color:#cbd5e1; }
  header[data-testid="stHeader"] { background:transparent; }

  /* hide default metric labels that duplicate our custom cards */
  [data-testid="stMetric"] { display:none; }

  /* ── Scrollbar ─────────────────────────────────── */
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:#0f1219; }
  ::-webkit-scrollbar-thumb { background:#334155; border-radius:4px; }

  /* ── Glass Card ────────────────────────────────── */
  .glass {
    background: rgba(15,18,25,0.72);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    transition: border-color .25s, box-shadow .25s;
  }
  .glass:hover {
    border-color: rgba(255,255,255,0.12);
    box-shadow: 0 0 24px rgba(99,179,237,0.08);
  }

  /* ── KPI card ──────────────────────────────────── */
  .kpi { text-align:center; }
  .kpi .label { font-size:.72rem; letter-spacing:1.6px; text-transform:uppercase; color:#64748b; font-weight:600; }
  .kpi .value { font-size:2rem; font-weight:800; margin:6px 0 2px; }
  .kpi .sub   { font-size:.74rem; color:#94a3b8; }

  /* ── Twin comparison cards ─────────────────────── */
  .twin-base {
    background: linear-gradient(135deg, rgba(14,43,75,.55), rgba(10,28,50,.8));
    border: 1px solid rgba(59,130,246,.3);
    border-radius: 14px;
    padding: 22px 24px;
    min-height: 270px;
  }
  .twin-curr {
    background: linear-gradient(135deg, rgba(75,20,30,.55), rgba(50,12,18,.8));
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 14px;
    padding: 22px 24px;
    min-height: 270px;
  }
  .twin-title { font-size:1.05rem; font-weight:700; margin-bottom:14px; display:flex; align-items:center; gap:8px; }

  /* ── Detail row ────────────────────────────────── */
  .drow { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid rgba(255,255,255,.04); font-size:.88rem; }
  .drow:last-child { border-bottom:none; }
  .drow .lbl { color:#94a3b8; }
  .drow .val { color:#f1f5f9; font-weight:600; text-align:right; max-width:60%; word-break:break-word; }

  /* ── Severity pill ─────────────────────────────── */
  .pill {
    display:inline-block; padding:4px 14px; border-radius:20px; font-weight:700; font-size:.82rem;
    letter-spacing:.4px;
  }
  .pill-crit  { background:rgba(239,68,68,.18); color:#f87171; border:1px solid rgba(239,68,68,.4); }
  .pill-warn  { background:rgba(245,158,11,.18); color:#fbbf24; border:1px solid rgba(245,158,11,.4); }
  .pill-ok    { background:rgba(34,197,94,.18);  color:#4ade80; border:1px solid rgba(34,197,94,.4); }

  /* ── Narrative card ────────────────────────────── */
  .narr {
    background: rgba(15,18,25,.72);
    backdrop-filter: blur(16px);
    border-left: 4px solid #3b82f6;
    border-radius: 0 14px 14px 0;
    padding: 20px 24px;
    line-height: 1.7;
    font-size: .92rem;
    color: #e2e8f0;
  }

  /* ── Action card ───────────────────────────────── */
  .action-card {
    background: rgba(15,18,25,.72);
    backdrop-filter: blur(16px);
    border-left: 4px solid #ef4444;
    border-radius: 0 14px 14px 0;
    padding: 20px 24px;
    text-align: center;
  }
  .action-card .ac-label { font-size:.74rem; letter-spacing:1.4px; text-transform:uppercase; color:#64748b; }
  .action-card .ac-value { font-size:1.4rem; font-weight:800; color:#f87171; margin:8px 0; }

  /* ── Section headings ──────────────────────────── */
  .sec { font-size:1.15rem; font-weight:700; color:#f8fafc; display:flex; align-items:center; gap:8px; margin-bottom:12px; }

  /* ── Sidebar tweaks ────────────────────────────── */
  section[data-testid="stSidebar"] { background:#0c0f17; }
  section[data-testid="stSidebar"] .stButton button {
    border: 1px solid #334155;
    border-radius: 8px;
    transition: border-color .2s, background .2s;
  }
  section[data-testid="stSidebar"] .stButton button:hover {
    border-color: #60a5fa;
    background: rgba(59,130,246,.08);
  }

  /* ── Tab styling ───────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
    font-size: .9rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "current_profiles" not in st.session_state:
    with open(PATH_PROFILES) as f:
        st.session_state.current_profiles = json.load(f)

if "alerts_df" not in st.session_state:
    features_df = pd.read_csv(PATH_FEATURES, parse_dates=["timestamp"])
    st.session_state.alerts_df = features_df[features_df["attack_type"].notna()].copy()

if "selected_entity" not in st.session_state:
    st.session_state.selected_entity = list(entity_lookup.keys())[0]

if "selected_alert" not in st.session_state:
    if not st.session_state.alerts_df.empty:
        st.session_state.selected_alert = st.session_state.alerts_df.iloc[-1]["session_id"]
    else:
        st.session_state.selected_alert = None

if "model" not in st.session_state:
    with open(PATH_MODEL, "rb") as f:
        st.session_state.model = pickle.load(f)

FEATURE_COLS = [
    "hour_deviation", "geo_distance", "is_new_resource", "is_new_device",
    "is_new_auth_method", "duration_zscore", "auth_fail_streak", "resource_breadth_1h",
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def _geo_str(geo_dict):
    """Render a home_geo dict as 'City, Country (lat, lon)' or just coords."""
    city = geo_dict.get("city", "")
    country = geo_dict.get("country", "")
    lat, lon = geo_dict.get("lat", 0), geo_dict.get("lon", 0)
    if city and city != "Unknown":
        return f"{city}, {country} ({lat}, {lon})"
    return f"({lat}, {lon})"


def _severity_pill(sim_val: float) -> str:
    if sim_val > 70:
        return '<span class="pill pill-ok">LOW RISK</span>'
    if sim_val > 40:
        return '<span class="pill pill-warn">MEDIUM</span>'
    return '<span class="pill pill-crit">HIGH RISK</span>'


def _gauge(value: float, title: str, color: str) -> go.Figure:
    """Mini donut-gauge 0-100."""
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
        hole=0.78,
        marker=dict(colors=[color, "rgba(30,35,50,.4)"]),
        textinfo="none", hoverinfo="none",
        sort=False, direction="clockwise",
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=160, width=160,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{value:.0f}%</b><br><span style='font-size:10px;color:#94a3b8'>{title}</span>",
            x=0.5, y=0.5, font_size=18, font_color="#f1f5f9",
            showarrow=False,
        )],
    )
    return fig


# ── Scorer ───────────────────────────────────────────────────────────────────
def score_single_session(row: dict, entity_id: str, entity_type: str) -> dict:
    profile = get_profile(entity_id, st.session_state.current_profiles, entity_type)

    hour_dev = abs(row["timestamp"].hour - profile["hour_mean"]) / max(profile["hour_std"], 1)
    geo_dist = haversine_km(row["geo_lat"], row["geo_lon"],
                            profile["home_geo"]["lat"], profile["home_geo"]["lon"])
    is_new_resource = int(row["resource_accessed"] not in profile["typical_resources"])
    is_new_device   = int(row["device_mac"] != profile["typical_device_mac"])
    is_new_auth     = int(row["auth_method"] != profile["typical_auth_method"])
    avg_dur = max(profile["avg_session_duration"], 1)
    dur_z   = (row["session_duration_sec"] - avg_dur) / max(avg_dur * 0.5, 1)

    feats = {
        "hour_deviation": hour_dev, "geo_distance": geo_dist,
        "is_new_resource": is_new_resource, "is_new_device": is_new_device,
        "is_new_auth_method": is_new_auth, "duration_zscore": dur_z,
        "auth_fail_streak": row.get("auth_fail_streak", 0),
        "resource_breadth_1h": row.get("resource_breadth_1h", 1),
    }

    X   = np.array([[feats[c] for c in FEATURE_COLS]])
    raw = st.session_state.model.decision_function(X)[0]
    anomaly_score = float(np.clip(1 - raw, 0, 1))
    if profile.get("is_cold_start"):
        anomaly_score *= COLD_START_DAMPENING

    w = {"hour": .25, "geo": .30, "res": .15, "dev": .15, "dur": .15}
    penalty = (w["hour"] * min(hour_dev / 3, 1) + w["geo"] * min(geo_dist / 5000, 1)
               + w["res"] * is_new_resource + w["dev"] * is_new_device
               + w["dur"] * min(abs(dur_z) / 3, 1))
    similarity_index = 100 * (1 - min(1, penalty))

    full = {**row, **feats,
            "anomaly_score": anomaly_score,
            "similarity_index": similarity_index,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "time_since_last_min": 999999}  # sentinel — no prior session in live mode
    full["attack_type"] = classify_row(pd.Series(full)) if anomaly_score > THRESHOLD_FLAG else None
    pred = predict_next_action(full["attack_type"])
    full["predicted_next_action"]  = pred["predicted_next"]
    full["prediction_confidence"]  = pred["confidence"]
    full["cumulative_risk_score"]  = anomaly_score
    full["explanation"]            = build_reason(pd.Series(full))
    full["session_id"]             = f"sim_{datetime.now(timezone.utc).timestamp():.0f}"
    full["profile"]                = profile
    return full


def simulate_and_append(rows, entity_id, entity_type):
    for row in rows:
        scored = score_single_session(row, entity_id, entity_type)
        st.session_state.alerts_df = pd.concat(
            [st.session_state.alerts_df, pd.DataFrame([scored])], ignore_index=True)
        st.session_state.selected_alert = scored["session_id"]


# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">
  <span style="font-size:2.2rem">🛡️</span>
  <div>
    <div style="font-size:1.6rem;font-weight:800;color:#f8fafc;letter-spacing:-.5px">
      AI SOC — Adaptive Digital Twin Copilot
    </div>
    <div style="font-size:.82rem;color:#64748b;margin-top:2px">
      Entity Baselining · Anomaly Detection · Kill-Chain Prediction · One-Click Response
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  KPI ROW
# ═════════════════════════════════════════════════════════════════════════════
n_twins = len([k for k in st.session_state.current_profiles if not k.startswith("__cohort_")])
n_alerts = len(st.session_state.alerts_df)
cold_n = sum(1 for v in st.session_state.current_profiles.values() if v.get("is_cold_start"))
peak_risk = st.session_state.alerts_df["anomaly_score"].max() if not st.session_state.alerts_df.empty else 0.0
avg_sim = st.session_state.alerts_df["similarity_index"].mean() if not st.session_state.alerts_df.empty else 100.0

kpis = [
    ("DIGITAL TWINS",  n_twins,          "#60a5fa", "Entities with learned baselines"),
    ("THREAT ALERTS",   n_alerts,         "#f87171", "Flagged behavioral deviations"),
    ("PEAK RISK",       f"{peak_risk:.2f}", "#fbbf24", "Highest anomaly score observed"),
    ("AVG SIMILARITY",  f"{avg_sim:.1f}%",  "#4ade80", "Mean baseline match across alerts"),
    ("COLD-START",      cold_n,           "#c084fc", "Entities using cohort fallback"),
]

cols = st.columns(len(kpis))
for col, (label, value, color, sub) in zip(cols, kpis):
    col.markdown(f"""
    <div class="glass kpi">
      <div class="label">{label}</div>
      <div class="value" style="color:{color}">{value}</div>
      <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Attack Simulator
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("### ⚡ Attack Simulator")
st.sidebar.caption("Inject synthetic attacks into the live stream.")
entity_ids = list(entity_lookup.keys())
st.session_state.selected_entity = st.sidebar.selectbox("🎯 Target Entity", entity_ids)
entity = entity_lookup[st.session_state.selected_entity]

st.sidebar.markdown("---")
st.sidebar.markdown("**Attack Vectors**")

if st.sidebar.button("💥  Brute Force", use_container_width=True):
    simulate_and_append(inject_brute_force(entity, n_attacks=1), entity["entity_id"], entity["entity_type"])
    st.rerun()
if st.sidebar.button("✈️  Impossible Travel", use_container_width=True):
    simulate_and_append(inject_impossible_travel(entity, n_events=1), entity["entity_id"], entity["entity_type"])
    st.rerun()
if st.sidebar.button("🔀  Lateral Movement", use_container_width=True):
    simulate_and_append(inject_lateral_movement(entity, n_events=1), entity["entity_id"], entity["entity_type"])
    st.rerun()
if st.sidebar.button("👤  Cold-Start User", use_container_width=True):
    fid = f"usr_NEW_{datetime.now(timezone.utc).timestamp():.0f}"
    simulate_and_append([sample_normal_session(entity)], fid, "user")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Drift Engine**")
if st.sidebar.button("⏩  Fast-Forward 10 Days", use_container_width=True):
    for _ in range(50):
        st.session_state.current_profiles = update_profile(
            st.session_state.current_profiles, entity["entity_id"], sample_normal_session(entity))
    st.toast(f"Baseline updated for {entity['entity_id']}", icon="✅")
    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_invest, tab_analytics, tab_queue = st.tabs([
    "🔍  Investigation",
    "📊  Analytics & Telemetry",
    "📋  Alert Queue",
])

# Auto-switch to Investigation tab when coming from "Load" button in Alert Queue
if st.session_state.get("active_tab") == "🔍  Investigation":
    st.session_state.pop("active_tab", None)
    st.components.v1.html("""
    <script>
    const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
    if (tabs.length > 0) { tabs[0].click(); }
    </script>
    """, height=0)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Investigation
# ─────────────────────────────────────────────────────────────────────────────
with tab_invest:
    if st.session_state.selected_alert is not None and not st.session_state.alerts_df.empty:
        match = st.session_state.alerts_df[
            st.session_state.alerts_df["session_id"] == st.session_state.selected_alert]
        alert = match.iloc[-1] if not match.empty else st.session_state.alerts_df.iloc[-1]
        profile = get_profile(alert["entity_id"], st.session_state.current_profiles,
                              alert.get("entity_type", "user"))

        sim_val = float(alert.get("similarity_index", 100))
        anom    = float(alert.get("anomaly_score", 0))

        # ── Status bar ───────────────────────────────────────────────────
        st.markdown(f"""
        <div class="glass" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
          <div>
            <span style="color:#64748b;font-size:.78rem">SESSION</span>&nbsp;
            <span style="color:#f1f5f9;font-weight:700">{alert['session_id']}</span>
            &nbsp;&nbsp;
            <span style="color:#64748b;font-size:.78rem">ENTITY</span>&nbsp;
            <span style="color:#f1f5f9;font-weight:700">{alert['entity_id']}</span>
            &nbsp;&nbsp;
            <span style="color:#64748b;font-size:.78rem">TYPE</span>&nbsp;
            <span style="color:#94a3b8;font-weight:600">{alert.get('attack_type','—')}</span>
          </div>
          <div>{_severity_pill(sim_val)}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauges row ───────────────────────────────────────────────────
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(_gauge(anom * 100, "Anomaly", "#f87171"), use_container_width=True, key="g_anom")
        with g2:
            st.plotly_chart(_gauge(sim_val, "Similarity", "#4ade80"), use_container_width=True, key="g_sim")
        with g3:
            risk_pct = float(alert.get("cumulative_risk_score", 0)) * 100
            st.plotly_chart(_gauge(risk_pct, "Risk", "#fbbf24"), use_container_width=True, key="g_risk")

        # ── Twin comparison ──────────────────────────────────────────────
        st.markdown('<div class="sec">🧬 Digital Twin Comparison</div>', unsafe_allow_html=True)
        left, right = st.columns(2)

        with left:
            home_geo = _geo_str(profile["home_geo"])
            st.markdown(f"""
            <div class="twin-base">
              <div class="twin-title" style="color:#93c5fd">📘 Learned Baseline</div>
              <div class="drow"><span class="lbl">Login Hour</span><span class="val">{profile['hour_mean']:.1f}:00 ± {profile['hour_std']:.1f}h</span></div>
              <div class="drow"><span class="lbl">Home Location</span><span class="val">{home_geo}</span></div>
              <div class="drow"><span class="lbl">Device / OS</span><span class="val">{profile['typical_device_os']}</span></div>
              <div class="drow"><span class="lbl">Auth Method</span><span class="val">{profile['typical_auth_method']}</span></div>
              <div class="drow"><span class="lbl">Sessions Seen</span><span class="val">{profile.get('n_sessions_seen','—')}</span></div>
              <div class="drow"><span class="lbl">Cold-Start?</span><span class="val">{'Yes — cohort fallback' if profile.get('is_cold_start') else 'No — warm baseline'}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            obs_geo = f"{alert.get('geo_city','?')}, {alert.get('geo_country','?')} ({float(alert.get('geo_lat',0)):.1f}, {float(alert.get('geo_lon',0)):.1f})"
            cur_hour = pd.Timestamp(alert["timestamp"]).hour
            st.markdown(f"""
            <div class="twin-curr">
              <div class="twin-title" style="color:#fca5a5">🚨 Current Session</div>
              <div class="drow"><span class="lbl">Access Hour</span><span class="val">{cur_hour}:00</span></div>
              <div class="drow"><span class="lbl">Geo Origin</span><span class="val">{obs_geo}</span></div>
              <div class="drow"><span class="lbl">Device / OS</span><span class="val">{alert.get('device_os','—')}</span></div>
              <div class="drow"><span class="lbl">Auth Method</span><span class="val">{alert.get('auth_method','—')}</span></div>
              <div class="drow"><span class="lbl">Resource</span><span class="val">{alert.get('resource_accessed','—')}</span></div>
              <div class="drow"><span class="lbl">Session Duration</span><span class="val">{alert.get('session_duration_sec','—')}s</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── AI Copilot — structured for quick scanning ───────────────────
        st.markdown('<div class="sec">🤖 AI Copilot — Incident Brief</div>', unsafe_allow_html=True)

        payload = build_alert_payload(alert)
        with st.spinner("Generating narrative …"):
            narrative = get_llm_narrative(str(alert["session_id"]), payload)

        summary_text = narrative.get("summary", "No narrative available.")
        rec = narrative.get("recommended_action", "Monitor Only")
        trigger_text = alert.get("explanation", "Statistical deviation detected.")
        attack_label = alert.get("attack_type", "Unknown")
        pred_next = alert.get("predicted_next_action")
        conf = alert.get("prediction_confidence", 0)

        # Colour the recommended action pill
        rec_lower = rec.lower()
        if any(w in rec_lower for w in ("isolate", "block", "revoke")):
            rec_color, rec_bg = "#f87171", "rgba(239,68,68,.12)"
        elif any(w in rec_lower for w in ("reset", "force", "escalate")):
            rec_color, rec_bg = "#fbbf24", "rgba(245,158,11,.12)"
        else:
            rec_color, rec_bg = "#4ade80", "rgba(34,197,94,.12)"

        # Build prediction line
        pred_html = ""
        if pred_next and str(pred_next).lower() not in ("nan", "none", ""):
            pred_html = f"""
            <div style="margin-top:16px;padding:14px 18px;background:rgba(245,158,11,.08);border-left:3px solid #f59e0b;border-radius:0 10px 10px 0">
              <span style="font-size:.72rem;letter-spacing:1.2px;text-transform:uppercase;color:#64748b">Predicted Next Attack Step</span>
              <div style="display:flex;align-items:baseline;gap:12px;margin-top:4px">
                <span style="font-size:1.15rem;font-weight:700;color:#fbbf24">{pred_next}</span>
                <span style="font-size:.88rem;color:#94a3b8">({conf*100:.0f}% confidence)</span>
              </div>
            </div>"""

        # Use st.columns so each panel gets its own st.markdown call
        cp_left, cp_right = st.columns(2)

        with cp_left:
            st.markdown(f"""
            <div class="glass" style="border-left:4px solid #3b82f6;border-radius:0 14px 14px 0;padding:20px 22px;min-height:220px">
              <div style="font-size:.96rem;font-weight:700;color:#93c5fd;margin-bottom:10px">📝 What Happened</div>
              <div style="font-size:.9rem;line-height:1.75;color:#e2e8f0">
                {summary_text}
              </div>
              <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,.06)">
                <span style="font-size:.78rem;color:#64748b;letter-spacing:.8px;text-transform:uppercase">Trigger Reason</span>
                <div style="font-size:.86rem;color:#cbd5e1;margin-top:4px">{trigger_text}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with cp_right:
            st.markdown(f"""
            <div class="glass" style="border-left:4px solid {rec_color};border-radius:0 14px 14px 0;padding:20px 22px;min-height:220px">
              <div style="font-size:.96rem;font-weight:700;color:#fca5a5;margin-bottom:10px">🛡️ What To Do</div>
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
                <span style="font-size:.72rem;letter-spacing:1.2px;text-transform:uppercase;color:#64748b">Recommended Action</span>
                <span style="background:{rec_bg};color:{rec_color};padding:4px 14px;border-radius:20px;font-weight:700;font-size:.88rem">{rec}</span>
              </div>
              <div style="font-size:.82rem;color:#94a3b8;margin-bottom:4px">
                <strong>Attack Classification:</strong>
                <span style="color:#e2e8f0;font-weight:600;margin-left:4px">{attack_label}</span>
              </div>
              <div style="font-size:.82rem;color:#94a3b8">
                <strong>Anomaly Score:</strong>
                <span style="color:#f87171;font-weight:700;margin-left:4px">{anom:.2f}</span>
                &nbsp;&nbsp;
                <strong>Similarity:</strong>
                <span style="color:#4ade80;font-weight:700;margin-left:4px">{sim_val:.1f}%</span>
              </div>
              {pred_html}
            </div>
            """, unsafe_allow_html=True)

        # ── One-click response row ───────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.caption("⚡ One-Click Response Actions")
        a1, a2, a3 = st.columns(3)
        if a1.button("🔒 Force Password Reset", use_container_width=True, help="Force password reset for this entity"):
            st.toast(f"Password reset → {alert['entity_id']}", icon="🔒")
        if a2.button("🚫 Revoke Session Token", use_container_width=True, help="Revoke all active session tokens"):
            st.toast(f"Token revoked → {alert['entity_id']}", icon="🚫")
        if a3.button("⚡ Isolate Endpoint", use_container_width=True, help="Network-isolate the endpoint"):
            st.toast(f"Endpoint isolated → {alert['entity_id']}", icon="⚡")
    else:
        st.info("No alert selected — trigger an attack from the sidebar or pick one from the Alert Queue tab.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Analytics & Telemetry
# ─────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    if st.session_state.alerts_df.empty:
        st.info("No alert data to visualize yet.")
    else:
        df = st.session_state.alerts_df.copy()

        # Row 1: Risk timeline + attack distribution
        r1a, r1b = st.columns([0.65, 0.35])

        with r1a:
            st.markdown('<div class="sec">📈 Risk Score Over Time</div>', unsafe_allow_html=True)

            # If we have a selected alert, show that entity's history
            if st.session_state.selected_alert is not None:
                match = df[df["session_id"] == st.session_state.selected_alert]
                focus_entity = match.iloc[-1]["entity_id"] if not match.empty else df.iloc[-1]["entity_id"]
            else:
                focus_entity = df.iloc[-1]["entity_id"]

            ent_df = df[df["entity_id"] == focus_entity].sort_values("timestamp")
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(
                x=list(range(1, len(ent_df) + 1)),
                y=ent_df["cumulative_risk_score"] * 100,
                mode="lines+markers",
                line=dict(color="#f87171", width=2.5, shape="spline"),
                marker=dict(size=6, color="#ef4444"),
                name="Risk %",
                hovertemplate="Action #%{x}<br>Risk: %{y:.1f}%<extra></extra>",
            ))
            fig_risk.add_hline(y=THRESHOLD_FLAG * 100, line_dash="dot",
                               line_color="#f59e0b", opacity=0.6,
                               annotation_text="Threshold", annotation_font_color="#fbbf24")
            fig_risk.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Cumulative Risk %", range=[0, 105], gridcolor="rgba(255,255,255,.04)"),
                xaxis=dict(title=f"Action # — {focus_entity}", gridcolor="rgba(255,255,255,.04)"),
                height=340, margin=dict(l=40, r=30, t=30, b=50),
                hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9"),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with r1b:
            st.markdown('<div class="sec">🎯 Attack Type Breakdown</div>', unsafe_allow_html=True)
            type_counts = df["attack_type"].value_counts().reset_index()
            type_counts.columns = ["Attack Type", "Count"]
            palette = ["#f87171", "#fbbf24", "#60a5fa", "#4ade80", "#c084fc", "#fb923c", "#f472b6"]
            fig_donut = go.Figure(go.Pie(
                labels=type_counts["Attack Type"],
                values=type_counts["Count"],
                hole=0.6,
                marker=dict(colors=palette[:len(type_counts)]),
                textinfo="label+percent",
                textfont=dict(size=11, color="#e2e8f0"),
                hovertemplate="%{label}: %{value} alerts<extra></extra>",
            ))
            fig_donut.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, height=340,
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Row 2: Anomaly distribution + Similarity histogram
        r2a, r2b = st.columns(2)

        with r2a:
            st.markdown('<div class="sec">📊 Anomaly Score Distribution</div>', unsafe_allow_html=True)
            fig_hist = go.Figure(go.Histogram(
                x=df["anomaly_score"],
                nbinsx=40,
                marker_color="#60a5fa",
                opacity=0.85,
                hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
            ))
            fig_hist.add_vline(x=THRESHOLD_FLAG, line_dash="dash", line_color="#f87171",
                               annotation_text="Threshold", annotation_font_color="#f87171")
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Anomaly Score", gridcolor="rgba(255,255,255,.04)"),
                yaxis=dict(title="Count", gridcolor="rgba(255,255,255,.04)"),
                height=280, margin=dict(l=40, r=20, t=30, b=40),
                bargap=0.08,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with r2b:
            st.markdown('<div class="sec">🧬 Similarity Index Spread</div>', unsafe_allow_html=True)
            fig_sim = go.Figure(go.Histogram(
                x=df["similarity_index"],
                nbinsx=40,
                marker_color="#4ade80",
                opacity=0.85,
                hovertemplate="Similarity: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ))
            fig_sim.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Similarity Index (%)", gridcolor="rgba(255,255,255,.04)"),
                yaxis=dict(title="Count", gridcolor="rgba(255,255,255,.04)"),
                height=280, margin=dict(l=40, r=20, t=30, b=40),
                bargap=0.08,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

        # Row 3: Top risky entities
        st.markdown('<div class="sec">🏆 Top 10 Riskiest Entities</div>', unsafe_allow_html=True)
        top_ent = (df.groupby("entity_id")["anomaly_score"]
                     .agg(["max", "mean", "count"])
                     .sort_values("max", ascending=False)
                     .head(10).reset_index())
        top_ent.columns = ["Entity", "Peak Score", "Avg Score", "Alert Count"]
        fig_bar = go.Figure(go.Bar(
            x=top_ent["Peak Score"],
            y=top_ent["Entity"],
            orientation="h",
            marker=dict(
                color=top_ent["Peak Score"],
                colorscale=[[0, "#3b82f6"], [0.5, "#f59e0b"], [1, "#ef4444"]],
            ),
            text=top_ent["Peak Score"].apply(lambda v: f"{v:.2f}"),
            textposition="auto",
            hovertemplate="%{y}<br>Peak: %{x:.2f}<extra></extra>",
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,.04)"),
            xaxis=dict(title="Peak Anomaly Score", gridcolor="rgba(255,255,255,.04)"),
            height=320, margin=dict(l=110, r=30, t=20, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Alert Queue
# ─────────────────────────────────────────────────────────────────────────────
with tab_queue:
    st.markdown('<div class="sec">📋 Threat Alert Queue</div>', unsafe_allow_html=True)

    if st.session_state.alerts_df.empty:
        st.info("No alerts in queue.")
    else:
        q = st.session_state.alerts_df.sort_values("cumulative_risk_score", ascending=False).copy()

        # Filter row
        fc1, fc2, fc3 = st.columns([0.4, 0.3, 0.3])
        with fc1:
            atk_opts = list(q["attack_type"].dropna().unique())
            atk_filt = st.multiselect("Attack Type", atk_opts, default=atk_opts, key="q_atk")
        with fc2:
            min_risk = st.slider("Min Risk Score", 0.0, 1.0, 0.0, 0.01, key="q_risk")
        with fc3:
            max_sim = st.slider("Max Similarity %", 0.0, 100.0, 100.0, 1.0, key="q_sim")

        filt = q[(q["attack_type"].isin(atk_filt))
                 & (q["anomaly_score"] >= min_risk)
                 & (q["similarity_index"] <= max_sim)]

        st.caption(f"Showing **{len(filt)}** of {len(q)} alerts")

        show_cols = ["session_id", "entity_id", "attack_type", "anomaly_score",
                     "cumulative_risk_score", "similarity_index"]
        # Add optional columns if present
        for c in ("geo_city", "device_os"):
            if c in filt.columns:
                show_cols.append(c)

        st.dataframe(
            filt[show_cols],
            use_container_width=True,
            height=420,
            column_config={
                "session_id": st.column_config.TextColumn("Session ID", width="medium"),
                "entity_id":  st.column_config.TextColumn("Entity ID", width="medium"),
                "attack_type": st.column_config.TextColumn("Attack Type", width="medium"),
                "anomaly_score": st.column_config.ProgressColumn(
                    "Anomaly Score", min_value=0, max_value=1, format="%.2f"),
                "cumulative_risk_score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=1, format="%.2f"),
                "similarity_index": st.column_config.NumberColumn(
                    "Similarity %", format="%.1f%%"),
                "geo_city":  st.column_config.TextColumn("City", width="small"),
                "device_os": st.column_config.TextColumn("OS", width="small"),
            },
        )

        sel_id = st.selectbox("Select a session to investigate", filt["session_id"].tolist(), key="q_sel")
        if st.button("🔍  Load into Investigation Tab", use_container_width=True, key="q_load"):
            st.session_state.selected_alert = sel_id
            st.session_state.active_tab = "🔍  Investigation"
            st.toast(f"Loaded {sel_id} — switching to Investigation tab", icon="🔍")
            st.rerun()

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 12px;font-size:.72rem;color:#475569;border-top:1px solid rgba(255,255,255,.04);margin-top:32px">
  AI SOC — Adaptive Digital Twin Copilot &nbsp;·&nbsp; Isolation Forest + Gemini AI Explainability &nbsp;·&nbsp; Action buttons are simulated for demo purposes
</div>
""", unsafe_allow_html=True)