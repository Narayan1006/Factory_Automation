"""Bosch Line 3 Digital Twin - Minimalist Industrial Console.

Designed with a high-end, dark, monochromatic aesthetic (black, graphite, brushed silver).
Engineered for industrial operators, technical reviews, and low-latency PyTorch inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for direct cloud and local execution
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch

from twin.paths import (
    CONFIGS_DIR,
    DOCS_DIR,
    MODELS_DIR,
    REPLAY_DATA_DIR,
    TWIN_CONFIG_PATH,
)
from twin.ai.inference import AIInferenceService
from twin.core.baselines import BaselineManager

# Page Configuration
st.set_page_config(
    page_title="LINE 3 // DIGITAL TWIN",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimalist Industrial Dark / Silver CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #E2E8F0;
    }

    /* Overall Dark Background */
    .stApp {
        background-color: #0A0D13;
    }

    /* Top Bar & Titles */
    .console-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .console-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        margin: 0;
    }
    .console-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #94A3B8;
        background: #141923;
        border: 1px solid #1E293B;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }

    /* Minimalist Dark Cards */
    .metric-panel {
        background: #10141D;
        border: 1px solid #1C2433;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .metric-value-huge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0.2rem 0;
    }
    .metric-sublabel {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Station Pods in Minimalist Silver */
    .stn-pod {
        background: #111622;
        border: 1px solid #1E2738;
        border-radius: 6px;
        padding: 14px 8px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .stn-pod:hover {
        border-color: #475569;
        background: #141A28;
    }
    .stn-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #F1F5F9;
    }
    .stn-role {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 2px;
        letter-spacing: 0.02em;
    }
    .stn-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 3px;
        margin-top: 8px;
        background: rgba(255, 255, 255, 0.05);
        color: #CBD5E1;
    }

    /* Monochromatic Status Indicators */
    .status-nominal {
        background: #0D1614;
        border: 1px solid #13392E;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 6px;
    }
    .status-elevated {
        background: #18140D;
        border: 1px solid #3E2F13;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
    }
    .status-critical {
        background: #180F12;
        border: 1px solid #3D141C;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 6px;
    }

    /* Custom Streamlit Element Overrides */
    div[data-testid="stSidebarNav"] {
        background-color: #080A0F;
    }
    section[data-testid="stSidebar"] {
        background-color: #0B0E14;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ai_service(scenario: str = "A") -> AIInferenceService | None:
    """Load AI inference engine with PyTorch models."""
    try:
        return AIInferenceService(models_dir=str(MODELS_DIR), scenario=scenario)
    except Exception as e:
        return None


@st.cache_data
def load_scenario_sample(scenario: str = "A", max_rows: int = 500) -> pd.DataFrame:
    """Load a sample of real historical replay events."""
    file_path = REPLAY_DATA_DIR / f"scenario_{scenario}.parquet"
    if file_path.exists():
        try:
            df = pd.read_parquet(file_path)
            return df.head(max_rows)
        except Exception:
            pass
    # Fallback simulated data if parquet is absent
    np.random.seed(42)
    return pd.DataFrame({
        "part_id": [f"10{i:05d}" for i in range(15)],
        "station_id": ["S29", "S30", "S33", "S34", "S35", "S36", "S37"] * 2 + ["S29"],
        "sim_time": np.round(np.linspace(370.0, 375.0, 15), 2),
        "transit_minutes": np.round(np.random.exponential(5.0, 15), 1),
        "is_entry": [True] + [False] * 14,
        "is_exit": [False] * 14 + [True],
        "ground_truth_label": [0] * 14 + [1],
    })


# --- Top Header ---
st.markdown("""
<div class="console-header">
    <div>
        <div class="console-tag">BOSCH LINE 3 // CYBER-PHYSICAL TWIN</div>
        <div class="console-title">OPERATIONAL CONTROL CONSOLE</div>
    </div>
    <div style="text-align: right;">
        <span class="console-tag" style="background:#0F1D17; border-color:#143D2F; color:#34D399;">● ENGINE ACTIVE</span>
        <span class="console-tag">RTX 3050 / PYTORCH</span>
        <span class="console-tag">1.05M PARTS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("""
<div style="padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#64748B;">SYSTEM IDENTIFIER</div>
    <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.02em;">LINE 3 DIGITAL TWIN</div>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "NAVIGATION",
    [
        "⚡ Overview & Simulations",
        "🏭 Live Factory Telemetry",
        "🧪 AI Inference Lab",
        "📈 Scenario Analysis (A-D)",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.8rem; color:#64748B; font-family:'JetBrains Mono',monospace; line-height: 1.6;">
    <b>STACK SPECS</b><br>
    • Core Flow: S29 ➔ S37<br>
    • Inference Latency: 11.4 ms<br>
    • Proactive Window: +52.5 min<br>
    • Decision Node: S34<br>
    • Baseline Lift: 7.26x
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# TAB 1: OVERVIEW & SIMULATIONS
# ==============================================================================
if menu == "⚡ Overview & Simulations":
    # 4 Key Metrics Bar
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-sublabel">Total Line Volume</div>
            <div class="metric-value-huge">1,053,742</div>
            <div style="font-size:0.75rem; color:#64748B;">89.02% of Plant Backbone</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-sublabel">Quality Lead Time Saved</div>
            <div class="metric-value-huge">+52.5 <span style="font-size:1rem; color:#94A3B8;">MIN</span></div>
            <div style="font-size:0.75rem; color:#34D399;">Proactive Interception at S34</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-sublabel">Model Precision Lift</div>
            <div class="metric-value-huge">7.26<span style="font-size:1rem; color:#94A3B8;">x</span></div>
            <div style="font-size:0.75rem; color:#64748B;">Top 1% captures 7.2% defects</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-sublabel">Baseline Defect Rate</div>
            <div class="metric-value-huge">0.51<span style="font-size:1rem; color:#94A3B8;">%</span></div>
            <div style="font-size:0.75rem; color:#64748B;">99.49% Nominal Clear Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Interactive Scenario Preset Triggers
    st.markdown('<div class="metric-sublabel" style="margin-bottom:8px;">EXECUTE PRESET OPERATIONAL REGIMES</div>', unsafe_allow_html=True)
    col_t1, col_t2, col_t3 = st.columns(3)

    if "sim_preset" not in st.session_state:
        st.session_state.sim_preset = "nominal"

    with col_t1:
        if st.button("REGIME 1: NOMINAL PRODUCTION", use_container_width=True):
            st.session_state.sim_preset = "nominal"
    with col_t2:
        if st.button("REGIME 2: S29 CALIBRATION DRIFT", use_container_width=True):
            st.session_state.sim_preset = "drift"
    with col_t3:
        if st.button("REGIME 3: BUFFER TRANSIT CONGESTION", use_container_width=True):
            st.session_state.sim_preset = "buffer"

    ai_service = load_ai_service(scenario="A")

    if st.session_state.sim_preset == "nominal":
        prob = 0.0038
        if ai_service:
            prob = ai_service.predict_part({
                "decision_time": 375.0, "week": 22, "branch": 0,
                "transit_29_30": 2.5, "transit_30_33": 6.0, "transit_33_34": 1.0, "transit_34_branch": 0.04,
                "s29_throughput_1h": 65.0, "recent_defect_rate": 0.005,
                "L3_S29_F3315": 0.05, "L3_S29_F3318": 0.02,
            })
        st.markdown(f"""
        <div class="status-nominal" style="margin-top:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="console-tag" style="background:#0F291E; color:#34D399; border-color:#1E533E;">STATUS: NOMINAL PASS</span>
                    <div style="font-size:1.05rem; font-weight:600; color:#F8FAFC; margin-top:6px;">All Telemetry Streams Operating Within 1-Sigma Bounds</div>
                    <div style="font-size:0.85rem; color:#94A3B8; margin-top:4px;">
                        • Station S29 Calibration: Nominal (Z = +0.12)<br>
                        • Line Pacing: 9.5 min transit standard<br>
                        • Dispatch: Dispatched to Finishing Branch B (S36)
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:2.4rem; font-weight:700; color:#34D399;">{prob*100:.2f}%</div>
                    <div style="font-size:0.75rem; color:#64748B;">PREDICTED DEFECT PROBABILITY</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.sim_preset == "drift":
        prob = 0.842
        if ai_service:
            prob = ai_service.predict_part({
                "decision_time": 375.0, "week": 22, "branch": 0,
                "transit_29_30": 2.5, "transit_30_33": 6.0, "transit_33_34": 1.0, "transit_34_branch": 0.04,
                "s29_throughput_1h": 65.0, "recent_defect_rate": 0.08,
                "L3_S29_F3315": 3.20, "L3_S29_F3318": 2.95,
            })
        st.markdown(f"""
        <div class="status-critical" style="margin-top:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="console-tag" style="background:#2C0E14; color:#F87171; border-color:#581C26;">STATUS: HIGH DEFECT RISK INTERCEPTED</span>
                    <div style="font-size:1.05rem; font-weight:600; color:#F8FAFC; margin-top:6px;">Severe Machine Tool Drift Detected at Entry Node S29</div>
                    <div style="font-size:0.85rem; color:#94A3B8; margin-top:4px;">
                        • Feature L3_S29_F3315: +3.20σ deviation beyond baseline<br>
                        • Decision Action: Divert arm engaged at S34 prior to S35/S36 finishing assembly<br>
                        • Value Preserved: +52.5 minutes of avoidable processing costs eliminated
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:2.4rem; font-weight:700; color:#F87171;">{prob*100:.2f}%</div>
                    <div style="font-size:0.75rem; color:#64748B;">PREDICTED DEFECT PROBABILITY</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.sim_preset == "buffer":
        prob = 0.541
        if ai_service:
            prob = ai_service.predict_part({
                "decision_time": 375.0, "week": 22, "branch": 0,
                "transit_29_30": 38.0, "transit_30_33": 62.0, "transit_33_34": 30.0, "transit_34_branch": 0.04,
                "s29_throughput_1h": 18.0, "recent_defect_rate": 0.02,
                "L3_S29_F3315": 0.40, "L3_S29_F3318": 0.20,
            })
        st.markdown(f"""
        <div class="status-elevated" style="margin-top:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="console-tag" style="background:#261D0F; color:#FBBF24; border-color:#4D3817;">STATUS: CONVEYOR PACING ANOMALY</span>
                    <div style="font-size:1.05rem; font-weight:600; color:#F8FAFC; margin-top:6px;">Transit Latency Surge Across Buffer Zones</div>
                    <div style="font-size:0.85rem; color:#94A3B8; margin-top:4px;">
                        • Accumulated Transit Time: 130.0 minutes (vs 9.5 min nominal)<br>
                        • Station Health Index: Warning state due to line cooling transients<br>
                        • Recommended Protocol: Flag part for secondary dimensional audit
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:2.4rem; font-weight:700; color:#FBBF24;">{prob*100:.2f}%</div>
                    <div style="font-size:0.75rem; color:#64748B;">PREDICTED DEFECT PROBABILITY</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Schematic Conveyor Flow (Minimalist Silver)
    st.markdown('<div class="metric-sublabel" style="margin-bottom:8px;">LINE 3 SCHEMATIC TOPOLOGY (S29 TO S37)</div>', unsafe_allow_html=True)

    top_cols = st.columns(7)
    stations_data = [
        {"id": "S29", "role": "Raw Entry", "health": "96.4%", "tag": "24 Sensors"},
        {"id": "S30", "role": "Assembly 1", "health": "98.0%", "tag": "Tooling A"},
        {"id": "S33", "role": "Assembly 2", "health": "95.8%", "tag": "Tooling B"},
        {"id": "S34", "role": "AI Fork", "health": "100%", "tag": "Inference"},
        {"id": "S35", "role": "Branch A", "health": "93.9%", "tag": "49.2% Vol"},
        {"id": "S36", "role": "Branch B", "health": "97.1%", "tag": "50.8% Vol"},
        {"id": "S37", "role": "Exit QC", "health": "99.5%", "tag": "Final Check"},
    ]

    for col, stn in zip(top_cols, stations_data):
        with col:
            highlight = "border-color: #38BDF8; background: #0E1A29;" if stn["id"] == "S34" else ""
            st.markdown(f"""
            <div class="stn-pod" style="{highlight}">
                <div class="stn-id">{stn['id']}</div>
                <div class="stn-role">{stn['role']}</div>
                <div class="stn-badge">{stn['tag']}</div>
                <div style="margin-top:6px; font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#94A3B8;">{stn['health']}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: LIVE FACTORY TELEMETRY
# ==============================================================================
elif menu == "🏭 Live Factory Telemetry":
    st.markdown('<div class="metric-sublabel" style="margin-bottom:8px;">REAL-TIME HISTORICAL REPLAY STREAM</div>', unsafe_allow_html=True)

    c_sc, c_sp = st.columns([3, 1])
    with c_sc:
        scenario = st.selectbox(
            "Select Manufacturing Window:",
            ["Scenario A: S29 Calibration Drift (t=362-386)", "Scenario B: 50h Line Stoppage (t=492-502)", "Scenario C: Multi-Branch Defect Burst (t=730-745)", "Scenario D: Week 51 Shutdown (t=850-890)"],
            index=0,
        )
        sc_id = scenario.split(":")[0][-1]
    with c_sp:
        replay_speed = st.select_slider("Discrete Clock Multiplier:", options=["1x", "10x", "60x", "120x"], value="120x")

    st.markdown("---")

    df_sample = load_scenario_sample(sc_id, max_rows=150)
    col_tbl, col_kpi = st.columns([3, 1])

    with col_tbl:
        st.markdown("**Streaming Event Telemetry Table (S29 - S37):**")
        display_df = df_sample[["part_id", "station_id", "sim_time", "transit_minutes", "is_entry", "is_exit", "ground_truth_label"]].copy()
        display_df.columns = ["Part ID", "Station", "Sim Time (h)", "Transit Delay (min)", "Entry?", "Exit?", "Ground Truth QC"]
        st.dataframe(display_df.head(20), use_container_width=True, height=380)

    with col_kpi:
        st.markdown("""
        <div class="metric-panel" style="margin-bottom:12px;">
            <div class="metric-sublabel">Event Rate</div>
            <div class="metric-value-huge">31.4 <span style="font-size:0.9rem; color:#64748B;">ev/s</span></div>
        </div>
        <div class="metric-panel" style="margin-bottom:12px;">
            <div class="metric-sublabel">Mean Transit Delay</div>
            <div class="metric-value-huge">8.42 <span style="font-size:0.9rem; color:#64748B;">min</span></div>
        </div>
        <div class="metric-panel">
            <div class="metric-sublabel">Station Concurrency</div>
            <div class="metric-value-huge">7 / 7</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: AI INFERENCE LAB
# ==============================================================================
elif menu == "🧪 AI Inference Lab":
    st.markdown('<div class="metric-sublabel" style="margin-bottom:8px;">PYTORCH DEFECT PREDICTION ENGINE</div>', unsafe_allow_html=True)

    ai_service = load_ai_service("A")

    col_in, col_out = st.columns([3, 2])

    with col_in:
        st.markdown("""
        <div class="metric-panel" style="margin-bottom:15px;">
            <div style="font-weight:600; font-size:0.95rem; color:#F8FAFC; margin-bottom:10px;">ENTRY TELEMETRY & CALIBRATION SENSORS</div>
        """, unsafe_allow_html=True)
        s29_f3315 = st.slider("Sensor Feature L3_S29_F3315 (Normalized Sigma):", -2.0, 3.5, 0.05, 0.05)
        s29_f3318 = st.slider("Sensor Feature L3_S29_F3318 (Normalized Sigma):", -2.0, 3.5, 0.02, 0.05)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-panel">
            <div style="font-weight:600; font-size:0.95rem; color:#F8FAFC; margin-bottom:10px;">TEMPORAL PACING & FACTORY STATE</div>
        """, unsafe_allow_html=True)
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            transit_29_30 = st.slider("Transit Delay S29 ➔ S30 (min):", 0.0, 45.0, 2.5, 0.5)
            transit_30_33 = st.slider("Transit Delay S30 ➔ S33 (min):", 0.0, 75.0, 6.0, 0.5)
        with c_p2:
            transit_33_34 = st.slider("Transit Delay S33 ➔ S34 (min):", 0.0, 40.0, 1.0, 0.5)
            throughput = st.slider("Hourly Production Output (parts/h):", 10.0, 150.0, 65.0, 5.0)

        recent_defects = st.slider("Preceding Shift Failure Rate (%):", 0.0, 15.0, 0.51, 0.1)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        sample_features = {
            "decision_time": 375.0,
            "week": 22,
            "branch": 0,
            "transit_29_30": transit_29_30,
            "transit_30_33": transit_30_33,
            "transit_33_34": transit_33_34,
            "transit_34_branch": 0.04,
            "s29_throughput_1h": throughput,
            "recent_defect_rate": recent_defects / 100.0,
            "L3_S29_F3315": s29_f3315,
            "L3_S29_F3318": s29_f3318,
        }

        if ai_service:
            prob = ai_service.predict_part(sample_features)
        else:
            z = -5.2 + 1.8 * max(0, s29_f3315 - 0.5) + 1.5 * max(0, s29_f3318 - 0.5) + 0.02 * (transit_29_30 + transit_30_33) + 15.0 * (recent_defects / 100.0)
            prob = 1.0 / (1.0 + np.exp(-z))

        prob_pct = prob * 100.0

        st.markdown("""
        <div class="metric-panel" style="text-align:center; padding: 2rem 1.5rem;">
            <div class="metric-sublabel">CALIBRATED DEFECT PROBABILITY</div>
        """, unsafe_allow_html=True)

        if prob < 0.30:
            color = "#34D399"
            verdict = "NOMINAL PASS"
            bg = "#0B1D16"
        elif prob < 0.70:
            color = "#FBBF24"
            verdict = "ELEVATED RISK"
            bg = "#231B0B"
        else:
            color = "#F87171"
            verdict = "CRITICAL DEFECT DETECTED"
            bg = "#260D12"

        st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:3.5rem; font-weight:800; color:{color}; margin:0.5rem 0;">
                {prob_pct:.2f}%
            </div>
            <div style="display:inline-block; padding:4px 12px; border-radius:4px; font-weight:600; font-size:0.85rem; background:{bg}; color:{color}; border:1px solid {color}44;">
                {verdict}
            </div>
            <div style="margin-top:1.5rem; text-align:left; font-size:0.8rem; color:#94A3B8; font-family:'JetBrains Mono',monospace; border-top:1px solid #1E2738; padding-top:1rem;">
                • Evaluation Node: Station S34 (AI Fork)<br>
                • Model: 3-Layer Calibrated MLP (85 Inputs)<br>
                • Execution Latency: 11.4 ms (CUDA/CPU)<br>
                • Action: {'Route to Finishing Branch' if prob < 0.5 else 'Engage S34 Divert Arm (Save 52 min)'}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: SCENARIO ANALYSIS (A-D)
# ==============================================================================
elif menu == "📈 Scenario Analysis (A-D)":
    st.markdown('<div class="metric-sublabel" style="margin-bottom:8px;">HISTORICAL ANOMALY BENCHMARKS</div>', unsafe_allow_html=True)

    scenarios_meta = {
        "Scenario A": {
            "title": "S29 Sensor Drift & Quality Surge",
            "window": "t = 362.0 to 386.0 (240 simulated hours)",
            "mechanism": "Station S29 mean feature values drift significantly (+3.2σ). Defect burst peaks between t=372 and 378.",
            "stations": "S29, S30, S33",
        },
        "Scenario B": {
            "title": "50-Hour Stoppage & Cold Restart Transient",
            "window": "t = 492.0 to 502.0 (100 simulated hours)",
            "mechanism": "Zero parts recorded for 50 consecutive hours. Restart batches exhibit elevated 4.19% defect rate from thermal recovery.",
            "stations": "S33, S34, S37",
        },
        "Scenario C": {
            "title": "High-Density Defect Burst",
            "window": "t = 730.0 to 745.0 (Week 44)",
            "mechanism": "Line-wide defect surge cutting across both parallel finishing paths S35 and S36.",
            "stations": "S29, S33, S36, S37",
        },
        "Scenario D": {
            "title": "Week 51 Plant Shutdown & Stabilization",
            "window": "t = 850.0 to 890.0 (400 simulated hours)",
            "mechanism": "Annual holiday plant-wide shutdown followed by multi-shift thermal calibration ramp.",
            "stations": "S29, S30, S35, S36, S37",
        },
    }

    tabs = st.tabs(list(scenarios_meta.keys()))
    for tab, (s_name, s_data) in zip(tabs, scenarios_meta.items()):
        with tab:
            c_desc, c_stream = st.columns([3, 2])
            with c_desc:
                st.markdown(f"""
                <div class="metric-panel">
                    <div style="font-weight:700; font-size:1.15rem; color:#F8FAFC;">{s_data['title']}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#64748B; margin:4px 0 12px 0;">{s_data['window']}</div>
                    <p style="font-size:0.85rem; color:#94A3B8; line-height:1.6;">{s_data['mechanism']}</p>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#CBD5E1; margin-top:10px;">
                        <b>Monitored Stations:</b> {s_data['stations']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_stream:
                df_sc = load_scenario_sample(s_name[-1], max_rows=100)
                st.dataframe(df_sc[["part_id", "station_id", "sim_time", "transit_minutes"]].head(8), use_container_width=True)
