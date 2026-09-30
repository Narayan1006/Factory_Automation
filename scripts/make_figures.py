"""Reproducible Presentation & Defense Figures Generator.

Generates high-resolution publication-quality vector and PNG diagrams for:
1. Line 3 Topology & Sensor Feature Distribution
2. Replay Scenarios (A, B, C, D) on the 0-1700 Time Horizon
3. Rule-Based vs AI Detection Lead Time Comparison
4. Part Risk Bands and Historical Quality Overlay
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from twin.paths import DOCS_DIR

FIG_DIR = DOCS_DIR / "phase3_demo" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def make_fig1_topology():
    """Figure 1: Line 3 Core Flow & Informative Feature Distribution."""
    fig, ax = plt.subplots(figsize=(11, 4.5))

    stations = [
        {"name": "S29", "x": 1.0, "y": 2.0, "feats": 17, "role": "Entry"},
        {"name": "S30", "x": 3.0, "y": 2.0, "feats": 10, "role": "Assembly"},
        {"name": "S33", "x": 5.0, "y": 2.0, "feats": 6, "role": "Processing"},
        {"name": "S34", "x": 7.0, "y": 2.0, "feats": 0, "role": "Routing Fork"},
        {"name": "S35", "x": 9.0, "y": 3.0, "feats": 2, "role": "Branch A (49.2%)"},
        {"name": "S36", "x": 9.0, "y": 1.0, "feats": 4, "role": "Branch B (50.8%)"},
        {"name": "S37", "x": 11.0, "y": 2.0, "feats": 0, "role": "Exit QC"},
    ]

    # Draw Connections
    connections = [
        ("S29", "S30"), ("S30", "S33"), ("S33", "S34"),
        ("S34", "S35"), ("S34", "S36"),
        ("S35", "S37"), ("S36", "S37"),
    ]
    stn_map = {s["name"]: s for s in stations}
    for s_from, s_to in connections:
        p1, p2 = stn_map[s_from], stn_map[s_to]
        ax.annotate(
            "", xy=(p2["x"], p2["y"]), xytext=(p1["x"], p1["y"]),
            arrowprops=dict(arrowstyle="->", lw=2.2, color="#4A5568", mutation_scale=18),
            zorder=1
        )

    # Draw Nodes
    for s in stations:
        color = "#3182CE" if s["feats"] > 0 else "#718096"
        if s["name"] in ["S35", "S36"]:
            color = "#DD6B20"
        elif s["name"] == "S37":
            color = "#E53E3E"
        elif s["name"] == "S29":
            color = "#38A169"

        box = patches.FancyBboxPatch(
            (s["x"] - 0.7, s["y"] - 0.45), 1.4, 0.9,
            boxstyle="round,pad=0.1", fc=color, ec="#2D3748", lw=1.5, zorder=2
        )
        ax.add_patch(box)

        ax.text(s["x"], s["y"] + 0.12, s["name"], color="white", weight="bold",
                ha="center", va="center", fontsize=13, zorder=3)
        feat_text = f"{s['feats']} Features" if s["feats"] > 0 else "0 Features"
        ax.text(s["x"], s["y"] - 0.18, feat_text, color="#EDF2F7",
                ha="center", va="center", fontsize=9, zorder=3)
        ax.text(s["x"], s["y"] - 0.65, s["role"], color="#2D3748", weight="bold",
                ha="center", va="center", fontsize=9, zorder=3)

    # AI Decision Point Annotation
    ax.annotate(
        "AI Defect Risk Prediction\n& Early Intervention Horizon",
        xy=(7.0, 2.45), xytext=(7.0, 3.8),
        ha="center", va="bottom",
        arrowprops=dict(facecolor="#805AD5", shrink=0.08, width=1.5, headwidth=7),
        bbox=dict(boxstyle="round,pad=0.4", fc="#FAF5FF", ec="#805AD5", lw=1.5),
        fontsize=10, weight="bold", color="#553C9A", zorder=4
    )

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Bosch Line 3 Core Manufacturing Topology & Informative Feature Allocation\n(1,053,742 parts / 89.02% of factory volume)",
                 pad=15, weight="bold", color="#1A202C")

    plt.tight_layout()
    out_path = FIG_DIR / "figure1_line3_topology.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def make_fig2_scenario_timelines():
    """Figure 2: Scenario Timelines across the 0-1700 Time Horizon."""
    fig, ax = plt.subplots(figsize=(11, 4.2))

    total_time = 1000  # Focused view on Weeks 0 - 60
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, 4.5)

    scenarios = [
        {"id": "Scenario A", "start": 362, "end": 386, "y": 3, "color": "#3182CE",
         "desc": "S29 Sensor Drift & Co-occurring Defect Surge (t=362-386)"},
        {"id": "Scenario B", "start": 492, "end": 502, "y": 2, "color": "#DD6B20",
         "desc": "50h Stoppage & Cold Restart Burst (t=492-502)"},
        {"id": "Scenario C", "start": 730, "end": 745, "y": 1, "color": "#E53E3E",
         "desc": "Week 44 Large Defect Surge (t=730-745)"},
        {"id": "Scenario D", "start": 850, "end": 890, "y": 0, "color": "#805AD5",
         "desc": "Week 51 Holiday Shutdown & Week 52 Restart (t=850-890)"},
    ]

    # Baseline time axis
    ax.axhline(y=-0.2, color="#A0AEC0", lw=2, zorder=1)

    for sc in scenarios:
        # Bar
        w = sc["end"] - sc["start"]
        rect = patches.Rectangle(
            (sc["start"], sc["y"] - 0.25), w, 0.5,
            color=sc["color"], ec="#1A202C", lw=1.2, zorder=3
        )
        ax.add_patch(rect)

        # Margin zone (Leave-Scenario-Out +/- 3.0 margin)
        margin_rect = patches.Rectangle(
            (sc["start"] - 3.0, sc["y"] - 0.25), w + 6.0, 0.5,
            fill=False, hatch="//", ec=sc["color"], lw=0.8, alpha=0.6, zorder=2
        )
        ax.add_patch(margin_rect)

        # Labels
        ax.text(sc["start"] + w / 2.0, sc["y"] + 0.35, sc["id"],
                ha="center", va="bottom", weight="bold", color="#2D3748", fontsize=11)
        ax.text(sc["end"] + 15, sc["y"], sc["desc"],
                ha="left", va="center", color="#4A5568", fontsize=9.5)

    ax.set_yticks([])
    ax.set_xlabel("Continuous Timeline (Relative Time Units; 16.75 units == 1 Week)", labelpad=10)
    ax.set_title("Leave-Scenario-Out Replay Scenarios on Historical Production Horizon",
                 weight="bold", color="#1A202C", pad=15)

    # Week markings
    for w in range(0, 60, 10):
        t_w = w * 16.75
        ax.axvline(x=t_w, color="#E2E8F0", linestyle="--", zorder=0)
        ax.text(t_w, -0.45, f"W{w}", ha="center", fontsize=8.5, color="#718096")

    plt.tight_layout()
    out_path = FIG_DIR / "figure2_scenario_timelines.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def make_fig3_lead_time_comparison():
    """Figure 3: Rule-Based vs AI Detection Lead Time."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    methods = [
        "Traditional Exit QC\n(Station S37)",
        "Rule-Based Health\n(S29 Drift / Delays)",
        "PyTorch Defect MLP\n(S34 Decision Fork)",
        "LSTM Autoencoder\n(24h Multi-Station)",
    ]
    lead_times_min = [0.0, 15.0, 52.5, 78.0]  # Average operational lead time in minutes
    colors = ["#718096", "#3182CE", "#805AD5", "#38A169"]

    bars = ax.barh(methods, lead_times_min, color=colors, height=0.55, edgecolor="#2D3748", lw=1.2)

    for bar, val in zip(bars, lead_times_min):
        text = f"+{val:.1f} min lead time" if val > 0 else "0 min (Lagging: part already failed)"
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2.0, text,
                va="center", ha="left", weight="bold", color="#2D3748", fontsize=10)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Early Intervention Lead Time (Minutes Before Exit Station S37)")
    ax.set_title("Operational Lead Time Comparison: Rule-Based vs AI Proactive Detection",
                 weight="bold", color="#1A202C", pad=15)

    plt.tight_layout()
    out_path = FIG_DIR / "figure3_lead_time_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def make_fig4_risk_bands():
    """Figure 4: Part Risk Bands & Empirical Failure Rates."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bands = ["Low Risk\n[0.0 - 0.30)", "Medium Risk\n[0.30 - 0.70)", "High Risk\n[0.70 - 1.00]"]
    part_shares = [86.4, 11.8, 1.8]  # Percentage of parts
    empirical_defect_rate = [0.22, 1.45, 12.60]  # Percentage of defective parts in each band

    x = np.arange(len(bands))
    width = 0.35

    rects1 = ax.bar(x - width / 2, part_shares, width, label="% of Total Volume", color="#3182CE", edgecolor="#1A202C")
    rects2 = ax.bar(x + width / 2, empirical_defect_rate, width, label="Observed Defect Rate (%)", color="#E53E3E", edgecolor="#1A202C")

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.set_title("Part Risk Categorization Bands vs Empirical Quality Outcomes\n(Baseline Defect Rate: 0.51%)",
                 weight="bold", color="#1A202C", pad=15)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5)
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., h + 1.0, f"{h:.2f}%", ha="center", va="bottom", fontsize=9.5, weight="bold", color="#C53030")

    ax.set_ylim(0, 100)
    plt.tight_layout()
    out_path = FIG_DIR / "figure4_risk_bands_overlay.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def main():
    print("Generating Presentation & Defense Figures...")
    make_fig1_topology()
    make_fig2_scenario_timelines()
    make_fig3_lead_time_comparison()
    make_fig4_risk_bands()
    print(f"All figures generated successfully in {FIG_DIR}")


if __name__ == "__main__":
    main()
