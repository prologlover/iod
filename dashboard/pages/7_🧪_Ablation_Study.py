"""
Stage 9 — Ablation Study Dashboard
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    inject_css, section, metric_tile, apply_dark_theme,
    COLORS, MODEL_COLORS, ATTACK_COLORS, PLOTLY_LAYOUT,
    get_paths, load_ablation_json,
)

st.set_page_config(page_title="Ablation Study", page_icon="🧪", layout="wide")
inject_css()

st.markdown("## 🧪 Ablation Study")
st.markdown("<p style='color:#9AA3B2'>Systematically evaluate components, attack types, attacker ratios, and graph topologies</p>",
            unsafe_allow_html=True)
st.markdown("---")

paths = get_paths()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")

# ── Load ablation data ────────────────────────────────────────────────────────
ablation = load_ablation_json()

# Also try individual study files
@st.cache_data(show_spinner=False)
def load_all_ablation():
    data = {}
    if ablation:
        data.update(ablation)
    for p in sorted(paths["tables"].glob("ablation_*.json")):
        key = p.stem.replace("ablation_", "")
        if key == "full_results":
            continue
        try:
            data[key] = json.loads(p.read_text())
        except Exception:
            pass
    return data

abl_all = load_all_ablation()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧩 Components", "⚔️ Attack Types", "📊 Attacker Ratio",
    "🕸️ Graph Topology", "🚀 Run Ablation"
])

METRIC_ORDER = ["f1", "accuracy", "precision", "recall", "roc_auc"]

def ablation_bar(data_dict: dict, title: str, metric: str = "f1", palette=None):
    """Create a bar chart from ablation result dict."""
    names = list(data_dict.keys())
    vals = [data_dict[n].get(metric, 0) if isinstance(data_dict[n], dict)
            else data_dict[n] for n in names]
    colors = palette or [COLORS["primary"]] * len(names)
    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker=dict(color=colors, opacity=0.85),
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    if vals:
        best = max(vals)
        fig.add_hline(y=best, line_dash="dot", line_color=COLORS["success"],
                      opacity=0.6, annotation_text=f"Best: {best:.4f}",
                      annotation_position="right")
    fig.update_layout(title=title, yaxis=dict(range=[0, min(1.2, max(vals, default=1) * 1.15)],
                                               gridcolor="#2D3250"), **PLOTLY_LAYOUT)
    return fig

def abl_heatmap(data_dict: dict, title: str, metrics: list = METRIC_ORDER):
    """Multi-metric heatmap for ablation."""
    names = list(data_dict.keys())
    z = []
    for m in metrics:
        row = [data_dict[n].get(m, 0) if isinstance(data_dict[n], dict) else 0 for n in names]
        z.append(row)
    fig = go.Figure(go.Heatmap(
        z=z, x=names, y=[m.upper() for m in metrics],
        colorscale="Blues", zmin=0, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
    ))
    fig.update_layout(title=title, height=280, **PLOTLY_LAYOUT)
    return fig


# ── TAB 1: Components ────────────────────────────────────────────────────────
with tab1:
    section("Component Ablation", "🧩")
    st.markdown("""
    Tests which components of the GAT+GRU architecture contribute to performance:
    - **Full model**: GAT + GRU + temporal attention
    - **No GAT**: Replace GAT with MLP (no graph structure)
    - **No GRU**: Replace GRU with mean pooling (no temporal)
    - **No Attn**: Remove temporal attention
    """)

    comp_data = abl_all.get("components", None)

    if comp_data is None:
        st.info("No component ablation data found. Use the Run Ablation tab to generate.")

        # Synthetic demo
        demo_comp = {
            "Full Model":   {"f1": 0.912, "accuracy": 0.928, "precision": 0.905, "recall": 0.919, "roc_auc": 0.967},
            "No GAT":       {"f1": 0.764, "accuracy": 0.798, "precision": 0.751, "recall": 0.777, "roc_auc": 0.841},
            "No GRU":       {"f1": 0.803, "accuracy": 0.821, "precision": 0.796, "recall": 0.810, "roc_auc": 0.889},
            "No Attention": {"f1": 0.884, "accuracy": 0.901, "precision": 0.878, "recall": 0.890, "roc_auc": 0.942},
        }
        comp_data = demo_comp
        st.caption("Showing synthetic demo data")

    metric_sel = st.selectbox("Metric", METRIC_ORDER, key="comp_metric")
    c1, c2 = st.columns(2)
    with c1:
        fig = ablation_bar(comp_data, "Component Ablation", metric_sel,
                            palette=[COLORS["primary"], "#FFB300", "#FF5252", "#00D4FF"])
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    with c2:
        fig_h = abl_heatmap(comp_data, "Component vs Metric Heatmap")
        st.plotly_chart(apply_dark_theme(fig_h), use_container_width=True)

    # Delta table
    section("Relative Performance Drop vs Full Model", "📉")
    full_f1 = comp_data.get("Full Model", {}).get("f1", 1.0) if isinstance(
        comp_data.get("Full Model"), dict) else 1.0
    import pandas as pd
    delta_rows = []
    for name, md in comp_data.items():
        f1 = md.get("f1", 0) if isinstance(md, dict) else md
        delta_rows.append({
            "Configuration": name,
            "F1": round(f1, 4),
            "F1 Drop": round(full_f1 - f1, 4),
            "F1 Drop %": f"{(full_f1 - f1) / max(full_f1, 1e-8) * 100:.1f}%",
        })
    df_d = pd.DataFrame(delta_rows).set_index("Configuration")
    st.dataframe(df_d.style.background_gradient(cmap="RdYlGn", subset=["F1"])
                 .background_gradient(cmap="Reds", subset=["F1 Drop"]),
                 use_container_width=True)

# ── TAB 2: Attack Types ──────────────────────────────────────────────────────
with tab2:
    section("Cross-Attack-Type Performance", "⚔️")
    st.markdown("Tests model robustness across all four Byzantine attack types.")

    atk_data = abl_all.get("attacks", None)

    if atk_data is None:
        st.info("No cross-attack ablation data. Use Run Ablation tab.")
        atk_data = {
            "false_state":  {"f1": 0.912, "accuracy": 0.928, "precision": 0.905, "recall": 0.919, "roc_auc": 0.967},
            "intermittent": {"f1": 0.834, "accuracy": 0.862, "precision": 0.821, "recall": 0.847, "roc_auc": 0.911},
            "colluding":    {"f1": 0.791, "accuracy": 0.823, "precision": 0.778, "recall": 0.804, "roc_auc": 0.878},
            "delay":        {"f1": 0.867, "accuracy": 0.889, "precision": 0.854, "recall": 0.880, "roc_auc": 0.934},
        }
        st.caption("Showing synthetic demo data")

    metric_a = st.selectbox("Metric", METRIC_ORDER, key="atk_metric")
    atk_colors = [ATTACK_COLORS.get(k, COLORS["primary"]) for k in atk_data.keys()]

    c1, c2 = st.columns(2)
    with c1:
        fig_a = ablation_bar(atk_data, "Performance by Attack Type", metric_a, atk_colors)
        st.plotly_chart(apply_dark_theme(fig_a), use_container_width=True)
    with c2:
        fig_ah = abl_heatmap(atk_data, "Attack Type vs Metric")
        st.plotly_chart(apply_dark_theme(fig_ah), use_container_width=True)

    # Radar comparison
    section("Attack Type Radar", "📡")
    radar_keys = ["f1", "accuracy", "precision", "recall", "roc_auc"]
    radar_labels = ["F1", "Accuracy", "Precision", "Recall", "ROC-AUC"]
    closed = radar_labels + [radar_labels[0]]

    fig_rad = go.Figure()
    for atk, md in atk_data.items():
        vals = [md.get(k, 0) for k in radar_keys]
        vals_c = vals + [vals[0]]
        color = ATTACK_COLORS.get(atk, COLORS["primary"])
        fig_rad.add_trace(go.Scatterpolar(
            r=vals_c, theta=closed, fill="toself",
            name=atk.replace("_", " ").title(),
            line_color=color, fillcolor=color + "20",
        ))
    fig_rad.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], gridcolor="#2D3250", color="#9AA3B2"),
            angularaxis=dict(gridcolor="#2D3250"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Attack Type Performance Radar", height=450,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "plot_bgcolor")},
    )
    st.plotly_chart(apply_dark_theme(fig_rad), use_container_width=True)

# ── TAB 3: Attacker Ratio ────────────────────────────────────────────────────
with tab3:
    section("Detection vs Attacker Ratio", "📊")
    st.markdown("How does model performance change as Byzantine agents become more numerous?")

    ratio_data = abl_all.get("ratios", None)

    if ratio_data is None:
        st.info("No ratio ablation data. Use Run Ablation tab.")
        # Synthetic
        ratios_demo = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        ratio_data = {
            f"{r:.0%}": {
                "f1": max(0.3, 0.95 - r * 1.2 + np.random.normal(0, 0.02)),
                "accuracy": max(0.5, 0.97 - r * 0.8),
            }
            for r in ratios_demo
        }
        st.caption("Showing synthetic demo data")

    metric_r = st.selectbox("Metric", METRIC_ORDER, key="ratio_metric")
    ratios = list(ratio_data.keys())
    vals_r = [v.get(metric_r, 0) if isinstance(v, dict) else v for v in ratio_data.values()]

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=ratios, y=vals_r, mode="lines+markers",
        name=metric_r.upper(),
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=8, color=COLORS["primary"]),
        fill="tozeroy", fillcolor=COLORS["primary"] + "20",
    ))
    if "accuracy" in (ratio_data.get(ratios[0]) or {}):
        acc_vals = [v.get("accuracy", 0) if isinstance(v, dict) else 0 for v in ratio_data.values()]
        fig_r.add_trace(go.Scatter(
            x=ratios, y=acc_vals, mode="lines+markers",
            name="Accuracy",
            line=dict(color=COLORS["secondary"], width=2, dash="dash"),
            marker=dict(size=6),
        ))

    fig_r.add_hline(y=0.8, line_dash="dot", line_color=COLORS["warning"],
                     annotation_text="0.80 threshold")
    fig_r.update_layout(
        title=f"{metric_r.upper()} vs Attacker Ratio",
        xaxis_title="Attacker Ratio",
        yaxis=dict(range=[0, 1.05], gridcolor="#2D3250"),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig_r), use_container_width=True)

    # Heatmap across all ratios and metrics
    section("All Metrics vs Ratio Heatmap", "🔥")
    if all(isinstance(v, dict) for v in ratio_data.values()):
        fig_rh = abl_heatmap(ratio_data, "Metrics vs Attacker Ratio")
        st.plotly_chart(apply_dark_theme(fig_rh), use_container_width=True)

# ── TAB 4: Graph Topology ────────────────────────────────────────────────────
with tab4:
    section("Graph Topology Comparison", "🕸️")
    st.markdown("Compare how different graph construction methods affect detection performance.")

    topo_data = abl_all.get("topology", None)

    if topo_data is None:
        st.info("No topology ablation data. Use Run Ablation tab.")
        topo_data = {
            "KNN":         {"f1": 0.912, "accuracy": 0.928, "precision": 0.905, "recall": 0.919, "roc_auc": 0.967},
            "Distance":    {"f1": 0.887, "accuracy": 0.903, "precision": 0.881, "recall": 0.893, "roc_auc": 0.951},
            "Hexagonal":   {"f1": 0.871, "accuracy": 0.889, "precision": 0.864, "recall": 0.878, "roc_auc": 0.938},
        }
        st.caption("Showing synthetic demo data")

    metric_t = st.selectbox("Metric", METRIC_ORDER, key="topo_metric")
    topo_colors = [COLORS["primary"], COLORS["secondary"], COLORS["warning"]]

    c1, c2 = st.columns(2)
    with c1:
        fig_t = ablation_bar(topo_data, "Performance by Graph Topology", metric_t, topo_colors)
        st.plotly_chart(apply_dark_theme(fig_t), use_container_width=True)
    with c2:
        fig_th = abl_heatmap(topo_data, "Topology vs Metric")
        st.plotly_chart(apply_dark_theme(fig_th), use_container_width=True)

    # Topology descriptions
    section("Topology Descriptions", "ℹ️")
    topo_info = {
        "KNN": {
            "color": COLORS["primary"],
            "desc": "Connect each drone to its k nearest neighbors by Euclidean distance. Adaptive to formation changes but requires continuous recalculation.",
            "pros": "Adaptive, maintains fixed degree",
            "cons": "Computationally intensive, topology changes with movement",
        },
        "Distance": {
            "color": COLORS["secondary"],
            "desc": "Connect drones within a fixed communication range r. Models physical radio communication realistically but can produce disconnected graphs.",
            "pros": "Physically motivated, simple",
            "cons": "Variable degree, disconnection risk",
        },
        "Hexagonal": {
            "color": COLORS["warning"],
            "desc": "Fixed lattice formation where drones maintain a hexagonal grid. Regular, symmetric topology ideal for formation flying but inflexible.",
            "pros": "Regular structure, predictable degree",
            "cons": "Inflexible, requires strict formation maintenance",
        },
    }
    cols = st.columns(3)
    for col, (name, info) in zip(cols, topo_info.items()):
        with col:
            st.markdown(f"""
            <div class='dash-card' style='border-left:4px solid {info["color"]}'>
                <div style='color:{info["color"]};font-weight:700;font-size:1rem'>{name}</div>
                <div style='color:#9AA3B2;font-size:0.82rem;margin:0.4rem 0'>{info["desc"]}</div>
                <div style='font-size:0.8rem'>
                    <span style='color:{COLORS["success"]}'>+ {info["pros"]}</span><br>
                    <span style='color:{COLORS["danger"]}'>- {info["cons"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 5: Run Ablation ───────────────────────────────────────────────────────
with tab5:
    section("Launch Ablation Study", "🚀")
    st.info("Configure and launch ablation studies. Results will be saved to outputs/tables/.")

    c1, c2 = st.columns(2)
    with c1:
        study_sel = st.selectbox(
            "Study type",
            ["all", "components", "attacks", "ratios", "topology"],
            key="abl_study"
        )
        n_epochs_abl = st.slider("Epochs per run", 5, 100, 20, key="abl_epochs")
    with c2:
        n_snaps_abl = st.slider("Snapshots per study (0 = full)", 0, 200, 40, key="abl_snaps",
                                 help="Use small values for quick ablation")

    snaps_arg = f"--snapshots {n_snaps_abl}" if n_snaps_abl > 0 else ""
    cmd = (f'"{VENV_PYTHON}" -u scripts/ablation.py '
           f'--study {study_sel} --epochs {n_epochs_abl} {snaps_arg}')
    st.code(cmd, language="bash")

    run_abl = st.button("▶ Run Ablation", type="primary", key="run_abl_btn")
    if run_abl:
        st.markdown("**Ablation output:**")
        placeholder = st.empty()
        log_text = ""
        try:
            import os
            proc = subprocess.Popen(
                cmd, shell=True, cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            for line in proc.stdout:
                log_text += line
                placeholder.code(log_text[-5000:], language="text")
            proc.wait()
            if proc.returncode == 0:
                st.success("Ablation completed! Refresh tabs to see results.")
                st.cache_data.clear()
            else:
                st.error(f"Ablation failed (exit code {proc.returncode})")
        except Exception as e:
            st.error(f"Failed to launch: {e}")

    # Show saved result files
    section("Available Result Files", "📂")
    result_files = sorted(paths["tables"].glob("ablation_*.json"))
    if result_files:
        success_color = COLORS["success"]
        for p in result_files:
            size = p.stat().st_size / 1024
            st.markdown(
                f"<div style='padding:4px 0'>"
                f"<span style='color:{success_color}'>OK</span> "
                f"<code>{p.name}</code> "
                f"<span style='color:#9AA3B2'>({size:.1f} KB)</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<span style='color:#555'>No ablation result files yet.</span>",
                    unsafe_allow_html=True)
