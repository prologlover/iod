"""
Dashboard shared utilities — theming, data loaders, plot helpers.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#6C63FF",
    "secondary": "#00D4FF",
    "success":   "#00E676",
    "warning":   "#FFB300",
    "danger":    "#FF5252",
    "benign":    "#00E676",
    "attack":    "#FF5252",
    "bg":        "#0E1117",
    "card":      "#1A1D2E",
    "border":    "#2D3250",
}

MODEL_COLORS = {
    "GAT+GRU":       "#6C63FF",
    "GraphSAGE+GRU": "#00D4FF",
    "MLP":           "#FFB300",
    "LSTM":          "#FF7043",
    "1D-CNN":        "#00E676",
    "GCN":           "#F06292",
}

ATTACK_COLORS = {
    "false_state":  "#FF5252",
    "intermittent": "#FFB300",
    "colluding":    "#FF7043",
    "delay":        "#F06292",
}

# ── CSS injection ────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* Remove default streamlit padding */
.block-container { padding-top: 1.5rem !important; }

/* Cards */
.dash-card {
    background: #1A1D2E;
    border: 1px solid #2D3250;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* Metric tile */
.metric-tile {
    background: linear-gradient(135deg, #1A1D2E 0%, #2D3250 100%);
    border: 1px solid #6C63FF40;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-tile .value {
    font-size: 2rem;
    font-weight: 700;
    color: #6C63FF;
    line-height: 1;
}
.metric-tile .label {
    font-size: 0.8rem;
    color: #9AA3B2;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-tile .delta {
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

/* Section header */
.section-header {
    border-left: 4px solid #6C63FF;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    font-size: 1.15rem;
    font-weight: 600;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-success { background:#00E67620; color:#00E676; border:1px solid #00E67640; }
.badge-warning { background:#FFB30020; color:#FFB300; border:1px solid #FFB30040; }
.badge-danger  { background:#FF525220; color:#FF5252; border:1px solid #FF525240; }
.badge-info    { background:#00D4FF20; color:#00D4FF; border:1px solid #00D4FF40; }

/* Sidebar title */
.sidebar-logo {
    text-align: center;
    padding: 0.5rem 0 1rem 0;
}
.sidebar-logo .title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #6C63FF;
}
.sidebar-logo .subtitle {
    font-size: 0.75rem;
    color: #9AA3B2;
}

/* Table overrides */
.dataframe th { background-color: #2D3250 !important; }
.dataframe tr:hover { background-color: #2D325040 !important; }

/* Progress bar label */
.stProgress > div > div { background: #6C63FF !important; }
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def card(content_fn, title: str = "", icon: str = ""):
    """Wrap content in a styled card."""
    header = f"<div class='section-header'>{icon} {title}</div>" if title else ""
    st.markdown(f"<div class='dash-card'>{header}</div>", unsafe_allow_html=True)
    content_fn()


def metric_tile(value, label: str, delta: str = "", color: str = "#6C63FF"):
    delta_html = f"<div class='delta' style='color:{color}'>{delta}</div>" if delta else ""
    st.markdown(f"""
    <div class='metric-tile'>
        <div class='value' style='color:{color}'>{value}</div>
        <div class='label'>{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, kind: str = "info"):
    st.markdown(f"<span class='badge badge-{kind}'>{text}</span>", unsafe_allow_html=True)


def section(title: str, icon: str = ""):
    st.markdown(f"<div class='section-header'>{icon} {title}</div>", unsafe_allow_html=True)


# ── Path helpers ─────────────────────────────────────────────────────────────
def get_paths():
    from src.config import (
        DATA_DIR, PROCESSED_DATA_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR,
        RAW_DATA_DIR,
    )
    return {
        "project":   PROJECT_ROOT,
        "data":      DATA_DIR,
        "raw":       RAW_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
        "figures":   FIGURE_DIR,
        "tables":    TABLE_DIR,
        "models":    MODEL_DIR,
    }


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_data() -> Optional[pd.DataFrame]:
    try:
        from src.data_loader import load_and_merge
        return load_and_merge()
    except Exception as e:
        st.error(f"Could not load raw data: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_processed_split(split: str) -> Optional[pd.DataFrame]:
    paths = get_paths()
    p = paths["processed"] / f"{split}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_metrics_json(name: str = "false_state_metrics") -> Optional[Dict]:
    paths = get_paths()
    p = paths["tables"] / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def load_ablation_json() -> Optional[Dict]:
    paths = get_paths()
    p = paths["tables"] / "ablation_full_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def load_training_history(model_name: str) -> Optional[Dict]:
    paths = get_paths()
    p = paths["models"] / f"history_{model_name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_figures() -> List[Path]:
    paths = get_paths()
    return sorted(paths["figures"].glob("*.png"))


def list_model_checkpoints() -> List[Path]:
    paths = get_paths()
    return sorted(paths["models"].glob("*.pt"))


# ── Plotly helpers ────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", family="sans-serif"),
    xaxis=dict(gridcolor="#2D3250", zerolinecolor="#2D3250"),
    yaxis=dict(gridcolor="#2D3250", zerolinecolor="#2D3250"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2D3250"),
)


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def bar_chart(data: Dict[str, float], title: str, color: str = "#6C63FF",
              horizontal: bool = False) -> go.Figure:
    names = list(data.keys())
    values = list(data.values())
    if horizontal:
        fig = go.Figure(go.Bar(y=names, x=values, orientation="h",
                               marker_color=color, text=[f"{v:.4f}" for v in values],
                               textposition="outside"))
    else:
        fig = go.Figure(go.Bar(x=names, y=values, marker_color=color,
                               text=[f"{v:.4f}" for v in values],
                               textposition="outside"))
    fig.update_layout(title=title, **PLOTLY_LAYOUT)
    return fig


def training_curve_chart(history: Dict, model_name: str) -> go.Figure:
    fig = go.Figure()
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"], mode="lines+markers",
        name="Train Loss", line=dict(color="#6C63FF", width=2),
        marker=dict(size=4),
    ))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"], mode="lines+markers",
            name="Val Loss", line=dict(color="#FF5252", width=2, dash="dash"),
            marker=dict(size=4),
        ))
    fig.update_layout(
        title=f"{model_name} — Training Loss",
        xaxis_title="Epoch", yaxis_title="Loss",
        **PLOTLY_LAYOUT,
    )
    return fig


def metrics_radar(metrics: Dict[str, float], model_name: str) -> go.Figure:
    keys = [k for k in ["accuracy", "precision", "recall", "f1", "roc_auc"] if k in metrics]
    vals = [metrics[k] for k in keys]
    vals_closed = vals + [vals[0]]
    keys_closed = keys + [keys[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_closed, theta=keys_closed, fill="toself",
        line_color=MODEL_COLORS.get(model_name, "#6C63FF"),
        fillcolor=MODEL_COLORS.get(model_name, "#6C63FF") + "30",
        name=model_name,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], gridcolor="#2D3250", color="#9AA3B2"),
            angularaxis=dict(gridcolor="#2D3250"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=f"{model_name} — Metrics",
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis", "plot_bgcolor")},
    )
    return fig


def confusion_matrix_chart(cm: List[List[int]], title: str) -> go.Figure:
    labels = ["Benign", "Attack"]
    z = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]
    text = [[str(v) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Pred Attack", "Pred Benign"],
        y=["True Attack", "True Benign"],
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#1A1D2E"], [1, "#6C63FF"]],
        showscale=False,
    ))
    fig.update_layout(title=title, **PLOTLY_LAYOUT)
    return fig


def gauge_chart(value: float, title: str, color: str = "#6C63FF") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number=dict(suffix="%", font=dict(size=28, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#9AA3B2"),
            bar=dict(color=color),
            bgcolor="#1A1D2E",
            bordercolor="#2D3250",
            steps=[
                dict(range=[0, 50],  color="#FF525215"),
                dict(range=[50, 75], color="#FFB30015"),
                dict(range=[75, 100], color="#00E67615"),
            ],
            threshold=dict(line=dict(color="#FAFAFA", width=2), thickness=0.75, value=value * 100),
        ),
        title=dict(text=title, font=dict(color="#9AA3B2", size=13)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(height=220, **{k: v for k, v in PLOTLY_LAYOUT.items()
                                      if k not in ("xaxis", "yaxis")})
    return fig


def feature_importance_chart(importance: List[float], feature_names: List[str],
                               top_k: int = 20) -> go.Figure:
    pairs = sorted(zip(importance, feature_names), reverse=True)[:top_k]
    vals, names = zip(*pairs)
    colors = px.colors.sequential.Plasma_r[:len(vals)]
    fig = go.Figure(go.Bar(
        y=list(names), x=list(vals), orientation="h",
        marker=dict(color=list(vals), colorscale="Viridis"),
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title=f"Top-{top_k} Feature Importances",
        xaxis_title="Importance Score",
        yaxis=dict(autorange="reversed", gridcolor="#2D3250"),
        **PLOTLY_LAYOUT,
    )
    return fig


# ── Status helpers ────────────────────────────────────────────────────────────
def file_status(path: Path, label: str):
    exists = path.exists()
    icon = "✅" if exists else "❌"
    color = COLORS["success"] if exists else COLORS["danger"]
    size = f" ({path.stat().st_size / 1024:.1f} KB)" if exists else ""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0'>"
        f"<span style='color:{color}'>{icon}</span>"
        f"<span style='font-size:0.9rem'>{label}</span>"
        f"<span style='color:#9AA3B2;font-size:0.8rem'>{size}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
