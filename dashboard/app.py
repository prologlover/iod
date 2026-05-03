"""
Byzantine Attack Detection in Drone Swarms — Research Dashboard
Main entry point.
Run with:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import inject_css, get_paths, COLORS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Byzantine Drone Detection",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <div style='font-size:2.5rem'>🛸</div>
        <div class='title'>Byzantine Detection</div>
        <div class='subtitle'>Drone Swarm Security · Research Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# ── Home page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem 0'>
    <div style='font-size:4rem'>🛸</div>
    <h1 style='font-size:2.2rem; font-weight:800; margin:0.5rem 0 0.3rem 0'>
        Explainable Detection of<br>Byzantine Attacks in Drone Swarms
    </h1>
    <p style='color:#9AA3B2; font-size:1.05rem; max-width:700px; margin:0 auto'>
        A spatio-temporal GAT+GRU pipeline with full explainability via GNNExplainer
        and permutation-based feature importance.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pipeline overview cards ───────────────────────────────────────────────────
stages = [
    ("📊", "Data Explorer",     "Explore the UAV dataset — 54K rows, 37 features, 5 attack types",          "Stage 2"),
    ("🕸️",  "Graph Builder",    "Visualise swarm topology: KNN, distance-threshold, hexagonal lattices",    "Stage 3"),
    ("⚔️",  "Attack Injection", "Inspect Byzantine attack effects — false-state, intermittent, colluding, delay", "Stage 4"),
    ("🏋️",  "Model Training",   "Monitor training curves and configure hyperparameters",                    "Stage 6"),
    ("📈", "Evaluation",        "Compare all models — confusion matrices, ROC/PR curves, metric tables",    "Stage 7"),
    ("🔍", "Explainability",    "GNNExplainer subgraphs, temporal importance heatmaps, feature rankings",   "Stage 8"),
    ("🧪", "Ablation Study",    "Component, attack-type, attacker-ratio and topology ablations",            "Stage 9"),
]

cols = st.columns(3)
for i, (icon, title, desc, stage) in enumerate(stages):
    with cols[i % 3]:
        st.markdown(f"""
        <div class='dash-card' style='min-height:160px'>
            <div style='font-size:2rem'>{icon}</div>
            <div style='font-weight:700; font-size:1rem; margin:0.4rem 0 0.2rem'>{title}</div>
            <div style='color:#9AA3B2; font-size:0.82rem; line-height:1.4'>{desc}</div>
            <div style='margin-top:0.6rem'>
                <span class='badge badge-info'>{stage}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Quick-status row ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>⚡ Pipeline Status</div>", unsafe_allow_html=True)

paths = get_paths()
status_items = [
    ("Dataset CSV",       paths["raw"] / "UAVs-Dataset-Under-Normal-and-Cyberattacks" / "Dataset_T-ITS.csv"),
    ("Train split",       paths["processed"] / "train.parquet"),
    ("Val split",         paths["processed"] / "val.parquet"),
    ("Test split",        paths["processed"] / "test.parquet"),
    ("GAT+GRU model",     paths["models"] / "best_gat_temporal.pt"),
    ("MLP baseline",      paths["models"] / "best_mlp.pt"),
    ("LSTM baseline",     paths["models"] / "best_lstm.pt"),
    ("CNN baseline",      paths["models"] / "best_cnn.pt"),
    ("GCN baseline",      paths["models"] / "best_gcn.pt"),
    ("Eval metrics JSON", paths["tables"] / "false_state_metrics.json"),
]

c1, c2 = st.columns(2)
for i, (label, path) in enumerate(status_items):
    col = c1 if i % 2 == 0 else c2
    with col:
        exists = path.exists()
        icon = "✅" if exists else "⬜"
        color = COLORS["success"] if exists else "#555"
        size = f"  ·  {path.stat().st_size/1024:.0f} KB" if exists else ""
        st.markdown(
            f"<div style='padding:3px 0; font-size:0.88rem'>"
            f"<span style='color:{color}'>{icon}</span> "
            f"<b>{label}</b><span style='color:#9AA3B2'>{size}</span></div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation hint ───────────────────────────────────────────────────────────
st.info("👈  Use the **sidebar** to navigate between pipeline stages.")

st.markdown("""
<div style='text-align:center; color:#555; font-size:0.8rem; padding:2rem 0 0.5rem'>
    Built with Streamlit · PyTorch Geometric · Explainable AI
</div>
""", unsafe_allow_html=True)
