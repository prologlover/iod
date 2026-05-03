"""
Stage 6 — Model Training Monitor & Launcher
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    inject_css, section, metric_tile, apply_dark_theme,
    COLORS, MODEL_COLORS, PLOTLY_LAYOUT, load_training_history,
    list_model_checkpoints, get_paths,
)

st.set_page_config(page_title="Model Training", page_icon="🏋️", layout="wide")
inject_css()

st.markdown("## 🏋️ Model Training")
st.markdown("<p style='color:#9AA3B2'>Configure, launch and monitor model training</p>",
            unsafe_allow_html=True)
st.markdown("---")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Training Curves", "🚀 Launch Training", "⚙️ Model Architectures"
])

# ── TAB 1: Training Curves ────────────────────────────────────────────────────
with tab1:
    paths = get_paths()
    checkpoints = list_model_checkpoints()

    section("Saved Checkpoints", "💾")
    if checkpoints:
        import pandas as pd
        cp_data = []
        for cp in checkpoints:
            hist_p = paths["models"] / f"history_{cp.stem.replace('best_', '')}.json"
            final_loss = None
            if hist_p.exists():
                h = json.loads(hist_p.read_text())
                final_loss = h.get("val_loss", [None])[-1]
            cp_data.append({
                "Checkpoint": cp.name,
                "Size (KB)": round(cp.stat().st_size / 1024, 1),
                "Modified": cp.stat().st_mtime,
                "Final Val Loss": round(final_loss, 5) if final_loss else "N/A",
            })
        cp_df = pd.DataFrame(cp_data)
        cp_df["Modified"] = pd.to_datetime(cp_df["Modified"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(cp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No model checkpoints found. Train models first using the 🚀 Launch Training tab.")

    section("Training History", "📈")
    history_files = sorted(paths["models"].glob("history_*.json"))
    model_names = [p.stem.replace("history_", "") for p in history_files]

    if not model_names:
        st.warning("No training histories found. Train at least one model to see curves.")
    else:
        selected = st.multiselect("Select models to compare", model_names,
                                   default=model_names[:min(3, len(model_names))],
                                   key="curve_select")
        metric_sel = st.radio("Metric", ["loss", "f1", "accuracy"], horizontal=True,
                               key="curve_metric")

        if selected:
            c1, c2 = st.columns(2)
            for col_idx, phase in enumerate(["train", "val"]):
                fig = go.Figure()
                for mname in selected:
                    h = load_training_history(mname)
                    if h is None:
                        continue
                    key = f"{phase}_{metric_sel}"
                    if key not in h:
                        key = f"{phase}_loss"
                    vals = h.get(key, [])
                    if not vals:
                        continue
                    color = MODEL_COLORS.get(mname.upper(), COLORS["primary"])
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(vals) + 1)),
                        y=vals, mode="lines+markers",
                        name=mname.upper(),
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                    ))
                fig.update_layout(
                    title=f"{phase.capitalize()} {metric_sel.upper()}",
                    xaxis_title="Epoch",
                    yaxis_title=metric_sel.upper(),
                    **PLOTLY_LAYOUT,
                )
                target = c1 if col_idx == 0 else c2
                with target:
                    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

        # Loss + F1 overlay for a single model
        if len(selected) == 1:
            section("Loss + F1 Overlay (dual axis)", "📉")
            mname = selected[0]
            h = load_training_history(mname)
            if h:
                fig_dual = go.Figure()
                if "train_loss" in h:
                    fig_dual.add_trace(go.Scatter(
                        x=list(range(1, len(h["train_loss"]) + 1)),
                        y=h["train_loss"], name="Train Loss",
                        line=dict(color=COLORS["primary"], width=2),
                        yaxis="y1",
                    ))
                if "val_loss" in h:
                    fig_dual.add_trace(go.Scatter(
                        x=list(range(1, len(h["val_loss"]) + 1)),
                        y=h["val_loss"], name="Val Loss",
                        line=dict(color=COLORS["danger"], width=2, dash="dash"),
                        yaxis="y1",
                    ))
                if "val_f1" in h:
                    fig_dual.add_trace(go.Scatter(
                        x=list(range(1, len(h["val_f1"]) + 1)),
                        y=h["val_f1"], name="Val F1",
                        line=dict(color=COLORS["success"], width=2),
                        yaxis="y2",
                    ))
                fig_dual.update_layout(
                    title=f"{mname.upper()} — Loss & F1",
                    yaxis=dict(title="Loss", gridcolor="#2D3250", side="left"),
                    yaxis2=dict(title="F1 Score", overlaying="y", side="right",
                                range=[0, 1], showgrid=False),
                    xaxis_title="Epoch",
                    **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis",)},
                )
                st.plotly_chart(apply_dark_theme(fig_dual), use_container_width=True)

# ── TAB 2: Launch Training ───────────────────────────────────────────────────
with tab2:
    section("Launch Training Pipeline", "🚀")
    st.info("This tab launches training scripts as subprocesses. Output will appear below.")

    c1, c2 = st.columns(2)
    with c1:
        model_choice = st.selectbox(
            "Model to train",
            ["GAT+GRU (Proposed)", "GraphSAGE+GRU", "All Baselines (MLP+LSTM+CNN+GCN)",
             "Single: MLP", "Single: LSTM", "Single: 1D-CNN", "Single: GCN"],
            key="train_model"
        )
        attack_type = st.selectbox(
            "Attack Type",
            ["false_state", "intermittent", "colluding", "delay"],
            key="train_atk"
        )
        graph_type = st.selectbox("Graph Topology",
                                   ["knn", "distance", "hexagonal"], key="train_graph")

    with c2:
        n_epochs = st.slider("Epochs", 5, 200, 30, key="train_epochs")
        n_snapshots = st.slider("Snapshots (0 = full dataset)", 0, 500, 50,
                                 key="train_snaps",
                                 help="Use small values for quick testing")
        lr = st.number_input("Learning rate", value=1e-3, format="%.5f", key="train_lr")

    st.markdown("<br>", unsafe_allow_html=True)
    col_run, col_stop = st.columns([2, 1])

    script_cmd = None
    snaps_arg = f"--snapshots {n_snapshots}" if n_snapshots > 0 else ""

    if "GAT+GRU" in model_choice or "GraphSAGE" in model_choice:
        model_arg = "gat" if "GAT" in model_choice else "graphsage"
        script_cmd = (
            f'"{VENV_PYTHON}" -u scripts/train_gnn_temporal.py '
            f'--model {model_arg} --attack_type {attack_type} '
            f'--graph_type {graph_type} --epochs {n_epochs} {snaps_arg}'
        )
    elif "All Baselines" in model_choice:
        script_cmd = (
            f'"{VENV_PYTHON}" -u scripts/train_baselines.py '
            f'--attack_type {attack_type} --epochs {n_epochs} {snaps_arg}'
        )
    else:
        single_map = {"MLP": "mlp", "LSTM": "lstm", "1D-CNN": "cnn", "GCN": "gcn"}
        for k, v in single_map.items():
            if k in model_choice:
                script_cmd = (
                    f'"{VENV_PYTHON}" -u scripts/train_baselines.py '
                    f'--model_types {v} --attack_type {attack_type} '
                    f'--epochs {n_epochs} {snaps_arg}'
                )
                break

    with col_run:
        run_btn = st.button("▶ Start Training", type="primary", use_container_width=True,
                             key="run_train")
    with col_stop:
        st.button("⏸ Training runs in background", disabled=True, use_container_width=True)

    if script_cmd:
        st.code(script_cmd, language="bash")

    if run_btn and script_cmd:
        st.markdown("**Training output:**")
        placeholder = st.empty()
        log_text = ""
        try:
            proc = subprocess.Popen(
                script_cmd, shell=True, cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
            )
            for line in proc.stdout:
                log_text += line
                placeholder.code(log_text[-5000:], language="text")
            proc.wait()
            if proc.returncode == 0:
                st.success("Training completed successfully! Refresh Training Curves tab.")
            else:
                st.error(f"Training failed with exit code {proc.returncode}")
        except Exception as e:
            st.error(f"Failed to launch: {e}")

# ── TAB 3: Architectures ──────────────────────────────────────────────────────
with tab3:
    section("Model Architecture Overview", "⚙️")

    architectures = {
        "GAT+GRU (Proposed)": {
            "color": MODEL_COLORS["GAT+GRU"],
            "layers": [
                ("Input", "Node features: shape (N, F)"),
                ("GAT Layer 1", "8 heads × 64 dim → 512 dim, ELU"),
                ("GAT Layer 2", "8 heads × 64 dim → 512 dim, ELU"),
                ("Dropout", "p=0.3"),
                ("GRU", "hidden=256, 2 layers, bidirectional"),
                ("Temporal Attn", "Learned timestep weighting"),
                ("Classifier", "FC 256→2, Softmax"),
            ],
            "params": "~2.1M",
            "innovation": "Combines spatial graph attention with temporal GRU processing",
        },
        "GraphSAGE+GRU": {
            "color": MODEL_COLORS["GraphSAGE+GRU"],
            "layers": [
                ("Input", "Node features: shape (N, F)"),
                ("SAGE Conv 1", "256 dim, mean aggregation, ReLU"),
                ("SAGE Conv 2", "256 dim, mean aggregation, ReLU"),
                ("GRU", "hidden=128, 2 layers"),
                ("Classifier", "FC 128→2, Softmax"),
            ],
            "params": "~0.9M",
            "innovation": "Inductive learning via neighbor sampling aggregation",
        },
        "MLP Baseline": {
            "color": MODEL_COLORS["MLP"],
            "layers": [
                ("Input", "Flattened features: shape (N×F)"),
                ("FC 1", "256 dim, BatchNorm, ReLU, Dropout(0.3)"),
                ("FC 2", "128 dim, BatchNorm, ReLU, Dropout(0.3)"),
                ("FC 3", "64 dim, ReLU"),
                ("Output", "2-class softmax"),
            ],
            "params": "~0.3M",
            "innovation": "No graph structure — pure feature-based baseline",
        },
        "LSTM Baseline": {
            "color": MODEL_COLORS["LSTM"],
            "layers": [
                ("Input", "Temporal features: shape (T, N, F)"),
                ("LSTM", "256 hidden, 2 layers, bidirectional"),
                ("Temporal Attn", "Additive attention over timesteps"),
                ("Classifier", "FC 512→2, Softmax"),
            ],
            "params": "~1.5M",
            "innovation": "Temporal LSTM without graph topology",
        },
        "1D-CNN Baseline": {
            "color": MODEL_COLORS["1D-CNN"],
            "layers": [
                ("Input", "Temporal features: shape (F, T)"),
                ("Conv 1", "64 filters, kernel=3, ReLU"),
                ("Conv 2", "128 filters, kernel=3, ReLU"),
                ("Conv 3", "256 filters, kernel=3, ReLU"),
                ("Global Avg Pool", "Temporal pooling"),
                ("Classifier", "FC 256→2, Softmax"),
            ],
            "params": "~0.4M",
            "innovation": "Convolutional feature extraction over time dimension",
        },
        "GCN Baseline": {
            "color": MODEL_COLORS["GCN"],
            "layers": [
                ("Input", "Single-snapshot features: (N, F)"),
                ("GCN Layer 1", "256 dim, ReLU"),
                ("GCN Layer 2", "128 dim, ReLU"),
                ("GCN Layer 3", "64 dim, ReLU"),
                ("Classifier", "FC 64→2, Softmax"),
            ],
            "params": "~0.2M",
            "innovation": "Graph convolution without temporal dimension",
        },
    }

    for name, info in architectures.items():
        with st.expander(f"**{name}** — {info['params']} params", expanded=False):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Innovation:** {info['innovation']}")
                st.markdown("**Layer Stack:**")
                for layer, desc in info["layers"]:
                    st.markdown(
                        f"<div style='display:flex;gap:12px;padding:4px 0;border-bottom:1px solid #2D3250'>"
                        f"<span style='color:{info['color']};font-weight:600;min-width:140px'>{layer}</span>"
                        f"<span style='color:#9AA3B2;font-size:0.88rem'>{desc}</span></div>",
                        unsafe_allow_html=True,
                    )
            with c2:
                st.markdown(f"**Parameters:** `{info['params']}`")
                st.markdown(f"**Color:**")
                st.markdown(f"<div style='width:40px;height:20px;background:{info['color']};border-radius:4px'></div>",
                            unsafe_allow_html=True)
