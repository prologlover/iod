"""
Stage 7 — Model Evaluation Dashboard
"""
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    inject_css, section, metric_tile, apply_dark_theme,
    COLORS, MODEL_COLORS, PLOTLY_LAYOUT, get_paths, list_figures,
    load_metrics_json,
)

st.set_page_config(page_title="Evaluation", page_icon="📈", layout="wide")
inject_css()

st.markdown("## 📈 Evaluation & Results")
st.markdown("<p style='color:#9AA3B2'>Compare all models across metrics, attack types and thresholds</p>",
            unsafe_allow_html=True)
st.markdown("---")

paths = get_paths()

# ── Load all available metrics ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_metrics():
    all_m = {}
    for p in sorted(paths["tables"].glob("*_metrics.json")):
        key = p.stem.replace("_metrics", "")
        try:
            all_m[key] = json.loads(p.read_text())
        except Exception:
            pass
    return all_m

all_metrics = load_all_metrics()

if not all_metrics:
    st.warning("No evaluation results found. Run `python scripts/evaluate.py` first.")
    st.info("Example: `python scripts/evaluate.py --attack_type false_state --snapshots 50`")
    st.stop()

# ── Filter controls ───────────────────────────────────────────────────────────
attack_keys = list(all_metrics.keys())
selected_attack = st.selectbox("Attack scenario", attack_keys, key="eval_atk")

metrics_data = all_metrics[selected_attack]

# ── Gather model results ──────────────────────────────────────────────────────
MODEL_KEYS = ["GAT+GRU", "GraphSAGE+GRU", "MLP", "LSTM", "1D-CNN", "GCN"]
available_models = [m for m in MODEL_KEYS if m in metrics_data]

if not available_models:
    available_models = list(metrics_data.keys())

# ── Top row: per-model metric tiles ──────────────────────────────────────────
section("Model Metric Overview", "🏆")
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

import pandas as pd

rows = []
for m in available_models:
    md = metrics_data.get(m, {})
    row = {"Model": m}
    for mc in METRIC_COLS:
        row[mc.upper()] = round(md.get(mc, 0.0), 4)
    rows.append(row)

if rows:
    df_m = pd.DataFrame(rows).set_index("Model")

    # Highlight best per column
    def highlight_max(s):
        is_max = s == s.max()
        return [f"background-color:{COLORS['primary']}30; color:{COLORS['primary']}; font-weight:700"
                if v else "" for v in is_max]

    styled = df_m.style.apply(highlight_max).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Bar Charts", "🎯 Confusion Matrix", "📉 ROC Curves",
    "📡 Radar Charts", "🖼️ Output Figures"
])

# ── TAB 1: Bar charts ─────────────────────────────────────────────────────────
with tab1:
    metric_sel = st.selectbox("Metric to compare",
                               ["F1", "ACCURACY", "PRECISION", "RECALL", "ROC_AUC"],
                               key="bar_metric")
    mc_map = {"F1": "f1", "ACCURACY": "accuracy", "PRECISION": "precision",
              "RECALL": "recall", "ROC_AUC": "roc_auc"}
    mc_key = mc_map[metric_sel]

    names = available_models
    vals = [metrics_data.get(m, {}).get(mc_key, 0) for m in names]
    colors = [MODEL_COLORS.get(m, COLORS["primary"]) for m in names]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker=dict(color=colors, opacity=0.85),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
    ))
    fig.add_hline(y=max(vals), line_dash="dot",
                  line_color=COLORS["success"], opacity=0.5,
                  annotation_text=f"Best: {max(vals):.4f}",
                  annotation_position="right")
    fig.update_layout(
        title=f"{metric_sel} Comparison — {selected_attack.replace('_', ' ').title()}",
        yaxis=dict(range=[0, min(1.15, max(vals) * 1.2)], gridcolor="#2D3250"),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    # All metrics grouped bar
    section("All Metrics — Grouped Bar", "📊")
    fig2 = go.Figure()
    for mc, mc_lbl in zip(METRIC_COLS, METRIC_LABELS):
        fig2.add_trace(go.Bar(
            name=mc_lbl,
            x=available_models,
            y=[metrics_data.get(m, {}).get(mc, 0) for m in available_models],
        ))
    fig2.update_layout(
        barmode="group", title="All Metrics Grouped",
        yaxis=dict(range=[0, 1.1], gridcolor="#2D3250"),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig2), use_container_width=True)

# ── TAB 2: Confusion Matrix ───────────────────────────────────────────────────
with tab2:
    section("Confusion Matrix", "🎯")
    model_cm = st.selectbox("Model", available_models, key="cm_model")
    md = metrics_data.get(model_cm, {})
    cm = md.get("confusion_matrix", None)

    if cm:
        cm = [[int(v) for v in row] for row in cm]
        tp = cm[1][1]; fn = cm[1][0]; fp = cm[0][1]; tn = cm[0][0]
        total = tp + fn + fp + tn

        metric_cols = st.columns(4)
        for col, (val, lbl, clr) in zip(metric_cols, [
            (f"{tp:,}", "True Positives",  COLORS["success"]),
            (f"{tn:,}", "True Negatives",  COLORS["primary"]),
            (f"{fp:,}", "False Positives", COLORS["warning"]),
            (f"{fn:,}", "False Negatives", COLORS["danger"]),
        ]):
            with col:
                metric_tile(val, lbl, color=clr)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            fig_cm = go.Figure(go.Heatmap(
                z=[[tp, fn], [fp, tn]],
                x=["Predicted Attack", "Predicted Benign"],
                y=["True Attack", "True Benign"],
                text=[[f"TP={tp}", f"FN={fn}"], [f"FP={fp}", f"TN={tn}"]],
                texttemplate="%{text}",
                colorscale=[[0, "#1A1D2E"], [1, "#6C63FF"]],
                showscale=True,
            ))
            fig_cm.update_layout(title=f"{model_cm} — Confusion Matrix",
                                   height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(apply_dark_theme(fig_cm), use_container_width=True)

        with c2:
            # Normalised CM
            norm_cm = [[tp/max(1,tp+fn), fn/max(1,tp+fn)],
                        [fp/max(1,fp+tn), tn/max(1,fp+tn)]]
            fig_ncm = go.Figure(go.Heatmap(
                z=norm_cm,
                x=["Predicted Attack", "Predicted Benign"],
                y=["True Attack", "True Benign"],
                text=[[f"{v:.2%}" for v in row] for row in norm_cm],
                texttemplate="%{text}",
                colorscale="Blues", zmin=0, zmax=1,
            ))
            fig_ncm.update_layout(title=f"{model_cm} — Normalised CM",
                                   height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(apply_dark_theme(fig_ncm), use_container_width=True)

        # Per-attack-type breakdown
        per_attack = md.get("per_attack_metrics", {})
        if per_attack:
            section("Per-Attack-Type Metrics", "⚔️")
            pa_rows = [{"Attack Type": k, **{mc: round(v.get(mc, 0), 4)
                        for mc in METRIC_COLS}} for k, v in per_attack.items()]
            st.dataframe(pd.DataFrame(pa_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Confusion matrix data not available for this model/attack combination.")

# ── TAB 3: ROC Curves ────────────────────────────────────────────────────────
with tab3:
    section("ROC / PR Curves", "📉")
    fig_roc = go.Figure()
    fig_pr = go.Figure()

    for m in available_models:
        md = metrics_data.get(m, {})
        roc = md.get("roc_curve", {})
        pr = md.get("pr_curve", {})
        color = MODEL_COLORS.get(m, COLORS["primary"])
        auc_val = md.get("roc_auc", 0)
        ap_val = md.get("average_precision", 0)

        if roc:
            fig_roc.add_trace(go.Scatter(
                x=roc.get("fpr", []), y=roc.get("tpr", []),
                mode="lines", name=f"{m} (AUC={auc_val:.3f})",
                line=dict(color=color, width=2),
            ))
        if pr:
            fig_pr.add_trace(go.Scatter(
                x=pr.get("recall", []), y=pr.get("precision", []),
                mode="lines", name=f"{m} (AP={ap_val:.3f})",
                line=dict(color=color, width=2),
            ))

    # Diagonal
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   name="Random", line=dict(color="#555", dash="dot")))
    fig_roc.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR",
                           **PLOTLY_LAYOUT)
    fig_pr.update_layout(title="Precision-Recall Curves",
                          xaxis_title="Recall", yaxis_title="Precision", **PLOTLY_LAYOUT)

    c1, c2 = st.columns(2)
    with c1:
        if any(metrics_data.get(m, {}).get("roc_curve") for m in available_models):
            st.plotly_chart(apply_dark_theme(fig_roc), use_container_width=True)
        else:
            st.info("ROC curve data not available. Ensure evaluator saves roc_curve dict.")
    with c2:
        if any(metrics_data.get(m, {}).get("pr_curve") for m in available_models):
            st.plotly_chart(apply_dark_theme(fig_pr), use_container_width=True)
        else:
            st.info("PR curve data not available.")

    # Metric threshold analysis
    section("Metric vs Threshold", "🎚️")
    thresh_model = st.selectbox("Model", available_models, key="thresh_model")
    thresh_data = metrics_data.get(thresh_model, {}).get("threshold_analysis", None)

    if thresh_data:
        thresholds = thresh_data.get("thresholds", [])
        fig_thr = go.Figure()
        for mc in ["f1", "precision", "recall"]:
            if mc in thresh_data:
                fig_thr.add_trace(go.Scatter(
                    x=thresholds, y=thresh_data[mc], mode="lines",
                    name=mc.upper(),
                ))
        fig_thr.update_layout(title=f"{thresh_model} — Metrics vs Threshold",
                               xaxis_title="Threshold", yaxis_title="Score",
                               **PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_thr), use_container_width=True)
    else:
        st.info("Threshold analysis data not found in metrics JSON.")

# ── TAB 4: Radar Charts ───────────────────────────────────────────────────────
with tab4:
    section("Model Radar Comparison", "📡")
    radar_models = st.multiselect("Models to overlay", available_models,
                                   default=available_models, key="radar_sel")
    radar_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    radar_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    closed = radar_labels + [radar_labels[0]]

    fig_rad = go.Figure()
    for m in radar_models:
        md = metrics_data.get(m, {})
        vals = [md.get(k, 0) for k in radar_keys]
        vals_c = vals + [vals[0]]
        color = MODEL_COLORS.get(m, COLORS["primary"])
        fig_rad.add_trace(go.Scatterpolar(
            r=vals_c, theta=closed, fill="toself",
            name=m, line_color=color,
            fillcolor=color + "20",
        ))

    fig_rad.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], gridcolor="#2D3250", color="#9AA3B2"),
            angularaxis=dict(gridcolor="#2D3250"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Model Performance Radar",
        height=500,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "plot_bgcolor")},
    )
    st.plotly_chart(apply_dark_theme(fig_rad), use_container_width=True)

    # Swarm-specific metrics
    section("Swarm-Specific Metrics", "🛸")
    swarm_keys = ["localization_accuracy", "detection_latency", "per_node_fpr"]
    swarm_rows = []
    for m in available_models:
        md = metrics_data.get(m, {})
        row = {"Model": m}
        for sk in swarm_keys:
            row[sk] = round(md.get(sk, float("nan")), 4)
        swarm_rows.append(row)
    if swarm_rows:
        st.dataframe(pd.DataFrame(swarm_rows).set_index("Model"), use_container_width=True)

# ── TAB 5: Figures ────────────────────────────────────────────────────────────
with tab5:
    section("Output Figures", "🖼️")
    figures = list_figures()
    if not figures:
        st.info("No output figures found. Run evaluate.py or explain.py to generate plots.")
    else:
        filter_kw = st.text_input("Filter by name", "", key="fig_filter")
        filtered = [f for f in figures if filter_kw.lower() in f.name.lower()]

        n_cols = st.slider("Columns", 1, 4, 2, key="fig_cols")
        cols = st.columns(n_cols)
        for i, fig_path in enumerate(filtered):
            with cols[i % n_cols]:
                st.markdown(f"**{fig_path.name}**")
                try:
                    st.image(str(fig_path), use_container_width=True)
                except Exception as e:
                    st.error(f"Cannot display: {e}")
