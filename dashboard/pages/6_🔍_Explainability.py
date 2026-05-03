"""
Stage 8 — Explainability Dashboard
GNNExplainer, temporal importance, feature rankings
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
    COLORS, PLOTLY_LAYOUT, get_paths, list_figures,
)

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
inject_css()

st.markdown("## 🔍 Explainability")
st.markdown("<p style='color:#9AA3B2'>GNNExplainer subgraphs · Temporal importance · Permutation feature ranking</p>",
            unsafe_allow_html=True)
st.markdown("---")

paths = get_paths()

# ── Load explanation data ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_explain_data():
    data = {}
    for p in sorted(paths["tables"].glob("explain_*.json")):
        try:
            data[p.stem.replace("explain_", "")] = json.loads(p.read_text())
        except Exception:
            pass
    return data

explain_data = load_explain_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "🕸️ GNNExplainer", "⏱️ Temporal Importance",
    "📊 Feature Rankings", "🔥 Attention Heatmaps"
])

# ── TAB 1: GNNExplainer ──────────────────────────────────────────────────────
with tab1:
    section("GNNExplainer — Explanatory Subgraphs", "🕸️")

    if not explain_data:
        st.info("No explanation data found. Run `python scripts/explain.py` to generate explanations.")
        st.markdown("**Command:**")
        st.code('python scripts/explain.py --attack_type false_state --snapshots 50', language='bash')

        # Demo: synthetic subgraph
        section("Demo: Synthetic Explanation Subgraph", "🎭")
        rng = np.random.default_rng(42)
        n_nodes = 12
        positions = rng.uniform(0, 100, (n_nodes, 2))
        labels = rng.choice([0, 1], n_nodes, p=[0.7, 0.3])
        importance = rng.dirichlet(np.ones(n_nodes))

        # Edges
        edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)
                 if np.linalg.norm(positions[i] - positions[j]) < 35]
        edge_importance = rng.uniform(0, 1, len(edges))

        fig = go.Figure()
        for (i, j), ew in zip(edges, edge_importance):
            alpha = max(0.1, ew)
            fig.add_trace(go.Scatter(
                x=[positions[i, 0], positions[j, 0], None],
                y=[positions[i, 1], positions[j, 1], None],
                mode="lines",
                line=dict(color=f"rgba(108,99,255,{alpha:.2f})", width=max(0.5, ew * 4)),
                showlegend=False, hoverinfo="skip",
            ))

        for label, color, name in [(0, COLORS["benign"], "Benign"), (1, COLORS["danger"], "Attacker")]:
            mask = labels == label
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=positions[mask, 0], y=positions[mask, 1],
                mode="markers+text",
                marker=dict(size=importance[mask] * 35 + 10, color=color,
                            line=dict(color="white", width=1.5)),
                text=[str(i) for i in np.where(mask)[0]],
                textposition="top center",
                name=f"{name} (demo)",
                hovertemplate=f"{name} %{{text}}<br>Importance: %{{marker.size:.2f}}<extra></extra>",
            ))

        fig.update_layout(
            title="Demo: GNNExplainer Subgraph (node size = importance)",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            height=450, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        st.caption("Node size reflects importance score. Edge width reflects edge importance.")

    else:
        scenario = st.selectbox("Explanation scenario", list(explain_data.keys()), key="exp_sc")
        exp = explain_data[scenario]

        # Node importance
        if "node_importance" in exp:
            ni = np.array(exp["node_importance"])
            labels_e = np.array(exp.get("labels", [0] * len(ni)))
            positions_e = np.array(exp.get("positions", np.random.uniform(0, 100, (len(ni), 2))))
            edges_e = exp.get("edges", [])
            edge_imp = exp.get("edge_importance", [0.5] * len(edges_e))

            fig = go.Figure()
            for (i, j), ew in zip(edges_e, edge_imp):
                alpha = max(0.1, float(ew))
                fig.add_trace(go.Scatter(
                    x=[positions_e[i, 0], positions_e[j, 0], None],
                    y=[positions_e[i, 1], positions_e[j, 1], None],
                    mode="lines",
                    line=dict(color=f"rgba(108,99,255,{alpha:.2f})", width=max(0.5, ew * 5)),
                    showlegend=False, hoverinfo="skip",
                ))

            for lbl, color, name in [(0, COLORS["benign"], "Benign"), (1, COLORS["danger"], "Attacker")]:
                mask = labels_e == lbl
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=positions_e[mask, 0], y=positions_e[mask, 1],
                    mode="markers",
                    marker=dict(
                        size=(ni[mask] * 30 + 8).clip(8, 40),
                        color=color, opacity=0.9,
                        line=dict(color="white", width=1.5),
                        colorscale="Plasma",
                    ),
                    name=name,
                ))

            fig.update_layout(
                title=f"GNNExplainer — {scenario}",
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                height=480, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

        # Edge importance distribution
        if "edge_importance" in exp:
            ei = np.array(exp["edge_importance"])
            fig_ei = go.Figure(go.Histogram(x=ei, marker_color=COLORS["primary"], opacity=0.8))
            fig_ei.update_layout(title="Edge Importance Distribution",
                                  xaxis_title="Importance", yaxis_title="Count", **PLOTLY_LAYOUT)
            st.plotly_chart(apply_dark_theme(fig_ei), use_container_width=True)

    # Display saved explanation figures
    exp_figs = [f for f in list_figures() if "explain" in f.name or "gnn" in f.name.lower()]
    if exp_figs:
        section("Saved Explanation Plots", "🖼️")
        cols = st.columns(2)
        for i, fp in enumerate(exp_figs[:6]):
            with cols[i % 2]:
                st.markdown(f"**{fp.name}**")
                st.image(str(fp), use_container_width=True)

# ── TAB 2: Temporal Importance ────────────────────────────────────────────────
with tab2:
    section("Temporal Contribution Analysis", "⏱️")

    # Check for saved temporal data
    temp_data = None
    for sc, exp in explain_data.items():
        if "temporal_importance" in exp:
            temp_data = (sc, exp["temporal_importance"])
            break

    if temp_data is None:
        # Synthetic demo
        st.info("No temporal explanation data found. Showing synthetic demo.")
        rng = np.random.default_rng(42)
        T = 20
        n_nodes = 8
        synth_temp = rng.dirichlet(np.ones(T), size=n_nodes)  # (N, T)
        sc_name = "Demo"
        temp_arr = synth_temp
    else:
        sc_name, raw = temp_data
        temp_arr = np.array(raw) if not isinstance(raw, np.ndarray) else raw

    fig_th = go.Figure(go.Heatmap(
        z=temp_arr,
        x=[f"t={i}" for i in range(temp_arr.shape[-1])],
        y=[f"Node {i}" for i in range(temp_arr.shape[0])],
        colorscale=[[0, "#0E1117"], [0.5, "#6C63FF"], [1, "#FF5252"]],
        colorbar=dict(title="Importance"),
    ))
    fig_th.update_layout(
        title=f"Temporal Importance Heatmap — {sc_name}",
        height=max(300, temp_arr.shape[0] * 30),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig_th), use_container_width=True)

    # Mean importance over time
    section("Mean Temporal Importance Over Time", "📉")
    mean_t = temp_arr.mean(axis=0)
    std_t = temp_arr.std(axis=0)
    t_range = list(range(len(mean_t)))

    fig_mt = go.Figure()
    fig_mt.add_trace(go.Scatter(
        x=t_range + t_range[::-1],
        y=(mean_t + std_t).tolist() + (mean_t - std_t).tolist()[::-1],
        fill="toself", fillcolor=COLORS["primary"] + "20",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig_mt.add_trace(go.Scatter(
        x=t_range, y=mean_t.tolist(), mode="lines+markers",
        name="Mean Importance",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=6),
    ))
    fig_mt.update_layout(title="Mean Temporal Importance ± 1σ",
                          xaxis_title="Timestep", yaxis_title="Importance", **PLOTLY_LAYOUT)
    st.plotly_chart(apply_dark_theme(fig_mt), use_container_width=True)

# ── TAB 3: Feature Rankings ───────────────────────────────────────────────────
with tab3:
    section("Permutation Feature Importance", "📊")

    feat_data = None
    for sc, exp in explain_data.items():
        if "feature_importance" in exp:
            feat_data = (sc, exp["feature_importance"])
            break

    if feat_data is None:
        st.info("No feature importance data found. Showing synthetic demo.")
        rng = np.random.default_rng(42)
        feature_names_d = [f"feature_{i}" for i in range(20)]
        importance_vals = rng.exponential(0.5, 20)
        importance_vals /= importance_vals.sum()
    else:
        sc_name, fi_raw = feat_data
        if isinstance(fi_raw, dict):
            feature_names_d = list(fi_raw.keys())
            importance_vals = np.array(list(fi_raw.values()))
        else:
            importance_vals = np.array(fi_raw)
            feature_names_d = [f"feature_{i}" for i in range(len(importance_vals))]

    top_k = st.slider("Top-K features", 5, min(30, len(importance_vals)), 20, key="feat_k")
    pairs = sorted(zip(importance_vals, feature_names_d), reverse=True)[:top_k]
    vals_f, names_f = zip(*pairs)

    fig_fi = go.Figure(go.Bar(
        y=list(names_f), x=list(vals_f), orientation="h",
        marker=dict(color=list(vals_f), colorscale="Viridis"),
        text=[f"{v:.4f}" for v in vals_f], textposition="outside",
    ))
    fig_fi.update_layout(
        title=f"Top-{top_k} Feature Importances (Permutation)",
        xaxis_title="Importance Drop (when permuted)",
        yaxis=dict(autorange="reversed", gridcolor="#2D3250"),
        height=max(400, top_k * 22),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig_fi), use_container_width=True)

    # Cumulative importance
    section("Cumulative Feature Importance", "📈")
    all_pairs = sorted(zip(importance_vals, feature_names_d), reverse=True)
    all_vals = [v for v, _ in all_pairs]
    cumulative = np.cumsum(all_vals) / sum(all_vals)

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=list(range(1, len(cumulative) + 1)),
        y=cumulative.tolist(), mode="lines+markers",
        name="Cumulative Importance",
        line=dict(color=COLORS["secondary"], width=2),
        fill="tozeroy", fillcolor=COLORS["secondary"] + "20",
    ))
    fig_cum.add_hline(y=0.8, line_dash="dot", line_color=COLORS["warning"],
                       annotation_text="80% threshold")
    fig_cum.add_hline(y=0.95, line_dash="dot", line_color=COLORS["danger"],
                       annotation_text="95% threshold")
    fig_cum.update_layout(
        title="Cumulative Feature Importance",
        xaxis_title="# Features", yaxis_title="Cumulative Importance",
        yaxis=dict(range=[0, 1.05], gridcolor="#2D3250"),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig_cum), use_container_width=True)

# ── TAB 4: Attention Heatmaps ────────────────────────────────────────────────
with tab4:
    section("GAT Attention Weight Heatmaps", "🔥")

    attn_data = None
    for sc, exp in explain_data.items():
        if "attention_weights" in exp:
            attn_data = (sc, exp["attention_weights"])
            break

    if attn_data is None:
        st.info("No attention weight data found. Showing synthetic demo.")
        rng = np.random.default_rng(42)
        n_nodes = 12
        attn_mat = rng.dirichlet(np.ones(n_nodes), size=n_nodes)  # NxN
        sc_name = "Demo (Synthetic)"
    else:
        sc_name, aw_raw = attn_data
        attn_mat = np.array(aw_raw)

    fig_attn = go.Figure(go.Heatmap(
        z=attn_mat,
        colorscale=[[0, "#0E1117"], [0.5, "#2D3250"], [1, "#6C63FF"]],
        colorbar=dict(title="Attention"),
    ))
    fig_attn.update_layout(
        title=f"GAT Attention Matrix — {sc_name}",
        xaxis_title="Target Node", yaxis_title="Source Node",
        height=500, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig_attn), use_container_width=True)

    # Attention distribution
    section("Attention Weight Distribution", "📊")
    flat_attn = attn_mat.flatten()
    fig_adist = go.Figure(go.Histogram(
        x=flat_attn, marker_color=COLORS["primary"], opacity=0.8, nbinsx=50,
    ))
    fig_adist.update_layout(title="Attention Weight Distribution",
                             xaxis_title="Weight", yaxis_title="Count", **PLOTLY_LAYOUT)
    st.plotly_chart(apply_dark_theme(fig_adist), use_container_width=True)

    # Display explanation figures
    exp_figs = [f for f in list_figures()
                if any(kw in f.name.lower() for kw in ["attn", "attention", "shap", "explain"])]
    if exp_figs:
        section("Saved Explanation Figures", "🖼️")
        cols = st.columns(2)
        for i, fp in enumerate(exp_figs[:6]):
            with cols[i % 2]:
                st.markdown(f"**{fp.name}**")
                st.image(str(fp), use_container_width=True)
