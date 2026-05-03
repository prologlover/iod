"""
Stage 2 — Data Explorer
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    inject_css, section, metric_tile, badge, apply_dark_theme,
    COLORS, PLOTLY_LAYOUT, load_raw_data, load_processed_split, get_paths,
)

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
inject_css()

st.markdown("## 📊 Data Explorer")
st.markdown("<p style='color:#9AA3B2'>UAV cyber-physical dataset · 54,783 rows · 37 features · 5 attack types</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    df = load_raw_data()

if df is None:
    st.error("Dataset not found. Run `python scripts/download_data.py` first.")
    st.stop()

meta_cols = ["attack_type", "is_attack", "domain", "label"]
feature_cols = [c for c in df.columns if c not in meta_cols]

# ── Top metrics ───────────────────────────────────────────────────────────────
cols = st.columns(5)
metrics = [
    (f"{len(df):,}",           "Total Samples",   COLORS["primary"]),
    (str(len(feature_cols)),   "Features",        COLORS["secondary"]),
    (str(df["attack_type"].nunique()), "Classes",  COLORS["warning"]),
    (f"{df['is_attack'].mean()*100:.1f}%", "Attack Rate", COLORS["danger"]),
    (f"{df.isna().mean().mean()*100:.1f}%", "Missing Values", COLORS["success"]),
]
for col, (val, lbl, clr) in zip(cols, metrics):
    with col:
        metric_tile(val, lbl, color=clr)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗂️ Overview", "📉 Distributions", "🔗 Correlations",
    "⚔️ Class Analysis", "🔄 Train/Val/Test Splits"
])

# ── TAB 1: Overview ──────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        section("Label Distribution", "📦")
        vc = df["attack_type"].value_counts().reset_index()
        vc.columns = ["Attack Type", "Count"]
        fig = go.Figure(go.Bar(
            x=vc["Attack Type"], y=vc["Count"],
            marker=dict(color=[
                COLORS["benign"] if t == "Benign" else COLORS["danger"]
                for t in vc["Attack Type"]
            ]),
            text=vc["Count"], textposition="outside",
        ))
        fig.update_layout(title="Sample Count per Class", **PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    with c2:
        section("Class Proportions", "🥧")
        fig2 = go.Figure(go.Pie(
            labels=vc["Attack Type"], values=vc["Count"],
            hole=0.55,
            marker=dict(colors=[
                COLORS["benign"], "#FF5252", "#FFB300", "#00D4FF", "#F06292"
            ]),
            textinfo="label+percent",
        ))
        fig2.update_layout(
            showlegend=False, height=320,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        )
        st.plotly_chart(apply_dark_theme(fig2), use_container_width=True)

    section("Dataset Sample", "👁️")
    st.dataframe(
        df[feature_cols[:10] + ["attack_type", "is_attack"]].head(100).style
        .applymap(lambda v: "background-color:#FF525220" if v == 1 else
                  "background-color:#00E67620" if v == 0 else "",
                  subset=["is_attack"]),
        use_container_width=True, height=280,
    )

    section("Feature Summary", "📋")
    desc = df[feature_cols].describe().T
    desc["missing%"] = (df[feature_cols].isna().sum() / len(df) * 100).values
    st.dataframe(desc.style.background_gradient(cmap="Blues", subset=["mean", "std"]),
                 use_container_width=True, height=320)

# ── TAB 2: Distributions ─────────────────────────────────────────────────────
with tab2:
    section("Feature Distributions by Attack Type", "📉")
    sel_feat = st.selectbox("Select feature", feature_cols[:20], key="dist_feat")

    fig = go.Figure()
    attack_types = df["attack_type"].unique()
    palette = [COLORS["benign"], "#FF5252", "#FFB300", "#00D4FF", "#F06292"]
    for color, atype in zip(palette, attack_types):
        vals = df.loc[df["attack_type"] == atype, sel_feat].dropna()
        fig.add_trace(go.Violin(
            y=vals, name=atype, box_visible=True,
            meanline_visible=True, fillcolor=color + "40",
            line_color=color, opacity=0.85,
        ))
    fig.update_layout(title=f"Distribution of '{sel_feat}' by Attack Type",
                      yaxis_title=sel_feat, **PLOTLY_LAYOUT)
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Histogram", "📊")
        n_bins = st.slider("Bins", 20, 100, 50, key="hist_bins")
        fig_h = px.histogram(df, x=sel_feat, color="attack_type", nbins=n_bins,
                             barmode="overlay", opacity=0.7,
                             color_discrete_sequence=palette)
        fig_h.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_h), use_container_width=True)

    with c2:
        section("Box Plot", "📦")
        fig_b = px.box(df, x="attack_type", y=sel_feat, color="attack_type",
                       color_discrete_sequence=palette)
        fig_b.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_b), use_container_width=True)

# ── TAB 3: Correlations ───────────────────────────────────────────────────────
with tab3:
    section("Feature Correlation Matrix", "🔗")
    top_n = st.slider("Number of features", 10, min(30, len(feature_cols)), 15, key="corr_n")
    top_feats = feature_cols[:top_n]
    corr = df[top_feats].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=top_feats, y=top_feats,
        colorscale=[[0, "#0E1117"], [0.5, "#2D3250"], [1, "#6C63FF"]],
        zmid=0,
    ))
    fig.update_layout(title=f"Top-{top_n} Feature Correlations",
                      height=600, **PLOTLY_LAYOUT)
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    section("Scatter Matrix (top 5)", "🔵")
    scatter_feats = st.multiselect("Features", feature_cols[:15], default=feature_cols[:5],
                                   key="scatter_feats")
    if len(scatter_feats) >= 2:
        fig_s = px.scatter_matrix(df.sample(min(1000, len(df)), random_state=42),
                                   dimensions=scatter_feats, color="attack_type",
                                   color_discrete_sequence=palette, opacity=0.5)
        fig_s.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_s), use_container_width=True)

# ── TAB 4: Class Analysis ─────────────────────────────────────────────────────
with tab4:
    section("Feature Means per Attack Type", "📊")
    means = df.groupby("attack_type")[feature_cols[:15]].mean()
    fig = px.imshow(means.T, color_continuous_scale="RdBu_r", aspect="auto",
                    labels=dict(x="Attack Type", y="Feature", color="Mean"))
    fig.update_layout(title="Normalised Feature Means Heatmap",
                      height=500, **PLOTLY_LAYOUT)
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    section("Pairwise Feature Separation (Benign vs Attack)", "⚔️")
    f1 = st.selectbox("Feature X", feature_cols[:20], index=0, key="pair_x")
    f2 = st.selectbox("Feature Y", feature_cols[:20], index=1, key="pair_y")
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig_p = px.scatter(sample, x=f1, y=f2, color="attack_type",
                       color_discrete_sequence=palette, opacity=0.6,
                       marginal_x="histogram", marginal_y="histogram")
    fig_p.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(apply_dark_theme(fig_p), use_container_width=True)

# ── TAB 5: Splits ────────────────────────────────────────────────────────────
with tab5:
    section("Train / Val / Test Split Overview", "🔄")
    paths = get_paths()
    split_data = {}
    for split in ["train", "val", "test"]:
        p = paths["processed"] / f"{split}.parquet"
        if p.exists():
            split_data[split] = pd.read_parquet(p)

    if split_data:
        # Size comparison
        sizes = {k: len(v) for k, v in split_data.items()}
        fig_s = go.Figure(go.Bar(
            x=list(sizes.keys()), y=list(sizes.values()),
            text=list(sizes.values()), textposition="outside",
            marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["warning"]],
        ))
        fig_s.update_layout(title="Split Sizes", **PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_s), use_container_width=True)

        # Class balance per split
        section("Class Balance per Split", "⚖️")
        c1, c2, c3 = st.columns(3)
        for col, (split, sdf) in zip([c1, c2, c3], split_data.items()):
            with col:
                if "attack_type" in sdf.columns:
                    vc2 = sdf["attack_type"].value_counts()
                    fig2 = go.Figure(go.Pie(
                        labels=vc2.index, values=vc2.values,
                        hole=0.5, textinfo="percent",
                        marker=dict(colors=palette),
                    ))
                    fig2.update_layout(
                        title=f"{split.capitalize()} ({len(sdf):,})",
                        height=280,
                        **{k: v for k, v in PLOTLY_LAYOUT.items()
                           if k not in ("xaxis", "yaxis")}
                    )
                    st.plotly_chart(apply_dark_theme(fig2), use_container_width=True)
    else:
        st.warning("Processed splits not found. Run `python scripts/preprocess.py` first.")
