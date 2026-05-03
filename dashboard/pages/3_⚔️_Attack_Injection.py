"""
Stage 4 — Byzantine Attack Injection Visualiser
"""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    inject_css, section, metric_tile, apply_dark_theme,
    COLORS, ATTACK_COLORS, PLOTLY_LAYOUT, load_raw_data,
)

st.set_page_config(page_title="Attack Injection", page_icon="⚔️", layout="wide")
inject_css()

st.markdown("## ⚔️ Byzantine Attack Injection")
st.markdown("<p style='color:#9AA3B2'>Visualise the effect of each Byzantine attack type on drone features</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Attack Config")
    attack_type = st.selectbox("Attack Type",
                               ["False State (FDI)", "Intermittent", "Colluding", "Delay"],
                               key="atk_type")
    n_drones = st.slider("Drones in swarm", 5, 30, 20, key="atk_drones")
    attacker_ratio = st.slider("Attacker ratio", 0.05, 0.50, 0.20, step=0.05, key="atk_ratio")
    n_timesteps = st.slider("Timesteps", 10, 60, 30, key="atk_steps")

    # Attack-specific params
    st.markdown("#### Attack Parameters")
    noise_scale = st.slider("Noise scale (FDI/Intermittent)", 0.5, 10.0, 3.0, step=0.5)
    p_active = st.slider("p_active (Intermittent)", 0.1, 1.0, 0.4, step=0.1)
    delay_steps = st.slider("Delay steps", 1, 10, 3)
    collusion_size = st.slider("Collusion group size", 2, 8, 4)
    seed = st.number_input("Seed", value=42, key="atk_seed")

ATTACK_MAP = {
    "False State (FDI)": "false_state",
    "Intermittent":      "intermittent",
    "Colluding":         "colluding",
    "Delay":             "delay",
}
atk_key = ATTACK_MAP[attack_type]
atk_color = ATTACK_COLORS[atk_key]

# ── Simulate data ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def simulate_features(n, n_t, ratio, noise, p_act, delay, coll, atk, seed):
    rng = np.random.default_rng(seed)
    n_feat = 5
    n_att = max(1, int(n * ratio))
    att_idx = set(rng.choice(n, n_att, replace=False).tolist())

    # Clean signal: sinusoidal + noise
    base = np.zeros((n_t, n, n_feat))
    for f in range(n_feat):
        freq = 0.1 + f * 0.05
        for i in range(n):
            base[:, i, f] = np.sin(np.arange(n_t) * freq + rng.uniform(0, 2*np.pi)) \
                            + rng.normal(0, 0.1, n_t)

    attacked = base.copy()
    att_list = sorted(att_idx)

    if atk == "false_state":
        for t in range(n_t):
            for i in att_list:
                attacked[t, i] += rng.normal(0, noise, n_feat)

    elif atk == "intermittent":
        for t in range(n_t):
            if rng.random() < p_act:
                for i in att_list:
                    attacked[t, i] += rng.normal(0, noise, n_feat)

    elif atk == "colluding":
        shared = rng.normal(5, 0.1, n_feat)
        for t in range(n_t):
            for i in att_list[:coll]:
                attacked[t, i] = shared + rng.normal(0, 0.05, n_feat)

    elif atk == "delay":
        for t in range(n_t):
            src_t = max(0, t - delay)
            for i in att_list:
                attacked[t, i] = base[src_t, i]

    return base, attacked, att_list


base_data, atk_data, att_nodes = simulate_features(
    n_drones, n_timesteps, attacker_ratio,
    noise_scale, p_active, delay_steps, collusion_size, atk_key, seed,
)

# ── Metric tiles ──────────────────────────────────────────────────────────────
cols = st.columns(4)
with cols[0]: metric_tile(str(len(att_nodes)), "Attackers", color=COLORS["danger"])
with cols[1]: metric_tile(f"{attacker_ratio:.0%}", "Attack Rate", color=COLORS["warning"])
with cols[2]:
    diff = np.abs(atk_data - base_data).mean()
    metric_tile(f"{diff:.3f}", "Mean Perturbation", color=atk_color)
with cols[3]:
    max_diff = np.abs(atk_data - base_data).max()
    metric_tile(f"{max_diff:.2f}", "Max Perturbation", color=COLORS["danger"])

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Feature Traces", "🔥 Perturbation Heatmap",
    "📊 Attack Comparison", "ℹ️ Attack Descriptions"
])

# ── TAB 1: Feature Traces ────────────────────────────────────────────────────
with tab1:
    section("Before vs After Attack — Feature Time Series", "📈")
    feat_idx = st.slider("Feature index", 0, 4, 0, key="trace_feat")
    node_to_show = st.selectbox("Compare node",
                                 [f"Attacker {i}" for i in att_nodes] +
                                 [f"Benign {i}" for i in range(n_drones) if i not in set(att_nodes)],
                                 key="trace_node")
    is_att = node_to_show.startswith("Attacker")
    node_idx = int(node_to_show.split()[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n_timesteps)), y=base_data[:, node_idx, feat_idx],
        mode="lines", name="Clean",
        line=dict(color=COLORS["secondary"], width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(n_timesteps)), y=atk_data[:, node_idx, feat_idx],
        mode="lines", name="After Attack",
        line=dict(color=atk_color, width=2.5),
    ))
    # Shade perturbation area
    diff_vals = (atk_data[:, node_idx, feat_idx] - base_data[:, node_idx, feat_idx])
    fig.add_trace(go.Scatter(
        x=list(range(n_timesteps)) + list(range(n_timesteps))[::-1],
        y=(base_data[:, node_idx, feat_idx] + diff_vals).tolist() +
           base_data[:, node_idx, feat_idx].tolist()[::-1],
        fill="toself", fillcolor=atk_color + "20",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    node_type = "Attacker" if is_att else "Benign"
    fig.update_layout(
        title=f"{node_type} Node {node_idx} — Feature {feat_idx} — Attack: {attack_type}",
        xaxis_title="Timestep", yaxis_title="Feature Value",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    # All attackers overview
    section("All Attacker Nodes — Feature Traces", "⚡")
    fig2 = go.Figure()
    for i in att_nodes[:6]:
        fig2.add_trace(go.Scatter(
            x=list(range(n_timesteps)), y=atk_data[:, i, feat_idx],
            mode="lines", name=f"Attacker {i}",
            line=dict(width=1.5, dash="dash"),
        ))
    for i in list(set(range(n_drones)) - set(att_nodes))[:3]:
        fig2.add_trace(go.Scatter(
            x=list(range(n_timesteps)), y=atk_data[:, i, feat_idx],
            mode="lines", name=f"Benign {i}",
            line=dict(color=COLORS["benign"], width=1),
        ))
    fig2.update_layout(title=f"Feature {feat_idx} — All Nodes", **PLOTLY_LAYOUT)
    st.plotly_chart(apply_dark_theme(fig2), use_container_width=True)

# ── TAB 2: Heatmap ───────────────────────────────────────────────────────────
with tab2:
    section("Perturbation Heatmap (nodes × timesteps)", "🔥")
    feat_h = st.slider("Feature index", 0, 4, 0, key="heat_feat")
    diff_mat = np.abs(atk_data[:, :, feat_h] - base_data[:, :, feat_h]).T  # (N, T)

    sorted_order = sorted(range(n_drones),
                          key=lambda i: (0 if i in set(att_nodes) else 1, i))
    diff_sorted = diff_mat[sorted_order]
    node_labels = [f"{'[A]' if i in set(att_nodes) else '   '} {i}"
                   for i in sorted_order]

    fig = go.Figure(go.Heatmap(
        z=diff_sorted,
        x=[f"t={t}" for t in range(n_timesteps)],
        y=node_labels,
        colorscale=[[0, "#0E1117"], [0.5, "#FFB300"], [1, "#FF5252"]],
        colorbar=dict(title="Perturbation"),
    ))
    fig.update_layout(
        title=f"Perturbation Magnitude — Feature {feat_h} ({attack_type})",
        height=max(350, n_drones * 22),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    # Mean perturbation per timestep
    section("Mean Perturbation Over Time", "📉")
    att_set = set(att_nodes)
    ben_nodes = [i for i in range(n_drones) if i not in att_set]

    mean_att = diff_mat[sorted(att_set)].mean(axis=0) if att_set else np.zeros(n_timesteps)
    mean_ben = diff_mat[ben_nodes].mean(axis=0) if ben_nodes else np.zeros(n_timesteps)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=list(range(n_timesteps)), y=mean_att,
                               mode="lines+markers", name="Attackers",
                               line=dict(color=COLORS["danger"], width=2.5),
                               fill="tozeroy", fillcolor=COLORS["danger"] + "20"))
    fig3.add_trace(go.Scatter(x=list(range(n_timesteps)), y=mean_ben,
                               mode="lines", name="Benign",
                               line=dict(color=COLORS["benign"], width=1.5, dash="dot")))
    fig3.update_layout(title="Mean Perturbation by Node Type", **PLOTLY_LAYOUT)
    st.plotly_chart(apply_dark_theme(fig3), use_container_width=True)

# ── TAB 3: Attack Comparison ─────────────────────────────────────────────────
with tab3:
    section("Compare All Attack Types", "📊")

    @st.cache_data(show_spinner=False)
    def all_attacks_data(n, n_t, ratio, noise, p_act, delay, coll, seed):
        results = {}
        for ak in ["false_state", "intermittent", "colluding", "delay"]:
            _, attacked, att = simulate_features(n, n_t, ratio, noise, p_act, delay, coll, ak, seed)
            _, clean, _ = simulate_features(n, n_t, ratio, noise, p_act, delay, coll, ak, seed)
            results[ak] = {
                "mean_pert": float(np.abs(attacked - clean).mean()),
                "max_pert": float(np.abs(attacked - clean).max()),
                "snr": float(np.abs(clean).mean() / (np.abs(attacked - clean).mean() + 1e-8)),
            }
        return results

    comp = all_attacks_data(n_drones, n_timesteps, attacker_ratio,
                            noise_scale, p_active, delay_steps, collusion_size, seed)

    c1, c2 = st.columns(2)
    with c1:
        fig_c1 = go.Figure(go.Bar(
            x=list(comp.keys()),
            y=[v["mean_pert"] for v in comp.values()],
            marker_color=list(ATTACK_COLORS.values()),
            text=[f"{v['mean_pert']:.4f}" for v in comp.values()],
            textposition="outside",
        ))
        fig_c1.update_layout(title="Mean Perturbation per Attack Type", **PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_c1), use_container_width=True)

    with c2:
        fig_c2 = go.Figure(go.Bar(
            x=list(comp.keys()),
            y=[v["snr"] for v in comp.values()],
            marker_color=list(ATTACK_COLORS.values()),
            text=[f"{v['snr']:.2f}" for v in comp.values()],
            textposition="outside",
        ))
        fig_c2.update_layout(title="Signal-to-Noise Ratio", **PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_c2), use_container_width=True)

# ── TAB 4: Descriptions ──────────────────────────────────────────────────────
with tab4:
    attacks_info = {
        "False State (FDI)": {
            "color": ATTACK_COLORS["false_state"],
            "desc": "Attackers continuously inject false sensor readings by adding Gaussian noise to their feature vectors. This simulates sensor spoofing where drones report incorrect position, velocity, or cyber data.",
            "params": f"Noise σ = {noise_scale}",
            "detectability": "Medium",
            "impact": "High",
        },
        "Intermittent": {
            "color": ATTACK_COLORS["intermittent"],
            "desc": "Attackers activate only with probability p_active per timestep, making detection harder. When active, they apply false-state perturbation. This models realistic covert Byzantine behavior.",
            "params": f"p_active = {p_active}",
            "detectability": "Hard",
            "impact": "Medium",
        },
        "Colluding": {
            "color": ATTACK_COLORS["colluding"],
            "desc": "A group of colluding drones report the same coordinated false value, amplifying the effect and making individual outlier detection ineffective. Requires graph-level analysis to detect.",
            "params": f"Group size = {collusion_size}",
            "detectability": "Very Hard",
            "impact": "Very High",
        },
        "Delay": {
            "color": ATTACK_COLORS["delay"],
            "desc": "Attackers replay stale data from d timesteps ago instead of current readings. This can cause the swarm to make decisions based on outdated information without triggering simple anomaly detectors.",
            "params": f"Delay = {delay_steps} steps",
            "detectability": "Medium",
            "impact": "Medium-High",
        },
    }

    for name, info in attacks_info.items():
        st.markdown(f"""
        <div class='dash-card' style='border-left:4px solid {info["color"]}'>
            <div style='display:flex;justify-content:space-between;align-items:start'>
                <div>
                    <div style='font-size:1.05rem;font-weight:700;color:{info["color"]}'>{name}</div>
                    <div style='color:#9AA3B2;font-size:0.85rem;margin:0.5rem 0'>{info["desc"]}</div>
                    <code style='background:#2D3250;padding:3px 8px;border-radius:4px;font-size:0.8rem'>{info["params"]}</code>
                </div>
                <div style='text-align:right;min-width:120px'>
                    <div style='margin-bottom:4px'>
                        <span style='color:#9AA3B2;font-size:0.75rem'>Detectability</span><br>
                        <span style='color:{info["color"]};font-weight:600'>{info["detectability"]}</span>
                    </div>
                    <div>
                        <span style='color:#9AA3B2;font-size:0.75rem'>Impact</span><br>
                        <span style='color:{COLORS["danger"]};font-weight:600'>{info["impact"]}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
