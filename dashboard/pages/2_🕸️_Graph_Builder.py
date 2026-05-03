"""
Stage 3 — Graph Builder & Visualiser
"""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import inject_css, section, metric_tile, apply_dark_theme, COLORS, PLOTLY_LAYOUT

st.set_page_config(page_title="Graph Builder", page_icon="🕸️", layout="wide")
inject_css()

st.markdown("## 🕸️ Graph Builder")
st.markdown("<p style='color:#9AA3B2'>Visualise swarm topologies and temporal graph snapshots</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Config sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Graph Config")
    n_drones = st.slider("Number of drones", 5, 30, 20, key="g_drones")
    graph_type = st.selectbox("Topology", ["KNN", "Distance", "Hexagonal"], key="g_type")
    attacker_ratio = st.slider("Attacker ratio", 0.05, 0.50, 0.20, step=0.05, key="g_ratio")
    knn_k = st.slider("KNN k", 2, 10, 5, key="g_k")
    comm_range = st.slider("Communication range (m)", 10, 100, 50, key="g_range")
    seed = st.number_input("Random seed", value=42, key="g_seed")
    regenerate = st.button("🔄 Regenerate", type="primary", use_container_width=True)

# ── Generate positions & topology ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def make_graph(n, ratio, g_type, k, r, seed):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 200, (n, 2))
    n_attack = max(1, int(n * ratio))
    labels = np.zeros(n, dtype=int)
    attack_idx = rng.choice(n, n_attack, replace=False)
    labels[attack_idx] = 1

    edges = []
    if g_type == "KNN":
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(positions)
        _, indices = nbrs.kneighbors(positions)
        for i, nbrs_i in enumerate(indices):
            for j in nbrs_i[1:]:
                edges.append((i, j))
    elif g_type == "Distance":
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < r:
                    edges.append((i, j))
    else:  # Hexagonal
        rows = max(2, int(np.sqrt(n)))
        cols = (n + rows - 1) // rows
        hex_positions = []
        for row in range(rows):
            for col in range(cols):
                if len(hex_positions) >= n:
                    break
                x = col * 25 + (12.5 if row % 2 else 0)
                y = row * 22
                hex_positions.append([x, y])
        positions = np.array(hex_positions[:n], dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 30:
                    edges.append((i, j))

    return positions, labels, edges


positions, labels, edges = make_graph(
    n_drones, attacker_ratio, graph_type, knn_k, comm_range, seed
)

# ── Metrics row ───────────────────────────────────────────────────────────────
cols = st.columns(4)
with cols[0]: metric_tile(n_drones, "Drones", color=COLORS["primary"])
with cols[1]: metric_tile(int(labels.sum()), "Attackers", color=COLORS["danger"])
with cols[2]: metric_tile(len(edges), "Edges", color=COLORS["secondary"])
with cols[3]:
    avg_deg = 2 * len(edges) / max(1, n_drones)
    metric_tile(f"{avg_deg:.1f}", "Avg Degree", color=COLORS["warning"])

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🗺️ Swarm Graph", "📊 Graph Statistics", "⏱️ Temporal Sequence"])

# ── TAB 1: Swarm Graph ────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([3, 1])
    with c1:
        section("Interactive Swarm Graph", "🕸️")

        fig = go.Figure()

        # Draw edges
        for i, j in edges:
            fig.add_trace(go.Scatter(
                x=[positions[i, 0], positions[j, 0], None],
                y=[positions[i, 1], positions[j, 1], None],
                mode="lines",
                line=dict(color="#2D3250", width=1.5),
                showlegend=False, hoverinfo="skip",
            ))

        # Draw nodes
        for label, color, name in [(0, COLORS["benign"], "Benign"), (1, COLORS["danger"], "Attacker")]:
            mask = labels == label
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=positions[mask, 0], y=positions[mask, 1],
                mode="markers+text",
                text=[str(i) for i in np.where(mask)[0]],
                textposition="top center",
                marker=dict(
                    size=18, color=color,
                    line=dict(color="white", width=1.5),
                    symbol="circle",
                ),
                name=name,
                hovertemplate=f"<b>{name}</b><br>x=%{{x:.1f}}<br>y=%{{y:.1f}}<extra></extra>",
            ))

        fig.update_layout(
            title=f"{graph_type} Topology — {n_drones} drones, {int(attacker_ratio*100)}% attackers",
            xaxis_title="X position (m)", yaxis_title="Y position (m)",
            height=500,
            legend=dict(orientation="h", y=-0.1),
            **PLOTLY_LAYOUT,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

    with c2:
        section("Legend", "ℹ️")
        st.markdown(f"""
        <div class='dash-card'>
            <div style='margin-bottom:0.8rem'>
                <span style='color:{COLORS["benign"]}; font-size:1.2rem'>●</span>
                <span style='margin-left:6px'>Benign drone</span>
            </div>
            <div>
                <span style='color:{COLORS["danger"]}; font-size:1.2rem'>●</span>
                <span style='margin-left:6px'>Byzantine attacker</span>
            </div>
            <hr style='border-color:#2D3250'>
            <div style='color:#9AA3B2; font-size:0.85rem'>
                <b>Edges</b> represent communication links.<br><br>
                <b>KNN</b>: connect k nearest neighbors<br>
                <b>Distance</b>: connect if dist &lt; range<br>
                <b>Hexagonal</b>: fixed lattice formation
            </div>
        </div>
        """, unsafe_allow_html=True)

        section("Node Table", "📋")
        import pandas as pd
        node_df = pd.DataFrame({
            "Node": range(n_drones),
            "X (m)": positions[:, 0].round(1),
            "Y (m)": positions[:, 1].round(1),
            "Label": ["Attacker" if l else "Benign" for l in labels],
        })
        st.dataframe(node_df.style.applymap(
            lambda v: f"color:{COLORS['danger']}" if v == "Attacker" else
                      f"color:{COLORS['benign']}", subset=["Label"]
        ), height=280, use_container_width=True)

# ── TAB 2: Graph Statistics ───────────────────────────────────────────────────
with tab2:
    section("Degree Distribution", "📊")

    import pandas as pd
    from collections import Counter
    degree = Counter()
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    degrees = [degree.get(i, 0) for i in range(n_drones)]

    c1, c2 = st.columns(2)
    with c1:
        fig_deg = go.Figure(go.Histogram(
            x=degrees, marker_color=COLORS["primary"], opacity=0.8,
            xbins=dict(size=1),
        ))
        fig_deg.update_layout(title="Degree Distribution", xaxis_title="Degree",
                               yaxis_title="Count", **PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_deg), use_container_width=True)

    with c2:
        deg_df = pd.DataFrame({
            "Node": range(n_drones),
            "Degree": degrees,
            "Type": ["Attacker" if l else "Benign" for l in labels],
        })
        fig_box = go.Figure()
        for ltype, color in [("Benign", COLORS["benign"]), ("Attacker", COLORS["danger"])]:
            sub = deg_df[deg_df["Type"] == ltype]["Degree"]
            fig_box.add_trace(go.Box(y=sub, name=ltype, fillcolor=color + "40",
                                      line_color=color, boxmean=True))
        fig_box.update_layout(title="Degree by Node Type", **PLOTLY_LAYOUT)
        st.plotly_chart(apply_dark_theme(fig_box), use_container_width=True)

    section("Topology Comparison", "🔭")
    topo_stats = {}
    for gt in ["KNN", "Distance", "Hexagonal"]:
        _, _, e = make_graph(n_drones, attacker_ratio, gt, knn_k, comm_range, seed)
        degs = Counter()
        for i, j in e:
            degs[i] += 1; degs[j] += 1
        d_vals = [degs.get(i, 0) for i in range(n_drones)]
        topo_stats[gt] = {
            "Edges": len(e),
            "Avg Degree": round(np.mean(d_vals), 2),
            "Max Degree": max(d_vals) if d_vals else 0,
            "Connectivity": round(len(e) / max(1, n_drones * (n_drones-1) / 2), 3),
        }
    st.dataframe(pd.DataFrame(topo_stats).T, use_container_width=True)

# ── TAB 3: Temporal Sequence ─────────────────────────────────────────────────
with tab3:
    section("Temporal Position Evolution", "⏱️")
    n_steps = st.slider("Timesteps to simulate", 5, 30, 15, key="t_steps")
    step_size = st.slider("Step noise (m)", 1, 15, 5, key="t_noise")

    @st.cache_data(show_spinner=False)
    def simulate_temporal(n, ratio, k, r, g_type, steps, noise, seed):
        rng = np.random.default_rng(seed)
        pos = rng.uniform(0, 200, (n, 2))
        n_attack = max(1, int(n * ratio))
        att_idx = set(rng.choice(n, n_attack, replace=False).tolist())
        trajectory = [pos.copy()]
        for _ in range(steps - 1):
            pos = pos + rng.normal(0, noise, pos.shape)
            pos = np.clip(pos, 0, 200)
            trajectory.append(pos.copy())
        return trajectory, att_idx

    traj, att_set = simulate_temporal(n_drones, attacker_ratio, knn_k, comm_range,
                                       graph_type, n_steps, step_size, seed)

    frames = []
    for t, pos in enumerate(traj):
        frame_data = []
        for node in range(n_drones):
            color = COLORS["danger"] if node in att_set else COLORS["benign"]
            frame_data.append(go.Scatter(
                x=[pos[node, 0]], y=[pos[node, 1]],
                mode="markers",
                marker=dict(size=14, color=color, opacity=0.9,
                            line=dict(color="white", width=1)),
                showlegend=False,
                hovertemplate=f"Node {node}<br>t={t}<extra></extra>",
            ))
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Initial frame
    init_pos = traj[0]
    fig_anim = go.Figure(
        data=[
            go.Scatter(
                x=init_pos[labels == 0, 0], y=init_pos[labels == 0, 1],
                mode="markers", name="Benign",
                marker=dict(size=14, color=COLORS["benign"],
                            line=dict(color="white", width=1)),
            ),
            go.Scatter(
                x=init_pos[labels == 1, 0], y=init_pos[labels == 1, 1],
                mode="markers", name="Attacker",
                marker=dict(size=14, color=COLORS["danger"],
                            line=dict(color="white", width=1)),
            ),
        ],
        frames=frames,
    )
    fig_anim.update_layout(
        title="Swarm Movement Simulation",
        xaxis=dict(range=[0, 200], gridcolor="#2D3250"),
        yaxis=dict(range=[0, 200], gridcolor="#2D3250"),
        height=480,
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05,
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=250, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], dict(mode="immediate")],
                        method="animate", label=f.name) for f in frames],
            transition=dict(duration=0),
            x=0, y=0, len=1.0,
            currentvalue=dict(prefix="Timestep: ", font=dict(color="#9AA3B2")),
        )],
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(apply_dark_theme(fig_anim), use_container_width=True)
