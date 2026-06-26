"""
scripts/dashboard.py
Streamlit dashboard for the Brain Tumor Surgical Planner.
Run: streamlit run scripts/dashboard.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

from src.causal.scm import BrainTumorSCM
from src.causal.do_calculus import DoCalculusEngine, SurgicalAction, ACTION_REGISTRY
from src.causal.counterfactual import CounterfactualEngine, CounterfactualQuery

API_BASE = os.getenv("API_BASE", "http://localhost:8002")


# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 NeuroPlan AI — Causal Surgical Planner for India",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.3rem 0;
    border-left: 4px solid #7c3aed;
    transition: transform 0.15s;
}
.metric-card:hover { transform: translateX(3px); }
.high-risk { border-left-color: #ef4444; }
.med-risk  { border-left-color: #f59e0b; }
.low-risk  { border-left-color: #22c55e; }
.attribution-box {
    background: #13131f;
    border: 1px solid #7c3aed44;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: 'Inter', monospace;
}
.india-stat {
    background: linear-gradient(135deg, #1e1e2e, #2d1b4e);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    border-left: 3px solid #f59e0b;
    margin: 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Page Navigation ──────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Surgical Planner", "👥 Patient Management", "📊 Simulation Monitor", "📈 Comparative Analysis", "📤 Export & Reports"],
    index=0,
)


# ─── Sidebar: Patient Parameters ─────────────────────────────────────────────
st.sidebar.title("🧠 Surgical Planner")
st.sidebar.header("Patient Parameters (SCM)")

tumor_size = st.sidebar.slider("Tumor Size", 0.0, 1.0, 0.30, 0.01)
edema      = st.sidebar.slider("Edema Volume", 0.0, 1.0, 0.20, 0.01)
icp        = st.sidebar.slider("Intracranial Pressure", 0.0, 1.0, 0.20, 0.01)
blood_flow = st.sidebar.slider("Blood Flow", 0.0, 1.0, 0.70, 0.01)
inflammation = st.sidebar.slider("Inflammatory Response", 0.0, 1.0, 0.30, 0.01)

st.sidebar.divider()
n_sims = st.sidebar.slider("Monte-Carlo Simulations", 50, 500, 100, 50)
st.sidebar.caption("More simulations = more stable estimates but slower")

patient_params = {
    "tumor_size":            tumor_size,
    "edema_volume":          edema,
    "intracranial_pressure": icp,
    "blood_flow":            blood_flow,
    "inflammatory_response": inflammation,
}


# ─── Surgical Planner Page ────────────────────────────────────────────────────
if page == "🏠 Surgical Planner":
    st.title("🧠 NeuroPlan AI — Causal Brain Tumor Surgical Planner")
    st.caption(
        "Pearl Do-Calculus × Counterfactual Simulation × Monte-Carlo Path Search "
        "| Built for India's 100,000+ annual brain tumor cases"
    )

    with st.sidebar.expander("🇮🇳 Why India Needs This", expanded=False):
        st.markdown("""
        <div class='india-stat'>📊 <b>100,000+</b> new brain tumor cases/year in India</div>
        <div class='india-stat'>👨‍⚕️ <b>1 neurosurgeon</b> per ~400,000 people</div>
        <div class='india-stat'>⏱️ Pre-op planning takes <b>4–8 hours</b> per case</div>
        <div class='india-stat'>🏥 Most tier-2/3 hospitals have <b>zero</b> surgical AI access</div>
        <br/>
        <small>NeuroPlan AI acts as a <b>causal AI co-pilot</b> — giving every
        surgeon the reasoning power of a specialized tumor board in &lt;60 seconds.</small>
        """, unsafe_allow_html=True)

    scm = BrainTumorSCM(patient_params=patient_params)
    baseline = scm.evaluate(noise=False)

    # Row 1: Baseline Metrics
    st.subheader("Baseline Physiological State")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Blood Flow",       f"{baseline['blood_flow']:.0%}",
                  delta=f"{baseline['blood_flow']-0.7:+.0%}")
    with col2:
        st.metric("Oxygen Sat",       f"{baseline['oxygen_saturation']:.0%}")
    with col3:
        st.metric("ICP",              f"{baseline['intracranial_pressure']:.0%}",
                  delta=f"{baseline['intracranial_pressure']-0.15:+.0%}", delta_color="inverse")
    with col4:
        st.metric("Neural Function",  f"{baseline['neural_function']:.0%}")
    with col5:
        st.metric("Recovery Score",   f"{baseline['recovery_score']:.0%}")

    # Row 2: SCM Causal Graph
    col_left, col_right = st.columns([1.3, 0.7])

    with col_left:
        st.subheader("Causal Variable Web")
        
        vars_ordered = [
            "tumor_size", "edema_volume", "vascular_compression",
            "blood_flow", "oxygen_saturation", "intracranial_pressure",
            "mass_effect", "metabolic_rate", "neural_function",
            "inflammatory_response", "recovery_score", "surgical_risk"
        ]
        angles = np.linspace(0, 2 * np.pi, len(vars_ordered), endpoint=False)
        radius = 1.0
        pos = {v: (radius * np.cos(a), radius * np.sin(a)) for v, a in zip(vars_ordered, angles)}

        edges = list(scm.dag.edges())

        edge_x, edge_y = [], []
        for u, v in edges:
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

        node_x = [pos[v][0] for v in vars_ordered if v in pos]
        node_y = [pos[v][1] for v in vars_ordered if v in pos]
        node_text = [v.replace("_", "<br>") for v in vars_ordered if v in pos]
        node_vals = [baseline.get(v, 0) for v in vars_ordered if v in pos]

        fig_dag = go.Figure()
        fig_dag.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#4b5563", width=1),
            hoverinfo="none",
        ))
        fig_dag.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=18, color=node_vals, colorscale="RdYlGn",
                        cmin=0, cmax=1, showscale=True,
                        colorbar=dict(title="Value", thickness=12)),
            text=node_text,
            textposition="top center",
            hovertemplate="%{text}<br>Value: %{marker.color:.3f}<extra></extra>",
        ))
        fig_dag.update_layout(
            height=380, showlegend=False,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="white", size=9),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_dag, use_container_width=True)

    with col_right:
        st.subheader("All do() Interventions")
        engine = DoCalculusEngine(scm)
        intervention_rows = []
        for action in SurgicalAction:
            r = engine.intervene(action, noise=False)
            intervention_rows.append({
                "Action": action.value.replace("_", " ").title(),
                "Recovery Gain": f"{r.recovery_gain:+.3f}",
                "Net Utility": f"{r.net_utility:+.3f}",
                "Risk": f"{r.risk_increase:.2f}",
            })
        intervention_rows.sort(key=lambda x: float(x["Net Utility"]), reverse=True)

        import pandas as pd
        df = pd.DataFrame(intervention_rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=360)

    st.divider()

    # ── Surgical Corridor Planner ─────────────────────────────────────────────
    st.subheader("🔬 Surgical Corridor Planner")
    st.caption("AI-computed approach trajectories ranked by risk to eloquent structures")

    # Corridor definitions (driven by patient tumor_size / ICP)
    _ts = patient_params["tumor_size"]
    _icp = patient_params["intracranial_pressure"]

    CORRIDORS = [
        {
            "id": "A",
            "name": "Pterional (Transsylvian)",
            "entry": "Frontotemporal Burr Hole",
            "trajectory": "Sylvian Fissure → Mesial Temporal",
            "length_mm": round(72.4 + _ts * 18, 1),
            "risk": round(0.08 + _ts * 0.06 + _icp * 0.04, 2),
            "eloquent": ["Optic Tract (18mm)"],
            "blood_vessel_crossings": 1,
            "brain_retraction_mm": 4,
            "recommended": True,
        },
        {
            "id": "B",
            "name": "Transcortical (Temporal)",
            "entry": "Parietal / Temporal Cortex",
            "trajectory": "Trans-cortical → Deep Temporal",
            "length_mm": round(58.1 + _ts * 14, 1),
            "risk": round(0.17 + _ts * 0.08 + _icp * 0.06, 2),
            "eloquent": ["Motor Cortex (12mm)", "Sensory Strip (22mm)"],
            "blood_vessel_crossings": 2,
            "brain_retraction_mm": 8,
            "recommended": False,
        },
        {
            "id": "C",
            "name": "Keyhole Supraorbital",
            "entry": "Supraorbital Eyebrow Incision",
            "trajectory": "Subfrontal → Anterior Temporal",
            "length_mm": round(91.7 + _ts * 22, 1),
            "risk": round(0.29 + _ts * 0.10 + _icp * 0.08, 2),
            "eloquent": ["Frontal Lobe", "Olfactory Nerve (9mm)"],
            "blood_vessel_crossings": 3,
            "brain_retraction_mm": 14,
            "recommended": False,
        },
    ]

    # Clamp risk to [0,1]
    for c in CORRIDORS:
        c["risk"] = min(max(c["risk"], 0.04), 0.95)

    # ── Corridor selector and summary cards ───────────────────────────────────
    card_cols = st.columns(3)
    sel_corr_id = st.session_state.get("active_corridor", "A")

    for col, corr in zip(card_cols, CORRIDORS):
        risk_pct = corr["risk"] * 100
        if risk_pct < 20:
            risk_color = "#22c55e"
            risk_bg = "#052e16"
            risk_border = "#166534"
        elif risk_pct < 30:
            risk_color = "#f59e0b"
            risk_bg = "#2d1f00"
            risk_border = "#92400e"
        else:
            risk_color = "#ef4444"
            risk_bg = "#2d0a0a"
            risk_border = "#991b1b"

        rec_html = "<span style='background:#14532d;color:#4ade80;border:1px solid #166534;border-radius:3px;padding:1px 6px;font-size:10px;margin-left:4px;'>★ RECOMMENDED</span>" if corr["recommended"] else ""
        selected_style = "border:2px solid #3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,0.2);" if corr["id"] == sel_corr_id else "border:1px solid #1e293b;"

        col.markdown(f"""
<div id="corridor-{corr['id']}" style="background:#0f1729;{selected_style}border-radius:12px;padding:1.1rem 1.2rem;margin-bottom:0.2rem;cursor:pointer;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">
    <span style="font-size:1rem;font-weight:700;color:#e2e8f0;">Corridor {corr['id']}</span>
    {rec_html}
  </div>
  <div style="font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:0.4rem;">{corr['name']}</div>
  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.7rem;">
    <div style="background:{risk_bg};border:1px solid {risk_border};border-radius:8px;padding:0.4rem 0.9rem;text-align:center;">
      <div style="font-size:1.5rem;font-weight:800;color:{risk_color};font-family:monospace;">{risk_pct:.0f}%</div>
      <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Risk</div>
    </div>
    <div>
      <div style="font-size:0.78rem;color:#64748b;">Length</div>
      <div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;">{corr['length_mm']} mm</div>
    </div>
  </div>
  <div style="font-size:0.73rem;color:#64748b;margin-bottom:0.3rem;">Entry point</div>
  <div style="font-size:0.8rem;color:#cbd5e1;margin-bottom:0.5rem;">{corr['entry']}</div>
  <div style="font-size:0.73rem;color:#64748b;margin-bottom:0.3rem;">Structures at risk</div>
  {"".join(f'<span style="background:#2d1f1f;color:#fca5a5;border:1px solid #7f1d1d;border-radius:4px;padding:1px 6px;font-size:0.7rem;margin:1px;display:inline-block;">{s}</span>' for s in corr['eloquent'])}
</div>
""", unsafe_allow_html=True)
        if col.button(f"Select Corridor {corr['id']}", key=f"sel_corr_{corr['id']}",
                      type="primary" if corr["id"] == sel_corr_id else "secondary",
                      use_container_width=True):
            st.session_state["active_corridor"] = corr["id"]
            st.rerun()

    # ── Entry → Trajectory → Tumor flow diagram + 3D path ────────────────────
    st.markdown("")
    active_corr = next(c for c in CORRIDORS if c["id"] == sel_corr_id)

    viz_col, detail_col = st.columns([1.8, 1.2])

    with viz_col:
        st.markdown(f"**3D Trajectory — Corridor {active_corr['id']}: {active_corr['name']}**")

        # Build a 3D figure: brain ellipsoid (transparent) + trajectory + tumor + entry point
        import numpy as np

        # Brain surface (simplified ellipsoid)
        u_b = np.linspace(0, 2 * np.pi, 40)
        v_b = np.linspace(0, np.pi, 20)
        rx, ry, rz = 85, 95, 70
        brain_x = rx * np.outer(np.cos(u_b), np.sin(v_b)).flatten()
        brain_y = ry * np.outer(np.sin(u_b), np.sin(v_b)).flatten()
        brain_z = rz * np.outer(np.ones_like(u_b), np.cos(v_b)).flatten()

        # Tumor sphere
        r_tumor = 18 + _ts * 22
        u_t = np.linspace(0, 2 * np.pi, 25)
        v_t = np.linspace(0, np.pi, 13)
        tx_c, ty_c, tz_c = -20, -15, 10  # left temporal lobe-ish
        tumor_x = tx_c + r_tumor * np.outer(np.cos(u_t), np.sin(v_t)).flatten()
        tumor_y = ty_c + r_tumor * np.outer(np.sin(u_t), np.sin(v_t)).flatten()
        tumor_z = tz_c + r_tumor * 0.85 * np.outer(np.ones_like(u_t), np.cos(v_t)).flatten()

        # Corridor entry points (on the skull surface, above the brain ellipsoid)
        ENTRY_POINTS = {
            "A": (-60, 40, 55),   # frontotemporal
            "B": (-30, -50, 65),  # parietal/temporal
            "C": (-55, 60, 30),   # supraorbital
        }
        entry_pt = np.array(ENTRY_POINTS[sel_corr_id], dtype=float)
        tumor_center = np.array([tx_c, ty_c, tz_c], dtype=float)

        # Trajectory: straight line from entry to tumor center, n=60 points
        n_pts = 60
        t_line = np.linspace(0, 1, n_pts)
        traj_x = entry_pt[0] + t_line * (tumor_center[0] - entry_pt[0])
        traj_y = entry_pt[1] + t_line * (tumor_center[1] - entry_pt[1])
        traj_z = entry_pt[2] + t_line * (tumor_center[2] - entry_pt[2])

        # Midpoint label position
        mid = n_pts // 2
        mid_x, mid_y, mid_z = traj_x[mid], traj_y[mid], traj_z[mid]

        CORR_COLORS = {"A": "#22c55e", "B": "#f59e0b", "C": "#ef4444"}
        corr_color = CORR_COLORS[sel_corr_id]

        fig_corr = go.Figure()

        # Brain surface (translucent)
        fig_corr.add_trace(go.Mesh3d(
            x=brain_x, y=brain_y, z=brain_z,
            alphahull=0,
            color="#5b8fcc", opacity=0.10,
            name="Brain", showlegend=True,
            lighting=dict(ambient=0.8, diffuse=0.5),
            hoverinfo="skip",
        ))

        # Tumor volume
        fig_corr.add_trace(go.Mesh3d(
            x=tumor_x, y=tumor_y, z=tumor_z,
            alphahull=0,
            color="#ef4444", opacity=0.80,
            name=f"Tumor  (r≈{r_tumor:.0f}mm)", showlegend=True,
            lighting=dict(ambient=0.5, diffuse=0.9, specular=0.4),
            hoverinfo="name",
        ))

        # Trajectory line
        fig_corr.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z,
            mode="lines",
            line=dict(color=corr_color, width=6),
            name=f"Corridor {sel_corr_id} Trajectory",
            hovertemplate="Trajectory<extra></extra>",
        ))

        # Entry point marker
        fig_corr.add_trace(go.Scatter3d(
            x=[entry_pt[0]], y=[entry_pt[1]], z=[entry_pt[2]],
            mode="markers+text",
            marker=dict(size=14, color=corr_color,
                        symbol="circle", line=dict(color="white", width=2)),
            text=["Entry"],
            textposition="top center",
            textfont=dict(size=12, color=corr_color),
            name="Entry Point",
        ))

        # Tumor center marker
        fig_corr.add_trace(go.Scatter3d(
            x=[tumor_center[0]], y=[tumor_center[1]], z=[tumor_center[2]],
            mode="markers+text",
            marker=dict(size=12, color="#ef4444",
                        symbol="diamond", line=dict(color="white", width=2)),
            text=["Tumor"],
            textposition="top center",
            textfont=dict(size=12, color="#ef4444"),
            name="Tumor Center",
        ))

        # Risk zone cone around trajectory (illustrative)
        if _ts > 0.2:
            cone_pts = 15
            cone_angles = np.linspace(0, 2 * np.pi, cone_pts)
            for frac in [0.3, 0.6]:
                center_pt = entry_pt + frac * (tumor_center - entry_pt)
                radius_cone = 8 * _ts
                perp1 = np.array([0, 0, 1])
                perp2 = np.cross(tumor_center - entry_pt, perp1)
                perp2 = perp2 / (np.linalg.norm(perp2) + 1e-8)
                perp1 = np.cross(perp2, tumor_center - entry_pt)
                perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
                ring_x = center_pt[0] + radius_cone * (np.cos(cone_angles) * perp1[0] + np.sin(cone_angles) * perp2[0])
                ring_y = center_pt[1] + radius_cone * (np.cos(cone_angles) * perp1[1] + np.sin(cone_angles) * perp2[1])
                ring_z = center_pt[2] + radius_cone * (np.cos(cone_angles) * perp1[2] + np.sin(cone_angles) * perp2[2])
                fig_corr.add_trace(go.Scatter3d(
                    x=ring_x, y=ring_y, z=ring_z,
                    mode="lines",
                    line=dict(color=corr_color, width=2, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                    opacity=0.4,
                ))

        fig_corr.update_layout(
            height=460,
            scene=dict(
                bgcolor="#070d1a",
                xaxis=dict(showgrid=False, showticklabels=False, showline=False, title=""),
                yaxis=dict(showgrid=False, showticklabels=False, showline=False, title=""),
                zaxis=dict(showgrid=False, showticklabels=False, showline=False, title=""),
                camera=dict(eye=dict(x=1.8, y=0.6, z=0.9), up=dict(x=0, y=0, z=1)),
                aspectmode="cube",
                annotations=[
                    dict(
                        x=mid_x, y=mid_y, z=mid_z,
                        text=f"  Corridor {sel_corr_id}<br>  {active_corr['risk']*100:.0f}% risk",
                        showarrow=False,
                        font=dict(color=corr_color, size=11),
                        bgcolor="rgba(0,0,0,0.6)",
                        bordercolor=corr_color,
                        borderwidth=1,
                    ),
                    dict(
                        x=entry_pt[0], y=entry_pt[1], z=entry_pt[2] + 15,
                        text=f"  ▼ {active_corr['entry']}",
                        showarrow=False,
                        font=dict(color="#94a3b8", size=9),
                    ),
                ],
            ),
            paper_bgcolor="#0f172a",
            font=dict(color="white", family="Inter"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1, font=dict(size=10), x=0, y=1),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with detail_col:
        st.markdown(f"**Approach: {active_corr['name']}**")

        # Flow diagram: Entry → Trajectory → Tumor
        flow_steps = [
            {"icon": "🎯", "label": "Entry Point",   "value": active_corr["entry"],       "color": corr_color},
            {"icon": "↓",  "label": "",               "value": "",                          "color": "#334155"},
            {"icon": "📏", "label": "Trajectory",    "value": active_corr["trajectory"],   "color": "#94a3b8"},
            {"icon": "↓",  "label": "",               "value": "",                          "color": "#334155"},
            {"icon": "🔴", "label": "Tumor Target",  "value": f"Depth: {active_corr['length_mm']} mm", "color": "#ef4444"},
        ]
        for step in flow_steps:
            if step["label"] == "":
                st.markdown(f"<div style='text-align:center;font-size:1.4rem;color:#334155;line-height:1;margin:0.1rem 0;'>│</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div style="background:#0f1729;border:1px solid #1e293b;border-left:3px solid {step['color']};
     border-radius:8px;padding:0.65rem 0.9rem;margin:0.15rem 0;">
  <div style="font-size:0.68rem;color:#64748b;font-weight:600;letter-spacing:0.08em;
       text-transform:uppercase;">{step['icon']} {step['label']}</div>
  <div style="font-size:0.85rem;color:#e2e8f0;margin-top:0.2rem;">{step['value']}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Corridor comparison table
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;'>All Corridors — Risk Comparison</div>", unsafe_allow_html=True)

        for corr in CORRIDORS:
            rp = corr["risk"] * 100
            bar_w = int(rp * 2.2)  # scale to max ~200px
            rc = "#22c55e" if rp < 20 else "#f59e0b" if rp < 30 else "#ef4444"
            sel_indicator = "▶ " if corr["id"] == sel_corr_id else "   "
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0.5rem;margin:0.2rem 0;
     background:{'#1e293b' if corr['id'] == sel_corr_id else 'transparent'};border-radius:6px;">
  <span style="font-size:0.8rem;font-weight:700;color:{rc};width:22px;">{sel_indicator}{corr['id']}</span>
  <div style="flex:1;">
    <div style="font-size:0.72rem;color:#94a3b8;margin-bottom:2px;">{corr['name']}</div>
    <div style="background:#1e293b;border-radius:3px;height:7px;overflow:hidden;">
      <div style="background:{rc};width:{bar_w}px;max-width:100%;height:100%;border-radius:3px;transition:width 0.4s;"></div>
    </div>
  </div>
  <span style="font-size:0.9rem;font-weight:700;color:{rc};font-family:monospace;min-width:36px;text-align:right;">{rp:.0f}%</span>
</div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Key metrics for selected corridor
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;'>Selected Corridor Metrics</div>", unsafe_allow_html=True)

        metrics_data = [
            ("Trajectory Length",       f"{active_corr['length_mm']} mm",   "📏"),
            ("Brain Retraction",         f"{active_corr['brain_retraction_mm']} mm", "🔩"),
            ("Vessel Crossings",         str(active_corr['blood_vessel_crossings']), "🩸"),
            ("Eloquent Structures",      str(len(active_corr['eloquent'])),   "⚡"),
        ]
        for label, val, icon in metrics_data:
            st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
     padding:0.4rem 0;border-bottom:1px solid #1e293b;font-size:0.82rem;">
  <span style="color:#94a3b8;">{icon} {label}</span>
  <span style="color:#e2e8f0;font-weight:600;">{val}</span>
</div>""", unsafe_allow_html=True)

        # AI recommendation box
        rec_corr = next(c for c in CORRIDORS if c["recommended"])
        is_rec = (sel_corr_id == rec_corr["id"])
        box_color = "#052e16" if is_rec else "#1c1917"
        box_border = "#166534" if is_rec else "#44403c"
        box_text_color = "#4ade80" if is_rec else "#a8a29e"
        box_icon = "✅" if is_rec else "⚠️"
        box_msg = f"AI recommends this corridor — lowest risk ({rec_corr['risk']*100:.0f}%) with minimal eloquent structure involvement." if is_rec else f"AI recommends Corridor {rec_corr['id']} ({rec_corr['risk']*100:.0f}% risk). Switch for lower risk approach."

        st.markdown(f"""
<div style="background:{box_color};border:1px solid {box_border};border-radius:8px;
     padding:0.8rem 1rem;margin-top:0.8rem;font-size:0.8rem;color:{box_text_color};line-height:1.5;">
  {box_icon} <b>AI Assessment</b><br>{box_msg}
</div>""", unsafe_allow_html=True)

    st.divider()

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Top 5 Surgical Plans
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🏆 Top 5 Surgical Plans")
    st.caption("AI-ranked recommendations based on causal Monte-Carlo counterfactual analysis")

    PLAN_NAMES = {
        "remove_tumor_full":        ("Gross Total Resection",     "GTR"),
        "remove_tumor_partial":     ("Subtotal Resection",        "STR"),
        "debulk_tumor":             ("Surgical Debulking",        "DBK"),
        "biopsy_only":              ("Stereotactic Biopsy",       "BX"),
        "ablate_tissue":            ("Laser Ablation (LITT)",     "LITT"),
        "radiosurgery":             ("Stereotactic Radiosurgery", "SRS"),
        "reduce_edema":             ("Medical Management + RT",   "MMR"),
        "drain_csf":                ("CSF Diversion",             "CSD"),
        "cortical_awake_craniotomy":("Awake Craniotomy",          "AWK"),
        "awake_mapping":            ("Awake Mapping",             "AWM"),
        "clamp_artery":             ("Vascular Control",          "VAC"),
        "hyperthermic_chemo":       ("Hyperthermic Chemotherapy", "HCC"),
        "photodynamic_therapy":     ("Photodynamic Therapy",      "PDT"),
        "laser_interstitial_thermal":("LITT Ablation",            "LIT"),
    }

    if st.button("▶  Generate Top 5 Plans", type="primary", key="gen_plans_btn"):
        with st.spinner("Running causal Monte-Carlo analysis..."):
            _cf_engine = CounterfactualEngine(
                BrainTumorSCM(patient_params=patient_params),
                n_simulations=n_sims,
            )
            _top_plans = _cf_engine.monte_carlo_search(top_k=5)
            st.session_state["top_plans"] = _top_plans
            st.session_state["selected_plan_idx"] = 0

    if "top_plans" in st.session_state:
        plans = st.session_state["top_plans"]
        sel_idx = st.session_state.get("selected_plan_idx", 0)

        plan_col, detail_col = st.columns([1.4, 1.6])

        with plan_col:
            st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;'>Ranked Recommendations</div>", unsafe_allow_html=True)

            rank_colors = ["#22c55e", "#3b82f6", "#f59e0b", "#f97316", "#ef4444"]
            rank_labels = ["★ BEST", "2nd", "3rd", "4th", "5th"]

            for i, plan in enumerate(plans):
                ak = plan.actions[0].value if plan.actions else ""
                pname, pcode = PLAN_NAMES.get(ak, (ak.replace("_", " ").title(), "—"))
                rc = rank_colors[i]
                rl = rank_labels[i]
                score = plan.net_utility
                is_sel = (i == sel_idx)
                sel_border = f"border:2px solid {rc};box-shadow:0 0 0 3px {rc}22;" if is_sel else "border:1px solid #1e293b;"

                st.markdown(f"""
<div style="background:#0f1729;{sel_border}border-radius:10px;padding:0.9rem 1.1rem;margin:0.35rem 0;cursor:pointer;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.35rem;">
    <div style="display:flex;align-items:center;gap:0.6rem;">
      <span style="background:{rc}22;color:{rc};border:1px solid {rc}55;border-radius:5px;padding:0.1rem 0.5rem;font-size:0.7rem;font-weight:700;">{rl}</span>
      <span style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">{pname}</span>
      <span style="font-size:0.68rem;color:#64748b;background:#1e293b;padding:0.1rem 0.4rem;border-radius:3px;">{pcode}</span>
    </div>
    <span style="font-size:1rem;font-weight:800;color:{rc};font-family:monospace;">{score:+.3f}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.4rem;font-size:0.72rem;">
    <div><span style="color:#64748b;">Recovery</span><br><b style="color:#22c55e;">{plan.expected_recovery:.0%}</b></div>
    <div><span style="color:#64748b;">Risk</span><br><b style="color:{'#ef4444' if plan.expected_risk>0.3 else '#f59e0b' if plan.expected_risk>0.15 else '#22c55e'};">{plan.expected_risk:.0%}</b></div>
    <div><span style="color:#64748b;">Blood Loss</span><br><b style="color:#e2e8f0;">{plan.blood_loss_ml:.0f}mL</b></div>
    <div><span style="color:#64748b;">ICU Days</span><br><b style="color:#e2e8f0;">{plan.icu_days:.1f}d</b></div>
  </div>
</div>""", unsafe_allow_html=True)

                if st.button(f"Select Plan #{plan.rank}", key=f"selplan_{i}", use_container_width=True,
                             type="primary" if is_sel else "secondary"):
                    st.session_state["selected_plan_idx"] = i
                    st.rerun()

        with detail_col:
            sel_plan = plans[sel_idx]
            sel_ak = sel_plan.actions[0].value if sel_plan.actions else ""
            sel_name, sel_code = PLAN_NAMES.get(sel_ak, (sel_ak.replace("_", " ").title(), "—"))
            rc = rank_colors[sel_idx]

            st.markdown(f"<div style='font-size:0.7rem;font-weight:700;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;'>Plan #{sel_plan.rank} Detail</div>", unsafe_allow_html=True)
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f1729,#111827);border:1px solid {rc}55;border-top:3px solid {rc};border-radius:12px;padding:1.2rem;">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">{sel_name} <span style="font-size:0.75rem;color:#64748b;background:#1e293b;padding:0.15rem 0.5rem;border-radius:4px;">{sel_code}</span></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-bottom:1rem;">
    <div style="background:#0d1117;border-radius:8px;padding:0.8rem;text-align:center;">
      <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Expected Recovery</div>
      <div style="font-size:2rem;font-weight:800;color:#22c55e;font-family:monospace;">{sel_plan.expected_recovery:.0%}</div>
      <div style="font-size:0.68rem;color:#64748b;">95% CI: {sel_plan.confidence_interval[0]:.0%}–{sel_plan.confidence_interval[1]:.0%}</div>
    </div>
    <div style="background:#0d1117;border-radius:8px;padding:0.8rem;text-align:center;">
      <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Mortality Risk</div>
      <div style="font-size:2rem;font-weight:800;color:{'#ef4444' if sel_plan.expected_risk>0.3 else '#f59e0b' if sel_plan.expected_risk>0.15 else '#22c55e'};font-family:monospace;">{sel_plan.expected_risk:.0%}</div>
      <div style="font-size:0.68rem;color:#64748b;">Surgical risk</div>
    </div>
    <div style="background:#0d1117;border-radius:8px;padding:0.8rem;text-align:center;">
      <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Predicted Blood Loss</div>
      <div style="font-size:2rem;font-weight:800;color:#f59e0b;font-family:monospace;">{sel_plan.blood_loss_ml:.0f}</div>
      <div style="font-size:0.68rem;color:#64748b;">mL estimated</div>
    </div>
    <div style="background:#0d1117;border-radius:8px;padding:0.8rem;text-align:center;">
      <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Neuro Deficit Risk</div>
      <div style="font-size:2rem;font-weight:800;color:{'#ef4444' if sel_plan.nerve_damage_prob>0.25 else '#f59e0b' if sel_plan.nerve_damage_prob>0.1 else '#22c55e'};font-family:monospace;">{sel_plan.nerve_damage_prob:.0%}</div>
      <div style="font-size:0.68rem;color:#64748b;">nerve damage probability</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.8rem;">
    <div style="padding:0.4rem 0;border-bottom:1px solid #1e293b;"><span style="color:#64748b;">ICU Stay</span><span style="float:right;color:#e2e8f0;font-weight:600;">{sel_plan.icu_days:.1f} days</span></div>
    <div style="padding:0.4rem 0;border-bottom:1px solid #1e293b;"><span style="color:#64748b;">Net Utility</span><span style="float:right;color:{rc};font-weight:700;">{sel_plan.net_utility:+.3f}</span></div>
  </div>
</div>""", unsafe_allow_html=True)

            # Radar chart for selected plan
            radar_cats = ["Recovery", "Low Risk", "Low Blood Loss", "Neuro Safety", "Short ICU"]
            radar_vals = [
                sel_plan.expected_recovery,
                1 - sel_plan.expected_risk,
                1 - min(sel_plan.blood_loss_ml / 800, 1),
                1 - sel_plan.nerve_damage_prob,
                1 - min(sel_plan.icu_days / 14, 1),
            ]
            radar_vals_c = radar_vals + [radar_vals[0]]
            # Convert hex to rgba for Plotly compatibility
            hex_c = rc.lstrip('#')
            r_val, g_val, b_val = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            rgba_fill = f"rgba({r_val},{g_val},{b_val},0.13)"

            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals_c, theta=radar_cats + [radar_cats[0]],
                fill="toself", fillcolor=rgba_fill,
                line=dict(color=rc, width=2),
                name=sel_name,
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0d1117",
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9), gridcolor="#1e293b"),
                    angularaxis=dict(gridcolor="#1e293b"),
                ),
                showlegend=False, height=260,
                paper_bgcolor="#0f172a",
                font=dict(color="white", size=10),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.markdown("""<div style="background:#0f1729;border:1px dashed #334155;border-radius:10px;
        padding:2.5rem;text-align:center;color:#64748b;font-size:0.9rem;">
        Click <b>Generate Top 5 Plans</b> to run causal Monte-Carlo analysis<br>
        <span style="font-size:0.75rem;">Typically takes 5–10 seconds</span></div>""", unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Why This Plan? (Causal Explainability)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔍 Why This Plan? — Causal Reasoning")
    st.caption("Transparent AI explanations — tracing how each factor drives the recommendation")

    if "top_plans" in st.session_state:
        plans = st.session_state["top_plans"]
        sel_idx = st.session_state.get("selected_plan_idx", 0)
        sel_plan = plans[sel_idx]
        sel_ak = sel_plan.actions[0].value if sel_plan.actions else ""
        sel_name, _ = PLAN_NAMES.get(sel_ak, (sel_ak.replace("_", " ").title(), "—"))

        explain_left, explain_right = st.columns([1.2, 1.8])

        with explain_left:
            st.markdown(f"**Causal Chain for: {sel_name}**")

            # Primary causal factors with directional arrows
            _ts = patient_params["tumor_size"]
            _icp = patient_params["intracranial_pressure"]
            _bf = patient_params["blood_flow"]

            causal_chains = [
                {
                    "factor": "Tumor Volume",
                    "factor_val": f"{_ts:.0%} normalized",
                    "arrow": "↓",
                    "effect": "Blood Loss Risk",
                    "effect_val": f"+{int(_ts * 280 + 120)} mL estimated",
                    "direction": "negative",
                },
                {
                    "factor": "Distance to Brainstem",
                    "factor_val": "24.7 mm proximity",
                    "arrow": "↓",
                    "effect": "Neurological Deficit Risk",
                    "effect_val": f"{min(0.08 + _ts * 0.18, 0.6):.0%} probability",
                    "direction": "negative",
                },
                {
                    "factor": "Cerebral Blood Flow",
                    "factor_val": f"{_bf:.0%} (reduced)",
                    "arrow": "↓",
                    "effect": "Oxygen Delivery",
                    "effect_val": "Limits resection window",
                    "direction": "negative" if _bf < 0.6 else "neutral",
                },
                {
                    "factor": f"{sel_name}",
                    "factor_val": "Causal intervention applied",
                    "arrow": "↓",
                    "effect": "Recovery Probability",
                    "effect_val": f"+{sel_plan.expected_recovery - baseline.get('recovery_score', 0.3):+.0%} vs baseline",
                    "direction": "positive",
                },
            ]

            for chain in causal_chains:
                border_c = "#166534" if chain["direction"] == "positive" else "#991b1b" if chain["direction"] == "negative" else "#92400e"
                bg_c = "#052e16" if chain["direction"] == "positive" else "#2d0a0a" if chain["direction"] == "negative" else "#2d1f00"
                text_c = "#4ade80" if chain["direction"] == "positive" else "#fca5a5" if chain["direction"] == "negative" else "#fcd34d"

                st.markdown(f"""
<div style="background:{bg_c};border:1px solid {border_c};border-radius:8px;padding:0.65rem 0.9rem;margin:0.25rem 0;">
  <div style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">{chain['factor']}</div>
  <div style="font-size:0.78rem;color:#cbd5e1;margin:0.1rem 0 0.3rem;">{chain['factor_val']}</div>
</div>
<div style="text-align:center;font-size:1.1rem;color:{border_c};line-height:1.2;margin:0.1rem 0;">│<br>{chain['arrow']}</div>
<div style="background:#0f1729;border:1px solid {border_c};border-left:3px solid {border_c};border-radius:8px;padding:0.65rem 0.9rem;margin:0.1rem 0 0.5rem;">
  <div style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">{chain['effect']}</div>
  <div style="font-size:0.82rem;font-weight:600;color:{text_c};margin-top:0.1rem;">{chain['effect_val']}</div>
</div>""", unsafe_allow_html=True)

        with explain_right:
            try:
                from src.causal.attribution import CausalAttributor
                attr_scm = BrainTumorSCM(patient_params=patient_params)
                attributor = CausalAttributor(attr_scm)
                chain_obj = attributor.explain_plan(sel_plan.actions, plan_rank=sel_plan.rank)

                st.markdown(f"""
<div style="background:#0f1729;border:1px solid #1e3a5f;border-left:4px solid #3b82f6;border-radius:8px;padding:1rem;margin-bottom:0.8rem;">
  <div style="font-size:0.65rem;color:#3b82f6;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;">AI Rationale</div>
  <div style="font-size:0.87rem;color:#e2e8f0;line-height:1.5;">{chain_obj.explanation}</div>
</div>""", unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Baseline Recovery", f"{chain_obj.baseline_recovery:.1%}")
                m2.metric("Post-Intervention", f"{chain_obj.final_recovery:.1%}", delta=f"{chain_obj.recovery_delta:+.1%}")
                m3.metric("Steps Applied", str(len(sel_plan.actions)))

                if chain_obj.variable_deltas:
                    var_labels = [d.clinical_label for d in chain_obj.variable_deltas[:8]]
                    var_deltas = [d.delta for d in chain_obj.variable_deltas[:8]]
                    bar_colors = ["#22c55e" if d.direction == "improved" else "#ef4444" for d in chain_obj.variable_deltas[:8]]

                    fig_attr = go.Figure(go.Bar(
                        x=var_deltas, y=var_labels, orientation="h",
                        marker=dict(color=bar_colors, line=dict(color="#1e293b", width=1)),
                        text=[f"{d:+.3f}" for d in var_deltas],
                        textposition="outside",
                        textfont=dict(size=9, color="#e2e8f0"),
                        hovertemplate="<b>%{y}</b><br>Δ = %{x:+.4f}<extra></extra>",
                    ))
                    fig_attr.update_layout(
                        title=dict(text="Physiological Variable Changes After Intervention", font=dict(size=11, color="#64748b")),
                        height=300,
                        plot_bgcolor="#0d1117", paper_bgcolor="#0f172a",
                        font=dict(color="#e2e8f0", size=10),
                        margin=dict(l=5, r=60, t=35, b=10),
                        xaxis=dict(gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155"),
                        yaxis=dict(gridcolor="#1e293b"),
                    )
                    st.plotly_chart(fig_attr, use_container_width=True)
                    st.caption("🟢 Improved after intervention  🔴 Worsened")

            except Exception as e:
                st.info(f"Causal attribution: {e}")
    else:
        st.info("Generate plans above to see causal explanations.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Risk Dashboard
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("⚠️ Risk Dashboard")
    st.caption("Color-coded risk gauges — Green < 15% · Yellow 15–30% · Red > 30%")

    # Compute live risk metrics from SCM
    _scm_risk = BrainTumorSCM(patient_params=patient_params)
    _baseline_risk = _scm_risk.evaluate(noise=False)
    _ts = patient_params["tumor_size"]
    _icp = patient_params["intracranial_pressure"]

    mortality_risk  = max(0.03, min(_baseline_risk.get("surgical_risk", 0.12), 0.95))
    neuro_deficit   = max(0.05, min(0.06 + _ts * 0.20 + _icp * 0.08, 0.85))
    blood_loss_ml   = 180 + _ts * 380 + _icp * 120
    los_days        = 3.0 + _ts * 5.5 + _icp * 2.5

    if "top_plans" in st.session_state and st.session_state.get("selected_plan_idx") is not None:
        sp = st.session_state["top_plans"][st.session_state["selected_plan_idx"]]
        mortality_risk = sp.expected_risk
        neuro_deficit  = sp.nerve_damage_prob
        blood_loss_ml  = sp.blood_loss_ml
        los_days       = sp.icu_days

    def gauge(val, title, suffix, max_val, inv=False):
        display = val / max_val
        if inv:
            c = "#22c55e" if display > 0.7 else "#f59e0b" if display > 0.4 else "#ef4444"
        else:
            c = "#ef4444" if display > 0.45 else "#f59e0b" if display > 0.2 else "#22c55e"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": suffix, "font": {"size": 26, "color": c, "family": "JetBrains Mono, monospace"}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#334155", "tickfont": {"color": "#64748b", "size": 8}},
                "bar": {"color": c, "thickness": 0.3},
                "bgcolor": "#0d1117",
                "bordercolor": "#1e293b",
                "steps": [
                    {"range": [0, max_val * 0.2], "color": "#052e16" if not inv else "#2d0a0a"},
                    {"range": [max_val * 0.2, max_val * 0.45], "color": "#2d2000"},
                    {"range": [max_val * 0.45, max_val], "color": "#2d0a0a" if not inv else "#052e16"},
                ],
                "threshold": {"line": {"color": c, "width": 2}, "thickness": 0.7, "value": val},
            },
            title={"text": title, "font": {"size": 11, "color": "#94a3b8"}},
        ))
        fig.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=5),
                         paper_bgcolor="#0f172a", font_color="#e2e8f0")
        return fig

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(gauge(mortality_risk * 100, "Mortality Risk", "%", 60), use_container_width=True)
    with g2:
        st.plotly_chart(gauge(neuro_deficit * 100, "Neurological Deficit", "%", 60), use_container_width=True)
    with g3:
        st.plotly_chart(gauge(blood_loss_ml, "Predicted Blood Loss", " mL", 800), use_container_width=True)
    with g4:
        st.plotly_chart(gauge(los_days, "Length of Stay", " d", 14), use_container_width=True)

    # Risk summary bar
    risk_metrics = [
        ("Mortality Risk",        f"{mortality_risk*100:.0f}%",  mortality_risk, False),
        ("Neurological Deficit",  f"{neuro_deficit*100:.0f}%",   neuro_deficit,  False),
        ("Blood Loss",            f"{blood_loss_ml:.0f} mL",     blood_loss_ml/800, False),
        ("Hospital Stay",         f"{los_days:.1f} days",        los_days/14,    False),
        ("Recovery Probability",  f"{_baseline_risk.get('recovery_score',0.5):.0%}", 1-_baseline_risk.get('recovery_score',0.5), False),
    ]
    st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;margin:0.5rem 0;'>Risk Summary</div>", unsafe_allow_html=True)
    for label, val_str, raw, inv in risk_metrics:
        bc = "#ef4444" if raw > 0.45 else "#f59e0b" if raw > 0.2 else "#22c55e"
        bar_w = int(raw * 100)
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.8rem;padding:0.4rem 0;border-bottom:1px solid #1e293b;">
  <span style="font-size:0.82rem;color:#94a3b8;min-width:180px;">{label}</span>
  <div style="flex:1;background:#1e293b;border-radius:3px;height:8px;">
    <div style="background:{bc};width:{bar_w}%;height:100%;border-radius:3px;"></div>
  </div>
  <span style="font-size:0.88rem;font-weight:700;color:{bc};min-width:75px;text-align:right;">{val_str}</span>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Outcome Simulator
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🧪 Outcome Simulator")
    st.caption("Adjust parameters and instantly see how outcomes change — counterfactual reasoning made visual")

    sim_a, sim_b = st.columns([1.2, 1.8])

    with sim_a:
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;'>Surgical Parameters</div>", unsafe_allow_html=True)
        out_resection   = st.slider("Tumor Resection", 0, 100, 75, 5, format="%d%%", key="out_resect")
        out_approach    = st.selectbox("Approach", ["Pterional", "Transcortical", "Keyhole", "Awake Craniotomy"], key="out_approach")
        out_edema       = st.slider("Edema Reduction", 0, 100, 40, 5, format="%d%%", key="out_edema")
        out_blood_ctrl  = st.slider("Blood Loss Control", 0, 100, 60, 5, format="%d%%", key="out_blood")

    with sim_b:
        # Compute current vs simulated outcomes
        base_rec = _baseline_risk.get("recovery_score", 0.35)

        _sim_params = {
            **patient_params,
            "tumor_size": patient_params["tumor_size"] * (1 - (out_resection / 100) * 0.88),
            "edema_volume": patient_params.get("edema_volume", 0.3) * (1 - (out_edema / 100) * 0.7),
        }
        _sim_scm = BrainTumorSCM(patient_params=_sim_params)
        _sim_state = _sim_scm.evaluate(noise=False)
        sim_rec = _sim_state.get("recovery_score", 0.5)
        sim_blood = max(50, blood_loss_ml * (1 - (out_blood_ctrl / 100) * 0.6))
        sim_neuro = max(0.02, neuro_deficit * (1 - (out_resection / 100) * 0.3))
        delta_rec = sim_rec - base_rec

        # Current vs Simulated side-by-side
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-bottom:1rem;">
  <div style="background:#0f1729;border:1px solid #1e293b;border-radius:10px;padding:1rem;text-align:center;">
    <div style="font-size:0.65rem;color:#64748b;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;">Current Plan</div>
    <div style="font-size:2.2rem;font-weight:800;color:#94a3b8;font-family:monospace;">{base_rec:.0%}</div>
    <div style="font-size:0.72rem;color:#64748b;">Recovery Score</div>
    <div style="margin-top:0.5rem;font-size:0.78rem;color:#94a3b8;">Blood Loss: {blood_loss_ml:.0f} mL</div>
  </div>
  <div style="background:#0f1729;border:1px solid {'#22c55e' if delta_rec>=0 else '#ef4444'}55;border-top:3px solid {'#22c55e' if delta_rec>=0 else '#ef4444'};border-radius:10px;padding:1rem;text-align:center;">
    <div style="font-size:0.65rem;color:#64748b;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;">Simulated ({out_resection}% Resection)</div>
    <div style="font-size:2.2rem;font-weight:800;color:{'#22c55e' if delta_rec>=0 else '#ef4444'};font-family:monospace;">{sim_rec:.0%}</div>
    <div style="font-size:0.72rem;color:#64748b;">Recovery Score</div>
    <div style="margin-top:0.5rem;font-size:0.78rem;color:{'#22c55e' if delta_rec>=0 else '#ef4444'};font-weight:600;">Δ {delta_rec:+.0%} vs current</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Recovery vs Resection curve
        res_vals = list(range(0, 101, 5))
        rec_curve, blood_curve = [], []
        for rv in res_vals:
            _p = {**patient_params, "tumor_size": patient_params["tumor_size"] * (1 - rv/100 * 0.88)}
            _r = BrainTumorSCM(patient_params=_p).evaluate(noise=False).get("recovery_score", 0.3)
            rec_curve.append(_r)
            blood_curve.append(max(50, blood_loss_ml * (1 - rv/100 * 0.5)))

        fig_sim = make_subplots(rows=1, cols=2,
                                subplot_titles=("Recovery Score vs Resection %", "Estimated Blood Loss vs Resection %"))
        fig_sim.add_trace(go.Scatter(x=res_vals, y=rec_curve, mode="lines",
            line=dict(color="#22c55e", width=2.5), fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
            name="Recovery"), row=1, col=1)
        fig_sim.add_vline(x=out_resection, line=dict(color="#3b82f6", dash="dash", width=2),
                          annotation_text=f"{out_resection}%", annotation_font_color="#3b82f6",
                          row=1, col=1)
        fig_sim.add_trace(go.Scatter(x=res_vals, y=blood_curve, mode="lines",
            line=dict(color="#f59e0b", width=2.5), fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
            name="Blood Loss (mL)"), row=1, col=2)
        fig_sim.add_vline(x=out_resection, line=dict(color="#3b82f6", dash="dash", width=2),
                          row=1, col=2)
        fig_sim.update_layout(
            height=230, showlegend=False,
            plot_bgcolor="#0d1117", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0", size=9),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig_sim.update_xaxes(title_text="Resection %", gridcolor="#1e293b")
        fig_sim.update_yaxes(tickformat=".0%", row=1, col=1, gridcolor="#1e293b")
        fig_sim.update_yaxes(title_text="mL", row=1, col=2, gridcolor="#1e293b")
        st.plotly_chart(fig_sim, use_container_width=True)

        # What-if scenario table
        scenarios = [
            ("90% Resection (Aggressive)", 90, 35),
            ("75% Resection (Moderate)",   75, 50),
            ("50% Resection (Conservative)", 50, 65),
            ("Biopsy Only (Diagnostic)",    5, 85),
        ]
        st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;margin:0.5rem 0;'>What-If Scenarios</div>", unsafe_allow_html=True)
        for sc_name, sc_res, sc_bc in scenarios:
            _sp = {**patient_params, "tumor_size": patient_params["tumor_size"] * (1 - sc_res/100 * 0.88)}
            _sr = BrainTumorSCM(patient_params=_sp).evaluate(noise=False).get("recovery_score", 0.35)
            _sb = max(50, blood_loss_ml * (1 - sc_bc/100 * 0.6))
            _d = _sr - base_rec
            dc = "#22c55e" if _d > 0 else "#ef4444"
            st.markdown(f"""
<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:0.5rem;
     padding:0.4rem 0;border-bottom:1px solid #1e293b;font-size:0.78rem;align-items:center;">
  <span style="color:#94a3b8;">{sc_name}</span>
  <span style="color:#e2e8f0;text-align:center;">{_sr:.0%}</span>
  <span style="color:#f59e0b;text-align:center;">{_sb:.0f}mL</span>
  <span style="color:{dc};font-weight:700;text-align:right;">{_d:+.0%}</span>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8 — Anatomical Risk Heatmap
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🗺️ Anatomical Risk Heatmap")
    st.caption("Surgical risk overlay by brain region — red = eloquent / high risk, green = safe tissue")

    hmap_left, hmap_right = st.columns([1.8, 1.2])

    with hmap_left:
        # 3D brain with color-coded risk regions
        _np = np

        def _ellipsoid(rx, ry, rz, cx, cy, cz, nu=30, nv=15, noise=0.0):
            u = _np.linspace(0, 2*_np.pi, nu)
            v = _np.linspace(0, _np.pi, nv)
            x = cx + rx*(1+noise*_np.random.randn(nu,nv)*0.05)*_np.outer(_np.cos(u),_np.sin(v))
            y = cy + ry*(1+noise*_np.random.randn(nu,nv)*0.05)*_np.outer(_np.sin(u),_np.sin(v))
            z = cz + rz*_np.outer(_np.ones_like(u),_np.cos(v))
            return x.flatten(), y.flatten(), z.flatten()

        # Risk regions: (name, rx,ry,rz, cx,cy,cz, risk_score, color)
        RISK_REGIONS = [
            ("Brainstem",            14, 14, 22,   0,-5,-55, 0.96, "#dc2626"),
            ("Motor Cortex (L)",     22, 12, 18,  -55, 0, 25, 0.88, "#ef4444"),
            ("Motor Cortex (R)",     22, 12, 18,   55, 0, 25, 0.88, "#ef4444"),
            ("Broca's Area",         18, 12, 14,  -42, 20, 10, 0.82, "#f87171"),
            ("Wernicke's Area",      16, 12, 12,  -52,-30, 10, 0.80, "#f97316"),
            ("Optic Chiasm",         14, 10,  8,   0, 8,-35, 0.75, "#fb923c"),
            ("Cerebellum",           35, 38, 28,   0,-50,-35, 0.60, "#fbbf24"),
            ("Ant. Cerebral A.",      8,  8,  6,  -10, 45, 20, 0.70, "#f59e0b"),
            ("Prefrontal Cortex",    28, 22, 18,   0, 55, 20, 0.35, "#4ade80"),
            ("Parietal Assoc.",      28, 22, 16,  -30,-20, 40, 0.30, "#22c55e"),
            ("Frontal Pole",         18, 20, 16,   0, 65, 15, 0.15, "#16a34a"),
            ("Occipital Pole",       20, 18, 14,   0,-65, 10, 0.20, "#22c55e"),
            ("Tumor Region",         int(16+patient_params["tumor_size"]*18),
                                     int(14+patient_params["tumor_size"]*14),
                                     int(12+patient_params["tumor_size"]*10),
                                     -22,-12, 8, 0.99, "#b91c1c"),
        ]

        fig_hmap = go.Figure()

        # Add brain shell (very transparent)
        bx, by, bz = _ellipsoid(85, 92, 70, 0, 0, 0, nu=50, nv=25)
        fig_hmap.add_trace(go.Mesh3d(x=bx, y=by, z=bz, alphahull=0,
            color="#4a6fa5", opacity=0.06, showlegend=False, hoverinfo="skip"))

        # Add each risk region
        for rname, rx, ry, rz, cx, cy, cz, risk, color in RISK_REGIONS:
            ex, ey, ez = _ellipsoid(rx, ry, rz, cx, cy, cz, nu=20, nv=12)
            fig_hmap.add_trace(go.Mesh3d(
                x=ex, y=ey, z=ez, alphahull=0,
                color=color, opacity=0.75,
                name=f"{rname} ({risk:.0%})",
                showlegend=True,
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
                hovertemplate=f"<b>{rname}</b><br>Risk: {risk:.0%}<extra></extra>",
            ))

        fig_hmap.update_layout(
            height=480,
            scene=dict(
                bgcolor="#060a12",
                xaxis=dict(showgrid=False, showticklabels=False, showline=False, title=""),
                yaxis=dict(showgrid=False, showticklabels=False, showline=False, title=""),
                zaxis=dict(showgrid=False, showticklabels=False, showline=False, title=""),
                camera=dict(eye=dict(x=1.5, y=0.5, z=0.8), up=dict(x=0, y=0, z=1)),
                aspectmode="cube",
            ),
            paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0", size=9, family="Inter"),
            legend=dict(bgcolor="#0d1117", bordercolor="#1e293b", borderwidth=1,
                       font=dict(size=9), x=0, y=1, itemsizing="constant"),
            margin=dict(l=0, r=0, t=5, b=0),
        )
        st.plotly_chart(fig_hmap, use_container_width=True)

    with hmap_right:
        st.markdown("**Risk by Anatomical Region**")

        risk_table = [
            ("Tumor Region",         0.99, "Critical"),
            ("Brainstem",            0.96, "Critical"),
            ("Motor Cortex",         0.88, "Eloquent"),
            ("Broca's Area (Speech)",0.82, "Eloquent"),
            ("Wernicke's Area",      0.80, "Eloquent"),
            ("Optic Chiasm",         0.75, "Critical"),
            ("Ant. Cerebral Artery", 0.70, "Vascular"),
            ("Cerebellum",           0.60, "Functional"),
            ("Prefrontal Cortex",    0.35, "Functional"),
            ("Parietal Association", 0.30, "Low Risk"),
            ("Occipital Pole",       0.20, "Low Risk"),
            ("Frontal Pole",         0.15, "Low Risk"),
        ]

        for struct, risk, category in risk_table:
            rc = "#ef4444" if risk > 0.70 else "#f59e0b" if risk > 0.45 else "#22c55e"
            bg = "#2d0a0a" if risk > 0.70 else "#2d2000" if risk > 0.45 else "#052e16"
            border = "#991b1b" if risk > 0.70 else "#92400e" if risk > 0.45 else "#166534"
            bar_w = int(risk * 100)
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0;border-bottom:1px solid #1e293b;">
  <div style="width:10px;height:10px;border-radius:50%;background:{rc};flex-shrink:0;"></div>
  <div style="flex:1;">
    <div style="font-size:0.78rem;color:#e2e8f0;font-weight:500;">{struct}</div>
    <div style="font-size:0.65rem;color:#64748b;">{category}</div>
  </div>
  <div style="background:#1e293b;border-radius:2px;height:6px;width:80px;">
    <div style="background:{rc};width:{bar_w}%;height:100%;border-radius:2px;"></div>
  </div>
  <span style="font-size:0.82rem;font-weight:700;color:{rc};min-width:36px;text-align:right;">{risk:.0%}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="background:#0f1729;border:1px solid #1e293b;border-radius:8px;padding:0.8rem;margin-top:0.8rem;font-size:0.75rem;color:#94a3b8;">
  <b style="color:#e2e8f0;">Legend</b><br>
  <span style="color:#ef4444;">■</span> Critical / Eloquent (>70%)<br>
  <span style="color:#f59e0b;">■</span> Functional (45–70%)<br>
  <span style="color:#22c55e;">■</span> Lower Risk (&lt;45%)
</div>""", unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 9 — Procedure Timeline
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📋 Procedure Timeline")
    st.caption("Full pre-operative to intraoperative workflow — powered by NeuroPlan AI")

    tl_left, tl_right = st.columns([1, 2])

    TIMELINE_STEPS = [
        {"step": 1, "icon": "📡", "title": "MRI Upload & DICOM Import",
         "detail": "T1, T2, FLAIR, and DWI sequences uploaded. Voxel spacing: 1×1×1mm", "status": "done", "time": "~2 min"},
        {"step": 2, "icon": "🧬", "title": "AI Segmentation (MONAI)",
         "detail": "Automated tumor delineation: GBM core, edema, necrotic center", "status": "done", "time": "~45 sec"},
        {"step": 3, "icon": "🏗️", "title": "3D Mesh Reconstruction",
         "detail": "Marching cubes → smoothed surface meshes for all structures", "status": "done", "time": "~30 sec"},
        {"step": 4, "icon": "🔗", "title": "Graph Neural Network Analysis",
         "detail": "Anatomical knowledge graph: vascular supply, compression chains, risk edges", "status": "done", "time": "~20 sec"},
        {"step": 5, "icon": "🔬", "title": "Corridor Search",
         "detail": "AI evaluates 3 surgical trajectories, ranks by proximity to eloquent structures", "status": "done", "time": "~15 sec"},
        {"step": 6, "icon": "⚠️", "title": "Causal Risk Assessment",
         "detail": "Pearl Do-Calculus × SCM: blood loss, neuro deficit, mortality estimates computed", "status": "active", "time": "~10 sec"},
        {"step": 7, "icon": "🏆", "title": "Final Surgical Plan Output",
         "detail": "Top 5 plans ranked by net utility. PDF surgical report generated.", "status": "pending", "time": "~5 sec"},
    ]

    with tl_left:
        for i, step in enumerate(TIMELINE_STEPS):
            if step["status"] == "done":
                dot_color = "#22c55e"
                dot_bg = "#052e16"
                dot_border = "#166534"
                icon_s = "✅"
                title_c = "#22c55e"
            elif step["status"] == "active":
                dot_color = "#3b82f6"
                dot_bg = "#0f1f3d"
                dot_border = "#1d4ed8"
                icon_s = "🔵"
                title_c = "#60a5fa"
            else:
                dot_color = "#475569"
                dot_bg = "#1e293b"
                dot_border = "#334155"
                icon_s = "⏸"
                title_c = "#64748b"

            connector = "" if i == len(TIMELINE_STEPS) - 1 else f"""
<div style="margin-left:0.7rem;width:2px;height:20px;background:linear-gradient({dot_border},{dot_border if TIMELINE_STEPS[i+1]['status']=='done' else '#334155'});"></div>"""

            st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:0.8rem;">
  <div style="display:flex;flex-direction:column;align-items:center;">
    <div style="width:26px;height:26px;border-radius:50%;background:{dot_bg};border:2px solid {dot_border};
         display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:{dot_color};flex-shrink:0;font-weight:700;">{step['step']}</div>
    {connector}
  </div>
  <div style="padding-bottom:0.3rem;">
    <div style="font-size:0.88rem;font-weight:600;color:{title_c};">{step['icon']} {step['title']}</div>
    <div style="font-size:0.72rem;color:#64748b;margin-top:0.1rem;">{step['detail']}</div>
    <div style="font-size:0.68rem;color:{dot_color};margin-top:0.15rem;">⏱ {step['time']}</div>
  </div>
</div>""", unsafe_allow_html=True)

    with tl_right:
        # Timeline Gantt bar chart
        import pandas as pd
        gantt_data = [
            {"Task": step["title"][:30], "Start": i * 1.0,
             "Duration": 0.8,
             "Color": "#22c55e" if step["status"] == "done" else "#3b82f6" if step["status"] == "active" else "#475569"}
            for i, step in enumerate(TIMELINE_STEPS)
        ]

        fig_tl = go.Figure()
        for d in gantt_data:
            fig_tl.add_trace(go.Bar(
                x=[d["Duration"]], y=[d["Task"]],
                orientation="h", base=d["Start"],
                marker=dict(color=d["Color"], line=dict(color="#1e293b", width=1)),
                showlegend=False,
                hovertemplate=f"<b>{d['Task']}</b><extra></extra>",
            ))

        total_time = 3.0  # minutes
        fig_tl.add_annotation(
            x=5.5, y=-0.5,
            text=f"Total AI Planning Time: <b>~{total_time:.0f} min</b>  vs  Traditional: <b>4–8 hours</b>",
            showarrow=False, font=dict(size=11, color="#22c55e"),
            xanchor="center",
        )
        fig_tl.update_layout(
            title=dict(text="NeuroPlan AI Pipeline — Execution Timeline", font=dict(size=12, color="#64748b")),
            height=320,
            plot_bgcolor="#0d1117", paper_bgcolor="#0f172a",
            font=dict(color="#e2e8f0", size=9),
            xaxis=dict(title="Time (seconds)", gridcolor="#1e293b",
                       tickvals=list(range(8)), ticktext=[f"Step {i+1}" for i in range(7)] + ["Done"]),
            yaxis=dict(gridcolor="#1e293b", autorange="reversed"),
            margin=dict(l=5, r=5, t=40, b=40),
            barmode="stack",
        )
        st.plotly_chart(fig_tl, use_container_width=True)

        # Time saved callout
        st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;margin-top:0.5rem;">
  <div style="background:#052e16;border:1px solid #166534;border-radius:10px;padding:0.9rem;text-align:center;">
    <div style="font-size:1.8rem;font-weight:800;color:#22c55e;font-family:monospace;">~3min</div>
    <div style="font-size:0.72rem;color:#64748b;">NeuroPlan AI</div>
  </div>
  <div style="background:#0f1729;border:1px solid #1e293b;border-radius:10px;padding:0.9rem;text-align:center;display:flex;align-items:center;justify-content:center;">
    <span style="font-size:1.5rem;color:#ef4444;">vs</span>
  </div>
  <div style="background:#2d0a0a;border:1px solid #991b1b;border-radius:10px;padding:0.9rem;text-align:center;">
    <div style="font-size:1.8rem;font-weight:800;color:#ef4444;font-family:monospace;">4–8h</div>
    <div style="font-size:0.72rem;color:#64748b;">Traditional Planning</div>
  </div>
</div>""", unsafe_allow_html=True)



# ─── Patient Management Page ──────────────────────────────────────────────────
elif page == "👥 Patient Management":
    st.title("👥 Patient Management")
    st.caption("Create, view, and manage patient records and surgical plans")

    # Create Patient Form
    with st.expander("➕ Create New Patient", expanded=False):
        with st.form("create_patient"):
            col1, col2 = st.columns(2)
            with col1:
                patient_id = st.text_input("Patient ID", value="P001")
                name = st.text_input("Patient Name", value="John Doe")
                date_of_birth = st.date_input("Date of Birth")
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                diagnosis = st.text_input("Diagnosis", value="Glioblastoma")
                tumor_type = st.selectbox("Tumor Type", ["glioblastoma", "meningioma", "metastasis", "schwannoma"])
                tumor_location = st.selectbox("Tumor Location", ["frontal_lobe", "parietal_lobe", "temporal_lobe", "occipital_lobe", "cerebellum", "brainstem"])
                tumor_size = st.slider("Tumor Size", 0.0, 1.0, 0.35, 0.01)
                grade = st.selectbox("Grade", ["I", "II", "III", "IV"])

            if st.form_submit_button("Create Patient"):
                try:
                    response = requests.post(
                        f"{API_BASE}/patients",
                        json={
                            "patient_id": patient_id,
                            "name": name,
                            "date_of_birth": date_of_birth.isoformat(),
                            "gender": gender,
                            "diagnosis": diagnosis,
                            "tumor_type": tumor_type,
                            "tumor_location": tumor_location,
                            "tumor_size": tumor_size,
                            "grade": grade,
                        }
                    )
                    if response.status_code == 200:
                        st.success(f"Patient {patient_id} created successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to create patient: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # List Patients
    st.subheader("📋 Patient Records")
    try:
        response = requests.get(f"{API_BASE}/patients")
        if response.status_code == 200:
            data = response.json()
            patients = data.get("patients", [])
            if patients:
                for patient in patients:
                    with st.expander(f"👤 {patient['name']} ({patient['patient_id']})"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Diagnosis:** {patient.get('diagnosis', 'N/A')}")
                            st.write(f"**Tumor Type:** {patient.get('tumor_type', 'N/A')}")
                        with col2:
                            st.write(f"**Location:** {patient.get('tumor_location', 'N/A')}")
                            st.write(f"**Size:** {patient.get('tumor_size', 'N/A')}")
                        with col3:
                            st.write(f"**Grade:** {patient.get('grade', 'N/A')}")
                            st.write(f"**Created:** {patient.get('created_at', 'N/A')}")

                        # View Studies
                        studies_response = requests.get(f"{API_BASE}/patients/{patient['patient_id']}/studies")
                        if studies_response.status_code == 200:
                            studies = studies_response.json().get("studies", [])
                            if studies:
                                st.write("**MRI Studies:**")
                                for study in studies:
                                    st.write(f"  - {study['study_id']}: {study.get('modality', 'N/A')} ({study.get('study_date', 'N/A')})")

                        # View Plans
                        plans_response = requests.get(f"{API_BASE}/patients/{patient['patient_id']}/plans")
                        if plans_response.status_code == 200:
                            plans = plans_response.json().get("plans", [])
                            if plans:
                                st.write("**Surgical Plans:**")
                                for plan in plans:
                                    st.write(f"  - {plan['plan_id']}: {plan.get('status', 'N/A')} (Recovery: {plan.get('expected_recovery', 'N/A'):.1%})")
            else:
                st.info("No patients found. Create a new patient above.")
        else:
            st.error("Failed to fetch patients")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")


# ─── Simulation Monitor Page ──────────────────────────────────────────────────
elif page == "📊 Simulation Monitor":
    st.title("📊 Simulation Monitor")
    st.caption("Run and monitor surgery simulations in real-time")

    # Simulation Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_duration = st.number_input("Duration (ms)", value=10000, min_value=1000, max_value=60000, step=1000)
    with col2:
        sim_dt = st.number_input("Time Step (ms)", value=500, min_value=100, max_value=5000, step=100)
    with col3:
        tumor_removal_ms = st.number_input("Tumor Removal (ms)", value=5000, min_value=0, max_value=30000, step=500)

    if st.button("▶  Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                response = requests.post(
                    f"{API_BASE}/simulation/simulate",
                    params={"duration_ms": sim_duration, "dt_ms": sim_dt, "tumor_removal_at_ms": tumor_removal_ms}
                )
                if response.status_code == 200:
                    data = response.json()
                    states = data.get("states", [])

                    if states:
                        timestamps = [s["timestamp_ms"] for s in states]
                        vitals = {ch: [s["vitals"].get(ch, 0) for s in states] for ch in ["blood_pressure_systolic", "blood_pressure_diastolic", "heart_rate", "spo2", "intracranial_pressure", "cerebral_blood_flow"]}

                        fig = make_subplots(rows=2, cols=2, subplot_titles=("Blood Pressure", "Heart Rate & SpO2", "ICP", "CBF"))

                        fig.add_trace(go.Scatter(x=timestamps, y=vitals["blood_pressure_systolic"], name="Systolic", line=dict(color="#ef4444")), row=1, col=1)
                        fig.add_trace(go.Scatter(x=timestamps, y=vitals["blood_pressure_diastolic"], name="Diastolic", line=dict(color="#f97316")), row=1, col=1)
                        fig.add_trace(go.Scatter(x=timestamps, y=vitals["heart_rate"], name="HR", line=dict(color="#3b82f6")), row=1, col=2)
                        fig.add_trace(go.Scatter(x=timestamps, y=vitals["spo2"], name="SpO2", line=dict(color="#10b981")), row=1, col=2)
                        fig.add_trace(go.Scatter(x=timestamps, y=vitals["intracranial_pressure"], name="ICP", line=dict(color="#f59e0b")), row=2, col=1)
                        fig.add_trace(go.Scatter(x=timestamps, y=vitals["cerebral_blood_flow"], name="CBF", line=dict(color="#8b5cf6")), row=2, col=2)

                        if tumor_removal_ms:
                            fig.add_vline(x=tumor_removal_ms, line_dash="dash", line_color="yellow", annotation_text="Tumor Removed")

                        fig.update_layout(height=600, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="white"))
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Final Outcome", f"{data.get('final_outcome', 0):.2%}")
                        m2.metric("Total Alerts", data.get("alerts_summary", {}).get("total", 0))
                        m3.metric("Critical", data.get("alerts_summary", {}).get("critical", 0))
                        m4.metric("Warnings", data.get("alerts_summary", {}).get("warning", 0))
                    else:
                        st.warning("No simulation data returned")
                else:
                    st.error(f"Simulation failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")


# ─── Comparative Analysis Page ─────────────────────────────────────────────────
elif page == "📈 Comparative Analysis":
    st.title("📈 Comparative Analysis")
    st.caption("Compare different surgical approaches side-by-side")

    # SCM Parameter Comparison
    st.subheader("🔬 SCM Parameter Impact Analysis")

    col1, col2 = st.columns(2)
    with col1:
        base_tumor = st.slider("Base Tumor Size", 0.0, 1.0, 0.30, 0.01, key="base_tumor")
        compare_tumor = st.slider("Compare Tumor Size", 0.0, 1.0, 0.60, 0.01, key="compare_tumor")
    with col2:
        base_icp = st.slider("Base ICP", 0.0, 1.0, 0.20, 0.01, key="base_icp")
        compare_icp = st.slider("Compare ICP", 0.0, 1.0, 0.50, 0.01, key="compare_icp")

    if st.button("Compare SCM States"):
        scm_base = BrainTumorSCM(patient_params={
            "tumor_size": base_tumor, "edema_volume": 0.2, "intracranial_pressure": base_icp,
            "blood_flow": 0.7, "inflammatory_response": 0.3
        })
        scm_compare = BrainTumorSCM(patient_params={
            "tumor_size": compare_tumor, "edema_volume": 0.2, "intracranial_pressure": compare_icp,
            "blood_flow": 0.7, "inflammatory_response": 0.3
        })

        state_base = scm_base.evaluate(noise=False)
        state_compare = scm_compare.evaluate(noise=False)

        # Comparison chart
        variables = list(state_base.keys())
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Base",
            x=variables,
            y=[state_base[v] for v in variables],
            marker_color="#3b82f6"
        ))
        fig_comp.add_trace(go.Bar(
            name="Compare",
            x=variables,
            y=[state_compare[v] for v in variables],
            marker_color="#ef4444"
        ))
        fig_comp.update_layout(
            title="SCM State Comparison",
            barmode="group",
            height=400,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Delta metrics
        st.subheader("📊 State Deltas")
        delta_cols = st.columns(4)
        for i, var in enumerate(variables[:8]):
            with delta_cols[i % 4]:
                delta = state_compare[var] - state_base[var]
                st.metric(var.replace("_", " ").title(),
                         f"{state_compare[var]:.2%}",
                         delta=f"{delta:+.2%}")

    st.divider()

    # Intervention Comparison
    st.subheader("⚡ Intervention Effectiveness Comparison")

    if st.button("Compare All Interventions"):
        scm = BrainTumorSCM(patient_params=patient_params)
        engine = DoCalculusEngine(scm)

        results = []
        for action in SurgicalAction:
            r = engine.intervene(action, noise=False)
            results.append({
                "Action": action.value.replace("_", " ").title(),
                "Recovery Gain": r.recovery_gain,
                "Risk Increase": r.risk_increase,
                "Net Utility": r.net_utility,
            })

        results.sort(key=lambda x: x["Net Utility"], reverse=True)

        fig_int = go.Figure()
        fig_int.add_trace(go.Bar(
            name="Recovery Gain",
            x=[r["Action"] for r in results],
            y=[r["Recovery Gain"] for r in results],
            marker_color="#22c55e"
        ))
        fig_int.add_trace(go.Bar(
            name="Risk Increase",
            x=[r["Action"] for r in results],
            y=[r["Risk Increase"] for r in results],
            marker_color="#ef4444"
        ))
        fig_int.update_layout(
            title="Intervention Comparison: Recovery Gain vs Risk",
            barmode="group",
            height=400,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="white"),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_int, use_container_width=True)

        # Net utility ranking
        fig_rank = go.Figure(go.Bar(
            x=[r["Net Utility"] for r in results],
            y=[r["Action"] for r in results],
            orientation="h",
            marker_color=["#22c55e" if x > 0 else "#ef4444" for x in [r["Net Utility"] for r in results]],
        ))
        fig_rank.update_layout(
            title="Net Utility Ranking",
            height=max(300, len(results) * 35),
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="white"),
            xaxis=dict(title="Net Utility"),
        )
        st.plotly_chart(fig_rank, use_container_width=True)


# ─── Export & Reports Page ─────────────────────────────────────────────────────
elif page == "📤 Export & Reports":
    st.title("📤 Export & Reports")
    st.caption("Generate and download surgical reports and plan data")

    # Report Generation
    st.subheader("📋 Surgical Report Generator")

    report_col1, report_col2 = st.columns([2, 1])
    with report_col1:
        report_patient = st.text_input("Patient ID", value="P001", key="report_patient")
        report_type = st.selectbox("Report Type", ["Full Surgical Plan", "Intervention Summary", "Risk Assessment", "Counterfactual Analysis"])

    with report_col2:
        include_charts = st.checkbox("Include Charts", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)

    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            scm = BrainTumorSCM(patient_params=patient_params)
            engine = DoCalculusEngine(scm)
            cf_engine = CounterfactualEngine(scm, n_simulations=100)

            # Generate report content
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("BRAIN TUMOR SURGICAL PLANNING REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Patient ID: {report_patient}")
            report_lines.append(f"Report Type: {report_type}")
            report_lines.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}")
            report_lines.append("")

            # Baseline state
            baseline = scm.evaluate(noise=False)
            report_lines.append("BASELINE STATE:")
            report_lines.append("-" * 40)
            for k, v in baseline.items():
                report_lines.append(f"  {k}: {v:.4f}")
            report_lines.append("")

            # Interventions
            report_lines.append("INTERVENTION ANALYSIS:")
            report_lines.append("-" * 40)
            for action in SurgicalAction:
                r = engine.intervene(action, noise=False)
                report_lines.append(f"  {action.value}:")
                report_lines.append(f"    Recovery Gain: {r.recovery_gain:+.4f}")
                report_lines.append(f"    Risk Increase: {r.risk_increase:.4f}")
                report_lines.append(f"    Net Utility: {r.net_utility:+.4f}")
            report_lines.append("")

            # Top Plans
            top_plans = cf_engine.monte_carlo_search(top_k=5)
            report_lines.append("TOP SURGICAL PLANS:")
            report_lines.append("-" * 40)
            for plan in top_plans:
                report_lines.append(f"  Rank #{plan.rank}:")
                report_lines.append(f"    Actions: {', '.join(a.value for a in plan.actions)}")
                report_lines.append(f"    Expected Recovery: {plan.expected_recovery:.2%}")
                report_lines.append(f"    Expected Risk: {plan.expected_risk:.2%}")
                report_lines.append(f"    Net Utility: {plan.net_utility:+.4f}")
                report_lines.append(f"    Blood Loss: {plan.blood_loss_ml:.0f} mL")
                report_lines.append(f"    ICU Days: {plan.icu_days:.1f}")
            report_lines.append("")

            if include_raw_data:
                report_lines.append("RAW DATA:")
                report_lines.append("-" * 40)
                report_lines.append(json.dumps({
                    "baseline": baseline,
                    "patient_params": patient_params,
                }, indent=2))

            report_text = "\n".join(report_lines)

            st.code(report_text, language=None)

            # PDF Download
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                try:
                    from src.reports.pdf_generator import PDFReportGenerator
                    pdf_gen = PDFReportGenerator()

                    # Gather attribution if plans exist
                    attr_chain = None
                    if st.session_state.get("top_plans"):
                        try:
                            from src.causal.attribution import CausalAttributor
                            attr_scm2 = BrainTumorSCM(patient_params=patient_params)
                            attributor2 = CausalAttributor(attr_scm2)
                            best_plan = st.session_state["top_plans"][0]
                            chain2 = attributor2.explain_plan(best_plan.actions)
                            attr_chain = attributor2.to_dict(chain2)
                        except Exception:
                            pass

                    pdf_bytes = pdf_gen.generate(
                        patient_id=report_patient,
                        baseline_scm=baseline,
                        top_plans=st.session_state.get("latest_plans", []),
                        gnn_prediction=None,
                        attribution_chain=attr_chain,
                        patient_params=patient_params,
                    )
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"surgical_plan_{report_patient}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )
                except ImportError:
                    st.warning("Install reportlab for PDF: `pip install reportlab`")
                except Exception as e:
                    st.error(f"PDF generation error: {e}")

            with col_dl2:
                st.download_button(
                    label="📥 Download TXT Report",
                    data=report_text,
                    file_name=f"surgical_report_{report_patient}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

    st.divider()

    # Export Plan Data
    st.subheader("📊 Export Plan Data (JSON)")

    export_format = st.selectbox("Export Format", ["JSON", "CSV"])
    export_scope = st.selectbox("Export Scope", ["All Plans", "Top 5 Plans", "Top 3 Plans"])

    if st.button("Prepare Export"):
        scm = BrainTumorSCM(patient_params=patient_params)
        cf_engine = CounterfactualEngine(scm, n_simulations=100)

        top_k = 5 if "5" in export_scope else (3 if "3" in export_scope else 10)
        plans = cf_engine.monte_carlo_search(top_k=top_k)

        export_data = {
            "patient_params": patient_params,
            "plans": [p.to_dict() for p in plans],
            "export_timestamp": __import__('datetime').datetime.now().isoformat(),
        }

        if export_format == "JSON":
            export_json = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=export_json,
                file_name=f"plans_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            import pandas as pd
            df = pd.DataFrame([{
                "Rank": p.rank,
                "Actions": ", ".join(a.value for a in p.actions),
                "Recovery": p.expected_recovery,
                "Risk": p.expected_risk,
                "Utility": p.net_utility,
                "Blood Loss (mL)": p.blood_loss_ml,
                "ICU Days": p.icu_days,
                "Nerve Damage": p.nerve_damage_prob,
            } for p in plans])

            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"plans_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        st.dataframe(df if export_format == "CSV" else export_data["plans"])

    st.divider()

    # Simulation Export
    st.subheader("🏥 Simulation Data Export")

    sim_duration_exp = st.number_input("Duration (ms)", value=5000, key="exp_duration")
    if st.button("Run & Export Simulation"):
        with st.spinner("Running simulation..."):
            try:
                response = requests.post(
                    f"{API_BASE}/simulation/simulate",
                    params={"duration_ms": sim_duration_exp, "dt_ms": 500}
                )
                if response.status_code == 200:
                    sim_data = response.json()
                    sim_json = json.dumps(sim_data, indent=2)
                    st.download_button(
                        label="📥 Download Simulation Data",
                        data=sim_json,
                        file_name=f"simulation_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )
                    st.success("Simulation complete! Click download above.")
                else:
                    st.error(f"Simulation failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠ Research prototype — not for clinical use. All decisions must be made by qualified surgeons.")
