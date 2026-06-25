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

    # Row 3: Monte-Carlo Plan Search
    st.subheader("🎲 Monte-Carlo Surgical Path Search")

    if st.button("▶  Run Search", type="primary"):
        with st.spinner(f"Running {n_sims} simulations per plan..."):
            cf_engine = CounterfactualEngine(
                BrainTumorSCM(patient_params=patient_params),
                n_simulations=n_sims
            )
            top_plans = cf_engine.monte_carlo_search(top_k=5)
            st.session_state["top_plans"] = top_plans

    if "top_plans" in st.session_state:
        plans = st.session_state["top_plans"]

        categories = ["Recovery", "Low Risk", "Low Blood Loss", "Neural Safety", "Fast ICU"]
        fig_radar = go.Figure()
        colors = ["#7c3aed", "#2563eb", "#16a34a", "#dc2626", "#d97706"]

        for i, plan in enumerate(plans):
            values = [
                plan.expected_recovery,
                1 - plan.expected_risk,
                1 - min(plan.blood_loss_ml / 1000, 1),
                1 - plan.nerve_damage_prob,
                1 - min(plan.icu_days / 14, 1),
            ]
            values += [values[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=f"#{plan.rank}: {plan.actions[0].value.replace('_',' ')}",
                line=dict(color=colors[i % len(colors)]),
                opacity=0.7,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=400,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font=dict(color="white"),
        )

        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_r2:
            for plan in plans:
                risk_class = (
                    "high-risk" if plan.expected_risk > 0.4 else
                    "med-risk"  if plan.expected_risk > 0.2 else "low-risk"
                )
                actions_str = " → ".join(a.value.replace("_", " ").title() for a in plan.actions)
                st.markdown(f"""
<div class="metric-card {risk_class}">
  <b>#{plan.rank}</b> {actions_str}<br>
  Recovery: <b>{plan.expected_recovery:.0%}</b> &nbsp;|&nbsp;
  Risk: <b>{plan.expected_risk:.0%}</b> &nbsp;|&nbsp;
  Utility: <b>{plan.net_utility:+.3f}</b><br>
  ICU: {plan.icu_days:.1f} days &nbsp;|&nbsp;
  Blood loss: {plan.blood_loss_ml:.0f} mL<br>
  95% CI: [{plan.confidence_interval[0]:.0%}, {plan.confidence_interval[1]:.0%}]
</div>
""", unsafe_allow_html=True)

        st.divider()

        # ── 3D Brain Visualization ──────────────────────────────────────────
        st.subheader("🧬 3D Brain Anatomy & Surgical Plan")
        st.caption("Interactive 3D view — drag to rotate, scroll to zoom")

        try:
            from src.graph.viz_3d import BrainMeshBuilder
            builder = BrainMeshBuilder()
            top_plan_dict = plans[0].to_dict() if plans else None
            fig_3d = builder.build_simple_figure(
                tumor_size=patient_params.get("tumor_size", 0.3),
                tumor_position="frontal",
                critical_proximity=patient_params.get("intracranial_pressure", 0.2),
            )
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
                st.caption("🟥 Tumor  🟠 Brainstem  🟣 Vessels  🟢 Approach corridor  🟣 Critical structures")
        except Exception as e:
            st.info(f"3D view loading... ({e})")

        st.divider()

        # ── Causal Attribution: Why This Plan? ──────────────────────────────
        st.subheader("🔬 Why This Plan? — Causal Attribution")
        st.caption(
            "Traces exactly which physiological variables changed (and by how much) "
            "when each surgical action is applied via Do-Calculus."
        )

        try:
            from src.causal.attribution import CausalAttributor
            from src.causal.do_calculus import SurgicalAction as SA

            attr_scm = BrainTumorSCM(patient_params=patient_params)
            attributor = CausalAttributor(attr_scm)

            plan_tabs = st.tabs([f"Plan #{p.rank}" for p in plans[:3]])
            for tab_idx, (tab, plan) in enumerate(zip(plan_tabs, plans[:3])):
                with tab:
                    chain = attributor.explain_plan(
                        plan.actions, plan_rank=plan.rank
                    )

                    # Explanation sentence
                    st.markdown(
                        f"<div class='attribution-box'>💡 {chain.explanation}</div>",
                        unsafe_allow_html=True
                    )

                    # Recovery delta metric
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric(
                        "Baseline Recovery",
                        f"{chain.baseline_recovery:.1%}",
                    )
                    col_m2.metric(
                        "Post-intervention Recovery",
                        f"{chain.final_recovery:.1%}",
                        delta=f"{chain.recovery_delta:+.1%}",
                    )
                    col_m3.metric(
                        "Actions",
                        str(len(plan.actions)),
                    )

                    # Waterfall chart
                    if chain.variable_deltas:
                        var_labels = [d.clinical_label for d in chain.variable_deltas[:8]]
                        var_deltas = [d.delta for d in chain.variable_deltas[:8]]
                        bar_colors = [
                            "#22c55e" if d.direction == "improved" else "#ef4444"
                            for d in chain.variable_deltas[:8]
                        ]

                        fig_attr = go.Figure(go.Bar(
                            x=var_deltas,
                            y=var_labels,
                            orientation="h",
                            marker_color=bar_colors,
                            text=[f"{d:+.3f}" for d in var_deltas],
                            textposition="outside",
                            hovertemplate="<b>%{y}</b><br>Delta: %{x:+.4f}<extra></extra>",
                        ))
                        fig_attr.update_layout(
                            height=320,
                            xaxis_title="Variable Change (after intervention)",
                            plot_bgcolor="#0f172a",
                            paper_bgcolor="#0f172a",
                            font=dict(color="white", size=11),
                            margin=dict(l=10, r=60, t=20, b=30),
                            xaxis=dict(
                                gridcolor="#1e2030",
                                zeroline=True,
                                zerolinecolor="#4b5563",
                            ),
                            yaxis=dict(gridcolor="#1e2030"),
                        )
                        st.plotly_chart(fig_attr, use_container_width=True)
                        st.caption("🟢 Green = clinically improved  🔴 Red = worsened")

        except Exception as e:
            st.info(f"Attribution engine: {e}")

        st.divider()

        # Store for PDF export
        st.session_state["latest_plans"] = [p.to_dict() for p in plans]
        st.session_state["latest_baseline"] = baseline
        st.session_state["latest_patient_params"] = patient_params

    st.divider()

    # Row 4: Counterfactual Explorer
    st.subheader("🔁 Counterfactual Explorer")
    st.caption("Ask: 'What would recovery have been if we had done X instead of Y?'")

    cf_col1, cf_col2, cf_col3 = st.columns([1, 1, 1])
    with cf_col1:
        factual_action = st.selectbox(
            "Factual action (what happened)",
            ["(no surgery)"] + [a.value for a in SurgicalAction],
            index=0,
        )
    with cf_col2:
        cf_action = st.selectbox(
            "Counterfactual action (what if...)",
            [a.value for a in SurgicalAction],
            index=0,
        )
    with cf_col3:
        if st.button("Run Counterfactual"):
            scm_cf = BrainTumorSCM(patient_params=patient_params)
            cf_eng = CounterfactualEngine(scm_cf, n_simulations=50)

            fa = None
            if factual_action != "(no surgery)":
                fa = SurgicalAction(factual_action)

            factual_outcome = scm_cf.evaluate(noise=False)
            if fa:
                DoCalculusEngine(BrainTumorSCM(patient_params=patient_params)).intervene(fa, noise=False)

            query = CounterfactualQuery(
                factual_action=fa,
                counterfactual_action=SurgicalAction(cf_action),
                observed_outcome=factual_outcome,
            )
            cf_result = cf_eng.run_counterfactual(query)
            st.session_state["cf_result"] = cf_result

    if "cf_result" in st.session_state:
        cfr = st.session_state["cf_result"]
        good = cfr.was_better
        icon = "✅" if good else "⚠️"
        delta_str = f"{cfr.recovery_delta:+.1%}"

        st.markdown(f"""
**{icon} {cfr.explanation}**

| World | Recovery Score |
|-------|---------------|
| Factual ({factual_action}) | {cfr.factual_state.get('recovery_score', 0):.1%} |
| Counterfactual ({cf_action}) | {cfr.counterfactual_state.get('recovery_score', 0):.1%} |
| **Delta** | **{delta_str}** |
""")

    st.divider()

    # Row 5: SNN Physiology Monitor
    st.subheader("🏥 Intraoperative Physiology Monitor (SNN)")

    from src.simulation.snn_physiology import IntraoperativeMonitor, PHYSIO_CHANNELS

    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1])

    with sim_col1:
        sim_duration = st.number_input("Simulation Duration (ms)", value=5000, min_value=1000, max_value=30000, step=1000)
    with sim_col2:
        sim_dt = st.number_input("Time Step (ms)", value=500, min_value=100, max_value=2000, step=100)
    with sim_col3:
        tumor_removal_ms = st.number_input("Tumor Removal at (ms)", value=2500, min_value=0, max_value=30000, step=500)

    if st.button("▶  Run Surgery Simulation", type="primary"):
        with st.spinner("Running SNN physiology simulation..."):
            monitor = IntraoperativeMonitor()
            states = monitor.simulate_surgery(
                duration_ms=sim_duration,
                dt_ms=sim_dt,
                tumor_removal_at_ms=tumor_removal_ms,
            )

            timestamps = [s.timestamp_ms for s in states]
            channels_data = {ch: [s.vitals.get(ch, 0) for s in states] for ch in PHYSIO_CHANNELS}
            alert_levels = [s.alert_level for s in states]
            predicted_outcomes = [s.predicted_outcome for s in states]

            fig_vitals = make_subplots(
                rows=3, cols=2,
                subplot_titles=("Blood Pressure", "Heart Rate & SpO2", "ICP & CBF", "EtO2 & EtCO2", "Predicted Outcome", "Alert Level"),
                vertical_spacing=0.08,
            )

            fig_vitals.add_trace(go.Scatter(x=timestamps, y=channels_data["blood_pressure_systolic"], name="Systolic", line=dict(color="#ef4444")), row=1, col=1)
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=channels_data["blood_pressure_diastolic"], name="Diastolic", line=dict(color="#f97316")), row=1, col=1)
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=channels_data["heart_rate"], name="HR", line=dict(color="#3b82f6")), row=1, col=2)
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=channels_data["spo2"], name="SpO2", line=dict(color="#10b981")), row=1, col=2)
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=channels_data["intracranial_pressure"], name="ICP", line=dict(color="#f59e0b")), row=2, col=1)
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=channels_data["cerebral_blood_flow"], name="CBF", line=dict(color="#8b5cf6")), row=2, col=1)
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=channels_data["eto2"], name="EtO2", line=dict(color="#06b6d4")), row=2, col=2)
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=channels_data["etco2"], name="EtCO2", line=dict(color="#64748b")), row=2, col=2)
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=predicted_outcomes, name="Outcome", line=dict(color="#22c55e", width=2)), row=3, col=1)

            alert_colors = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
            alert_numeric = [alert_colors.get(a, 0) for a in alert_levels]
            fig_vitals.add_trace(go.Scatter(x=timestamps, y=alert_numeric, name="Alert", fill="tozeroy", line=dict(color="#ef4444")), row=3, col=2)

            if tumor_removal_ms:
                fig_vitals.add_vline(x=tumor_removal_ms, line_dash="dash", line_color="yellow", annotation_text="Tumor Removed")

            fig_vitals.update_layout(
                height=800, showlegend=True,
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_vitals, use_container_width=True)

            alert_summary = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
            for s in states:
                alert_summary[s.alert_level] = alert_summary.get(s.alert_level, 0) + 1

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Outcome", f"{states[-1].predicted_outcome:.2%}")
            m2.metric("Critical Alerts", alert_summary.get("CRITICAL", 0))
            m3.metric("Warnings", alert_summary.get("WARNING", 0))
            m4.metric("Normal Readings", alert_summary.get("NORMAL", 0))

    st.divider()

    # Row 6: Knowledge Graph Explorer
    st.subheader("🔬 Anatomical Knowledge Graph Explorer")

    from src.graph.knowledge_graph import AnatomicalKnowledgeGraph

    @st.cache_resource
    def load_kg():
        return AnatomicalKnowledgeGraph()

    kg = load_kg()

    kg_tab1, kg_tab2, kg_tab3 = st.tabs(["Blood Supply Chain", "Compression Chain", "Surgical Risks"])

    with kg_tab1:
        st.markdown("Query blood supply relationships for anatomical structures.")
        supply_structure = st.selectbox(
            "Select structure",
            ["white_matter", "gray_matter", "frontal_lobe", "cerebellum", "brainstem"],
            key="supply_tab"
        )
        if st.button("Query Blood Supply", key="btn_supply"):
            suppliers = kg.get_blood_supply_chain(supply_structure)
            if suppliers:
                fig_supply = go.Figure()
                fig_supply.add_trace(go.Scatter(
                    x=list(range(len(suppliers))),
                    y=[1] * len(suppliers),
                    mode="markers+text",
                    marker=dict(size=30, color="#ef4444"),
                    text=suppliers,
                    textposition="top center",
                ))
                fig_supply.update_layout(
                    title=f"Blood Supply Chain for {supply_structure}",
                    height=200, showlegend=False,
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font=dict(color="white"),
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 2]),
                )
                st.plotly_chart(fig_supply, use_container_width=True)
                st.write(f"**Suppliers:** {', '.join(suppliers)}")
            else:
                st.info("No blood supply relationships found.")

    with kg_tab2:
        st.markdown("Query structures compressed by tumors.")
        compress_tumor = st.selectbox(
            "Select tumor/compression source",
            ["enhancing_tumor", "peritumoral_edema"],
            key="compress_tab"
        )
        if st.button("Query Compression", key="btn_compress"):
            compressed = kg.get_compression_chain(compress_tumor)
            if compressed:
                st.write(f"**{compress_tumor} compresses:**")
                for s in compressed:
                    st.write(f"  - {s}")
            else:
                st.info("No compression relationships found.")

    with kg_tab3:
        st.markdown("Query risks associated with surgical actions.")
        risk_action = st.selectbox(
            "Select surgical action",
            ["remove_tumor_full", "remove_tumor_partial", "clamp_artery", "drain_csf", "radiosurgery", "reduce_edema", "cortical_awake_craniotomy", "awake_mapping", "laser_interstitial_thermal", "hyperthermic_chemo", "photodynamic_therapy"],
            key="risk_tab"
        )
        if st.button("Query Risks", key="btn_risks"):
            risks = kg.get_action_risks(risk_action)
            if risks:
                fig_risks = go.Figure(go.Bar(
                    x=[r["risk"] for r in risks],
                    y=[r["structure"] for r in risks],
                    orientation="h",
                    marker_color="#ef4444",
                ))
                fig_risks.update_layout(
                    title=f"Risks for {risk_action}",
                    height=max(200, len(risks) * 40),
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font=dict(color="white"),
                    xaxis=dict(title="Risk Factor", range=[0, 1]),
                )
                st.plotly_chart(fig_risks, use_container_width=True)
            else:
                st.info("No risk relationships found for this action.")

    st.divider()

    # Row 7: 3D Brain Mesh Viewer
    st.subheader("🧊 3D Brain Mesh Viewer")

    col_patient, col_refresh = st.columns([3, 1])
    with col_patient:
        try:
            resp = requests.get(f"{API_BASE}/outputs", timeout=3)
            patients = resp.json().get("patients", [])
        except Exception:
            patients = []
            if os.path.isdir("/app/outputs"):
                patients = sorted([
                    d.name for d in Path("/app/outputs").iterdir()
                    if d.is_dir() and (d / "surgical_report.txt").exists()
                ])
        if not patients:
            st.info("No pipeline outputs found. Run the pipeline first to generate 3D brain meshes.")
            st.code("python scripts/run_pipeline.py --patient-id BRATS001 --mri data/adapted/BRATS001 --output outputs", language="bash")
            st.stop()
        selected_patient = st.selectbox("Select patient", patients)

    import glob
    from pathlib import Path

    mesh_dir = Path("/app/outputs") / selected_patient / "meshes"
    mesh_files = sorted(mesh_dir.glob("*.obj")) if mesh_dir.exists() else []

    if not mesh_files:
        st.info(f"No mesh files found for {selected_patient}.")
        st.stop()

    mesh_colors = {
        "background": "#636efa",
        "enhancing_tumor": "#ef553b",
        "necrotic_tumor_core": "#00cc96",
        "peritumoral_edema": "#ffa15a",
        "tumor_core": "#ab63fa",
        "edema": "#ffa15a",
        "brain": "#636efa",
    }

    mesh_labels = {f.name: f.stem.replace(f"{selected_patient}_", "").replace("_", " ").title()
                   for f in mesh_files}

    selected_mesh = st.selectbox("Select mesh structure", list(mesh_labels.values()))
    mesh_path = [f for f in mesh_files if mesh_labels[f] == selected_mesh][0]

    import trimesh
    try:
        mesh = trimesh.load(str(mesh_path))
        vertices = mesh.vertices
        faces = mesh.faces

        color_key = next((k for k in mesh_colors if k in mesh_path.stem), "#7c3aed")
        color = mesh_colors.get(color_key, "#7c3aed")

        fig_3d = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=0.85,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
            )
        ])
        fig_3d.update_layout(
            title=f"{selected_mesh} — {selected_patient}",
            height=500,
            scene=dict(
                bgcolor="#0f172a",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                zaxis=dict(showgrid=False, showticklabels=False),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            paper_bgcolor="#0f172a",
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption(f"Vertices: {len(vertices):,} | Faces: {len(faces):,} | {mesh_path.name}")
    except Exception as e:
        st.error(f"Error loading mesh: {e}")


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
