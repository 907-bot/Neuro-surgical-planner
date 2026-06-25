"""
huggingface/app.py
Gradio demo app for Hugging Face Spaces deployment.
Runs fully on CPU — no MRI data required.

Deploy:
    1. Create repo at huggingface.co/spaces/<username>/neuroplan-ai
    2. Upload this file + requirements.txt
    3. HF auto-builds and deploys

Demo URL: https://huggingface.co/spaces/<username>/neuroplan-ai
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from src.causal.scm import BrainTumorSCM
from src.causal.do_calculus import DoCalculusEngine, SurgicalAction
from src.causal.counterfactual import CounterfactualEngine
from src.causal.attribution import CausalAttributor
from src.graph.viz_3d import BrainMeshBuilder


# ─── Core inference function ──────────────────────────────────────────────────
def run_surgical_planner(
    tumor_size: float,
    edema_volume: float,
    intracranial_pressure: float,
    blood_flow: float,
    inflammatory_response: float,
    n_sims: int = 100,
) -> tuple:
    """
    Run the full causal surgical planning pipeline.
    Returns: (plans_table, causal_attribution_text, 3d_figure, report_text)
    """
    patient_params = {
        "tumor_size":            tumor_size,
        "edema_volume":          edema_volume,
        "intracranial_pressure": intracranial_pressure,
        "blood_flow":            blood_flow,
        "inflammatory_response": inflammatory_response,
    }

    # Causal pipeline
    scm = BrainTumorSCM(patient_params=patient_params)
    baseline = scm.evaluate(noise=False)

    # Monte-Carlo plan search
    cf_engine = CounterfactualEngine(scm, n_simulations=n_sims)
    top_plans = cf_engine.monte_carlo_search(top_k=3)

    # Causal attribution for top plan
    scm2 = BrainTumorSCM(patient_params=patient_params)
    attributor = CausalAttributor(scm2)
    attribution = attributor.explain_plan(top_plans[0].actions) if top_plans else None

    # Format plans table
    plans_data = []
    for plan in top_plans:
        actions_str = " → ".join(a.value.replace("_", " ").title() for a in plan.actions)
        risk_icon = "🔴" if plan.expected_risk > 0.4 else ("🟡" if plan.expected_risk > 0.2 else "🟢")
        plans_data.append([
            f"#{plan.rank}",
            actions_str,
            f"{plan.expected_recovery:.1%}",
            f"{plan.expected_risk:.1%} {risk_icon}",
            f"{plan.net_utility:+.3f}",
            f"{plan.blood_loss_ml:.0f} mL",
            f"{plan.icu_days:.1f} days",
            f"{plan.confidence_interval[0]:.0%}–{plan.confidence_interval[1]:.0%}",
        ])

    # Attribution text
    attr_text = ""
    if attribution:
        attr_text = f"## 🔬 Causal Attribution — Why Plan #1 Works\n\n"
        attr_text += f"**{attribution.explanation}**\n\n"
        attr_text += f"| Variable | Before | After | Change | Effect |\n"
        attr_text += f"|----------|--------|-------|--------|---------|\n"
        for d in attribution.variable_deltas[:6]:
            arrow = "↑" if d.delta > 0 else "↓"
            icon = "✅" if d.direction == "improved" else "⚠️"
            attr_text += (
                f"| {d.clinical_label} | {d.before:.1%} | {d.after:.1%} | "
                f"{arrow}{abs(d.delta_pct):.0f}% | {icon} {d.direction.title()} |\n"
            )

    # 3D brain figure
    builder = BrainMeshBuilder()
    fig_3d = builder.build_simple_figure(
        tumor_size=tumor_size,
        tumor_position="frontal",
    )

    # Plain text report
    report = f"""
CAUSALNEURO — SURGICAL PLANNING REPORT
======================================
Tumor Size: {tumor_size:.0%}  |  ICP: {intracranial_pressure:.0%}  |  Blood Flow: {blood_flow:.0%}

BASELINE PHYSIOLOGICAL STATE
  Blood Flow:            {baseline['blood_flow']:.1%}
  Oxygen Saturation:     {baseline['oxygen_saturation']:.1%}
  Intracranial Pressure: {baseline['intracranial_pressure']:.1%}
  Neural Function:       {baseline['neural_function']:.1%}
  Baseline Recovery:     {baseline['recovery_score']:.1%}

TOP 3 SURGICAL PLANS
{'-' * 50}"""
    for plan in top_plans:
        actions_str = " → ".join(a.value for a in plan.actions)
        report += f"""
Rank #{plan.rank}: {actions_str}
  Recovery: {plan.expected_recovery:.1%}  |  Risk: {plan.expected_risk:.1%}
  Utility: {plan.net_utility:+.3f}  |  Blood Loss: {plan.blood_loss_ml:.0f} mL
  95% CI: [{plan.confidence_interval[0]:.0%}, {plan.confidence_interval[1]:.0%}]
"""
    if attribution:
        report += f"\nCAUSAL CHAIN:\n{attribution.explanation}\n"

    report += "\n⚠ RESEARCH PROTOTYPE — NOT FOR CLINICAL USE"

    return plans_data, attr_text, fig_3d, report


# ─── Gradio App ───────────────────────────────────────────────────────────────
def build_app() -> "gr.Blocks":
    with gr.Blocks(
        title="🧠 NeuroPlan AI — Causal Brain Tumor Surgical Planner",
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        #title { text-align: center; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🧠 NeuroPlan AI — Causal Brain Tumor Surgical Planner
            ### Pearl Do-Calculus × Counterfactual Simulation × Monte-Carlo Path Search

            > Built for India's 100,000+ annual brain tumor cases.
            > 1 neurosurgeon serves ~400,000 patients — this tool acts as a **Causal AI co-pilot**.

            **Not prediction. Counterfactual reasoning: *"What would recovery be if we intervene at X?"***

            ---
            """,
            elem_id="title",
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛 Patient Parameters (SCM)")

                tumor_size = gr.Slider(0.0, 1.0, value=0.35, step=0.01,
                                       label="Tumor Size (normalized)")
                edema = gr.Slider(0.0, 1.0, value=0.25, step=0.01,
                                  label="Peritumoral Edema Volume")
                icp = gr.Slider(0.0, 1.0, value=0.30, step=0.01,
                                label="Intracranial Pressure (normalized)")
                blood_flow = gr.Slider(0.0, 1.0, value=0.60, step=0.01,
                                       label="Cerebral Blood Flow")
                inflammation = gr.Slider(0.0, 1.0, value=0.35, step=0.01,
                                         label="Inflammatory Response")
                n_sims = gr.Slider(50, 300, value=100, step=50,
                                   label="Monte-Carlo Simulations (more = stable but slower)")

                run_btn = gr.Button("▶ Run Causal Surgical Planner", variant="primary",
                                    size="lg")

                gr.Markdown(
                    """
                    **India Stats**
                    - 📊 100,000+ new brain tumor cases/year
                    - 👨‍⚕️ 1 neurosurgeon per ~400,000 people
                    - ⏱️ Pre-op planning: 4–8 hours/case
                    - 🏥 Tier-2/3 hospitals: zero surgical AI access

                    Target: AIIMS · Apollo · NIMHANS · CMC Vellore
                    """
                )

            with gr.Column(scale=2):
                gr.Markdown("### 🏆 Top 3 Surgical Plans")
                plans_table = gr.Dataframe(
                    headers=["Rank", "Actions", "Recovery", "Risk", "Utility",
                              "Blood Loss", "ICU", "95% CI Recovery"],
                    datatype=["str"] * 8,
                    interactive=False,
                )

                gr.Markdown("### 🧬 3D Brain Anatomy")
                brain_3d = gr.Plot(label="3D Brain — drag to rotate, scroll to zoom")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔬 Causal Attribution")
                attribution_md = gr.Markdown("*Run the planner to see causal attribution.*")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📄 Full Surgical Report")
                report_text = gr.Textbox(label="Report", lines=20, max_lines=30,
                                          interactive=False)

        # Examples
        gr.Examples(
            examples=[
                [0.35, 0.25, 0.30, 0.60, 0.35, 100],   # Moderate case
                [0.70, 0.50, 0.55, 0.40, 0.60, 150],   # High-risk case
                [0.15, 0.10, 0.15, 0.85, 0.15, 100],   # Low-risk case
            ],
            inputs=[tumor_size, edema, icp, blood_flow, inflammation, n_sims],
            label="Example Cases",
        )

        gr.Markdown(
            """
            ---
            ⚠️ **IMPORTANT**: This is a **research prototype** — NOT for clinical use.
            All surgical decisions must be made by qualified neurosurgeons.

            📂 **Source**: [github.com/abhishekadari/brain-surgical-planner](https://github.com/abhishekadari/brain-surgical-planner)
            📄 **Paper**: See `docs/preprint/paper_draft.md`
            """
        )

        run_btn.click(
            fn=run_surgical_planner,
            inputs=[tumor_size, edema, icp, blood_flow, inflammation, n_sims],
            outputs=[plans_table, attribution_md, brain_3d, report_text],
        )

    return demo


if __name__ == "__main__":
    if not GRADIO_AVAILABLE:
        print("Install gradio: pip install gradio")
        sys.exit(1)
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
