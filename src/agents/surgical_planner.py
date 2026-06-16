"""
src/agents/surgical_planner.py
LangGraph-based agentic surgical planning system.

Agents:
  1. AnatomyAgent   — parses anatomical graph, identifies key structures
  2. CausalAgent    — runs SCM, applies do() operators
  3. RiskAgent      — evaluates surgical risks from GNN
  4. PlannerAgent   — generates surgical plan candidates
  5. RankingAgent   — ranks plans by utility, selects top 5
  6. ReportAgent    — generates human-readable surgical report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger

from ..causal.scm import BrainTumorSCM
from ..causal.do_calculus import DoCalculusEngine, SurgicalAction
from ..causal.counterfactual import CounterfactualEngine, SurgicalPathScore

try:
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available — using rule-based agent fallback")


# ─── Shared Agent State ───────────────────────────────────────────────────────
class PlannerState(TypedDict):
    patient_id:         str
    twin_summary:       Dict
    graph_summary:      Dict
    gnn_prediction:     Dict
    scm_state:          Dict
    intervention_results: List[Dict]
    candidate_plans:    List[Dict]
    top_plans:          List[Dict]
    surgical_report:    str
    error:              Optional[str]
    step:               str


# ─── Individual Agents ────────────────────────────────────────────────────────
class AnatomyAgent:
    """Parses the anatomical digital twin and identifies surgical targets."""

    def run(self, state: PlannerState) -> Dict:
        logger.info("AnatomyAgent: analyzing twin summary")
        twin = state.get("twin_summary", {})
        structures = twin.get("structures", [])

        critical = [s for s in structures if s.get("name") in (
            "brainstem", "internal_carotid_artery", "basilar_artery"
        )]
        tumors = [s for s in structures if s.get("is_tumor")]
        high_risk_proximity = []

        for tumor in tumors:
            t_center = tumor.get("centroid_mm") or tumor.get("centroid_voxel", [0, 0, 0])
            for crit in critical:
                c_center = crit.get("centroid_mm") or crit.get("centroid_voxel", [0, 0, 0])
                import numpy as np
                dist = float(np.linalg.norm(np.array(t_center) - np.array(c_center)))
                if dist < 20.0:
                    high_risk_proximity.append({
                        "tumor": tumor["name"],
                        "critical_structure": crit["name"],
                        "distance_mm": round(dist, 2),
                    })

        graph_summary = {
            "total_structures": len(structures),
            "tumor_structures": len(tumors),
            "critical_structures": len(critical),
            "tumor_volume_mm3": twin.get("total_tumor_volume_mm3", 0),
            "high_risk_proximities": high_risk_proximity,
            "primary_tumor": tumors[0] if tumors else None,
        }

        return {"graph_summary": graph_summary, "step": "anatomy_done"}


class CausalAgent:
    """Runs the SCM and evaluates all available surgical interventions."""

    def __init__(self, scm: BrainTumorSCM):
        self.scm = scm

    def run(self, state: PlannerState) -> Dict:
        logger.info("CausalAgent: evaluating do() operators")
        engine = DoCalculusEngine(self.scm)
        scm_state = self.scm.evaluate(noise=False)

        results = []
        for action in SurgicalAction:
            try:
                result = engine.intervene(action, noise=False)
                results.append({
                    "action": action.value,
                    "recovery_gain": round(result.recovery_gain, 4),
                    "risk_increase": round(result.risk_increase, 4),
                    "net_utility": round(result.net_utility, 4),
                    "pre_recovery": round(result.pre_state.get("recovery_score", 0), 4),
                    "post_recovery": round(result.post_state.get("recovery_score", 0), 4),
                    "downstream_effects": result.downstream_effects,
                })
            except Exception as e:
                logger.warning(f"CausalAgent: {action.value} failed — {e}")

        # Sort by net_utility
        results.sort(key=lambda r: r["net_utility"], reverse=True)

        return {
            "scm_state": scm_state,
            "intervention_results": results,
            "step": "causal_done",
        }


class PlannerAgent:
    """Generates and evaluates multi-step surgical plans via Monte-Carlo search."""

    def __init__(self, scm: BrainTumorSCM, n_simulations: int = 200):
        self.scm = scm
        self.n_simulations = n_simulations

    def run(self, state: PlannerState) -> Dict:
        logger.info(f"PlannerAgent: Monte-Carlo search ({self.n_simulations} sims/plan)")
        engine = CounterfactualEngine(self.scm, n_simulations=self.n_simulations)
        top_plans = engine.monte_carlo_search(top_k=5)

        plans_dict = [p.to_dict() for p in top_plans]
        return {"candidate_plans": plans_dict, "top_plans": plans_dict, "step": "planning_done"}


class RiskAgent:
    """Evaluates and annotates risk factors from GNN predictions + SCM state."""

    def run(self, state: PlannerState) -> Dict:
        logger.info("RiskAgent: annotating risk factors")
        gnn = state.get("gnn_prediction", {})
        scm = state.get("scm_state", {})
        plans = state.get("top_plans", [])

        risk_annotations = {
            "overall_surgical_risk": round(scm.get("surgical_risk", 0.5), 3),
            "baseline_recovery":     round(scm.get("recovery_score", 0.5), 3),
            "icp_concern":           scm.get("intracranial_pressure", 0.0) > 0.5,
            "vascular_concern":      scm.get("vascular_compression", 0.0) > 0.4,
            "gnn_mortality_risk":    gnn.get("mortality_risk", 0.05),
            "gnn_nerve_damage":      gnn.get("nerve_damage_prob", 0.1),
            "gnn_blood_loss_ml":     gnn.get("blood_loss_ml", 200),
        }

        # Annotate each plan with risk level
        for plan in plans:
            plan["risk_level"] = (
                "HIGH" if plan.get("expected_risk", 0) > 0.4 else
                "MEDIUM" if plan.get("expected_risk", 0) > 0.2 else "LOW"
            )

        return {"top_plans": plans, "step": "risk_done", **{"risk_annotations": risk_annotations}}


class ReportAgent:
    """Generates a structured surgical planning report."""

    def run(self, state: PlannerState) -> Dict:
        logger.info("ReportAgent: generating surgical report")

        patient_id   = state.get("patient_id", "UNKNOWN")
        graph        = state.get("graph_summary", {})
        scm          = state.get("scm_state", {})
        top_plans    = state.get("top_plans", [])
        gnn          = state.get("gnn_prediction", {})

        lines = [
            "=" * 60,
            f"CAUSAL SURGICAL PLANNING REPORT",
            f"Patient: {patient_id}",
            "=" * 60,
            "",
            "ANATOMICAL SUMMARY",
            f"  Tumor structures:     {graph.get('tumor_structures', 0)}",
            f"  Total tumor volume:   {graph.get('tumor_volume_mm3', 0):.1f} mm³",
            f"  Critical structures:  {graph.get('critical_structures', 0)}",
            f"  High-risk proximities: {len(graph.get('high_risk_proximities', []))}",
            "",
            "BASELINE PHYSIOLOGICAL STATE (SCM)",
            f"  Blood flow:           {scm.get('blood_flow', 0):.0%}",
            f"  Oxygen saturation:    {scm.get('oxygen_saturation', 0):.0%}",
            f"  Intracranial pressure:{scm.get('intracranial_pressure', 0):.0%} (normalized)",
            f"  Neural function:      {scm.get('neural_function', 0):.0%}",
            f"  Baseline recovery:    {scm.get('recovery_score', 0):.0%}",
            "",
            "GNN RISK ASSESSMENT",
            f"  Blood loss estimate:  {gnn.get('blood_loss_ml', 'N/A')} mL",
            f"  Nerve damage prob:    {gnn.get('nerve_damage_prob', 'N/A'):.0%}" if gnn else "  (GNN not available)",
            f"  Mortality risk:       {gnn.get('mortality_risk', 'N/A'):.0%}" if gnn else "",
            "",
            "TOP 5 SURGICAL PLANS (Monte-Carlo Counterfactual Search)",
            "-" * 60,
        ]

        for plan in top_plans:
            lines += [
                f"",
                f"  RANK {plan.get('rank', '?')} [{plan.get('risk_level', '')}]",
                f"  Actions: {' → '.join(plan.get('actions', []))}",
                f"  Expected recovery:  {plan.get('expected_recovery', 0):.0%}",
                f"  Expected risk:      {plan.get('expected_risk', 0):.0%}",
                f"  Net utility:        {plan.get('net_utility', 0):+.3f}",
                f"  Blood loss:         {plan.get('blood_loss_ml', 0):.0f} mL",
                f"  Nerve damage prob:  {plan.get('nerve_damage_prob', 0):.0%}",
                f"  ICU days:           {plan.get('icu_days', 0):.1f}",
                f"  95% CI recovery:    {plan.get('confidence_95', [0,0])[0]:.0%} – {plan.get('confidence_95', [0,0])[1]:.0%}",
            ]

        lines += [
            "",
            "=" * 60,
            "RECOMMENDED PLAN: " + (
                " → ".join(top_plans[0].get("actions", [])) if top_plans else "UNDETERMINED"
            ),
            "=" * 60,
            "",
            "⚠  This report is generated by a research AI system.",
            "   All surgical decisions must be made by qualified surgeons.",
        ]

        report = "\n".join(lines)
        return {"surgical_report": report, "step": "report_done"}


# ─── Orchestrator ─────────────────────────────────────────────────────────────
class SurgicalPlannerOrchestrator:
    """
    Runs all agents in sequence (LangGraph state machine or simple pipeline).
    Falls back to direct pipeline if LangGraph unavailable.
    """

    def __init__(
        self,
        scm: BrainTumorSCM,
        n_simulations: int = 200,
        use_llm: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        self.scm = scm
        self.n_simulations = n_simulations
        self.use_llm = use_llm

        self.anatomy_agent  = AnatomyAgent()
        self.causal_agent   = CausalAgent(scm)
        self.planner_agent  = PlannerAgent(scm, n_simulations)
        self.risk_agent     = RiskAgent()
        self.report_agent   = ReportAgent()

    def run(
        self,
        patient_id: str,
        twin_summary: Dict,
        gnn_prediction: Optional[Dict] = None,
    ) -> PlannerState:
        """Execute the full multi-agent surgical planning pipeline."""

        state: PlannerState = {
            "patient_id":           patient_id,
            "twin_summary":         twin_summary,
            "graph_summary":        {},
            "gnn_prediction":       gnn_prediction or {},
            "scm_state":            {},
            "intervention_results": [],
            "candidate_plans":      [],
            "top_plans":            [],
            "surgical_report":      "",
            "error":                None,
            "step":                 "init",
        }

        agents = [
            ("anatomy",  self.anatomy_agent),
            ("causal",   self.causal_agent),
            ("planning", self.planner_agent),
            ("risk",     self.risk_agent),
            ("report",   self.report_agent),
        ]

        for name, agent in agents:
            try:
                logger.info(f"Running agent: {name}")
                updates = agent.run(state)
                state.update(updates)
            except Exception as e:
                logger.error(f"Agent {name} failed: {e}")
                state["error"] = f"{name}: {str(e)}"
                break

        return state
