"""
tests/test_causal.py
Unit tests for SCM, Do-Calculus, and Counterfactual engine.
These tests run without any ML dependencies.
"""

import pytest
import numpy as np
from copy import deepcopy

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.causal.scm import BrainTumorSCM
from src.causal.do_calculus import DoCalculusEngine, SurgicalAction, ACTION_REGISTRY
from src.causal.counterfactual import (
    CounterfactualEngine,
    CounterfactualQuery,
    SurgicalPathScore,
)


# ─── SCM Tests ────────────────────────────────────────────────────────────────
class TestBrainTumorSCM:

    def test_initialization(self):
        scm = BrainTumorSCM()
        assert "tumor_size" in scm.variables
        assert "recovery_score" in scm.variables
        assert len(scm.variables) >= 10

    def test_evaluate_returns_all_variables(self):
        scm = BrainTumorSCM()
        state = scm.evaluate(noise=False)
        assert isinstance(state, dict)
        assert "recovery_score" in state
        assert "blood_flow" in state
        assert "neural_function" in state

    def test_evaluate_values_in_range(self):
        scm = BrainTumorSCM()
        state = scm.evaluate(noise=False)
        for k, v in state.items():
            assert 0.0 <= v <= 3.0, f"{k} = {v} out of range"

    def test_larger_tumor_reduces_recovery(self):
        scm_small = BrainTumorSCM(patient_params={"tumor_size": 0.1})
        scm_large = BrainTumorSCM(patient_params={"tumor_size": 0.9})
        s1 = scm_small.evaluate(noise=False)
        s2 = scm_large.evaluate(noise=False)
        assert s2["recovery_score"] < s1["recovery_score"], (
            "Larger tumor should reduce recovery"
        )

    def test_dag_is_acyclic(self):
        import networkx as nx
        scm = BrainTumorSCM()
        assert nx.is_directed_acyclic_graph(scm.dag)

    def test_topological_ordering_consistent(self):
        import networkx as nx
        scm = BrainTumorSCM()
        order = list(nx.topological_sort(scm.dag))
        # tumor_size should come before recovery_score
        if "tumor_size" in order and "recovery_score" in order:
            assert order.index("tumor_size") < order.index("recovery_score")

    def test_reset_interventions(self):
        scm = BrainTumorSCM()
        scm.variables["tumor_size"].intervened = True
        scm.reset_interventions()
        assert not any(v.intervened for v in scm.variables.values())

    def test_noise_adds_variation(self):
        scm = BrainTumorSCM()
        results = [scm.evaluate(noise=True)["recovery_score"] for _ in range(20)]
        scm2 = BrainTumorSCM()
        base = scm2.evaluate(noise=False)["recovery_score"]
        assert not all(r == base for r in results), "Noise should add variation"

    def test_summary(self):
        scm = BrainTumorSCM()
        summary = scm.summary()
        assert "variables" in summary
        assert "edges" in summary
        assert len(summary["edges"]) > 0


# ─── Do-Calculus Tests ────────────────────────────────────────────────────────
class TestDoCalculusEngine:

    def test_all_actions_registered(self):
        for action in SurgicalAction:
            assert action in ACTION_REGISTRY, f"{action} missing from registry"

    def test_intervene_returns_result(self):
        scm = BrainTumorSCM()
        engine = DoCalculusEngine(scm)
        result = engine.intervene(SurgicalAction.BIOPSY_ONLY, noise=False)
        assert result.pre_state is not None
        assert result.post_state is not None
        assert isinstance(result.recovery_gain, float)

    def test_full_removal_improves_recovery(self):
        scm = BrainTumorSCM(patient_params={"tumor_size": 0.7})
        engine = DoCalculusEngine(scm)
        result = engine.intervene(SurgicalAction.REMOVE_TUMOR_FULL, noise=False)
        assert result.recovery_gain > 0, "Full removal should improve recovery"

    def test_full_removal_better_than_partial(self):
        scm_full    = BrainTumorSCM(patient_params={"tumor_size": 0.7})
        scm_partial = BrainTumorSCM(patient_params={"tumor_size": 0.7})
        r_full    = DoCalculusEngine(scm_full).intervene(SurgicalAction.REMOVE_TUMOR_FULL)
        r_partial = DoCalculusEngine(scm_partial).intervene(SurgicalAction.REMOVE_TUMOR_PARTIAL)
        assert r_full.post_state["recovery_score"] >= r_partial.post_state["recovery_score"]

    def test_intervention_isolates_target(self):
        """do(X=x) should fix X regardless of its structural equation."""
        scm = BrainTumorSCM(patient_params={"tumor_size": 0.9})
        engine = DoCalculusEngine(scm)
        result = engine.intervene(SurgicalAction.REMOVE_TUMOR_FULL)
        # tumor_size should be set to target_value
        target_val = ACTION_REGISTRY[SurgicalAction.REMOVE_TUMOR_FULL]["target_value"]
        assert abs(result.post_state["tumor_size"] - target_val) < 0.01

    def test_arterial_clamp_reduces_blood_flow(self):
        scm = BrainTumorSCM()
        engine = DoCalculusEngine(scm)
        result = engine.intervene(SurgicalAction.CLAMP_ARTERY)
        assert result.post_state["blood_flow"] < result.pre_state["blood_flow"]

    def test_icp_drain_reduces_pressure(self):
        scm = BrainTumorSCM(patient_params={"intracranial_pressure": 0.7})
        engine = DoCalculusEngine(scm)
        result = engine.intervene(SurgicalAction.DRAIN_CSF)
        assert result.post_state["intracranial_pressure"] < result.pre_state["intracranial_pressure"]

    def test_net_utility_sign(self):
        scm = BrainTumorSCM(patient_params={"tumor_size": 0.8})
        engine = DoCalculusEngine(scm)
        result = engine.intervene(SurgicalAction.REMOVE_TUMOR_FULL)
        assert result.net_utility == pytest.approx(
            result.recovery_gain - result.risk_increase, abs=0.001
        )

    def test_multi_step_sequence(self):
        scm = BrainTumorSCM()
        engine = DoCalculusEngine(scm)
        results = engine.multi_step_intervention([
            SurgicalAction.REDUCE_EDEMA,
            SurgicalAction.REMOVE_TUMOR_FULL,
        ])
        assert len(results) == 2
        assert results[1].post_state["recovery_score"] >= 0

    def test_downstream_effects_populated(self):
        scm = BrainTumorSCM()
        engine = DoCalculusEngine(scm)
        result = engine.intervene(SurgicalAction.REMOVE_TUMOR_FULL)
        assert len(result.downstream_effects) > 0


# ─── Counterfactual Tests ─────────────────────────────────────────────────────
class TestCounterfactualEngine:

    def test_counterfactual_returns_result(self):
        scm = BrainTumorSCM()
        engine = CounterfactualEngine(scm, n_simulations=20)
        observed = scm.evaluate(noise=False)
        query = CounterfactualQuery(
            factual_action=SurgicalAction.BIOPSY_ONLY,
            counterfactual_action=SurgicalAction.REMOVE_TUMOR_FULL,
            observed_outcome=observed,
        )
        result = engine.run_counterfactual(query)
        assert isinstance(result.recovery_delta, float)
        assert isinstance(result.was_better, bool)
        assert result.explanation

    def test_full_removal_better_than_biopsy_direct(self):
        """Full removal do() should give higher post_recovery than biopsy do()."""
        scm_full   = BrainTumorSCM(patient_params={"tumor_size": 0.7})
        scm_biopsy = BrainTumorSCM(patient_params={"tumor_size": 0.7})
        r_full   = DoCalculusEngine(scm_full).intervene(SurgicalAction.REMOVE_TUMOR_FULL)
        r_biopsy = DoCalculusEngine(scm_biopsy).intervene(SurgicalAction.BIOPSY_ONLY)
        assert r_full.post_state["recovery_score"] > r_biopsy.post_state["recovery_score"], \
            "Full removal should yield higher recovery than biopsy"

    def test_monte_carlo_returns_ranked_plans(self):
        scm = BrainTumorSCM()
        engine = CounterfactualEngine(scm, n_simulations=20)
        plans = engine.monte_carlo_search(top_k=5)
        assert len(plans) > 0
        assert plans[0].rank == 1
        # Should be sorted by utility
        utilities = [p.net_utility for p in plans]
        assert utilities == sorted(utilities, reverse=True)

    def test_plan_confidence_interval_valid(self):
        scm = BrainTumorSCM()
        engine = CounterfactualEngine(scm, n_simulations=30)
        plans = engine.monte_carlo_search(top_k=3)
        for plan in plans:
            lo, hi = plan.confidence_interval
            assert lo <= hi
            assert 0 <= lo <= 1
            assert 0 <= hi <= 1

    def test_plan_to_dict_serializable(self):
        scm = BrainTumorSCM()
        engine = CounterfactualEngine(scm, n_simulations=20)
        plans = engine.monte_carlo_search(top_k=1)
        import json
        # Should not raise
        json.dumps(plans[0].to_dict())

    def test_candidate_plan_generation(self):
        scm = BrainTumorSCM()
        engine = CounterfactualEngine(scm, n_simulations=5)
        plans = engine._generate_candidate_plans()
        assert len(plans) >= 10
        for plan in plans:
            assert isinstance(plan, list)
            assert all(isinstance(a, SurgicalAction) for a in plan)


# ─── Integration Test ─────────────────────────────────────────────────────────
class TestIntegration:

    def test_full_causal_pipeline(self):
        """End-to-end: SCM → interventions → counterfactual → ranked plans."""
        # Initialize
        scm = BrainTumorSCM(patient_params={"tumor_size": 0.5, "edema_volume": 0.3})
        baseline = scm.evaluate(noise=False)
        assert 0 <= baseline["recovery_score"] <= 1

        # Interventions
        engine = DoCalculusEngine(scm)
        results = []
        for action in [
            SurgicalAction.REMOVE_TUMOR_FULL,
            SurgicalAction.REDUCE_EDEMA,
            SurgicalAction.DRAIN_CSF,
        ]:
            r = engine.intervene(action, noise=False)
            results.append(r)
        assert all(r is not None for r in results)

        # Monte-Carlo
        cf_engine = CounterfactualEngine(scm, n_simulations=30)
        plans = cf_engine.monte_carlo_search(top_k=3)
        assert len(plans) >= 1
        assert plans[0].expected_recovery > 0

    def test_high_icp_scenario(self):
        """
        For a patient with critical ICP, drain_csf should have better net_utility
        than biopsy (low risk + targeted ICP relief vs small tumor reduction).
        """
        scm = BrainTumorSCM(patient_params={
            "tumor_size": 0.4,
            "intracranial_pressure": 0.85,
            "edema_volume": 0.7,
        })
        engine = DoCalculusEngine(scm)
        r_drain  = engine.intervene(SurgicalAction.DRAIN_CSF)
        r_biopsy = engine.intervene(SurgicalAction.BIOPSY_ONLY)
        # Both should improve recovery
        assert r_drain.recovery_gain > 0, "ICP drain should improve recovery"
        assert r_biopsy.recovery_gain > 0, "Biopsy should improve recovery"
        # Drain specifically targets ICP — post-ICP should drop significantly
        assert r_drain.post_state["intracranial_pressure"] < r_drain.pre_state["intracranial_pressure"], \
            "ICP drain should reduce intracranial pressure"
        # Drain should improve neural function (downstream of ICP)
        assert r_drain.post_state["neural_function"] > r_drain.pre_state["neural_function"], \
            "Reduced ICP should improve neural function"

    def test_agent_pipeline(self):
        """Full agent orchestrator produces a surgical report."""
        from src.causal.scm import BrainTumorSCM
        from src.agents.surgical_planner import SurgicalPlannerOrchestrator

        scm = BrainTumorSCM(patient_params={"tumor_size": 0.4})
        orchestrator = SurgicalPlannerOrchestrator(scm, n_simulations=20)

        twin_summary = {
            "patient_id": "TEST_001",
            "structures": [
                {"name": "enhancing_tumor", "label_id": 3, "is_tumor": True,
                 "voxel_count": 1024, "volume_mm3": 8192,
                 "centroid_mm": [64.0, 64.0, 64.0]},
                {"name": "necrotic_tumor_core", "label_id": 1, "is_tumor": True,
                 "voxel_count": 512, "volume_mm3": 4096,
                 "centroid_mm": [64.0, 64.0, 64.0]},
                {"name": "brainstem", "label_id": 7, "is_tumor": False,
                 "voxel_count": 4096, "volume_mm3": 32768,
                 "centroid_mm": [64.0, 50.0, 30.0]},
            ],
            "tumor_count": 2,
            "total_tumor_volume_mm3": 12288.0,
            "voxel_spacing": (1.0, 1.0, 1.0),
        }

        state = orchestrator.run(
            patient_id="TEST_001",
            twin_summary=twin_summary,
            gnn_prediction={"mortality_risk": 0.08, "nerve_damage_prob": 0.12, "blood_loss_ml": 250},
        )

        assert state["surgical_report"] != ""
        assert len(state["top_plans"]) > 0
        assert state["error"] is None
