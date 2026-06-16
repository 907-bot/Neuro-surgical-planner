"""
src/causal/do_calculus.py
Do-Calculus intervention engine — implements Pearl's do(X=x) operator.
Mutilates the SCM graph and propagates effects downstream.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from loguru import logger

from .scm import BrainTumorSCM


# ─── Surgical Action Space ────────────────────────────────────────────────────
class SurgicalAction(str, Enum):
    """Atomic surgical interventions mapped to do() operators."""
    REMOVE_TUMOR_PARTIAL  = "remove_tumor_partial"
    REMOVE_TUMOR_FULL     = "remove_tumor_full"
    DEBULK_TUMOR          = "debulk_tumor"
    CLAMP_ARTERY          = "clamp_artery"
    REROUTE_BLOOD_FLOW    = "reroute_blood_flow"
    REDUCE_EDEMA          = "reduce_edema"          # mannitol / steroids
    DRAIN_CSF             = "drain_csf"             # EVD placement
    ABLATE_TISSUE         = "ablate_tissue"
    BIOPSY_ONLY           = "biopsy_only"
    RADIOSURGERY          = "radiosurgery"          # Gamma Knife / CyberKnife
    DECOMPRESS_ICP        = "decompress_icp"        # craniectomy


@dataclass
class SurgicalIntervention:
    """
    A do() operator: force a variable to a specific value,
    cut incoming edges (graph mutilation), propagate effects.
    """
    action:           SurgicalAction
    target_variable:  str        # SCM variable to intervene on
    target_value:     float      # the value set by do()
    affected_structures: List[str] = field(default_factory=list)
    description:      str        = ""
    risk_factor:      float      = 0.0   # inherent risk of the intervention itself

    def __str__(self):
        return f"do({self.target_variable} = {self.target_value:.2f}) [{self.action.value}]"


# Maps SurgicalAction → SCM intervention parameters
ACTION_REGISTRY: Dict[SurgicalAction, Dict] = {
    SurgicalAction.REMOVE_TUMOR_FULL: {
        "target_variable": "tumor_size",
        "target_value": 0.0,
        "description": "Complete tumor resection — set tumor_size = 0",
        "risk_factor": 0.35,
        "affected_structures": ["enhancing_tumor", "necrotic_tumor_core", "peritumoral_edema"],
    },
    SurgicalAction.REMOVE_TUMOR_PARTIAL: {
        "target_variable": "tumor_size",
        "target_value": 0.15,
        "description": "Partial resection — reduce tumor_size by ~50%",
        "risk_factor": 0.20,
        "affected_structures": ["enhancing_tumor"],
    },
    SurgicalAction.DEBULK_TUMOR: {
        "target_variable": "tumor_size",
        "target_value": 0.10,
        "description": "Aggressive debulking — near-total resection",
        "risk_factor": 0.28,
        "affected_structures": ["enhancing_tumor", "necrotic_tumor_core"],
    },
    SurgicalAction.CLAMP_ARTERY: {
        "target_variable": "blood_flow",
        "target_value": 0.1,
        "description": "Temporarily clamp feeding artery",
        "risk_factor": 0.45,
        "affected_structures": ["middle_cerebral_artery"],
    },
    SurgicalAction.REROUTE_BLOOD_FLOW: {
        "target_variable": "vascular_compression",
        "target_value": 0.1,
        "description": "Reroute blood flow around compressed vessel",
        "risk_factor": 0.30,
        "affected_structures": ["middle_cerebral_artery", "anterior_cerebral_artery"],
    },
    SurgicalAction.REDUCE_EDEMA: {
        "target_variable": "edema_volume",
        "target_value": 0.05,
        "description": "Medical edema reduction (mannitol/dexamethasone)",
        "risk_factor": 0.05,
        "affected_structures": ["peritumoral_edema"],
    },
    SurgicalAction.DRAIN_CSF: {
        "target_variable": "intracranial_pressure",
        "target_value": 0.1,
        "description": "EVD placement — normalize ICP",
        "risk_factor": 0.10,
        "affected_structures": ["ventricles"],
    },
    SurgicalAction.ABLATE_TISSUE: {
        "target_variable": "tumor_size",
        "target_value": 0.2,
        "description": "Laser / thermal ablation of tumor core",
        "risk_factor": 0.15,
        "affected_structures": ["enhancing_tumor"],
    },
    SurgicalAction.BIOPSY_ONLY: {
        "target_variable": "tumor_size",
        "target_value": 0.29,  # negligible change
        "description": "Stereotactic biopsy — minimal tissue removal",
        "risk_factor": 0.08,
        "affected_structures": ["enhancing_tumor"],
    },
    SurgicalAction.RADIOSURGERY: {
        "target_variable": "inflammatory_response",
        "target_value": 0.6,  # transient inflammation increase
        "description": "Stereotactic radiosurgery — no open surgery",
        "risk_factor": 0.12,
        "affected_structures": ["enhancing_tumor"],
    },
    SurgicalAction.DECOMPRESS_ICP: {
        "target_variable": "intracranial_pressure",
        "target_value": 0.05,
        "description": "Decompressive craniectomy",
        "risk_factor": 0.25,
        "affected_structures": [],
    },
}


# ─── Intervention Engine ──────────────────────────────────────────────────────
@dataclass
class InterventionResult:
    """Result of running do(X=x) through the SCM."""
    action:           SurgicalAction
    intervention:     SurgicalIntervention
    pre_state:        Dict[str, float]
    post_state:       Dict[str, float]
    delta:            Dict[str, float]    # post - pre
    recovery_gain:    float               # Δ recovery_score
    risk_increase:    float               # inherent risk of action
    net_utility:      float               # recovery_gain - risk_increase
    downstream_effects: Dict[str, float]  # which vars changed and by how much

    def summary(self) -> str:
        lines = [
            f"Action: {self.action.value}",
            f"Intervention: {self.intervention}",
            f"Recovery gain: {self.recovery_gain:+.3f}",
            f"Risk increase: {self.risk_increase:.3f}",
            f"Net utility: {self.net_utility:+.3f}",
            "Key effects:",
        ]
        for var, delta in sorted(self.downstream_effects.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            arrow = "↑" if delta > 0 else "↓"
            lines.append(f"  {var}: {arrow} {abs(delta):.3f}")
        return "\n".join(lines)


class DoCalculusEngine:
    """
    Implements Pearl's do-calculus intervention operator.

    do(X=x):
        1. Copy the SCM
        2. Mutilate: remove all incoming edges to X in the DAG
        3. Set X = x (hard intervention)
        4. Propagate: re-evaluate all downstream variables
        5. Compare with pre-intervention state
    """

    def __init__(self, scm: BrainTumorSCM):
        self.scm = scm

    def intervene(
        self,
        action: SurgicalAction,
        noise: bool = False,
    ) -> InterventionResult:
        """
        Apply a surgical action as a do() intervention.

        Args:
            action: which surgical action to simulate
            noise:  whether to add stochastic noise

        Returns:
            InterventionResult with pre/post states and deltas
        """
        if action not in ACTION_REGISTRY:
            raise ValueError(f"Unknown action: {action}")

        params = ACTION_REGISTRY[action]
        intervention = SurgicalIntervention(action=action, **params)

        logger.info(f"Applying: {intervention}")

        # Step 1: Record pre-intervention state
        pre_state = deepcopy(self.scm.evaluate(noise=False))

        # Step 2: Create SCM copy for mutilation
        scm_copy = deepcopy(self.scm)

        # Step 3: Graph mutilation — cut all incoming edges to target
        target = intervention.target_variable
        in_edges = list(scm_copy.dag.in_edges(target))
        scm_copy.dag.remove_edges_from(in_edges)
        logger.debug(f"Graph mutilation: removed {len(in_edges)} incoming edges to '{target}'")

        # Step 4: Hard set target variable (the do() itself)
        scm_copy.set_variable(target, intervention.target_value)
        scm_copy.variables[target].intervened = True

        # Step 5: Propagate through the mutilated DAG
        post_state = scm_copy.evaluate(noise=noise)

        # Step 6: Compute deltas and utilities
        delta = {k: post_state[k] - pre_state[k] for k in pre_state}

        downstream = {
            k: delta[k] for k in delta
            if abs(delta[k]) > 0.001 and k != target
        }

        recovery_gain = post_state.get("recovery_score", 0) - pre_state.get("recovery_score", 0)
        risk_increase = intervention.risk_factor

        net_utility = recovery_gain - risk_increase

        return InterventionResult(
            action=action,
            intervention=intervention,
            pre_state=pre_state,
            post_state=post_state,
            delta=delta,
            recovery_gain=recovery_gain,
            risk_increase=risk_increase,
            net_utility=net_utility,
            downstream_effects=downstream,
        )

    def multi_step_intervention(
        self,
        actions: List[SurgicalAction],
        noise: bool = False,
    ) -> List[InterventionResult]:
        """
        Simulate a sequence of surgical actions.
        Each action modifies the SCM state before the next.
        """
        results = []
        scm_working = deepcopy(self.scm)
        engine = DoCalculusEngine(scm_working)

        for action in actions:
            result = engine.intervene(action, noise=noise)
            results.append(result)
            # Apply the change to working SCM for next step
            engine.scm.set_variable(
                result.intervention.target_variable,
                result.post_state[result.intervention.target_variable]
            )

        return results

    def available_actions(self) -> List[SurgicalAction]:
        return list(SurgicalAction)

    def action_preview(self, action: SurgicalAction) -> Dict:
        """Return action metadata without running the full SCM."""
        if action not in ACTION_REGISTRY:
            return {}
        return {**ACTION_REGISTRY[action], "action": action.value}
