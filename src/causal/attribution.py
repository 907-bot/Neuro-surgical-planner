"""
src/causal/attribution.py
Causal Attribution Engine — traces WHICH variables changed and BY HOW MUCH
when a surgical intervention is applied via do().

Answers: "Why is Plan A better than Plan B?"
→ "Resecting the tumor reduced ICP by 34% → neural function improved by 22%
   → recovery score improved by 18%"
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .scm import BrainTumorSCM
from .do_calculus import DoCalculusEngine, SurgicalAction, ACTION_REGISTRY


@dataclass
class VariableDelta:
    """Change in a single SCM variable after an intervention."""
    variable:    str
    before:      float
    after:       float
    delta:       float        # after - before
    delta_pct:   float        # percentage change
    direction:   str          # "improved" | "worsened" | "unchanged"
    clinical_label: str = ""  # human-readable name


@dataclass
class AttributionChain:
    """Full causal attribution for a single intervention or plan."""
    action_sequence: List[str]
    baseline_recovery: float
    final_recovery:    float
    recovery_delta:    float
    variable_deltas:   List[VariableDelta]   # sorted by |delta| descending
    explanation:       str                   # human-readable summary
    plan_rank:         Optional[int] = None


# ─── Clinical labels for SCM variables ──────────────────────────────────────
CLINICAL_LABELS = {
    "tumor_size":            "Tumor Size",
    "blood_flow":            "Cerebral Blood Flow",
    "oxygen_saturation":     "O₂ Saturation",
    "intracranial_pressure": "Intracranial Pressure (ICP)",
    "metabolic_rate":        "Metabolic Rate",
    "edema_volume":          "Peritumoral Edema",
    "vascular_compression":  "Vascular Compression",
    "neural_function":       "Neural Function",
    "mass_effect":           "Mass Effect",
    "inflammatory_response": "Inflammatory Response",
    "recovery_score":        "Recovery Score",
    "surgical_risk":         "Surgical Risk",
}

# Variables where LOWER is better
LOWER_IS_BETTER = {
    "tumor_size", "intracranial_pressure", "edema_volume",
    "vascular_compression", "mass_effect", "inflammatory_response",
    "surgical_risk",
}


class CausalAttributor:
    """
    Traces causal pathways through the SCM after an intervention.

    Usage:
        attributor = CausalAttributor(scm)
        chain = attributor.explain_action(SurgicalAction.TUMOR_RESECTION)
        print(chain.explanation)

        # Compare two plans
        diff = attributor.compare_plans(plan_a_actions, plan_b_actions)
    """

    def __init__(self, scm: BrainTumorSCM):
        self.scm = deepcopy(scm)  # never mutate the original
        self.baseline = self.scm.evaluate(noise=False)

    def _apply_action_sequence(
        self, actions: List[SurgicalAction], n_simulations: int = 50
    ) -> Dict[str, float]:
        """Apply a sequence of do() operators and return the final state."""
        scm_copy = deepcopy(self.scm)
        engine = DoCalculusEngine(scm_copy)

        current_state = deepcopy(self.baseline)

        for action in actions:
            try:
                result = engine.intervene(action, noise=False)
                current_state = result.post_state
                # Update SCM variables to reflect this state
                for var_name, val in current_state.items():
                    if var_name in scm_copy.variables:
                        scm_copy.variables[var_name].value = val
            except Exception as e:
                logger.warning(f"Attribution: action {action.value} failed — {e}")

        return current_state

    def explain_action(
        self, action: SurgicalAction, threshold: float = 0.005
    ) -> AttributionChain:
        """
        Explain the causal effect of a single surgical action.

        Args:
            action: the SurgicalAction to explain
            threshold: minimum |delta| to include in explanation
        """
        post_state = self._apply_action_sequence([action])
        return self._build_chain(
            action_sequence=[action.value],
            post_state=post_state,
            threshold=threshold,
        )

    def explain_plan(
        self,
        action_sequence: List[SurgicalAction],
        threshold: float = 0.005,
        plan_rank: Optional[int] = None,
    ) -> AttributionChain:
        """
        Explain the causal effect of a multi-step surgical plan.

        Args:
            action_sequence: ordered list of SurgicalAction
            threshold: minimum |delta| to include
            plan_rank: rank of this plan (for display)
        """
        post_state = self._apply_action_sequence(action_sequence)
        chain = self._build_chain(
            action_sequence=[a.value for a in action_sequence],
            post_state=post_state,
            threshold=threshold,
        )
        chain.plan_rank = plan_rank
        return chain

    def _build_chain(
        self,
        action_sequence: List[str],
        post_state: Dict[str, float],
        threshold: float = 0.005,
    ) -> AttributionChain:
        """Build an AttributionChain from baseline → post-intervention states."""
        deltas = []

        for var, post_val in post_state.items():
            pre_val = self.baseline.get(var, 0.0)
            delta = post_val - pre_val

            if abs(delta) < threshold:
                continue

            # Direction: is the change clinically good or bad?
            if var in LOWER_IS_BETTER:
                direction = "improved" if delta < 0 else "worsened"
            else:
                direction = "improved" if delta > 0 else "worsened"

            delta_pct = (delta / max(abs(pre_val), 1e-6)) * 100

            deltas.append(VariableDelta(
                variable=var,
                before=round(pre_val, 4),
                after=round(post_val, 4),
                delta=round(delta, 4),
                delta_pct=round(delta_pct, 1),
                direction=direction,
                clinical_label=CLINICAL_LABELS.get(var, var.replace("_", " ").title()),
            ))

        # Sort by absolute magnitude of delta
        deltas.sort(key=lambda d: abs(d.delta), reverse=True)

        baseline_recovery = self.baseline.get("recovery_score", 0.0)
        final_recovery = post_state.get("recovery_score", baseline_recovery)
        recovery_delta = final_recovery - baseline_recovery

        explanation = self._format_explanation(action_sequence, deltas, recovery_delta)

        return AttributionChain(
            action_sequence=action_sequence,
            baseline_recovery=round(baseline_recovery, 4),
            final_recovery=round(final_recovery, 4),
            recovery_delta=round(recovery_delta, 4),
            variable_deltas=deltas,
            explanation=explanation,
        )

    def _format_explanation(
        self,
        actions: List[str],
        deltas: List[VariableDelta],
        recovery_delta: float,
    ) -> str:
        """
        Format a causal chain explanation as a human-readable sentence.
        Example: "Tumor resection → ICP ↓34% → Neural function ↑22% → Recovery ↑18%"
        """
        if not deltas:
            return f"{' + '.join(actions)} → No significant physiological changes detected."

        # Pick the 3 most impactful variables (excluding recovery itself)
        top_vars = [d for d in deltas[:4] if d.variable != "recovery_score"][:3]

        parts = [" + ".join(a.replace("_", " ").title() for a in actions)]

        for d in top_vars:
            arrow = "↓" if d.delta < 0 else "↑"
            icon = "✓" if d.direction == "improved" else "⚠"
            parts.append(f"{icon} {d.clinical_label} {arrow}{abs(d.delta_pct):.0f}%")

        # Final recovery
        rec_arrow = "↑" if recovery_delta > 0 else "↓"
        rec_pct = abs(recovery_delta) * 100
        parts.append(f"{'✓' if recovery_delta > 0 else '⚠'} Recovery {rec_arrow}{rec_pct:.0f}%")

        return " → ".join(parts)

    def compare_plans(
        self,
        plan_a: List[SurgicalAction],
        plan_b: List[SurgicalAction],
    ) -> Dict:
        """
        Compare two plans side-by-side, returning the difference in each variable.
        Useful for "Plan A vs Plan B" waterfall charts.
        """
        chain_a = self.explain_plan(plan_a, plan_rank=1)
        chain_b = self.explain_plan(plan_b, plan_rank=2)

        all_vars = set(
            [d.variable for d in chain_a.variable_deltas] +
            [d.variable for d in chain_b.variable_deltas]
        )

        comparison = []
        delta_a = {d.variable: d for d in chain_a.variable_deltas}
        delta_b = {d.variable: d for d in chain_b.variable_deltas}

        for var in all_vars:
            da = delta_a.get(var)
            db = delta_b.get(var)
            comparison.append({
                "variable": var,
                "label": CLINICAL_LABELS.get(var, var),
                "plan_a_delta": da.delta if da else 0.0,
                "plan_b_delta": db.delta if db else 0.0,
                "advantage": "A" if (da and db and abs(da.delta) > abs(db.delta)) else "B",
            })

        return {
            "plan_a": {
                "actions": chain_a.action_sequence,
                "recovery_delta": chain_a.recovery_delta,
                "explanation": chain_a.explanation,
            },
            "plan_b": {
                "actions": chain_b.action_sequence,
                "recovery_delta": chain_b.recovery_delta,
                "explanation": chain_b.explanation,
            },
            "variable_comparison": sorted(
                comparison, key=lambda x: abs(x["plan_a_delta"] - x["plan_b_delta"]),
                reverse=True
            ),
            "winner": "A" if chain_a.recovery_delta > chain_b.recovery_delta else "B",
        }

    def to_dict(self, chain: AttributionChain) -> Dict:
        """Serialize an AttributionChain for JSON / API response."""
        return {
            "action_sequence": chain.action_sequence,
            "baseline_recovery": chain.baseline_recovery,
            "final_recovery": chain.final_recovery,
            "recovery_delta": chain.recovery_delta,
            "recovery_delta_pct": round(chain.recovery_delta * 100, 1),
            "explanation": chain.explanation,
            "plan_rank": chain.plan_rank,
            "variable_deltas": [
                {
                    "variable": d.variable,
                    "label": d.clinical_label,
                    "before": d.before,
                    "after": d.after,
                    "delta": d.delta,
                    "delta_pct": d.delta_pct,
                    "direction": d.direction,
                }
                for d in chain.variable_deltas
            ],
        }
