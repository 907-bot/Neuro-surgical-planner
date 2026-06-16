"""
src/causal/counterfactual.py
Counterfactual simulation engine — answers:
  "What *would* have happened if the surgeon had chosen path B instead of A?"

Implements Pearl's Three-Step Counterfactual Algorithm:
  1. Abduction  — infer exogenous noise values from observed state
  2. Action     — apply do(X=x) to mutilated SCM
  3. Prediction — propagate with fixed noise to get counterfactual outcome
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .scm import BrainTumorSCM
from .do_calculus import DoCalculusEngine, InterventionResult, SurgicalAction


# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class CounterfactualQuery:
    """
    A counterfactual question:
      "Given we observed Y' in factual world,
       what would Y be if we had done X=x instead?"
    """
    factual_action:         Optional[SurgicalAction]    # what actually happened
    counterfactual_action:  SurgicalAction              # hypothetical
    observed_outcome:       Dict[str, float]            # factual outcome
    question:               str = ""

    def __str__(self):
        factual = self.factual_action.value if self.factual_action else "no_surgery"
        return (f"CF Query: Given {factual} → observed outcome, "
                f"what if {self.counterfactual_action.value} instead?")


@dataclass
class CounterfactualResult:
    """Result of a counterfactual simulation."""
    query:              CounterfactualQuery
    factual_state:      Dict[str, float]
    counterfactual_state: Dict[str, float]
    recovery_delta:     float   # CF_recovery - factual_recovery
    was_better:         bool
    exogenous_noise:    Dict[str, float]
    explanation:        str


@dataclass
class SurgicalPathScore:
    """Scored surgical plan from Monte-Carlo search."""
    rank:               int
    actions:            List[SurgicalAction]
    expected_recovery:  float
    expected_risk:      float
    net_utility:        float
    blood_loss_ml:      float
    nerve_damage_prob:  float
    icu_days:           float
    simulation_count:   int
    confidence_interval: Tuple[float, float]
    step_results:       List[InterventionResult]
    description:        str = ""

    def to_dict(self) -> Dict:
        return {
            "rank":              self.rank,
            "actions":           [a.value for a in self.actions],
            "expected_recovery": round(self.expected_recovery, 4),
            "expected_risk":     round(self.expected_risk, 4),
            "net_utility":       round(self.net_utility, 4),
            "blood_loss_ml":     round(self.blood_loss_ml, 1),
            "nerve_damage_prob": round(self.nerve_damage_prob, 4),
            "icu_days":          round(self.icu_days, 1),
            "confidence_95":     [round(x, 4) for x in self.confidence_interval],
            "description":       self.description,
        }


# ─── Counterfactual Engine ────────────────────────────────────────────────────
class CounterfactualEngine:
    """
    Implements Pearl's Three-Step Counterfactual Algorithm
    and Monte-Carlo Surgical Path Search.
    """

    def __init__(self, scm: BrainTumorSCM, n_simulations: int = 500):
        self.scm = scm
        self.n_simulations = n_simulations

    # ── Pearl Three-Step Algorithm ───────────────────────────────────────────
    def run_counterfactual(self, query: CounterfactualQuery) -> CounterfactualResult:
        """
        Pearl's three-step:
          Step 1 — Abduction: find noise U consistent with observed world
          Step 2 — Action: apply do(X=x) (mutilate SCM)
          Step 3 — Prediction: propagate with fixed U
        """
        logger.info(f"Running: {query}")

        # Step 1: Abduction — back-calculate noise from observed outcome
        noise_values = self._abduct_noise(query.observed_outcome)

        # Step 2: Action — build counterfactual SCM with do() applied
        cf_scm = deepcopy(self.scm)
        cf_engine = DoCalculusEngine(cf_scm)
        cf_result = cf_engine.intervene(query.counterfactual_action, noise=False)

        # Step 3: Prediction — re-run with abducted noise
        post_state_with_noise = self._predict_with_noise(cf_scm, noise_values)

        factual_recovery = query.observed_outcome.get("recovery_score", 0.0)
        cf_recovery = post_state_with_noise.get("recovery_score", 0.0)
        delta = cf_recovery - factual_recovery

        # Build explanation
        key_changes = {
            k: post_state_with_noise[k] - query.observed_outcome.get(k, 0.0)
            for k in post_state_with_noise
            if abs(post_state_with_noise[k] - query.observed_outcome.get(k, 0.0)) > 0.01
        }
        explanation = self._build_explanation(query, delta, key_changes)

        return CounterfactualResult(
            query=query,
            factual_state=query.observed_outcome,
            counterfactual_state=post_state_with_noise,
            recovery_delta=delta,
            was_better=(delta > 0),
            exogenous_noise=noise_values,
            explanation=explanation,
        )

    def _abduct_noise(self, observed: Dict[str, float]) -> Dict[str, float]:
        """
        Step 1: Abduction — infer exogenous noise values.
        Simplified: compute residuals between SCM prediction and observed.
        """
        scm_copy = deepcopy(self.scm)
        # Set observed values
        for k, v in observed.items():
            if k in scm_copy.variables:
                scm_copy.set_variable(k, v)

        predicted = scm_copy.evaluate(noise=False)
        noise = {}
        for k in observed:
            if k in predicted:
                noise[k] = observed[k] - predicted.get(k, observed[k])

        return noise

    def _predict_with_noise(
        self,
        scm: BrainTumorSCM,
        noise: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Step 3: Prediction — evaluate with abducted noise applied.
        """
        # Inject abducted noise as variable offsets
        for k, n in noise.items():
            if k in scm.variables and not scm.variables[k].intervened:
                scm.variables[k].value = float(np.clip(
                    scm.variables[k].value + n * 0.5,  # damped injection
                    scm.variables[k].min_val,
                    scm.variables[k].max_val
                ))

        return scm.evaluate(noise=False)

    def _build_explanation(
        self,
        query: CounterfactualQuery,
        delta: float,
        key_changes: Dict[str, float],
    ) -> str:
        direction = "better" if delta > 0 else "worse"
        lines = [
            f"Counterfactual: {query.counterfactual_action.value}",
            f"Recovery would have been {direction} by {abs(delta):.1%}",
            "Key causal changes:",
        ]
        for var, change in sorted(key_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:4]:
            arrow = "↑" if change > 0 else "↓"
            lines.append(f"  {var}: {arrow} {abs(change):.3f}")
        return "\n".join(lines)

    # ── Monte-Carlo Surgical Path Search ────────────────────────────────────
    def monte_carlo_search(
        self,
        candidate_plans: Optional[List[List[SurgicalAction]]] = None,
        top_k: int = 5,
    ) -> List[SurgicalPathScore]:
        """
        Run Monte-Carlo simulations over candidate surgical plans.
        For each plan, simulate n_simulations times with stochastic noise.
        Rank by expected utility = E[recovery] - E[risk].

        Args:
            candidate_plans: list of action sequences to evaluate
                             (auto-generated if None)
            top_k: return top K plans

        Returns:
            Top K ranked SurgicalPathScore objects
        """
        if candidate_plans is None:
            candidate_plans = self._generate_candidate_plans()

        logger.info(f"Monte-Carlo search: {len(candidate_plans)} plans × "
                    f"{self.n_simulations} simulations")

        t0 = time.time()
        scores = []

        for plan in candidate_plans:
            score = self._evaluate_plan(plan)
            if score:
                scores.append(score)

        # Rank by net utility
        scores.sort(key=lambda s: s.net_utility, reverse=True)

        # Assign ranks
        for i, s in enumerate(scores[:top_k]):
            s.rank = i + 1

        elapsed = time.time() - t0
        logger.info(f"Search complete in {elapsed:.2f}s | "
                    f"Best: {scores[0].actions[0].value if scores else 'none'} "
                    f"(utility={scores[0].net_utility:.3f})" if scores else "No valid plans")

        return scores[:top_k]

    def _evaluate_plan(
        self,
        actions: List[SurgicalAction],
    ) -> Optional[SurgicalPathScore]:
        """Run n_simulations for one surgical plan and aggregate metrics."""
        recoveries, risks, blood_losses, nerve_probs, icu_days_list = [], [], [], [], []
        step_results_sample = None

        for sim in range(self.n_simulations):
            scm_sim = deepcopy(self.scm)
            engine = DoCalculusEngine(scm_sim)

            try:
                results = engine.multi_step_intervention(actions, noise=True)
            except Exception as e:
                logger.debug(f"Simulation failed: {e}")
                continue

            if not results:
                continue

            final = results[-1].post_state
            recoveries.append(final.get("recovery_score", 0.0))
            risks.append(sum(r.risk_increase for r in results))

            # Aggregate GNN-style estimates (approximated from SCM)
            blood_losses.append(final.get("blood_flow", 0.5) * 300 + 100)
            nerve_probs.append(1.0 - final.get("neural_function", 0.8))
            icu_days_list.append(max(1.0, (1.0 - final.get("recovery_score", 0.5)) * 10))

            if step_results_sample is None:
                step_results_sample = results

        if not recoveries:
            return None

        r_arr = np.array(recoveries)
        ci_low = float(np.percentile(r_arr, 2.5))
        ci_high = float(np.percentile(r_arr, 97.5))

        mean_recovery = float(np.mean(r_arr))
        mean_risk = float(np.mean(risks))
        net_utility = mean_recovery - mean_risk

        return SurgicalPathScore(
            rank=0,
            actions=actions,
            expected_recovery=mean_recovery,
            expected_risk=mean_risk,
            net_utility=net_utility,
            blood_loss_ml=float(np.mean(blood_losses)),
            nerve_damage_prob=float(np.mean(nerve_probs)),
            icu_days=float(np.mean(icu_days_list)),
            simulation_count=len(recoveries),
            confidence_interval=(ci_low, ci_high),
            step_results=step_results_sample or [],
            description=self._describe_plan(actions, mean_recovery, mean_risk),
        )

    def _generate_candidate_plans(self) -> List[List[SurgicalAction]]:
        """Generate a diverse set of candidate surgical plans."""
        return [
            # Single-action plans
            [SurgicalAction.REMOVE_TUMOR_FULL],
            [SurgicalAction.REMOVE_TUMOR_PARTIAL],
            [SurgicalAction.DEBULK_TUMOR],
            [SurgicalAction.BIOPSY_ONLY],
            [SurgicalAction.RADIOSURGERY],
            [SurgicalAction.ABLATE_TISSUE],

            # Two-step plans
            [SurgicalAction.REDUCE_EDEMA, SurgicalAction.REMOVE_TUMOR_FULL],
            [SurgicalAction.DRAIN_CSF, SurgicalAction.REMOVE_TUMOR_FULL],
            [SurgicalAction.DECOMPRESS_ICP, SurgicalAction.REMOVE_TUMOR_PARTIAL],
            [SurgicalAction.REDUCE_EDEMA, SurgicalAction.DEBULK_TUMOR],
            [SurgicalAction.DRAIN_CSF, SurgicalAction.DEBULK_TUMOR],
            [SurgicalAction.REDUCE_EDEMA, SurgicalAction.RADIOSURGERY],

            # Three-step plans
            [SurgicalAction.REDUCE_EDEMA, SurgicalAction.DRAIN_CSF, SurgicalAction.REMOVE_TUMOR_FULL],
            [SurgicalAction.REDUCE_EDEMA, SurgicalAction.DRAIN_CSF, SurgicalAction.DEBULK_TUMOR],
            [SurgicalAction.DECOMPRESS_ICP, SurgicalAction.REDUCE_EDEMA, SurgicalAction.REMOVE_TUMOR_PARTIAL],

            # Conservative + follow-up
            [SurgicalAction.BIOPSY_ONLY, SurgicalAction.RADIOSURGERY],
            [SurgicalAction.DRAIN_CSF, SurgicalAction.RADIOSURGERY],
        ]

    def _describe_plan(
        self,
        actions: List[SurgicalAction],
        recovery: float,
        risk: float,
    ) -> str:
        action_names = " → ".join(a.value.replace("_", " ").title() for a in actions)
        return (f"{action_names} | "
                f"Recovery: {recovery:.0%} | Risk: {risk:.0%}")
