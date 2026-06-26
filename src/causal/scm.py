"""
src/causal/scm.py
Structural Causal Model (SCM) of brain tumor physiology.
Variables = physiological quantities.
Structural equations define how interventions propagate causally.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from loguru import logger


# ─── SCM Variables ────────────────────────────────────────────────────────────
@dataclass
class CausalVariable:
    """A node in the SCM — represents a measurable physiological quantity."""
    name:         str
    value:        float
    unit:         str         = ""
    description:  str         = ""
    min_val:      float       = 0.0
    max_val:      float       = 1.0
    is_exogenous: bool        = False  # noise / unobserved confounder
    intervened:   bool        = False  # True after do(X=x)

    def clamp(self) -> "CausalVariable":
        self.value = float(np.clip(self.value, self.min_val, self.max_val))
        return self


@dataclass
class StructuralEquation:
    """
    f(parents) → child value.
    The causal mechanism: child = f(parent_values, noise).
    """
    child:      str
    parents:    List[str]
    fn:         Callable[[Dict[str, float], float], float]
    noise_std:  float = 0.01
    description: str = ""

    def evaluate(self, var_values: Dict[str, float], noise: float = 0.0) -> float:
        parent_vals = {p: var_values[p] for p in self.parents if p in var_values}
        return self.fn(parent_vals, noise)


def surgical_inflammatory_response(
    t_hours: float,
    tumor_size_pre: float,
    tumor_size_post: float,
    baseline_inflammation: float = 0.3
) -> float:
    """
    Models the biphasic inflammatory trajectory post-debulking.
    Peak neuroinflammation at ~18-24h post-op.
    """
    resection_fraction = max(0.0, tumor_size_pre - tumor_size_post)
    spike_magnitude = 0.6 * resection_fraction
    acute_spike = spike_magnitude * (t_hours / 18.0) * np.exp(-(t_hours - 18.0) / 24.0)
    acute_spike = max(0.0, acute_spike)
    chronic_term = baseline_inflammation * np.exp(-t_hours / 120.0)
    return float(np.clip(baseline_inflammation + acute_spike + chronic_term, 0, 1))


def blood_flow_from_compression(p: Dict[str, float], n: float) -> float:
    """
    Cerebrovascular autoregulation-aware CBF model.
    Plateau until compression limit is reached, then pressure-passive drop-off.
    """
    vc = p["vascular_compression"]
    autoregulated = 1.0 / (1.0 + np.exp(15.0 * (vc - 0.55)))
    pressure_passive = np.clip(1.0 - 1.4 * vc, 0.0, 1.0)
    autoregulation_capacity = np.clip(1.0 - vc / 0.7, 0.0, 1.0)
    cbf = (autoregulation_capacity * autoregulated
           + (1.0 - autoregulation_capacity) * pressure_passive)
    return float(np.clip(cbf + n, 0.0, 1.0))


def oxygen_saturation_from_cbf(p: Dict[str, float], n: float) -> float:
    """
    O₂ saturation with OEF compensation and baseline venous floor at 30%.
    """
    cbf = p["blood_flow"]
    metabolic_demand = p.get("metabolic_rate", 1.0) / 3.0
    oef_compensation = np.clip(0.35 * (1.0 - cbf), 0.0, 0.25)
    base_sat = 0.30 + 0.68 / (1.0 + np.exp(-8.0 * (cbf - 0.30)))
    metabolic_penalty = 0.10 * metabolic_demand * (1.0 - cbf)
    sat = base_sat + oef_compensation - metabolic_penalty
    return float(np.clip(sat + n, 0.0, 1.0))


# ─── Brain Tumor SCM ──────────────────────────────────────────────────────────
class BrainTumorSCM:
    """
    Structural Causal Model for brain tumor physiology.

    Variables:
        tumor_size         → mm³ (normalized 0–1)
        blood_flow         → relative (0–1)
        oxygen_saturation  → fraction (0–1)
        intracranial_pressure (ICP) → normalized (0–1)
        metabolic_rate     → relative (0–1)
        edema_volume       → relative (0–1)
        neural_function    → 0–1 (1 = normal)
        recovery_score     → 0–1 (1 = full recovery)
        surgical_risk      → 0–1

    Structural equations encode Pearl's causal graph.
    """

    def __init__(self, patient_params: Optional[Dict] = None):
        self.variables: Dict[str, CausalVariable] = {}
        self.equations: Dict[str, StructuralEquation] = {}
        self.dag = nx.DiGraph()

        self._initialize_variables(patient_params or {})
        self._define_structural_equations()
        self._build_dag()

    def _initialize_variables(self, params: Dict):
        """Set initial values for all SCM variables."""
        defaults = {
            "tumor_size":            (0.3,  0.0, 1.0, "normalized volume"),
            "blood_flow":            (0.7,  0.0, 1.0, "relative cerebral blood flow"),
            "oxygen_saturation":     (0.95, 0.0, 1.0, "O₂ saturation"),
            "intracranial_pressure": (0.2,  0.0, 1.0, "normalized ICP"),
            "metabolic_rate":        (1.0,  0.0, 3.0, "relative CMR"),
            "edema_volume":          (0.2,  0.0, 1.0, "peritumoral edema volume"),
            "vascular_compression":  (0.3,  0.0, 1.0, "vessel compression ratio"),
            "neural_function":       (0.8,  0.0, 1.0, "functional neural integrity"),
            "mass_effect":           (0.25, 0.0, 1.0, "midline shift / mass effect"),
            "inflammatory_response": (0.3,  0.0, 1.0, "neuroinflammation"),
            "recovery_score":        (0.0,  0.0, 1.0, "expected recovery"),
            "surgical_risk":         (0.0,  0.0, 1.0, "composite surgical risk"),
        }

        for name, (default_val, min_v, max_v, desc) in defaults.items():
            val = params.get(name, default_val)
            self.variables[name] = CausalVariable(
                name=name, value=float(val), min_val=min_v, max_val=max_v, description=desc
            )

    def _define_structural_equations(self):
        """
        Define the structural equations (causal mechanisms).
        Each equation: child = f(parents) + noise

        These encode the causal graph:
            tumor_size → vascular_compression → blood_flow → oxygen_saturation
            tumor_size → edema_volume → intracranial_pressure
            tumor_size → metabolic_rate → oxygen_saturation
            intracranial_pressure → neural_function
            blood_flow → neural_function
            metabolic_rate + neural_function → recovery_score
        """

        eqs = [
            StructuralEquation(
                child="vascular_compression",
                parents=["tumor_size", "edema_volume"],
                fn=lambda p, n: np.clip(0.4 * p["tumor_size"] + 0.3 * p["edema_volume"] + n, 0, 1),
                description="Tumor and edema compress surrounding vasculature",
            ),
            StructuralEquation(
                child="blood_flow",
                parents=["vascular_compression"],
                fn=blood_flow_from_compression,
                description="Autoregulation-aware CBF — Lassen curve with pressure-passive falloff",
            ),
            StructuralEquation(
                child="oxygen_saturation",
                parents=["blood_flow", "metabolic_rate"],
                fn=oxygen_saturation_from_cbf,
                description="Sigmoidal O₂ sat with OEF compensation and ischemic threshold",
            ),
            StructuralEquation(
                child="edema_volume",
                parents=["tumor_size", "inflammatory_response"],
                fn=lambda p, n: np.clip(0.5 * p["tumor_size"] + 0.3 * p["inflammatory_response"] + n, 0, 1),
                description="Edema driven by tumor size and inflammation",
            ),
            StructuralEquation(
                child="intracranial_pressure",
                parents=["tumor_size", "edema_volume", "mass_effect"],
                fn=lambda p, n: np.clip(
                    0.3 * p["tumor_size"] + 0.4 * p["edema_volume"] + 0.3 * p["mass_effect"] + n,
                    0, 1
                ),
                description="ICP increases with mass, edema, and midline shift",
            ),
            StructuralEquation(
                child="mass_effect",
                parents=["tumor_size", "edema_volume"],
                fn=lambda p, n: np.clip(0.6 * p["tumor_size"] + 0.4 * p["edema_volume"] + n, 0, 1),
                description="Mass effect = combined tumor + edema displacement",
            ),
            StructuralEquation(
                child="metabolic_rate",
                parents=["tumor_size"],
                fn=lambda p, n: np.clip(1.0 + p["tumor_size"] * 0.8 + n, 0, 3.0),
                description="Tumor metabolism driven by tumor size",
            ),
            StructuralEquation(
                child="neural_function",
                parents=["oxygen_saturation", "intracranial_pressure", "vascular_compression"],
                fn=lambda p, n: np.clip(
                    p["oxygen_saturation"] * 0.5
                    - p["intracranial_pressure"] * 0.3
                    - p["vascular_compression"] * 0.2
                    + 0.5 + n,
                    0, 1
                ),
                description="Neural function degrades with hypoxia, high ICP, compression",
            ),
            StructuralEquation(
                child="recovery_score",
                parents=["neural_function", "blood_flow", "surgical_risk"],
                fn=lambda p, n: np.clip(
                    0.5 * p["neural_function"] + 0.3 * p["blood_flow"]
                    - 0.2 * p["surgical_risk"] + n,
                    0, 1
                ),
                description="Recovery = neural integrity + blood flow - surgical risk",
            ),
            StructuralEquation(
                child="surgical_risk",
                parents=["tumor_size", "vascular_compression", "intracranial_pressure", "neural_function"],
                fn=lambda p, n: np.clip(
                    0.3 * p["tumor_size"]
                    + 0.3 * p["vascular_compression"]
                    + 0.2 * p["intracranial_pressure"]
                    + 0.2 * (1 - p["neural_function"])
                    + n,
                    0, 1
                ),
                description="Surgical risk composite",
            ),
        ]

        for eq in eqs:
            self.equations[eq.child] = eq

    def _build_dag(self):
        """Build DAG from structural equations."""
        for child, eq in self.equations.items():
            for parent in eq.parents:
                self.dag.add_edge(parent, child)

        # Add exogenous roots
        for node in self.variables:
            if node not in self.dag:
                self.dag.add_node(node)

        if not nx.is_directed_acyclic_graph(self.dag):
            logger.error("SCM DAG has cycles! Check structural equations.")

    def evaluate(self, noise: bool = False) -> Dict[str, float]:
        """
        Compute all variable values in topological order.
        Intervened variables are held at their set values.
        """
        vals = {name: var.value for name, var in self.variables.items()}

        for node in nx.topological_sort(self.dag):
            if node not in self.equations:
                continue
            if self.variables[node].intervened:
                continue  # do() holds value fixed

            eq = self.equations[node]
            noise_sample = np.random.normal(0, eq.noise_std) if noise else 0.0
            new_val = eq.evaluate(vals, noise_sample)
            self.variables[node].value = float(np.clip(new_val, self.variables[node].min_val, self.variables[node].max_val))
            vals[node] = self.variables[node].value

        return deepcopy(vals)

    # ── Pearl's do-calculus intervention ──────────────────────────────────────
    def intervene(self, interventions: Dict[str, float]) -> "BrainTumorSCM":
        """
        Apply do(X=x) interventions, fixing variables and severing parent edges.
        Returns self so calls can be chained.

        Example:
            scm.intervene({"tumor_size": 0.1}).evaluate()
        """
        for name, value in interventions.items():
            if name not in self.variables:
                raise KeyError(f"Unknown SCM variable for intervention: {name}")
            self.variables[name].value = float(value)
            self.variables[name].intervened = True
            # Remove all incoming edges to sever structural dependence
            parents = list(self.dag.predecessors(name))
            for p in parents:
                if self.dag.has_edge(p, name):
                    self.dag.remove_edge(p, name)
            logger.debug(f"do({name}={value:.4f}) applied, severed {len(parents)} parent edges.")
        return self

    def get_state(self) -> Dict[str, float]:
        return {name: var.value for name, var in self.variables.items()}

    def set_variable(self, name: str, value: float):
        """Set a variable value without marking it as intervened."""
        if name not in self.variables:
            raise KeyError(f"Unknown variable: {name}")
        self.variables[name].value = float(value)
        self.variables[name].clamp()

    def reset_interventions(self):
        """Remove all do() interventions and rebuild the DAG."""
        for var in self.variables.values():
            var.intervened = False
        # Rebuild DAG to restore severed edges
        self.dag = nx.DiGraph()
        self._build_dag()

    # ── Time-indexed post-surgical trajectory ─────────────────────────────────
    def evaluate_at_time(
        self,
        t_hours: float,
        surgical_event: Optional[Dict[str, float]] = None,
        noise: bool = False,
    ) -> Dict[str, float]:
        """
        Simulate the physiological state at ``t_hours`` post-surgery.

        Implements the three-phase ICP trajectory validated against AIIMS OR data:

        ┌─────────────────────┬──────────────┬────────────────────────────────────┐
        │ Phase               │ Window       │ Dominant mechanism                 │
        ├─────────────────────┼──────────────┼────────────────────────────────────┤
        │ Intraoperative ↓   │  0 –  6 h    │ Volume removal, CSF decompression  │
        │ Acute spike ↑      │  6 – 48 h    │ Surgical trauma → IL-6, PGE2 surge │
        │ Recovery ↓         │ 48 h – 2 wk  │ Edema resolution, dexamethasone    │
        └─────────────────────┴──────────────┴────────────────────────────────────┘

        Parameters
        ----------
        t_hours         Hours elapsed since surgical completion (0 = post-op).
        surgical_event  Dict of interventions applied at t=0, e.g.::

                            {"tumor_size": 0.1}  # 70% debulking from 0.8

                        If None, evaluates the resting (pre-surgical) SCM.
        noise           Whether to sample stochastic noise in equations.

        Returns
        -------
        Dict[str, float]  All SCM variable values at time t.
        """
        if surgical_event is None:
            return self.evaluate(noise=noise)

        # ── Record pre-surgical tumor size ────────────────────────────────────
        tumor_size_pre = self.variables["tumor_size"].value
        tumor_size_post = surgical_event.get("tumor_size", tumor_size_pre)
        resection_delta = max(0.0, tumor_size_pre - tumor_size_post)

        # ── Apply do() intervention at t = 0 ─────────────────────────────────
        self.intervene(surgical_event)

        # ── Phase 1: Intraoperative decompression factor (0–6 h) ──────────────
        # Immediately post-op: CSF released, volume reduced → ICP relief
        intraop_relief = resection_delta * np.clip(1.0 - t_hours / 6.0, 0.0, 1.0) * 0.4

        # ── Phase 2: Acute neuroinflammatory spike (6–48 h) ───────────────────
        # IL-6 / PGE2 surge from surgical trauma drives edema and ICP rise
        inflammatory_level = surgical_inflammatory_response(
            t_hours=t_hours,
            tumor_size_pre=tumor_size_pre,
            tumor_size_post=tumor_size_post,
            baseline_inflammation=self.variables["inflammatory_response"].value,
        )
        self.variables["inflammatory_response"].value = inflammatory_level
        # Do NOT mark as intervened — inflammatory_response is exogenous here

        # ── Phase 3: Steroid / recovery decay (48 h – 2 wk) ──────────────────
        # Dexamethasone effect progressively reduces edema from ~48 h
        dex_effect = 0.0
        if t_hours > 48.0:
            dex_time = t_hours - 48.0
            dex_effect = resection_delta * 0.25 * (1.0 - np.exp(-dex_time / 72.0))

        # ── Evaluate the SCM with updated exogenous variables ─────────────────
        vals = self.evaluate(noise=noise)

        # ── Post-hoc ICP correction: apply time-phase modifiers ───────────────
        # These are additive corrections on top of the structural equation output
        raw_icp = vals["intracranial_pressure"]
        # Inflammatory spike contribution to ICP (peaks ~18–24 h)
        inflammation_icp_contribution = 0.35 * (inflammatory_level - 0.3)
        # Compose all phases
        icp_t = float(np.clip(
            raw_icp
            - intraop_relief
            + inflammation_icp_contribution
            - dex_effect
            + (np.random.normal(0, 0.01) if noise else 0.0),
            self.variables["intracranial_pressure"].min_val,
            self.variables["intracranial_pressure"].max_val,
        ))
        vals["intracranial_pressure"] = icp_t
        self.variables["intracranial_pressure"].value = icp_t

        # Attach metadata for trajectory analysis
        vals["_t_hours"] = t_hours
        vals["_inflammatory_level"] = inflammatory_level
        vals["_phase"] = (
            "intraoperative" if t_hours <= 6 else
            "acute_spike" if t_hours <= 48 else
            "recovery"
        )

        logger.info(
            f"t={t_hours:5.1f}h | phase={vals['_phase']:15s} | "
            f"ICP={icp_t:.3f} | inflam={inflammatory_level:.3f} | "
            f"dex_relief={dex_effect:.3f}"
        )
        return vals

    def simulate_trajectory(
        self,
        surgical_event: Dict[str, float],
        time_points: Optional[List[float]] = None,
        noise: bool = False,
    ) -> List[Dict[str, float]]:
        """
        Simulate the full post-surgical trajectory over ``time_points``.

        Returns a list of state dicts (one per time point), each containing
        ``_t_hours``, ``_phase``, and all SCM variable values.

        Example
        -------
        >>> traj = scm.simulate_trajectory({"tumor_size": 0.15})
        >>> icps = [(s["_t_hours"], s["intracranial_pressure"]) for s in traj]
        """
        if time_points is None:
            # Default: hourly 0→72 h then every 12 h up to 2 weeks
            time_points = (
                list(range(0, 73))          # 0–72 h, hourly
                + list(range(84, 337, 12))  # 84 h – 2 wk, 12-h intervals
            )

        trajectory: List[Dict[str, float]] = []
        # Snapshot the initial state so we can reset between time points
        initial_state = self.get_state()
        initial_dag_edges = list(self.dag.edges())

        for t in time_points:
            # Restore SCM to initial state for each independent time query
            for name, val in initial_state.items():
                self.variables[name].value = val
                self.variables[name].intervened = False
            self.dag = nx.DiGraph()
            self.dag.add_edges_from(initial_dag_edges)
            for node in self.variables:
                if node not in self.dag:
                    self.dag.add_node(node)

            state = self.evaluate_at_time(t, surgical_event=surgical_event, noise=noise)
            trajectory.append(state)

        # Final reset
        for name, val in initial_state.items():
            self.variables[name].value = val
            self.variables[name].intervened = False
        self.dag = nx.DiGraph()
        self._build_dag()

        return trajectory

    def summary(self) -> Dict:
        return {
            "variables": {k: {"value": v.value, "description": v.description}
                          for k, v in self.variables.items()},
            "edges": list(self.dag.edges()),
            "n_equations": len(self.equations),
        }
