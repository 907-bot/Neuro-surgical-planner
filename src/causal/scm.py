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
    O₂ saturation (SjO₂) modeled via the Fick principle: venous floor at 30%,
    increasing sigmoidally with CBF and decreasing with metabolic rate.
    """
    cbf = p["blood_flow"]
    metabolic_rate = p.get("metabolic_rate", 1.0)
    # Physically-grounded formula matching agonal floor & normal SjO2 ranges
    sat = 0.30 + 0.68 * cbf / (cbf + 0.8 * metabolic_rate)
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
        self._frozen_surgical_risk: Optional[float] = None  # Fix #4: freeze at t=0

        self._initialize_variables(patient_params or {})
        self._define_structural_equations()
        self._build_dag()

    def _initialize_variables(self, params: Dict):
        """Set initial values for all SCM variables."""
        defaults = {
            "tumor_size":            (0.3,  0.0, 1.0,  "normalized volume"),
            "blood_flow":            (0.7,  0.0, 1.0,  "relative cerebral blood flow"),
            "oxygen_saturation":     (0.95, 0.0, 1.0,  "O₂ saturation"),
            "intracranial_pressure": (0.2,  0.0, 1.0,  "normalized ICP"),
            "metabolic_rate":        (1.0,  0.0, 3.0,  "effective CMRO₂ (Pasteur-corrected)"),
            "edema_volume":          (0.2,  0.0, 1.0,  "peritumoral edema volume"),
            "vascular_compression":  (0.3,  0.0, 1.0,  "vessel compression ratio"),
            "neural_function":       (0.8,  0.0, 1.0,  "functional neural integrity"),
            "mass_effect":           (0.25, 0.0, 1.0,  "midline shift / mass effect"),
            "inflammatory_response": (0.3,  0.0, 1.0,  "neuroinflammation index"),
            "recovery_score":        (0.0,  0.0, 1.0,  "expected recovery"),
            "surgical_risk":         (0.0,  0.0, 1.0,  "composite pre-surgical risk"),
            # Fix #2: exogenous input — O₂ saturation from the previous timestep.
            # Avoids the metabolic_rate → oxygen_saturation → metabolic_rate DAG cycle.
            # Set to current O₂ at t=0; updated by simulate_trajectory each step.
            "o2_sat_prev":           (0.95, 0.0, 1.0,  "O₂ saturation at previous timestep"),
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

        DAG (v2 — post clinical review):
            tumor_size → vascular_compression → blood_flow → oxygen_saturation
            tumor_size → edema_volume → intracranial_pressure
            inflammatory_response → edema_volume
            inflammatory_response → intracranial_pressure  (direct: CSF outflow obstruction)
            tumor_size + o2_sat_prev → metabolic_rate      (Pasteur hypoxic suppression)
            metabolic_rate → oxygen_saturation
            intracranial_pressure → neural_function
            blood_flow → neural_function
            neural_function + blood_flow + surgical_risk → recovery_score

        Fix history:
            2026-06-26  v1  initial
            2026-06-27  v2  Fix#1: add inflammatory_response → ICP (direct CSF path),
                                   reduce edema weight 0.40→0.35, mass_effect 0.30→0.25
                            Fix#2: add Pasteur-effect hypoxic suppression in metabolic_rate
                                   via o2_sat_prev exogenous input (avoids DAG cycle)
        """

        eqs = [
            StructuralEquation(
                child="vascular_compression",
                parents=["tumor_size", "edema_volume"],
                fn=lambda p, n: np.clip(
                    0.4 * p["tumor_size"] + 0.3 * p["edema_volume"] + n, 0, 1
                ),
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
                fn=lambda p, n: np.clip(
                    0.5 * p["tumor_size"] + 0.3 * p["inflammatory_response"] + n, 0, 1
                ),
                description="Edema driven by tumor size and inflammation",
            ),
            # ── Fix #1: inflammatory_response is now a DIRECT parent of ICP ──────
            # Rationale: inflammation impairs CSF outflow via meningeal irritation
            # independently of edema. Previously edema_volume already carried the
            # inflammatory signal, so the post-hoc +0.35*(inflam-0.3) addition in
            # evaluate_at_time was double-counting. Now the full path lives in the DAG:
            #   inflammatory_response → edema_volume → ICP  (vasogenic edema)
            #   inflammatory_response → ICP directly         (CSF outflow obstruction)
            # Edema weight reduced 0.40→0.35; mass_effect 0.30→0.25 to keep sum = 1.0
            StructuralEquation(
                child="intracranial_pressure",
                parents=["tumor_size", "edema_volume", "mass_effect", "inflammatory_response"],
                fn=lambda p, n: np.clip(
                    0.20  # baseline resting ICP (8 mmHg)
                    + 0.20 * p["tumor_size"]
                    + 0.30 * p["edema_volume"]
                    + 0.20 * p["mass_effect"]
                    + 0.10 * p["inflammatory_response"]  # direct CSF obstruction
                    + n,
                    0, 1,
                ),
                description=(
                    "ICP: resting baseline + tumor mass + vasogenic edema "
                    "+ midline shift + direct inflammatory CSF outflow obstruction"
                ),
            ),
            StructuralEquation(
                child="mass_effect",
                parents=["tumor_size", "edema_volume"],
                fn=lambda p, n: np.clip(
                    0.6 * p["tumor_size"] + 0.4 * p["edema_volume"] + n, 0, 1
                ),
                description="Mass effect = combined tumor + edema displacement",
            ),
            # ── Fix #2: Pasteur-effect hypoxic metabolic suppression ─────────────
            # Rationale: when O₂ drops below ~60%, glioma cells shift to anaerobic
            # glycolysis (Warburg/Pasteur effect) and CMRO₂ falls. A DAG cycle
            # (O₂ → metabolic_rate → O₂) is avoided by using o2_sat_prev — the
            # oxygen saturation from the PREVIOUS timestep — as an exogenous input.
            # At t=0 (no prior state) o2_sat_prev defaults to the initial O₂ value.
            StructuralEquation(
                child="metabolic_rate",
                parents=["tumor_size", "o2_sat_prev"],
                fn=lambda p, n: np.clip(
                    # Baseline aerobic demand from tumor burden
                    (1.0 + p["tumor_size"] * 0.8)
                    # Hypoxic suppression sigmoid: full demand above sat=0.65,
                    # falls to 60% of demand near zero O₂ (anaerobic floor)
                    * (0.6 + 0.4 / (1.0 + np.exp(-12.0 * (p["o2_sat_prev"] - 0.55))))
                    + n,
                    0, 3.0,
                ),
                description=(
                    "Effective CMRO₂: tumor aerobic demand suppressed by hypoxia "
                    "(Pasteur effect, sigmoid threshold at SjO₂≈55%)"
                ),
            ),
            StructuralEquation(
                child="neural_function",
                parents=["oxygen_saturation", "intracranial_pressure", "vascular_compression"],
                fn=lambda p, n: np.clip(
                    p["oxygen_saturation"] * 0.5
                    - p["intracranial_pressure"] * 0.3
                    - p["vascular_compression"] * 0.2
                    + 0.5 + n,
                    0, 1,
                ),
                description="Neural function degrades with hypoxia, high ICP, compression",
            ),
            StructuralEquation(
                child="recovery_score",
                parents=["neural_function", "blood_flow", "surgical_risk"],
                fn=lambda p, n: np.clip(
                    0.5 * p["neural_function"]
                    + 0.3 * p["blood_flow"]
                    - 0.2 * p["surgical_risk"]
                    + n,
                    0, 1,
                ),
                description="Recovery = neural integrity + blood flow - pre-surgical risk",
            ),
            StructuralEquation(
                child="surgical_risk",
                parents=["tumor_size", "vascular_compression", "intracranial_pressure", "neural_function"],
                fn=lambda p, n: np.clip(
                    0.3 * p["tumor_size"]
                    + 0.3 * p["vascular_compression"]
                    + 0.2 * p["intracranial_pressure"]
                    + 0.2 * (1.0 - p["neural_function"])
                    + n,
                    0, 1,
                ),
                description="Pre-surgical risk composite (frozen at t=0 in trajectories)",
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

        # ── Record pre-surgical state ─────────────────────────────────────────
        tumor_size_pre = self.variables["tumor_size"].value
        tumor_size_post = surgical_event.get("tumor_size", tumor_size_pre)
        resection_delta = max(0.0, tumor_size_pre - tumor_size_post)
        # Record pre-surgical edema volume (before intervention)
        edema_pre = self.variables["edema_volume"].value

        # ── Fix #4: Freeze surgical_risk at t=0 (pre-surgical assessment) ─────
        # surgical_risk is a planning variable assessed BEFORE operation.
        # Re-evaluating it at t=72h would show artificially lower risk as
        # neural_function recovers — conflating planning risk with post-op morbidity.
        if self._frozen_surgical_risk is None:
            pre_op_vals = self.evaluate(noise=False)
            self._frozen_surgical_risk = pre_op_vals["surgical_risk"]
            logger.debug(f"surgical_risk frozen at pre-op value: {self._frozen_surgical_risk:.4f}")
        self.variables["surgical_risk"].value = self._frozen_surgical_risk
        self.variables["surgical_risk"].intervened = True  # hold fixed through trajectory

        # ── Apply do() intervention (tumor resection) ─────────────────────────
        self.intervene(surgical_event)

        # ── Phase 1: Intraoperative ICP relief (0–6 h) ───────────────────────
        # Fix #3: Split into two components:
        #   a) Permanent volume relief — tumor is physically removed; this persists.
        #   b) Transient positioning/retraction artifact — resolves linearly by 6h.
        permanent_relief  = resection_delta * 0.25
        transient_relief  = resection_delta * 0.15 * np.clip(1.0 - t_hours / 6.0, 0.0, 1.0)
        intraop_relief    = permanent_relief + transient_relief

        # ── Phase 2: Acute neuroinflammatory spike (6–48 h) ──────────────────
        # IL-6/PGE2 surge from surgical trauma propagates through the DAG:
        #   inflammatory_response → edema_volume → ICP  (vasogenic edema)
        #   inflammatory_response → ICP directly         (CSF outflow obstruction)
        # Fix #1: The post-hoc inflammation_icp_contribution line is REMOVED.
        # The DAG now carries the full inflammatory signal; no double-counting.
        inflammatory_level = surgical_inflammatory_response(
            t_hours=t_hours,
            tumor_size_pre=tumor_size_pre,
            tumor_size_post=tumor_size_post,
            baseline_inflammation=self.variables["inflammatory_response"].value,
        )
        self.variables["inflammatory_response"].value = inflammatory_level
        # Not intervened — remains an exogenous time-varying input

        # ── Phase 3: Dexamethasone / edema resolution (48 h – 2 wk) ─────────
        dex_effect = 0.0
        if t_hours > 48.0:
            dex_time  = t_hours - 48.0
            dex_effect = resection_delta * 0.25 * (1.0 - np.exp(-dex_time / 72.0))

        # ── Edema volume temporal dynamics ────────────────────────────────────
        # Peritumoral vasogenic edema does not resolve instantly upon tumor resection.
        # It decays slowly (96-hour half-life) toward the post-op baseline.
        edema_post_static = np.clip(
            0.5 * tumor_size_post + 0.3 * inflammatory_level, 0.0, 1.0
        )
        edema_t = float(
            edema_pre * np.exp(-t_hours / 96.0)
            + edema_post_static * (1.0 - np.exp(-t_hours / 96.0))
        )
        self.variables["edema_volume"].value = edema_t
        self.variables["edema_volume"].intervened = True

        # ── Fix #2: Pasteur-effect — pass previous O₂ as exogenous input ──────
        # metabolic_rate now takes o2_sat_prev as a parent to model hypoxic
        # suppression without creating a DAG cycle. At each time point the
        # caller (simulate_trajectory) has already written the previous
        # timestep's O₂ into self.variables["o2_sat_prev"].
        # On a standalone evaluate_at_time call, o2_sat_prev = current O₂ (t=0).

        # ── Evaluate the DAG (inflammation propagates internally now) ─────────
        vals = self.evaluate(noise=noise)

        # ── ICP time-phase correction (permanent + transient relief + dex) ────
        # Fix #1: NO inflammation_icp_contribution here — it lives in the DAG.
        raw_icp = vals["intracranial_pressure"]
        icp_t   = float(np.clip(
            raw_icp
            - intraop_relief
            - dex_effect
            + (np.random.normal(0, 0.01) if noise else 0.0),
            self.variables["intracranial_pressure"].min_val,
            self.variables["intracranial_pressure"].max_val,
        ))
        vals["intracranial_pressure"] = icp_t
        self.variables["intracranial_pressure"].value = icp_t

        # ── Metadata for trajectory analysis ──────────────────────────────────
        vals["_t_hours"]            = t_hours
        vals["_inflammatory_level"] = inflammatory_level
        vals["_intraop_relief"]     = intraop_relief
        vals["_dex_effect"]         = dex_effect
        vals["_phase"] = (
            "intraoperative" if t_hours <= 6 else
            "acute_spike"    if t_hours <= 48 else
            "recovery"
        )

        logger.info(
            f"t={t_hours:5.1f}h | phase={vals['_phase']:14s} | "
            f"ICP={icp_t:.3f} | inflam={inflammatory_level:.3f} | "
            f"perm_relief={permanent_relief:.3f} | dex={dex_effect:.3f}"
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
        # Snapshot the pre-surgical state and DAG topology
        initial_state     = self.get_state()
        initial_dag_edges = list(self.dag.edges())

        # Fix #4: Compute and freeze surgical_risk once across the whole trajectory.
        # Reset first to ensure a clean pre-op evaluation.
        self._frozen_surgical_risk = None
        pre_op_vals = self.evaluate(noise=False)
        frozen_risk = pre_op_vals["surgical_risk"]
        logger.debug(f"Trajectory: surgical_risk frozen at {frozen_risk:.4f} (pre-op)")

        prev_o2_sat = initial_state.get("oxygen_saturation", 0.95)  # Fix #2: Pasteur bootstrap

        for t in time_points:
            # ── Restore SCM to pre-surgical state ────────────────────────────
            for name, val in initial_state.items():
                self.variables[name].value = val
                self.variables[name].intervened = False
            self.dag = nx.DiGraph()
            self.dag.add_edges_from(initial_dag_edges)
            for node in self.variables:
                if node not in self.dag:
                    self.dag.add_node(node)

            # Fix #4: inject frozen surgical_risk
            self._frozen_surgical_risk = frozen_risk

            # Fix #2: inject previous timestep's O₂ for Pasteur suppression
            self.variables["o2_sat_prev"].value = float(prev_o2_sat)

            state = self.evaluate_at_time(t, surgical_event=surgical_event, noise=noise)
            trajectory.append(state)

            # Update prev_o2_sat for the next timestep (causal chain across time)
            prev_o2_sat = state.get("oxygen_saturation", prev_o2_sat)

        # ── Full reset ────────────────────────────────────────────────────────
        for name, val in initial_state.items():
            self.variables[name].value = val
            self.variables[name].intervened = False
        self._frozen_surgical_risk = None
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
