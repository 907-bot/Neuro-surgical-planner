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
            intracranial_pressure → neural_function
            blood_flow → neural_function
            oxygen_saturation → metabolic_rate
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
                fn=lambda p, n: np.clip(1.0 - 0.8 * p["vascular_compression"] + n, 0, 1),
                description="Compression reduces cerebral blood flow",
            ),
            StructuralEquation(
                child="oxygen_saturation",
                parents=["blood_flow"],
                fn=lambda p, n: np.clip(0.6 + 0.4 * p["blood_flow"] + n, 0, 1),
                description="Oxygen delivery depends on blood flow",
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
                parents=["oxygen_saturation", "tumor_size"],
                fn=lambda p, n: np.clip(
                    p["oxygen_saturation"] * 0.7 + p["tumor_size"] * 0.5 + n,
                    0, 3.0
                ),
                description="Tumor hypoxia alters metabolic rate",
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

    def get_state(self) -> Dict[str, float]:
        return {name: var.value for name, var in self.variables.items()}

    def set_variable(self, name: str, value: float):
        """Set a variable value without marking it as intervened."""
        if name not in self.variables:
            raise KeyError(f"Unknown variable: {name}")
        self.variables[name].value = float(value)
        self.variables[name].clamp()

    def reset_interventions(self):
        """Remove all do() interventions."""
        for var in self.variables.values():
            var.intervened = False

    def summary(self) -> Dict:
        return {
            "variables": {k: {"value": v.value, "description": v.description}
                          for k, v in self.variables.items()},
            "edges": list(self.dag.edges()),
            "n_equations": len(self.equations),
        }
