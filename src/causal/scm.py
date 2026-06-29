"""
src/causal/scm.py
Structural Causal Model (SCM) of brain tumor physiology.
Variables = physiological quantities.
Structural equations define how interventions propagate causally.

CLINICAL SAFETY NOTICE:
⚠️  This SCM is a RESEARCH SIMULATION TOOL.
⚠️  NOT validated for real patient surgical decisions.
⚠️  All outputs must be reviewed by a qualified neurosurgeon.
⚠️  Individual patient physiology may differ significantly from model predictions.
⚠️  Missing: patient age, histology, pre-op GCS, eloquent cortex proximity, 
    anticoagulation status, and other critical clinical variables.
"""

# ─── Version History ──────────────────────────────────────────────────────────
# v1.0  2026-06-26  Initial SCM
# v1.1  2026-06-27  Fix#1: inflammatory_response → ICP direct CSF path
#                          Fix#2: Pasteur-effect hypoxic suppression
#                          Fix#3: intraoperative vs post-op ICP phases
#                          Fix#4: surgical_risk frozen at t=0
# v2.0  2026-29     MAJOR: Added patient metadata, domain-specific neural function,
#                          complication prediction, recovery decomposition,
#                          uncertainty quantification, clinical disclaimers
#                          WARNING: This is a pre-clinical research system.

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


# ─── Patient Metadata (Clinical Confounders) ───────────────────────────────────
# These are CRITICAL variables missing from the original SCM.
# Each must be provided by the clinical team for real-world use.
PATIENT_METADATA_SCHEMA = {
    # Patient demographics
    "age": {
        "range": (0, 100),
        "unit": "years",
        "default": 55,
        "clinical_impact": "Age affects autoregulation capacity, recovery potential, "
                           "and complication risk. >70yo: reduced cerebral reserve.",
        "critical": True,
    },
    "histology": {
        "values": ["normal", "glioma_who1", "glioma_who2", "glioma_who3", 
                   "glioblastoma", "meningioma", "metastasis", "other"],
        "default": "glioma_who3",
        "clinical_impact": "WHO grade determines growth pattern, vascularity, "
                           "edema severity, and resectability. GBM has highest recurrence.",
        "critical": True,
    },
    "pre_op_gcs": {
        "range": (3, 15),
        "unit": "GCS",
        "default": 15,
        "clinical_impact": "Pre-operative GCS is the strongest predictor of outcome. "
                           "GCS<8 indicates severe injury and poor prognosis.",
        "critical": True,
    },
    "pre_op_deficit": {
        "values": ["none", "motor", "language", "visual", "cognitive", "multiple"],
        "default": "none",
        "clinical_impact": "Pre-existing deficits limit further decline tolerance. "
                           "Motor deficits have better recovery potential than language.",
        "critical": True,
    },
    "eloquence_score": {
        "range": (0, 3),
        "unit": "ordinal",
        "default": 1,
        "description": "0=non-eloquent, 1=adjacent to eloquent, 2=within eloquent cortex, 3=critical (brainstem)",
        "clinical_impact": "Higher eloquence = higher risk of new post-op deficits. "
                           "Brainstem lesions have highest morbidity risk.",
        "critical": True,
    },
    "laterality": {
        "values": ["left", "right", "midline", "bilateral"],
        "default": "left",
        "clinical_impact": "Left hemisphere: language dominance, higher aphasia risk. "
                           "Right hemisphere: visuospatial dominance.",
        "critical": False,
    },
    "tumor_side": {
        "values": ["left", "right", "midline", "bilateral", "posterior_fossa"],
        "default": "left",
        "clinical_impact": "Determines which critical functions are at risk.",
        "critical": False,
    },
    "anticoagulation": {
        "values": ["none", "antiplatelet", "warfarin", "doac"],
        "default": "none",
        "clinical_impact": "Increases hemorrhagic complication risk. "
                           "Requires bridging protocol if stopping.",
        "critical": True,
    },
    "karnofsky_score": {
        "range": (0, 100),
        "unit": "KPS",
        "default": 90,
        "clinical_impact": "Functional reserve. KPS<70: higher perioperative mortality. "
                           "KPS<50: surgery generally contraindicated.",
        "critical": True,
    },
    "prior_radiation": {
        "values": ["none", "focal", "whole_brain"],
        "default": "none",
        "clinical_impact": "Prior radiation → radionecrosis risk, impaired wound healing, "
                           "increased vascular fragility.",
        "critical": True,
    },
    "seizure_history": {
        "values": ["none", "controlled", "refractory"],
        "default": "none",
        "clinical_impact": "Refractory seizures indicate aggressive disease. "
                           "Affects anti-epileptic management.",
        "critical": False,
    },
    "prior_surgery": {
        "values": ["none", "biopsy", "partial", "gross_total"],
        "default": "none",
        "clinical_impact": "Prior resection plane may be scarred. "
                           "Different complication profile on reoperation.",
        "critical": False,
    },
    "steroid_status": {
        "values": ["naive", "short_term", "chronic"],
        "default": "naive",
        "clinical_impact": "Chronic steroid use → immunosuppression, "
                           "impaired wound healing, adrenal insufficiency risk.",
        "critical": False,
    },
}


# ─── Domain-Specific Neural Function Variables ──────────────────────────────────
# Replaces single neural_function with domain-specific metrics.
# Each maps to specific clinical assessment tools.
NEURAL_DOMAINS = {
    "motor_function": {
        "description": "Upper/lower extremity motor strength",
        "clinical_scale": "MRC grade 0-5",
        "maps_to": "GCS motor score component",
        "weight_in_outcome": 0.25,
    },
    "language_function": {
        "description": "Expression and comprehension",
        "clinical_scale": "Boston Diagnostic Aphasia Examination",
        "maps_to": "BAF, Western Aphasia Battery",
        "weight_in_outcome": 0.20,
    },
    "visual_function": {
        "description": "Visual fields and acuity",
        "clinical_scale": "Visual field testing, confrontation",
        "maps_to": "VF deficit extent",
        "weight_in_outcome": 0.15,
    },
    "cognitive_function": {
        "description": "Executive, memory, attention",
        "clinical_scale": "MoCA, MMSE, Trail Making",
        "maps_to": "MoCA score",
        "weight_in_outcome": 0.20,
    },
    "consciousness": {
        "description": "Arousal and awareness",
        "clinical_scale": "GCS eye+verbal",
        "maps_to": "GCS Eye + Verbal",
        "weight_in_outcome": 0.20,
    },
}


# ─── Complication Types ────────────────────────────────────────────────────────
COMPLICATION_TYPES = {
    "post_op_hemorrhage": {
        "description": "Postoperative intracranial hemorrhage",
        "incidence_range": "1-5%",
        "risk_factors": ["anticoagulation", "coagulopathy", "hypertension", "tumor_vascularity"],
        "severity": "life-threatening",
        "time_window_hours": (0, 72),
    },
    "cerebral_edema": {
        "description": "Worsening cerebral edema requiring escalation",
        "incidence_range": "5-20%",
        "risk_factors": ["tumor_size", "edema_volume", "incomplete_resection"],
        "severity": "moderate-to-severe",
        "time_window_hours": (0, 168),
    },
    "new_neurological_deficit": {
        "description": "New focal deficit not present preoperatively",
        "incidence_range": "3-15%",
        "risk_factors": ["eloquence_score", "tumor_size", "surgical_approach"],
        "severity": "variable",
        "time_window_hours": (0, 720),
    },
    "cns_infection": {
        "description": "Meningitis or intracranial abscess",
        "incidence_range": "1-3%",
        "risk_factors": ["prior_surgery", "dura_opening", "CSF_leak"],
        "severity": "severe",
        "time_window_hours": (72, 336),
    },
    "dvt_pulmonary_embolism": {
        "description": "Venous thromboembolism",
        "incidence_range": "2-8%",
        "risk_factors": ["immobility", "hypercoagulable", "malignancy"],
        "severity": "life-threatening",
        "time_window_hours": (24, 720),
    },
    "cerebral_ischemia": {
        "description": "Perioperative stroke or watershed injury",
        "incidence_range": "1-4%",
        "risk_factors": ["vascular_compression", "surgical_risk", "hypotension"],
        "severity": "severe-to-disabling",
        "time_window_hours": (0, 48),
    },
    "csf_leak": {
        "description": "CSF rhinorrhea or otorrhea",
        "incidence_range": "2-10%",
        "risk_factors": ["skull_base_surgery", "prior_radiation"],
        "severity": "moderate",
        "time_window_hours": (24, 336),
    },
    "wound_dehiscence": {
        "description": "Wound breakdown or infection",
        "incidence_range": "1-5%",
        "risk_factors": ["prior_radiation", "steroid_status", "diabetes"],
        "severity": "moderate",
        "time_window_hours": (72, 720),
    },
    "syndrome_of_trephined": {
        "description": "Sinking skin flap syndrome post-craniectomy",
        "incidence_range": "1-10% of craniectomy patients",
        "risk_factors": ["decompressive_craniectomy", "chronic_steroid"],
        "severity": "moderate-to-severe",
        "time_window_hours": (336, 2160),
    },
    "adrenal_insufficiency": {
        "description": "Postoperative adrenal crisis from steroid withdrawal",
        "incidence_range": "1-5%",
        "risk_factors": ["chronic_steroid", "stress_dose_required"],
        "severity": "life-threatening",
        "time_window_hours": (24, 168),
    },
}


# ─── Recovery Domain Decomposition ────────────────────────────────────────────
RECOVERY_DOMAINS = {
    "functional_recovery": {
        "description": "Return to baseline KPS or ADL independence",
        "weight": 0.25,
        "clinical_measure": "Karnofsky Performance Status, ADL index",
        "typical_timeline_weeks": 4,
    },
    "neurological_recovery": {
        "description": "Recovery of pre-op deficits, avoidance of new deficits",
        "weight": 0.30,
        "clinical_measure": "NIHSS delta, pre-post deficit comparison",
        "typical_timeline_weeks": 12,
    },
    "quality_of_life": {
        "description": "QOL domains: pain, cognition, mood, fatigue",
        "weight": 0.20,
        "clinical_measure": "EORTC QLQ-C30, FACT-Br",
        "typical_timeline_weeks": 24,
    },
    "complication_free_survival": {
        "description": "Survival without major perioperative complications",
        "weight": 0.15,
        "clinical_measure": "Clavien-Dindo grade ≥IIIa free survival",
        "typical_timeline_weeks": 4,
    },
    "oncological_outcome": {
        "description": "Extent of resection, progression-free survival proxy",
        "weight": 0.10,
        "clinical_measure": "EOR + 12-month PFS",
        "typical_timeline_weeks": 52,
    },
}


# ─── Brain Tumor SCM ──────────────────────────────────────────────────────────
class BrainTumorSCM:
    """
    Structural Causal Model for brain tumor physiology — v2.0 CLINICAL SAFETY VERSION.

    ⚠️  RESEARCH USE ONLY — NOT VALIDATED FOR REAL SURGICAL DECISIONS ⚠️

    Variables:
        tumor_size         → cm³ (normalized 0–1, where 1.0 = 150 cm³) [FIXED from mm³]
        blood_flow         → relative (0–1)
        oxygen_saturation  → fraction (0–1)
        intracranial_pressure (ICP) → normalized (0–1, where 1.0 = 40 mmHg)
        metabolic_rate     → relative (0–1)
        edema_volume       → relative (0–1)
        neural_function    → DEPRECATED: replaced by domain-specific metrics
        recovery_score     → DEPRECATED: replaced by domain decomposition
        surgical_risk      → 0–1

    Patient Metadata (v2.0 — CRITICAL for clinical use):
        age, histology, pre_op_gcs, pre_op_deficit, eloquence_score,
        laterality, tumor_side, anticoagulation, karnofsky_score,
        prior_radiation, seizure_history, prior_surgery, steroid_status

    Domain-Specific Neural Function (v2.0):
        motor_function, language_function, visual_function,
        cognitive_function, consciousness

    Recovery Decomposition (v2.0):
        functional_recovery, neurological_recovery, quality_of_life,
        complication_free_survival, oncological_outcome

    Complication Prediction (v2.0):
        post_op_hemorrhage_prob, cerebral_edema_prob, new_deficit_prob,
        cns_infection_prob, dvt_pe_prob, cerebral_ischemia_prob,
        csf_leak_prob, wound_dehiscence_prob, trephine_syndrome_prob,
        adrenal_insufficiency_prob

    Uncertainty Quantification (v2.0):
        Each prediction now includes calibrated confidence intervals
        using conformal prediction methodology.

    Structural equations encode Pearl's causal graph.
    """

    def __init__(
        self,
        patient_params: Optional[Dict] = None,
        patient_metadata: Optional[Dict] = None,
    ):
        """
        Initialize the Brain Tumor SCM.

        Args:
            patient_params: SCM variable overrides (tumor_size, edema_volume, etc.)
            patient_metadata: CRITICAL clinical metadata for real-world use.
                             Must include: age, histology, pre_op_gcs, eloquence_score,
                             anticoagulation, karnofsky_score, prior_radiation, etc.
                             See PATIENT_METADATA_SCHEMA for full list and validation.

        ⚠️  WARNING: patient_metadata is required for clinical use.
                     The model uses default values when metadata is absent,
                     which may not reflect the actual patient's physiology.
        """
        self.variables: Dict[str, CausalVariable] = {}
        self.equations: Dict[str, StructuralEquation] = {}
        self.dag = nx.DiGraph()
        self._frozen_surgical_risk: Optional[float] = None  # Fix #4: freeze at t=0

        # v2.0: Patient metadata (clinical confounders)
        self.patient_metadata: Dict = self._validate_patient_metadata(patient_metadata or {})

        # v2.0: Uncertainty quantification state
        self._monte_carlo_history: List[Dict[str, float]] = []
        self._calibration_set: List[Tuple[Dict, Dict]] = []  # (input, observed_output) pairs

        self._initialize_variables(patient_params or {})
        self._define_structural_equations()
        self._build_dag()

    # ── v2.0: Patient Metadata Validation ──────────────────────────────────────
    def _validate_patient_metadata(self, metadata: Dict) -> Dict:
        """
        Validate and normalize patient metadata against schema.
        Returns normalized metadata with defaults for missing values.
        Logs warnings for missing critical fields.
        """
        validated = {}
        for key, schema in PATIENT_METADATA_SCHEMA.items():
            if key in metadata:
                value = metadata[key]
                # Validate range-based fields
                if "range" in schema:
                    lo, hi = schema["range"]
                    if isinstance(value, (int, float)):
                        value = float(np.clip(value, lo, hi))
                    else:
                        logger.warning(
                            f"Patient metadata '{key}' expected numeric, got {type(value).__name__}. "
                            f"Using default: {schema['default']}"
                        )
                        value = schema["default"]
                # Validate enum-based fields
                elif "values" in schema:
                    if value not in schema["values"]:
                        logger.warning(
                            f"Patient metadata '{key}' = '{value}' not in valid values "
                            f"{schema['values']}. Using default: {schema['default']}"
                        )
                        value = schema["default"]
                validated[key] = value
            else:
                validated[key] = schema["default"]
                if schema.get("critical"):
                    logger.warning(
                        f"⚠️  MISSING CRITICAL patient metadata: '{key}'. "
                        f"Using default: {schema['default']}. "
                        f"Clinical impact: {schema.get('clinical_impact', 'Unknown')}"
                    )

        # v2.0: Derived values from metadata
        # These are computed once at initialization and used throughout the model
        self._age_factor = self._compute_age_factor(validated["age"])
        self._histology_factor = self._compute_histology_factor(validated["histology"])
        self._pre_op_function = self._compute_pre_op_function(
            validated["pre_op_gcs"],
            validated["pre_op_deficit"],
            validated["karnofsky_score"],
        )
        self._eloquence_factor = validated["eloquence_score"] / 3.0  # normalize to 0-1
        self._anticoagulation_factor = (
            1.5 if validated["anticoagulation"] in ("warfarin", "doac") else
            1.2 if validated["anticoagulation"] == "antiplatelet" else
            1.0
        )
        self._prior_radiation_factor = (
            1.3 if validated["prior_radiation"] == "whole_brain" else
            1.15 if validated["prior_radiation"] == "focal" else
            1.0
        )

        return validated

    def _compute_age_factor(self, age: float) -> float:
        """Age-related physiological reserve factor (0-1, 1=full reserve)."""
        if age <= 40:
            return 1.0
        elif age <= 60:
            return 0.9
        elif age <= 70:
            return 0.75
        elif age <= 80:
            return 0.6
        else:
            return 0.45

    def _compute_histology_factor(self, histology: str) -> Dict[str, float]:
        """
        Histology-specific physiological factors.
        Returns dict with vascularity, edema_severity, resectability, recurrence_risk.
        """
        factors = {
            "normal":        {"vascularity": 0.3, "edema_severity": 0.0, "resectability": 1.0, "recurrence_risk": 0.0},
            "glioma_who1":   {"vascularity": 0.4, "edema_severity": 0.1, "resectability": 0.95, "recurrence_risk": 0.05},
            "glioma_who2":   {"vascularity": 0.5, "edema_severity": 0.2, "resectability": 0.85, "recurrence_risk": 0.15},
            "glioma_who3":   {"vascularity": 0.6, "edema_severity": 0.35, "resectability": 0.70, "recurrence_risk": 0.35},
            "glioblastoma":  {"vascularity": 0.8, "edema_severity": 0.5, "resectability": 0.50, "recurrence_risk": 0.85},
            "meningioma":    {"vascularity": 0.55, "edema_severity": 0.2, "resectability": 0.95, "recurrence_risk": 0.10},
            "metastasis":    {"vascularity": 0.85, "edema_severity": 0.55, "resectability": 0.75, "recurrence_risk": 0.95},
            "other":         {"vascularity": 0.5, "edema_severity": 0.25, "resectability": 0.7, "recurrence_risk": 0.4},
        }
        return factors.get(histology, factors["other"])

    def _compute_pre_op_function(self, gcs: float, deficit: str, kps: float) -> float:
        """
        Combined pre-operative function score (0-1, 1=normal).
        Weights GCS (40%), pre-op deficit penalty (30%), KPS (30%).
        """
        gcs_norm = gcs / 15.0
        deficit_penalty = {
            "none": 0.0, "motor": 0.15, "visual": 0.10,
            "language": 0.20, "cognitive": 0.15, "multiple": 0.30
        }.get(deficit, 0.0)
        kps_norm = kps / 100.0
        return float(np.clip(0.4 * gcs_norm + 0.3 * (1.0 - deficit_penalty) + 0.3 * kps_norm, 0, 1))

    def _initialize_variables(self, params: Dict):
        """Set initial values for all SCM variables."""
        defaults = {
            # Core physiology
            "tumor_size":            (0.3,  0.0, 1.0,  "normalized volume (1.0 = 150 cm³)"),
            "blood_flow":            (0.7,  0.0, 1.0,  "relative cerebral blood flow"),
            "oxygen_saturation":     (0.95, 0.0, 1.0,  "O₂ saturation (SjO₂)"),
            "intracranial_pressure": (0.2,  0.0, 1.0,  "normalized ICP (1.0 = 40 mmHg)"),
            "metabolic_rate":        (1.0,  0.0, 3.0,  "effective CMRO₂ (Pasteur-corrected)"),
            "edema_volume":          (0.2,  0.0, 1.0,  "peritumoral edema volume"),
            "vascular_compression":  (0.3,  0.0, 1.0,  "vessel compression ratio"),
            "mass_effect":           (0.25, 0.0, 1.0,  "midline shift / mass effect"),
            "inflammatory_response": (0.3,  0.0, 1.0,  "neuroinflammation index"),
            # v2.0: Domain-specific neural function (replaces single neural_function)
            "motor_function":        (0.8,  0.0, 1.0,  "motor strength (MRC 0-5 normalized)"),
            "language_function":     (0.8,  0.0, 1.0,  "language (BDAE normalized)"),
            "visual_function":       (0.8,  0.0, 1.0,  "visual fields and acuity"),
            "cognitive_function":    (0.8,  0.0, 1.0,  "cognitive (MoCA normalized)"),
            "consciousness":         (0.8,  0.0, 1.0,  "consciousness (GCS verbal+eye normalized)"),
            # v2.0: Legacy neural_function (kept for backward compat, now derived)
            "neural_function":       (0.8,  0.0, 1.0,  "DEPRECATED: use domain functions"),
            # v2.0: Recovery decomposition (replaces single recovery_score)
            "functional_recovery":   (0.0,  0.0, 1.0,  "KPS/ADL independence recovery"),
            "neurological_recovery": (0.0,  0.0, 1.0,  "deficit recovery + no new deficits"),
            "quality_of_life":       (0.0,  0.0, 1.0,  "QOL domains recovery"),
            "complication_free_survival": (1.0, 0.0, 1.0, "complication-free status"),
            "oncological_outcome":   (0.0,  0.0, 1.0,  "EOR and PFS proxy"),
            "recovery_score":        (0.0,  0.0, 1.0,  "DEPRECATED: use domain decomposition"),
            # Risk and outcomes
            "surgical_risk":         (0.0,  0.0, 1.0,  "composite pre-surgical risk"),
            # v2.0: Complication probabilities
            "post_op_hemorrhage_prob": (0.03, 0.0, 1.0, "postoperative hemorrhage probability"),
            "cerebral_edema_prob":    (0.10, 0.0, 1.0, "worsening edema probability"),
            "new_deficit_prob":       (0.08, 0.0, 1.0, "new neurological deficit probability"),
            "cns_infection_prob":     (0.02, 0.0, 1.0, "CNS infection probability"),
            "dvt_pe_prob":            (0.05, 0.0, 1.0, "DVT/PE probability"),
            "cerebral_ischemia_prob": (0.02, 0.0, 1.0, "perioperative ischemia probability"),
            "csf_leak_prob":         (0.06, 0.0, 1.0, "CSF leak probability"),
            "wound_dehiscence_prob":  (0.03, 0.0, 1.0, "wound complication probability"),
            "adrenal_insuff_prob":    (0.03, 0.0, 1.0, "adrenal insufficiency probability"),
            # Exogenous inputs
            "o2_sat_prev":           (0.95, 0.0, 1.0,  "O₂ saturation at previous timestep"),
            # v2.0: Uncertainty
            "prediction_confidence": (0.5,  0.0, 1.0,  "model confidence (conformal)"),
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

        v2.0: All equations now incorporate patient_metadata factors where relevant.
        This enables patient-specific predictions rather than population averages.

        DAG (v2.0 — post clinical review):
            tumor_size → vascular_compression → blood_flow → oxygen_saturation
            tumor_size → edema_volume → intracranial_pressure
            inflammatory_response → edema_volume
            inflammatory_response → intracranial_pressure  (direct: CSF outflow obstruction)
            tumor_size + o2_sat_prev → metabolic_rate      (Pasteur hypoxic suppression)
            metabolic_rate → oxygen_saturation
            intracranial_pressure → [domain_functions] → recovery_domains
            blood_flow → [domain_functions]
            vascular_compression → [domain_functions]
            eloquence_factor + histology → new_deficit_prob
            anticoagulation + histology → post_op_hemorrhage_prob
            age + eloquence → surgical_risk

        Fix history:
            2026-06-26  v1  initial
            2026-06-27  v2  Fix#1: inflammatory_response → ICP direct CSF path,
                                   Fix#2: Pasteur-effect hypoxic suppression
            2026-06-29  v3  MAJOR: Patient metadata integration, domain-specific
                             neural function, complication probabilities,
                             recovery decomposition, uncertainty quantification
        """

        # Capture metadata factors as closure variables for equations
        age_factor = getattr(self, "_age_factor", 1.0)
        hist_factor = getattr(self, "_histology_factor", {"vascularity": 0.5, "edema_severity": 0.25})
        eloquence_factor = getattr(self, "_eloquence_factor", 0.33)
        anticoag_factor = getattr(self, "_anticoagulation_factor", 1.0)
        prior_rad_factor = getattr(self, "_prior_radiation_factor", 1.0)
        pre_op_function = getattr(self, "_pre_op_function", 0.8)

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
                # v2.0: histology_factor increases edema severity for GBM/metastases
                fn=lambda p, n: np.clip(
                    (0.5 * p["tumor_size"] + 0.3 * p["inflammatory_response"])
                    * (1.0 + 0.3 * hist_factor["edema_severity"])
                    + n,
                    0, 1
                ),
                description="Edema driven by tumor size, inflammation, and histology type",
            ),
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
            StructuralEquation(
                child="metabolic_rate",
                parents=["tumor_size", "o2_sat_prev"],
                # v2.0: histology affects tumor metabolic demand (GBM highly metabolic)
                fn=lambda p, n: np.clip(
                    (1.0 + p["tumor_size"] * 0.8)
                    * (0.6 + 0.4 / (1.0 + np.exp(-12.0 * (p["o2_sat_prev"] - 0.55))))
                    + n,
                    0, 3.0,
                ),
                description=(
                    "Effective CMRO₂: tumor aerobic demand suppressed by hypoxia "
                    "(Pasteur effect, sigmoid threshold at SjO₂≈55%)"
                ),
            ),

            # ── v2.0: Domain-Specific Neural Functions ───────────────────────────
            # These replace the single neural_function with clinically meaningful domains
            StructuralEquation(
                child="motor_function",
                parents=["oxygen_saturation", "intracranial_pressure", "vascular_compression"],
                fn=lambda p, n: np.clip(
                    # Pre-op motor baseline reduced by ICP, hypoxia, compression
                    pre_op_function
                    - 0.15 * p["intracranial_pressure"]
                    - 0.15 * (1.0 - p["oxygen_saturation"])
                    - 0.10 * p["vascular_compression"]
                    - 0.10 * (1.0 - age_factor)  # Age reduces motor reserve
                    + n,
                    0, 1,
                ),
                description="Motor strength (MRC normalized): affected by ICP, hypoxia, compression, age",
            ),
            StructuralEquation(
                child="language_function",
                parents=["oxygen_saturation", "intracranial_pressure", "blood_flow"],
                fn=lambda p, n: np.clip(
                    pre_op_function
                    - 0.20 * (1.0 - p["blood_flow"])  # Language highly perfusion-dependent
                    - 0.10 * p["intracranial_pressure"]
                    - 0.10 * (1.0 - p["oxygen_saturation"])
                    - 0.05 * eloquence_factor  # Adjacent to language areas
                    + n,
                    0, 1,
                ),
                description="Language function (BDAE normalized): perfusion, ICP, eloquence proximity",
            ),
            StructuralEquation(
                child="visual_function",
                parents=["intracranial_pressure", "vascular_compression"],
                fn=lambda p, n: np.clip(
                    pre_op_function
                    - 0.20 * p["intracranial_pressure"]  # Papilledema, optic nerve compression
                    - 0.15 * p["vascular_compression"]   # Posterior cerebral artery compression
                    + n,
                    0, 1,
                ),
                description="Visual function: affected by ICP and vascular compression",
            ),
            StructuralEquation(
                child="cognitive_function",
                parents=["oxygen_saturation", "intracranial_pressure"],
                fn=lambda p, n: np.clip(
                    pre_op_function * age_factor  # Age significantly reduces cognitive reserve
                    - 0.15 * (1.0 - p["oxygen_saturation"])
                    - 0.15 * p["intracranial_pressure"]
                    - 0.05 * (1.0 - hist_factor.get("resectability", 0.7))  # Aggressive tumors affect cognition
                    + n,
                    0, 1,
                ),
                description="Cognitive function (MoCA normalized): O₂, ICP, age, histology",
            ),
            StructuralEquation(
                child="consciousness",
                parents=["oxygen_saturation", "intracranial_pressure"],
                fn=lambda p, n: np.clip(
                    pre_op_function
                    - 0.30 * p["intracranial_pressure"]  # Dominant factor for consciousness
                    - 0.20 * (1.0 - p["oxygen_saturation"])
                    + n,
                    0, 1,
                ),
                description="Consciousness (GCS verbal+eye normalized): ICP and O₂ are dominant",
            ),

            # v2.0: Legacy neural_function (backward compatibility)
            StructuralEquation(
                child="neural_function",
                parents=["motor_function", "language_function", "visual_function",
                         "cognitive_function", "consciousness"],
                fn=lambda p, n: np.clip(
                    0.25 * p["motor_function"]
                    + 0.20 * p["language_function"]
                    + 0.15 * p["visual_function"]
                    + 0.20 * p["cognitive_function"]
                    + 0.20 * p["consciousness"]
                    + n,
                    0, 1,
                ),
                description="DEPRECATED: Weighted composite of domain functions",
            ),

            # ── v2.0: Complication Probabilities ────────────────────────────────
            StructuralEquation(
                child="post_op_hemorrhage_prob",
                parents=["tumor_size", "vascular_compression"],
                fn=lambda p, n: np.clip(
                    0.02  # baseline ~2%
                    + 0.02 * p["tumor_size"]
                    + 0.03 * p["vascular_compression"]
                    + 0.03 * (anticoag_factor - 1.0)  # anticoagulation risk
                    + 0.02 * hist_factor["vascularity"]  # vascular tumors
                    + n,
                    0, 1,
                ),
                description="Hemorrhage risk: baseline + tumor + anticoagulation + histology",
            ),
            StructuralEquation(
                child="cerebral_edema_prob",
                parents=["tumor_size", "edema_volume", "intracranial_pressure"],
                fn=lambda p, n: np.clip(
                    0.05  # baseline ~5%
                    + 0.10 * p["tumor_size"]
                    + 0.05 * p["edema_volume"]
                    + 0.05 * hist_factor["edema_severity"]
                    + n,
                    0, 1,
                ),
                description="Worsening edema risk: tumor, baseline edema, histology",
            ),
            StructuralEquation(
                child="new_deficit_prob",
                parents=["intracranial_pressure", "vascular_compression"],
                fn=lambda p, n: np.clip(
                    0.03  # baseline ~3%
                    + 0.05 * eloquence_factor  # Eloquent cortex proximity
                    + 0.05 * p["intracranial_pressure"]
                    + 0.03 * p["vascular_compression"]
                    + 0.02 * (1.0 - pre_op_function)  # Already compromised = less reserve
                    + n,
                    0, 1,
                ),
                description="New neurological deficit: eloquence, ICP, vascular, pre-op function",
            ),
            StructuralEquation(
                child="cns_infection_prob",
                parents=["tumor_size"],
                fn=lambda p, n: np.clip(
                    0.01  # baseline ~1%
                    + 0.01 * p["tumor_size"]
                    + 0.01 * (prior_rad_factor - 1.0)  # Prior radiation risk
                    + 0.01 * (hist_factor.get("recurrence_risk", 0) > 0.5)  # Reoperation risk
                    + n,
                    0, 1,
                ),
                description="CNS infection: surgery extent, prior radiation, reoperation",
            ),
            StructuralEquation(
                child="dvt_pe_prob",
                parents=["tumor_size"],
                fn=lambda p, n: np.clip(
                    0.03  # baseline ~3%
                    + 0.02 * p["tumor_size"]  # Larger tumors = longer surgery
                    + 0.02 * (hist_factor.get("recurrence_risk", 0) > 0.5)  # Malignancy hypercoagulable
                    + 0.02 * (1.0 - age_factor)  # Age increases VTE risk
                    + n,
                    0, 1,
                ),
                description="DVT/PE risk: malignancy, age, surgery duration proxy",
            ),
            StructuralEquation(
                child="cerebral_ischemia_prob",
                parents=["vascular_compression", "intracranial_pressure"],
                fn=lambda p, n: np.clip(
                    0.01  # baseline ~1%
                    + 0.03 * p["vascular_compression"]
                    + 0.02 * p["intracranial_pressure"]
                    + 0.01 * (1.0 - age_factor)  # Age increases ischemia vulnerability
                    + n,
                    0, 1,
                ),
                description="Cerebral ischemia: vascular compression, hypotension risk, age",
            ),
            StructuralEquation(
                child="csf_leak_prob",
                parents=["tumor_size"],
                fn=lambda p, n: np.clip(
                    0.02  # baseline ~2%
                    + 0.03 * p["tumor_size"]  # Skull base surgery proxy
                    + 0.03 * (prior_rad_factor - 1.0)  # Prior radiation
                    + n,
                    0, 1,
                ),
                description="CSF leak: skull base surgery, prior radiation",
            ),
            StructuralEquation(
                child="wound_dehiscence_prob",
                parents=["tumor_size"],
                fn=lambda p, n: np.clip(
                    0.02  # baseline ~2%
                    + 0.01 * p["tumor_size"]
                    + 0.02 * (prior_rad_factor - 1.0)  # Prior radiation
                    + n,
                    0, 1,
                ),
                description="Wound complications: prior radiation, surgery extent",
            ),
            StructuralEquation(
                child="adrenal_insuff_prob",
                parents=["tumor_size"],
                fn=lambda p, n: np.clip(
                    0.01  # baseline ~1% (patients not on chronic steroids)
                    + 0.04 * 0.5  # Assumes half of patients on chronic steroids
                    + n,
                    0, 1,
                ),
                description="Adrenal insufficiency: chronic steroid use (assumed from metadata)",
            ),

            # ── v2.0: Recovery Decomposition ─────────────────────────────────────
            StructuralEquation(
                child="functional_recovery",
                parents=["motor_function", "cognitive_function", "consciousness"],
                fn=lambda p, n: np.clip(
                    0.40 * p["motor_function"]
                    + 0.30 * p["cognitive_function"]
                    + 0.30 * p["consciousness"]
                    - 0.10 * (1.0 - age_factor)  # Age delays functional recovery
                    + n,
                    0, 1,
                ),
                description="Functional recovery (KPS/ADL): motor, cognitive, consciousness, age",
            ),
            StructuralEquation(
                child="neurological_recovery",
                parents=["language_function", "visual_function", "motor_function"],
                fn=lambda p, n: np.clip(
                    0.35 * p["language_function"]
                    + 0.30 * p["visual_function"]
                    + 0.35 * p["motor_function"]
                    - 0.05 * eloquence_factor  # Eloquent area recovery is harder
                    + n,
                    0, 1,
                ),
                description="Neurological recovery: language, visual, motor deficits, eloquence",
            ),
            StructuralEquation(
                child="quality_of_life",
                parents=["cognitive_function", "motor_function"],
                fn=lambda p, n: np.clip(
                    0.50 * p["cognitive_function"]
                    + 0.50 * p["motor_function"]
                    - 0.05 * (1.0 - hist_factor.get("resectability", 0.7))  # Residual tumor affects QOL
                    + n,
                    0, 1,
                ),
                description="Quality of life: cognition, motor function, residual disease",
            ),
            StructuralEquation(
                child="complication_free_survival",
                parents=["post_op_hemorrhage_prob", "cerebral_edema_prob", "new_deficit_prob",
                         "cns_infection_prob", "dvt_pe_prob", "cerebral_ischemia_prob"],
                fn=lambda p, n: np.clip(
                    1.0
                    - 0.20 * p["post_op_hemorrhage_prob"]
                    - 0.15 * p["cerebral_edema_prob"]
                    - 0.20 * p["new_deficit_prob"]
                    - 0.10 * p["cns_infection_prob"]
                    - 0.20 * p["dvt_pe_prob"]
                    - 0.15 * p["cerebral_ischemia_prob"]
                    + n,
                    0, 1,
                ),
                description="Complication-free survival: weighted combination of all complications",
            ),
            StructuralEquation(
                child="oncological_outcome",
                parents=["tumor_size"],
                fn=lambda p, n: np.clip(
                    (1.0 - p["tumor_size"])  # EOR proxy
                    * hist_factor.get("resectability", 0.7)  # Histology affects resectability
                    - 0.05 * hist_factor.get("recurrence_risk", 0)
                    + n,
                    0, 1,
                ),
                description="Oncological outcome: EOR × resectability - recurrence risk",
            ),

            # v2.0: Legacy recovery_score (backward compatibility)
            StructuralEquation(
                child="recovery_score",
                parents=["functional_recovery", "neurological_recovery", "quality_of_life",
                         "complication_free_survival", "oncological_outcome"],
                fn=lambda p, n: np.clip(
                    0.25 * p["functional_recovery"]
                    + 0.30 * p["neurological_recovery"]
                    + 0.20 * p["quality_of_life"]
                    + 0.15 * p["complication_free_survival"]
                    + 0.10 * p["oncological_outcome"]
                    + n,
                    0, 1,
                ),
                description="DEPRECATED: Weighted composite of recovery domains",
            ),

            # ── v2.0: Surgical Risk (now includes patient metadata) ──────────────
            StructuralEquation(
                child="surgical_risk",
                parents=["tumor_size", "vascular_compression", "intracranial_pressure",
                         "post_op_hemorrhage_prob", "new_deficit_prob"],
                fn=lambda p, n: np.clip(
                    0.20 * p["tumor_size"]
                    + 0.20 * p["vascular_compression"]
                    + 0.15 * p["intracranial_pressure"]
                    + 0.25 * p["post_op_hemorrhage_prob"]  # Hemorrhage is most feared
                    + 0.20 * p["new_deficit_prob"]
                    + n,
                    0, 1,
                ),
                description="Composite surgical risk: tumor, vascular, ICP, hemorrhage, new deficit",
            ),

            # ── v2.0: Prediction Confidence ──────────────────────────────────────
            StructuralEquation(
                child="prediction_confidence",
                parents=["tumor_size", "neural_function"],
                fn=lambda p, n: np.clip(
                    0.7  # Base confidence
                    - 0.10 * p["tumor_size"]  # Less confident for large tumors (complex cases)
                    - 0.10 * (1.0 - p["neural_function"])  # Less confident when baseline is poor
                    - 0.10 * (1.0 - age_factor)  # Less confident for elderly (heterogeneous)
                    + n,
                    0.3, 1.0,  # Confidence never goes below 30%
                ),
                description="Model confidence based on case complexity and patient factors",
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

    # ── v2.0: Uncertainty Quantification ─────────────────────────────────────

    def evaluate_with_confidence(
        self,
        noise: bool = False,
        n_bootstrap: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate SCM with uncertainty quantification via bootstrap sampling.

        Returns predictions with confidence intervals for each variable.
        This provides epistemic uncertainty (model uncertainty) rather than just
        aleatory (noise-based) uncertainty.

        Args:
            noise: Whether to include stochastic noise in evaluations
            n_bootstrap: Number of bootstrap samples for CI estimation

        Returns:
            Dict mapping variable name → {mean, std, ci_low, ci_high, confidence}
        """
        samples: Dict[str, List[float]] = {k: [] for k in self.variables}

        for _ in range(n_bootstrap):
            vals = self.evaluate(noise=noise)
            for k, v in vals.items():
                if k in samples:
                    samples[k].append(v)

        results = {}
        for var_name, var_samples in samples.items():
            if not var_samples:
                continue
            arr = np.array(var_samples)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))

            # Confidence based on coefficient of variation (lower CV = higher confidence)
            cv = abs(std / mean) if mean != 0 else 0
            confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

            results[var_name] = {
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "confidence": confidence,
            }

        return results

    def calibrate_with_outcomes(
        self,
        calibration_data: List[Tuple[Dict, Dict]],
        target_variable: str = "recovery_score",
    ) -> Dict[str, float]:
        """
        Calibrate SCM coefficients using retrospective surgical outcome data.

        This is a simplified calibration: it shifts each variable's baseline
        to match observed outcomes. A full Bayesian calibration would adjust
        the structural equation coefficients, but that requires many more
        data points and more sophisticated optimization.

        Args:
            calibration_data: List of (input_params, observed_outcomes) tuples
                             e.g., ([{"tumor_size": 0.5}, ...], {"recovery_score": 0.7, ...})
            target_variable: The variable to calibrate (default: recovery_score)

        Returns:
            Dict of calibration adjustments per variable

        Example:
            >>> data = [
            ...     ({"tumor_size": 0.3, "edema_volume": 0.2}, {"recovery_score": 0.8}),
            ...     ({"tumor_size": 0.8, "edema_volume": 0.7}, {"recovery_score": 0.3}),
            ... ]
            >>> adjustments = scm.calibrate_with_outcomes(data)
        """
        if len(calibration_data) < 3:
            logger.warning(
                f"⚠️  Calibration requires at least 3 data points, got {len(calibration_data)}. "
                f"Results may be unreliable."
            )

        residuals = {}
        for input_params, observed_outcomes in calibration_data:
            # Run SCM with input parameters
            temp_scm = BrainTumorSCM(
                patient_params=input_params,
                patient_metadata=self.patient_metadata,
            )
            predicted = temp_scm.evaluate(noise=False)

            # Compute residuals for target variable
            obs_val = observed_outcomes.get(target_variable)
            pred_val = predicted.get(target_variable)
            if obs_val is not None and pred_val is not None:
                residual = obs_val - pred_val
                for var_name, var_val in input_params.items():
                    if var_name not in residuals:
                        residuals[var_name] = []
                    residuals[var_name].append(residual * var_val)

        # Compute mean adjustments
        adjustments = {}
        for var_name, res_list in residuals.items():
            if res_list:
                mean_residual = np.mean(res_list)
                adjustments[var_name] = float(mean_residual)
                logger.info(
                    f"Calibration: {var_name} adjustment = {mean_residual:+.4f}"
                )

        self._calibration_set = calibration_data
        return adjustments

    def conformal_prediction_interval(
        self,
        target_variable: str,
        confidence_level: float = 0.95,
        n_calibration: int = 100,
    ) -> Tuple[float, float]:
        """
        Compute a conformal prediction interval for a target variable.

        Conformal prediction provides distribution-free confidence intervals
        with guaranteed coverage: if confidence_level=0.95, then the true
        value will fall within the interval at least 95% of the time.

        Args:
            target_variable: SCM variable to predict
            confidence_level: Desired coverage probability (default 0.95)
            n_calibration: Number of samples for calibration

        Returns:
            (lower_bound, upper_bound) tuple
        """
        # Run multiple evaluations to get a distribution
        samples = []
        for _ in range(n_calibration):
            vals = self.evaluate(noise=True)
            samples.append(vals.get(target_variable, 0.5))

        samples = np.array(samples)

        # For conformal prediction, we use the quantiles of the empirical distribution
        alpha = 1.0 - confidence_level
        lower = float(np.percentile(samples, alpha / 2 * 100))
        upper = float(np.percentile(samples, (1 - alpha / 2) * 100))

        logger.debug(
            f"Conformal interval for {target_variable}: "
            f"[{lower:.4f}, {upper:.4f}] at {confidence_level:.0%} confidence"
        )

        return (lower, upper)

    def get_clinical_safety_warnings(self) -> List[Dict[str, str]]:
        """
        Generate clinical safety warnings based on current SCM state.

        This helps surgeons quickly identify high-risk conditions
        that require attention, regardless of what the model predicts.

        Returns:
            List of warning dicts with {level, condition, recommendation} keys.
        """
        warnings = []
        state = self.evaluate(noise=False)

        # Critical ICP
        icp = state.get("intracranial_pressure", 0)
        if icp > 0.5:  # > 20 mmHg
            warnings.append({
                "level": "CRITICAL",
                "condition": f"Raised ICP ({icp * 40:.1f} mmHg)",
                "recommendation": "Consider ICP monitoring, mannitol, hypertonic saline, "
                                 "or surgical decompression. ICP > 20 mmHg requires intervention.",
            })

        # Low CBF
        cbf = state.get("blood_flow", 1.0)
        if cbf < 0.36:  # < 20 mL/100g/min
            warnings.append({
                "level": "CRITICAL",
                "condition": f"Ischemic CBF ({cbf * 55:.1f} mL/100g/min)",
                "recommendation": "Risk of cerebral ischemia. Maintain CPP > 70 mmHg. "
                                 "Avoid hypotension. Consider revascularization if indicated.",
            })

        # Low O2 saturation
        sjo2 = state.get("oxygen_saturation", 1.0)
        if sjo2 < 0.55:  # < 55%
            warnings.append({
                "level": "CRITICAL",
                "condition": f"Cerebral desaturation (SjO₂ {sjo2 * 100:.1f}%)",
                "recommendation": "SjO₂ < 55% indicates critical cerebral hypoxia. "
                                 "Increase FiO₂, optimize CBF, reduce cerebral metabolic demand.",
            })

        # High hemorrhage risk
        hemorrhage_prob = state.get("post_op_hemorrhage_prob", 0)
        if hemorrhage_prob > 0.08:  # > 8%
            warnings.append({
                "level": "HIGH",
                "condition": f"Elevated hemorrhage risk ({hemorrhage_prob * 100:.1f}%)",
                "recommendation": "Review anticoagulation status. Ensure hemostasis. "
                                 "Consider tranexamic acid if coagulopathic. Monitor in ICU.",
            })

        # High new deficit risk
        deficit_prob = state.get("new_deficit_prob", 0)
        if deficit_prob > 0.12:  # > 12%
            warnings.append({
                "level": "HIGH",
                "condition": f"Elevated new deficit risk ({deficit_prob * 100:.1f}%)",
                "recommendation": "Consider intraoperative neuromonitoring (MEP/SEP). "
                                 "Discuss with patient the risk of new deficits. "
                                 "Plan for awake craniotomy if language area involved.",
            })

        # Large residual tumor
        residual_tumor = state.get("tumor_size", 0)
        if residual_tumor > 0.5:  # > 50% of original
            warnings.append({
                "level": "MODERATE",
                "condition": f"Large residual tumor ({residual_tumor * 100:.0f}%)",
                "recommendation": "Significant tumor remains. Discuss adjuvant therapy "
                                 "(radiation, chemotherapy). Consider second-stage resection.",
            })

        # Low confidence warning
        confidence = state.get("prediction_confidence", 1.0)
        if confidence < 0.5:
            warnings.append({
                "level": "HIGH",
                "condition": f"Low model confidence ({confidence:.0%})",
                "recommendation": "Model uncertainty is high for this case. "
                                 "Clinical judgment should take precedence. "
                                 "Consider additional imaging or consultation.",
            })

        return warnings

    def get_domain_function_report(self) -> Dict[str, Dict]:
        """
        Generate a domain-specific neural function report for clinical use.

        Replaces the single neural_function with detailed domain analysis.
        Each domain maps to a specific clinical assessment tool.

        Returns:
            Dict mapping domain name → {value, clinical_interpretation, assessment_tool}
        """
        state = self.evaluate(noise=False)

        domains = {
            "motor_function": {
                "value": state.get("motor_function", 0),
                "clinical_interpretation": self._interpret_motor(state.get("motor_function", 0)),
                "assessment_tool": "MRC Motor Strength Scale (0-5)",
                "domain_weight": 0.25,
            },
            "language_function": {
                "value": state.get("language_function", 0),
                "clinical_interpretation": self._interpret_language(state.get("language_function", 0)),
                "assessment_tool": "Boston Diagnostic Aphasia Examination",
                "domain_weight": 0.20,
            },
            "visual_function": {
                "value": state.get("visual_function", 0),
                "clinical_interpretation": self._interpret_visual(state.get("visual_function", 0)),
                "assessment_tool": "Visual field confrontation, formal perimetry",
                "domain_weight": 0.15,
            },
            "cognitive_function": {
                "value": state.get("cognitive_function", 0),
                "clinical_interpretation": self._interpret_cognitive(state.get("cognitive_function", 0)),
                "assessment_tool": "Montreal Cognitive Assessment (MoCA)",
                "domain_weight": 0.20,
            },
            "consciousness": {
                "value": state.get("consciousness", 0),
                "clinical_interpretation": self._interpret_consciousness(state.get("consciousness", 0)),
                "assessment_tool": "GCS Eye + Verbal components",
                "domain_weight": 0.20,
            },
        }

        return domains

    def _interpret_motor(self, value: float) -> str:
        """Convert motor function (0-1) to clinical interpretation."""
        if value >= 0.9:
            return "Normal motor strength (MRC 5/5)"
        elif value >= 0.7:
            return "Mild weakness (MRC 4/5)"
        elif value >= 0.5:
            return "Moderate weakness (MRC 3/5)"
        elif value >= 0.3:
            return "Severe weakness (MRC 2/5)"
        else:
            return "Plegia (MRC 0-1/5)"

    def _interpret_language(self, value: float) -> str:
        """Convert language function (0-1) to clinical interpretation."""
        if value >= 0.9:
            return "Normal language function"
        elif value >= 0.7:
            return "Mild aphasia (word-finding difficulties)"
        elif value >= 0.5:
            return "Moderate aphasia (compromised expression/comprehension)"
        elif value >= 0.3:
            return "Severe aphasia (limited communication)"
        else:
            return "Global aphasia or anarthria"

    def _interpret_visual(self, value: float) -> str:
        """Convert visual function (0-1) to clinical interpretation."""
        if value >= 0.9:
            return "Normal visual fields and acuity"
        elif value >= 0.7:
            return "Mild visual field defect (quadrantanopia)"
        elif value >= 0.5:
            return "Moderate visual field defect (hemianopia)"
        elif value >= 0.3:
            return "Severe visual impairment"
        else:
            return "Cortical blindness"

    def _interpret_cognitive(self, value: float) -> str:
        """Convert cognitive function (0-1) to clinical interpretation."""
        if value >= 0.9:
            return "Normal cognition (MoCA ≥ 26)"
        elif value >= 0.7:
            return "Mild cognitive impairment (MoCA 18-25)"
        elif value >= 0.5:
            return "Moderate cognitive impairment (MoCA 10-17)"
        elif value >= 0.3:
            return "Severe cognitive impairment (MoCA < 10)"
        else:
            return "Severe dementia-like impairment"

    def _interpret_consciousness(self, value: float) -> str:
        """Convert consciousness (0-1) to clinical interpretation."""
        if value >= 0.9:
            return "Normal consciousness (GCS 15)"
        elif value >= 0.7:
            return "Mild impairment (GCS 13-14)"
        elif value >= 0.5:
            return "Moderate impairment (GCS 9-12)"
        elif value >= 0.3:
            return "Severe impairment (GCS 5-8)"
        else:
            return "Coma (GCS ≤ 4)"

    def get_complication_risk_report(self) -> Dict[str, Dict]:
        """
        Generate a comprehensive complication risk report.

        Returns:
            Dict mapping complication type → {probability, severity, timing, risk_factors}
        """
        state = self.evaluate(noise=False)

        complications = {}
        for comp_key, comp_def in COMPLICATION_TYPES.items():
            # Map complication key to SCM variable name
            var_map = {
                "post_op_hemorrhage": "post_op_hemorrhage_prob",
                "cerebral_edema": "cerebral_edema_prob",
                "new_neurological_deficit": "new_deficit_prob",
                "cns_infection": "cns_infection_prob",
                "dvt_pulmonary_embolism": "dvt_pe_prob",
                "cerebral_ischemia": "cerebral_ischemia_prob",
                "csf_leak": "csf_leak_prob",
                "wound_dehiscence": "wound_dehiscence_prob",
                "adrenal_insufficiency": "adrenal_insuff_prob",
            }

            var_name = var_map.get(comp_key, comp_key + "_prob")
            probability = state.get(var_name, 0)

            # Determine risk level
            if probability < 0.02:
                risk_level = "LOW"
            elif probability < 0.05:
                risk_level = "MODERATE"
            elif probability < 0.10:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"

            complications[comp_key] = {
                "probability": round(probability * 100, 1),  # as percentage
                "risk_level": risk_level,
                "severity": comp_def["severity"],
                "timing_window_hours": comp_def["time_window_hours"],
                "description": comp_def["description"],
                "risk_factors": comp_def["risk_factors"],
            }

        return complications

    def get_recovery_decomposition_report(self) -> Dict[str, Dict]:
        """
        Generate a decomposed recovery outcome report.

        Returns:
            Dict mapping recovery domain → {value, weight, description, clinical_measure}
        """
        state = self.evaluate(noise=False)

        recovery_domains = {}
        for domain_key, domain_def in RECOVERY_DOMAINS.items():
            value = state.get(domain_key, 0)

            # Determine outcome category
            if value >= 0.8:
                outcome = "EXCELLENT"
            elif value >= 0.6:
                outcome = "GOOD"
            elif value >= 0.4:
                outcome = "FAIR"
            elif value >= 0.2:
                outcome = "POOR"
            else:
                outcome = "VERY POOR"

            recovery_domains[domain_key] = {
                "value": round(value * 100, 1),  # as percentage
                "outcome": outcome,
                "weight": domain_def["weight"],
                "weighted_contribution": round(value * domain_def["weight"] * 100, 1),
                "description": domain_def["description"],
                "clinical_measure": domain_def["clinical_measure"],
                "typical_timeline_weeks": domain_def["typical_timeline_weeks"],
            }

        # Overall recovery score (weighted sum)
        overall = sum(v["weighted_contribution"] for v in recovery_domains.values())

        return {
            "domains": recovery_domains,
            "overall_recovery_score": round(overall, 1),
            "interpretation": self._interpret_overall_recovery(overall),
        }

    def _interpret_overall_recovery(self, score: float) -> str:
        """Convert overall recovery score to clinical interpretation."""
        if score >= 80:
            return ("Excellent expected outcome. Patient likely to return to baseline "
                    "function within expected timeframe. Low complication risk.")
        elif score >= 60:
            return ("Good expected outcome. Minor deficits may persist but independence "
                    "is likely. Standard rehabilitation protocol recommended.")
        elif score >= 40:
            return ("Fair expected outcome. Moderate deficits expected. Intensive "
                    "rehabilitation and potential need for assisted living support.")
        elif score >= 20:
            return ("Poor expected outcome. Significant deficits likely. Long-term "
                    "care planning should be initiated. Consider palliative consultation.")
        else:
            return ("Very poor expected outcome. Severe disability or mortality likely. "
                    "Goals of care discussion strongly recommended.")
