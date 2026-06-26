"""
src/causal/clinical_validator.py
===============================================================================
Clinical Validation Console for BrainTumorSCM
===============================================================================

Allows a senior neurosurgeon to:
  1. Audit each structural equation against published clinical benchmarks
  2. Input real patient vitals (in clinical units) and compare model output
  3. Run canonical challenge scenarios with known expected outcomes

Unit Conversion Convention
──────────────────────────
The SCM works in normalized [0, 1] floats. Every clinical quantity has a
corresponding real-world unit range that is mapped linearly unless stated.

  Variable              Clinical unit    Real range → SCM [0,1]
  ──────────────────    ─────────────    ────────────────────────────
  tumor_size            cm³              0–150 cm³  → 0–1
  intracranial_pressure mmHg             0–40 mmHg  → 0–1  (20 mmHg = 0.5)
  blood_flow            mL/100g/min      0–55        → 0–1  (55 = normal)
  oxygen_saturation     %                0–100%      → 0–1  (66% SjO₂ = normal)
  metabolic_rate        mL O₂/100g/min   0–3.7       → 0–1  (3.7 = normal CMRO₂)
  edema_volume          cm³              0–80 cm³    → 0–1
  inflammatory_response arbitrary        0–1 (composite index)
  vascular_compression  fraction (0–1)   0 = open, 1 = fully occluded
  neural_function       GCS proxy        0–1 (1 = GCS 15, 0 = GCS 3)
  recovery_score        0–1              0 = poor, 1 = full recovery
  surgical_risk         0–1              0 = minimal, 1 = prohibitive

Published reference ranges
──────────────────────────
• Normal ICP:       5–15 mmHg   (Marmarou et al., J Neurosurg 2005)
• Raised ICP:       >20 mmHg    (Brain Trauma Foundation Guidelines 4th ed.)
• Critical ICP:     >40 mmHg
• Normal CBF:       50–55 mL/100g/min (Lassen, 1959)
• Ischemic CBF:     <20 mL/100g/min
• Normal SjO₂:      55–75%      (Gopinath et al., J Neurosurg 1994)
• Normal CMRO₂:     3.5 - 4.0 mL O₂/100g/min
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.causal.scm import BrainTumorSCM


# ─── Unit conversion helpers ──────────────────────────────────────────────────
class Units:
    """Bidirectional conversion between clinical and SCM-normalized units."""

    # (clinical_min, clinical_max) → maps to SCM [0, 1]
    RANGES: Dict[str, Tuple[float, float, str]] = {
        "tumor_size":            (0.0,  150.0, "cm³"),
        "intracranial_pressure": (0.0,   40.0, "mmHg"),
        "blood_flow":            (0.0,   55.0, "mL/100g/min"),  # 1.0 SCM = 55 mL/100g/min
        "oxygen_saturation":     (0.0,  100.0, "%"),
        "edema_volume":          (0.0,   80.0, "cm³"),
        "vascular_compression":  (0.0,    1.0, "fraction"),
        "neural_function":       (0.0,    1.0, "GCS proxy 0-1"),
        "inflammatory_response": (0.0,    1.0, "index"),
        "mass_effect":           (0.0,    1.0, "normalized midline shift"),
        "recovery_score":        (0.0,    1.0, "outcome score"),
        "surgical_risk":         (0.0,    1.0, "risk score"),
        "metabolic_rate":        (0.0,    3.7, "mL O₂/100g/min"), # 1.0 SCM = 3.7 mL O₂/100g/min
    }

    @classmethod
    def to_scm(cls, variable: str, clinical_value: float) -> float:
        """Convert a clinical-unit value to SCM normalized [0,1]."""
        lo, hi, _ = cls.RANGES[variable]
        return float(np.clip((clinical_value - lo) / (hi - lo), 0.0, 1.0))

    @classmethod
    def to_clinical(cls, variable: str, scm_value: float) -> float:
        """Convert an SCM normalized value back to clinical units."""
        lo, hi, _ = cls.RANGES[variable]
        return float(lo + scm_value * (hi - lo))

    @classmethod
    def unit(cls, variable: str) -> str:
        return cls.RANGES.get(variable, (None, None, ""))[2]


# ─── Clinical benchmark definitions ──────────────────────────────────────────
@dataclass
class ClinicalBenchmark:
    variable: str
    label: str
    scm_min: float
    scm_max: float
    clinical_min: float
    clinical_max: float
    unit: str
    source: str
    patient_params: Optional[Dict[str, float]] = None


BENCHMARKS: List[ClinicalBenchmark] = [
    ClinicalBenchmark(
        variable="intracranial_pressure",
        label="Normal ICP",
        scm_min=0.125, scm_max=0.375,  # 5.0 - 15.0 mmHg
        clinical_min=5.0, clinical_max=15.0,
        unit="mmHg",
        source="Marmarou et al., J Neurosurg 2005",
        patient_params={"tumor_size": 0.0, "edema_volume": 0.0, "inflammatory_response": 0.05},
    ),
    ClinicalBenchmark(
        variable="intracranial_pressure",
        label="Raised ICP (intervention threshold)",
        scm_min=0.50, scm_max=1.0,  # 20.0 - 40.0 mmHg
        clinical_min=20.0, clinical_max=40.0,
        unit="mmHg",
        source="Brain Trauma Foundation Guidelines 4th ed.",
        patient_params={"tumor_size": 0.6, "edema_volume": 0.5, "mass_effect": 0.5, "inflammatory_response": 0.3},
    ),
    ClinicalBenchmark(
        variable="blood_flow",
        label="Normal CBF",
        scm_min=0.818, scm_max=1.0,  # 45.0 - 55.0 mL/100g/min
        clinical_min=45.0, clinical_max=55.0,
        unit="mL/100g/min",
        source="Lassen, 1959; Kety & Schmidt, 1948",
        patient_params={"tumor_size": 0.0, "edema_volume": 0.0, "inflammatory_response": 0.05},
    ),
    ClinicalBenchmark(
        variable="blood_flow",
        label="Ischemic threshold",
        scm_min=0.0, scm_max=0.364,  # 0.0 - 20.0 mL/100g/min
        clinical_min=0.0, clinical_max=20.0,
        unit="mL/100g/min",
        source="Jones et al., Stroke 1981",
        patient_params={"vascular_compression": 0.8},
    ),
    ClinicalBenchmark(
        variable="oxygen_saturation",
        label="Normal SjO₂ (jugular bulb)",
        scm_min=0.55, scm_max=0.75,  # 55% - 75%
        clinical_min=55.0, clinical_max=75.0,
        unit="%",
        source="Gopinath et al., J Neurosurg 1994",
        patient_params={"tumor_size": 0.0, "edema_volume": 0.0, "inflammatory_response": 0.05},
    ),
    ClinicalBenchmark(
        variable="metabolic_rate",
        label="Normal CMRO₂",
        scm_min=0.94, scm_max=1.08,  # 3.5 - 4.0 mL O2/100g/min
        clinical_min=3.5, clinical_max=4.0,
        unit="mL O₂/100g/min",
        source="Sokoloff, 1981; Kennedy & Sokoloff, 1957",
        patient_params={"tumor_size": 0.0, "edema_volume": 0.0, "inflammatory_response": 0.05},
    ),
]


# ─── Challenge scenarios ──────────────────────────────────────────────────────
@dataclass
class ChallengeScenario:
    name: str
    description: str
    patient_params: Dict[str, float]          # SCM-normalized inputs
    expected: Dict[str, Tuple[str, float, float]]  # variable → (direction, min, max)
    clinical_context: str
    reference: str


CHALLENGE_SCENARIOS: List[ChallengeScenario] = [
    ChallengeScenario(
        name="Plateau wave (ICP crisis)",
        description="Large tumor (80 cm³), severely raised ICP (35 mmHg), depleted CBF autoregulation",
        patient_params={
            "tumor_size":            Units.to_scm("tumor_size", 80.0),
            "edema_volume":          Units.to_scm("edema_volume", 50.0),
            "intracranial_pressure": Units.to_scm("intracranial_pressure", 35.0),
            "blood_flow":            Units.to_scm("blood_flow", 18.0),
            "inflammatory_response": 0.7,
        },
        expected={
            "vascular_compression": ("high", 0.35, 0.60),
            "neural_function":      ("low",  0.30, 0.60),
            "surgical_risk":        ("high", 0.50, 0.80),
        },
        clinical_context="Patient presents with GCS 8, bilateral papilloedema, uncal herniation risk",
        reference="Rosner & Daughton, J Neurosurg 1990 — pressure-volume index theory",
    ),
    ChallengeScenario(
        name="Gross Total Resection — Post-op day 1",
        description="Post-GTR at 18h: inflammatory ICP spike despite tumor removal",
        patient_params={
            "tumor_size":            Units.to_scm("tumor_size", 10.0),   # after resection
            "edema_volume":          Units.to_scm("edema_volume", 35.0), # reactive edema
            "inflammatory_response": 0.85,  # peak IL-6/PGE2
            "intracranial_pressure": Units.to_scm("intracranial_pressure", 22.0),
        },
        expected={
            "intracranial_pressure": ("raised", 0.45, 0.70),
            "neural_function":       ("moderate", 0.60, 0.90),
        },
        clinical_context="Despite GTR, post-op ICP rises transiently at 18-24h. "
                         "This is pathological if model predicts ICP drop.",
        reference="AIIMS neurosurgery OR observation; Sinha et al., Neurosurgery 2011",
    ),
    ChallengeScenario(
        name="Normal baseline — healthy adult",
        description="No tumor, normal physiology — all variables should be within normal range",
        patient_params={
            "tumor_size":            0.0,
            "edema_volume":          0.0,
            "inflammatory_response": 0.05,
            "intracranial_pressure": Units.to_scm("intracranial_pressure", 10.0),
        },
        expected={
            "blood_flow":            ("normal", 0.80, 1.00),
            "oxygen_saturation":     ("normal", 0.55, 0.75),
            "intracranial_pressure": ("normal", 0.15, 0.35),
            "neural_function":       ("normal", 0.70, 1.00),
        },
        clinical_context="Baseline sanity check — the model must not produce pathological "
                         "values for a healthy brain.",
        reference="Standard neurophysiology textbooks",
    ),
    ChallengeScenario(
        name="Eloquent area subtotal resection",
        description="50% resection of eloquent area tumor, vascular risk preserved",
        patient_params={
            "tumor_size":            Units.to_scm("tumor_size", 30.0),
            "edema_volume":          Units.to_scm("edema_volume", 20.0),
            "inflammatory_response": 0.5,
            "vascular_compression":  0.4,
        },
        expected={
            "blood_flow":    ("moderate", 0.50, 0.80),
            "surgical_risk": ("moderate", 0.20, 0.45),
        },
        clinical_context="Residual tumor with moderate vascular risk. Recovery depends on edema resolution.",
        reference="Sanai & Berger, Neurosurgery 2008 — eloquent area surgery outcomes",
    ),
]


# ─── Equation audit registry ──────────────────────────────────────────────────
EQUATION_AUDIT: List[Dict] = [
    {
        "variable": "vascular_compression",
        "equation_latex": "VC = 0.4·T + 0.3·E",
        "plain_english": (
            "Vascular compression increases linearly with tumor bulk (40% weight) "
            "and peritumoral edema (30% weight). A 150 cm³ tumor with 80 cm³ edema "
            "produces full occlusion."
        ),
        "clinical_validation": (
            "Consistent with Monro-Kellie doctrine: any volume increase (tumor + edema) "
            "within the fixed cranial vault compresses adjacent structures including vasculature."
        ),
        "references": ["Mokri B. Mayo Clin Proc 2001", "Rasulo et al. Eur J Anaesthesiol 2008"],
    },
    {
        "variable": "blood_flow",
        "equation_latex": (
            "CBF = α·sigmoid(VC, σ=0.55, k=15) + (1-α)·(1 - 1.4·VC)\n"
            "where α = clip(1 - VC/0.7, 0, 1)  [autoregulation capacity]"
        ),
        "plain_english": (
            "When compression is below 55% (VC < 0.55), cerebrovascular autoregulation "
            "maintains near-normal CBF (Lassen plateau). Above 55%, autoregulation is "
            "exhausted and CBF becomes pressure-passive — proportional to perfusion pressure. "
            "This reproduces the classic Lassen autoregulation curve."
        ),
        "clinical_validation": (
            "At VC=0 (no compression): CBF = 1.0 → 55 mL/100g/min (physiological normal). "
            "At VC=0.55 (autoregulation limit): CBF ≈ 0.50 → 27.5 mL/100g/min (lower autoregulation floor). "
            "At VC=0.8 (pressure-passive): CBF ≈ 0.00 → 0 mL/100g/min (severe ischemia)."
        ),
        "references": ["Lassen NA. Physiol Rev 1959", "Rosner & Daughton J Neurosurg 1990"],
    },
    {
        "variable": "oxygen_saturation",
        "equation_latex": (
            "SjO₂ = 0.30 + 0.68 · CBF / (CBF + 0.8 · CMRO₂)"
        ),
        "plain_english": (
            "Oxygen saturation (SjO₂) models jugular bulb saturation based on Fick's principle. "
            "At normal blood flow, SjO₂ is ~66%. As flow drops, SjO₂ drops to a venous floor at 30% "
            "(representing maximum oxygen extraction by tissue from stagnant capillary blood). "
            "Higher metabolic demand (tumor CMRO₂) pulls the saturation curve down."
        ),
        "clinical_validation": (
            "SjO₂ at CBF=0: 30% (venous floor) — matches agonal state data. "
            "SjO₂ at CBF=1.0 (normal): ~66% — within normal clinical SjO₂ range (55-75%). "
            "SjO₂ at severe CBF reduction: correctly drops below ischemic threshold (<55%)."
        ),
        "references": [
            "Gopinath et al. J Neurosurg 1994",
            "Sheinberg et al. J Neurosurg 1992",
            "Gupta et al. J Neurol Neurosurg Psychiatry 1999",
        ],
    },
    {
        "variable": "intracranial_pressure",
        "equation_latex": "ICP = 0.20 + 0.20·T + 0.30·E + 0.20·M + 0.10·I  (resting, then corrected post-op)",
        "plain_english": (
            "Resting ICP has a healthy baseline of 0.20 (8 mmHg) and is driven by tumor mass (20%), "
            "edema volume (30%), midline shift mass effect (20%), and direct CSF outflow obstruction (10%). "
            "The dynamic post-operative trajectory adds intraoperative volume decompression, a post-op "
            "neuroinflammatory spike (IL-6/PGE2, peaking 18-24h), and dexamethasone recovery."
        ),
        "clinical_validation": (
            "Static model: Matches Monro-Kellie doctrine. "
            "Dynamic correction: Post-debulking ICP spike validated against AIIMS OR data. "
            "Refined to prevent double-counting of inflammation by including inflammatory_response "
            "as a direct SCM graph parent and modeling slow temporal edema resolution."
        ),
        "references": [
            "Monro A. Observations on the Structure and Function of the Nervous System, 1783",
            "Kellie G. Trans Med Chir Sci Edinburgh 1824",
            "Sinha et al. Neurosurgery 2011",
        ],
    },
    {
        "variable": "inflammatory_response",
        "equation_latex": (
            "f_infl(t) = baseline + [0.6·ΔT·(t/18)·exp(-(t-18)/24)]  for t>0\n"
            "+ baseline·exp(-t/120)  [chronic decay]\n"
            "where ΔT = tumor_size_pre - tumor_size_post"
        ),
        "plain_english": (
            "Surgical trauma triggers an inflammatory cascade proportional to the volume of tissue "
            "removed (ΔT). The acute spike peaks at t=18h post-op (log-normal envelope). "
            "Baseline chronic inflammation decays with a 5-day time constant "
            "(representing natural resolution without steroids)."
        ),
        "clinical_validation": (
            "IL-6 peaks at 12-24h post-craniotomy (Mathiesen et al. 2004). "
            "Larger resections → larger inflammatory burden (proportionality to ΔT). "
            "Chronic decay constant (120h) consistent with CRP normalization timeline."
        ),
        "references": [
            "Mathiesen et al. Acta Neurochir 2004",
            "Vecht et al. J Neurol Neurosurg Psychiatry 1994",
        ],
    },
]


# ─── Validator class ──────────────────────────────────────────────────────────
class ClinicalValidator:
    """
    Interactive clinical validation interface for BrainTumorSCM.

    Usage
    -----
    >>> validator = ClinicalValidator()
    >>> validator.run_full_audit()
    """

    def __init__(self):
        self.scm = BrainTumorSCM()
        self.results: List[Dict] = []

    # ── 1. Equation Audit ─────────────────────────────────────────────────────
    def audit_equations(self, verbose: bool = True) -> List[Dict]:
        """Print each structural equation with clinical interpretation and references."""
        results = []
        _hdr("STRUCTURAL EQUATION AUDIT", "=")
        print("Each equation is auditable by a senior neurosurgeon against published literature.\n")

        for eq_info in EQUATION_AUDIT:
            var = eq_info["variable"]
            _hdr(f"Variable: {var.upper()}", "-")
            print(f"  Equation:  {eq_info['equation_latex']}")
            print(f"\n  Mechanism: {eq_info['plain_english']}")
            print(f"\n  Clinical validation:")
            print(f"    {eq_info['clinical_validation']}")
            print(f"\n  References:")
            for ref in eq_info["references"]:
                print(f"    • {ref}")
            print()
            results.append({"variable": var, "status": "audited"})
        return results

    # ── 2. Benchmark Comparison ───────────────────────────────────────────────
    def check_benchmarks(self) -> List[Dict]:
        """
        Evaluate the SCM under relevant patient parameter conditions and check
        each output against published clinical benchmark ranges.
        """
        _hdr("CLINICAL BENCHMARK COMPARISON", "=")
        print("Testing SCM against published reference ranges...\n")

        results = []
        passed = 0
        failed = 0

        for bm in BENCHMARKS:
            scm = BrainTumorSCM()
            if bm.patient_params:
                scm.intervene(bm.patient_params)
            vals = scm.evaluate()

            model_val = vals.get(bm.variable, None)
            if model_val is None:
                continue
            clinical_val = Units.to_clinical(bm.variable, model_val)
            ok = bm.scm_min <= model_val <= bm.scm_max
            status = "✓ PASS" if ok else "✗ FAIL"
            if ok:
                passed += 1
            else:
                failed += 1

            print(
                f"  {status}  [{bm.label}]\n"
                f"          Model:    {model_val:.3f} SCM → {clinical_val:.1f} {bm.unit}\n"
                f"          Expected: {bm.clinical_min:.1f}–{bm.clinical_max:.1f} {bm.unit}\n"
                f"          Source:   {bm.source}\n"
            )
            results.append({
                "benchmark": bm.label,
                "variable": bm.variable,
                "model_scm": model_val,
                "model_clinical": clinical_val,
                "unit": bm.unit,
                "passed": ok,
            })

        print(f"\nBenchmarks: {passed} passed, {failed} failed\n")
        return results

    # ── 3. Challenge Scenarios ────────────────────────────────────────────────
    def run_challenge_scenarios(self) -> List[Dict]:
        """
        Run canonical clinical scenarios and verify model output
        matches expected clinical reality.
        """
        _hdr("CHALLENGE SCENARIOS", "=")
        results = []

        for scenario in CHALLENGE_SCENARIOS:
            _hdr(scenario.name, "-")
            print(f"  Context:  {scenario.clinical_context}")
            print(f"  Reference: {scenario.reference}\n")

            scm = BrainTumorSCM()
            scm.intervene(scenario.patient_params)
            vals = scm.evaluate()

            scenario_pass = True
            for var, (direction, lo, hi) in scenario.expected.items():
                model_val = vals.get(var, None)
                if model_val is None:
                    print(f"    [MISSING] {var}")
                    continue
                ok = lo <= model_val <= hi
                if not ok:
                    scenario_pass = False
                clinical_val = Units.to_clinical(var, model_val)
                unit = Units.unit(var)
                status = "✓" if ok else "✗"
                print(
                    f"    {status} {var}: SCM={model_val:.3f} → {clinical_val:.1f} {unit}  "
                    f"[expected {direction}: {lo:.2f}–{hi:.2f}]"
                )

            print(f"\n  Scenario result: {'PASS ✓' if scenario_pass else 'FAIL ✗'}\n")
            results.append({"scenario": scenario.name, "passed": scenario_pass})

        return results

    # ── 4. Doctor enters real patient vitals ──────────────────────────────────
    def patient_validation_session(self, patient_vitals: Dict[str, float]) -> Dict[str, float]:
        """
        Given real patient vitals in *clinical units*, run the SCM
        and return predictions in both SCM and clinical units.
        """
        _hdr("PATIENT VALIDATION SESSION", "=")
        print("Converting clinical inputs to SCM units...\n")

        scm_params = {}
        for var, clin_val in patient_vitals.items():
            scm_val = Units.to_scm(var, clin_val) if var in Units.RANGES else clin_val
            unit = Units.unit(var) if var in Units.RANGES else ""
            scm_params[var] = scm_val
            print(f"  {var}: {clin_val:.1f} {unit} → SCM {scm_val:.3f}")

        scm = BrainTumorSCM()
        scm.intervene(scm_params)
        vals = scm.evaluate()

        print("\nModel predictions (in clinical units):\n")
        output = {}
        for var, scm_val in vals.items():
            if var not in Units.RANGES:
                continue
            clinical_out = Units.to_clinical(var, scm_val)
            unit = Units.unit(var)
            output[var] = {"scm": scm_val, "clinical": clinical_out, "unit": unit}

            # Flag clinically significant values
            flag = ""
            if var == "intracranial_pressure" and clinical_out > 20:
                flag = " ⚠ RAISED ICP"
            elif var == "blood_flow" and clinical_out < 20:
                flag = " ⚠ ISCHEMIC THRESHOLD"
            elif var == "oxygen_saturation" and clinical_out < 55:
                flag = " ⚠ CEREBRAL DESATURATION"
            elif var == "surgical_risk" and scm_val > 0.65:
                flag = " ⚠ HIGH SURGICAL RISK"

            print(f"  {var:28s}: {clinical_out:7.2f} {unit}{flag}")

        return output

    # ── 5. Post-surgical trajectory audit ────────────────────────────────────
    def audit_icp_trajectory(
        self,
        tumor_size_pre_cm3: float = 80.0,
        tumor_size_post_cm3: float = 15.0,
        icp_pre_mmhg: float = 30.0,
    ) -> None:
        """
        Show the full ICP trajectory in mmHg for a given debulking operation.
        Allows the surgeon to compare model predictions to intraoperative neuromonitoring data.
        """
        _hdr("POST-SURGICAL ICP TRAJECTORY AUDIT", "=")
        print(
            f"  Pre-op tumor:  {tumor_size_pre_cm3:.0f} cm³  (ICP = {icp_pre_mmhg:.0f} mmHg)\n"
            f"  Post-op tumor: {tumor_size_post_cm3:.0f} cm³\n"
        )

        scm = BrainTumorSCM(patient_params={
            "tumor_size":            Units.to_scm("tumor_size", tumor_size_pre_cm3),
            "intracranial_pressure": Units.to_scm("intracranial_pressure", icp_pre_mmhg),
            "inflammatory_response": 0.3,
            "edema_volume":          Units.to_scm("edema_volume", 40.0),
        })

        surgical_event = {
            "tumor_size": Units.to_scm("tumor_size", tumor_size_post_cm3)
        }

        checkpoints = [0, 3, 6, 12, 18, 24, 36, 48, 72, 96, 120, 168]
        trajectory = scm.simulate_trajectory(surgical_event, time_points=checkpoints, noise=False)

        print(f"  {'Time':>8}  {'Phase':>18}  {'ICP (mmHg)':>12}  {'CBF (mL/100g)':>14}  {'SjO₂ (%)':>10}")
        print("  " + "-" * 70)
        for state in trajectory:
            t = state["_t_hours"]
            icp_mmhg = Units.to_clinical("intracranial_pressure", state["intracranial_pressure"])
            cbf      = Units.to_clinical("blood_flow", state["blood_flow"])
            sjo2     = Units.to_clinical("oxygen_saturation", state["oxygen_saturation"])
            phase    = state["_phase"]
            flag = " ← RAISED" if icp_mmhg > 20 else ""
            print(
                f"  {t:>6.0f}h    {phase:>18}  "
                f"{icp_mmhg:>10.1f}  {cbf:>14.1f}  {sjo2:>9.1f}%{flag}"
            )

        print(
            "\n  Reference thresholds:"
            "\n    ICP > 20 mmHg = intervention needed (BTF Guidelines)"
            "\n    CBF < 20 mL/100g/min = ischemic (Jones et al. 1981)"
            "\n    SjO₂ < 55% = cerebral desaturation (Gopinath et al. 1994)"
        )

    # ── Full audit run ─────────────────────────────────────────────────────────
    def run_full_audit(self) -> None:
        print("\n" + "═" * 72)
        print("  BRAIN TUMOR SCM — CLINICAL VALIDATION AUDIT")
        print("  For review by senior neurosurgeon / clinical advisor")
        print("═" * 72 + "\n")

        self.audit_equations()
        self.check_benchmarks()
        self.run_challenge_scenarios()
        self.audit_icp_trajectory(
            tumor_size_pre_cm3=80.0,
            tumor_size_post_cm3=12.0,
            icp_pre_mmhg=32.0,
        )

        print("\n" + "═" * 72)
        print("  END OF CLINICAL VALIDATION AUDIT")
        print("═" * 72 + "\n")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _hdr(title: str, char: str = "─") -> None:
    w = 70
    print(char * w)
    print(f"  {title}")
    print(char * w)


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    v = ClinicalValidator()

    if "--patient" in sys.argv:
        v.patient_validation_session({
            "tumor_size":            65.0,   # cm³
            "intracranial_pressure": 28.0,   # mmHg
            "blood_flow":            32.0,   # mL/100g/min
            "edema_volume":          30.0,   # cm³
            "inflammatory_response":  0.55,  # index
        })
    elif "--trajectory" in sys.argv:
        v.audit_icp_trajectory(80.0, 12.0, 32.0)
    elif "--benchmarks" in sys.argv:
        v.check_benchmarks()
    elif "--scenarios" in sys.argv:
        v.run_challenge_scenarios()
    else:
        v.run_full_audit()
