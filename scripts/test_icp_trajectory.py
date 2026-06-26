#!/usr/bin/env python3
"""
scripts/test_icp_trajectory.py
Validates the three-phase post-debulking ICP model against clinical expectations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.causal.scm import BrainTumorSCM

def main():
    scm = BrainTumorSCM(patient_params={
        "tumor_size":            0.75,
        "inflammatory_response": 0.3,
        "edema_volume":          0.4,
        "intracranial_pressure": 0.55,
    })
    surgical_event = {"tumor_size": 0.10}
    checkpoints = [0, 3, 6, 12, 18, 24, 36, 48, 72, 120, 168, 336]
    print(f"\n{'Time (h)':>9} | {'Phase':>16} | {'ICP':>6} | {'Inflammation':>12} | {'Blood Flow':>10}")
    print("-" * 70)
    trajectory = scm.simulate_trajectory(surgical_event, time_points=checkpoints, noise=False)
    icp_values = []
    for state in trajectory:
        t = state["_t_hours"]
        phase = state["_phase"]
        icp = state["intracranial_pressure"]
        inflam = state["_inflammatory_level"]
        bf = state["blood_flow"]
        icp_values.append((t, icp))
        print(f"{t:9.1f} | {phase:>16} | {icp:6.3f} | {inflam:12.3f} | {bf:10.3f}")

    print("\n--- Clinical Validation ---")
    icp_dict = dict(icp_values)
    early_icp  = icp_dict.get(3, icp_dict.get(6))
    peak_icp   = max(icp_dict.get(18, 0), icp_dict.get(24, 0))
    late_icp   = icp_dict.get(168, icp_dict.get(120))
    pre_op_icp = 0.55
    print(f"Pre-op ICP:           {pre_op_icp:.3f}")
    print(f"Intraoperative ICP:   {early_icp:.3f}  (expected < pre-op)")
    print(f"Peak ICP (18-24h):    {peak_icp:.3f}  (expected > intraop)")
    print(f"Recovery ICP (7d):    {late_icp:.3f}  (expected < peak)")
    ok1 = early_icp < pre_op_icp
    ok2 = peak_icp > early_icp
    ok3 = late_icp < peak_icp
    print(f"\n  Intraop decompression:    {'PASS' if ok1 else 'FAIL'}")
    print(f"  Acute inflammatory spike: {'PASS' if ok2 else 'FAIL'}")
    print(f"  Recovery phase:           {'PASS' if ok3 else 'FAIL'}")
    all_pass = ok1 and ok2 and ok3
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
