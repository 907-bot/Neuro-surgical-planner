"""
scripts/run_pipeline.py
CLI entry point for the Brain Tumor Surgical Planner pipeline.

Usage:
    # Full pipeline on real MRI
    python scripts/run_pipeline.py --mri data/raw/patient_001 --patient-id P001

    # Mock run (no MRI needed)
    python scripts/run_pipeline.py --mock --patient-id DEMO

    # Just the causal search (no imaging)
    python scripts/run_pipeline.py --causal-only --tumor-size 0.4
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def run_causal_only(args):
    """Run just the causal SCM + Monte-Carlo search without imaging."""
    from src.causal.scm import BrainTumorSCM
    from src.causal.do_calculus import DoCalculusEngine, SurgicalAction
    from src.causal.counterfactual import CounterfactualEngine

    console.print(Panel(
        f"[bold purple]Causal Surgical Planner[/bold purple]\n"
        f"Tumor size: {args.tumor_size:.0%} | "
        f"Simulations: {args.n_simulations}",
        title="🧠 Brain Tumor Surgical Planner",
    ))

    params = {"tumor_size": args.tumor_size}
    scm = BrainTumorSCM(patient_params=params)
    baseline = scm.evaluate(noise=False)

    # Print baseline
    table = Table(title="Baseline SCM State")
    table.add_column("Variable", style="cyan")
    table.add_column("Value", justify="right", style="green")
    for k, v in baseline.items():
        table.add_row(k.replace("_", " ").title(), f"{v:.4f}")
    console.print(table)

    # All single interventions
    console.print("\n[bold]Do() Intervention Results[/bold]")
    engine = DoCalculusEngine(scm)
    intervention_table = Table()
    intervention_table.add_column("Action", style="cyan")
    intervention_table.add_column("Recovery Gain", justify="right")
    intervention_table.add_column("Net Utility", justify="right")
    intervention_table.add_column("Risk", justify="right")

    results = []
    for action in SurgicalAction:
        r = engine.intervene(action, noise=False)
        results.append((action, r))

    results.sort(key=lambda x: x[1].net_utility, reverse=True)
    for action, r in results:
        color = "green" if r.net_utility > 0 else "red"
        intervention_table.add_row(
            action.value.replace("_", " ").title(),
            f"[{color}]{r.recovery_gain:+.4f}[/{color}]",
            f"[{color}]{r.net_utility:+.4f}[/{color}]",
            f"{r.risk_increase:.3f}",
        )
    console.print(intervention_table)

    # Monte-Carlo search
    console.print(f"\n[bold]Monte-Carlo Search ({args.n_simulations} sims/plan)[/bold]")
    cf_engine = CounterfactualEngine(
        BrainTumorSCM(patient_params=params),
        n_simulations=args.n_simulations
    )

    with console.status("Searching surgical paths..."):
        top_plans = cf_engine.monte_carlo_search(top_k=5)

    plan_table = Table(title="Top 5 Surgical Plans")
    plan_table.add_column("Rank", justify="center")
    plan_table.add_column("Actions", style="cyan")
    plan_table.add_column("Recovery", justify="right", style="green")
    plan_table.add_column("Risk", justify="right", style="red")
    plan_table.add_column("Utility", justify="right")
    plan_table.add_column("ICU Days", justify="right")

    for plan in top_plans:
        actions_str = " → ".join(a.value.replace("_", " ")[:20] for a in plan.actions)
        util_color = "green" if plan.net_utility > 0 else "red"
        plan_table.add_row(
            f"#{plan.rank}",
            actions_str,
            f"{plan.expected_recovery:.1%}",
            f"{plan.expected_risk:.1%}",
            f"[{util_color}]{plan.net_utility:+.4f}[/{util_color}]",
            f"{plan.icu_days:.1f}",
        )
    console.print(plan_table)

    console.print(Panel(
        f"[bold green]✓ Recommended:[/bold green] "
        + " → ".join(a.value.replace("_", " ").title() for a in top_plans[0].actions)
        + f"\n  Expected recovery: {top_plans[0].expected_recovery:.0%} "
        + f"(95% CI: {top_plans[0].confidence_interval[0]:.0%}–{top_plans[0].confidence_interval[1]:.0%})"
        if top_plans else "[red]No valid plans found[/red]",
        title="Recommendation",
    ))

    if args.output:
        out = {"top_plans": [p.to_dict() for p in top_plans], "baseline": baseline}
        Path(args.output).write_text(json.dumps(out, indent=2))
        console.print(f"[dim]Results saved to {args.output}[/dim]")


def run_full_pipeline(args):
    """Run the complete imaging + causal pipeline."""
    from src.pipeline import BrainSurgicalPlannerPipeline, PipelineConfig

    mri_dir = Path(args.mri)
    mri_paths = {}

    for modality in ["t1", "t1ce", "t2", "flair"]:
        candidates = list(mri_dir.glob(f"*{modality}*.nii*")) + \
                     list(mri_dir.glob(f"*{modality.upper()}*.nii*"))
        if candidates:
            mri_paths[modality] = str(candidates[0])
            logger.info(f"Found {modality}: {candidates[0]}")

    if not mri_paths:
        # Try single file
        niis = list(mri_dir.glob("*.nii*"))
        if niis:
            mri_paths["image"] = str(niis[0])

    if not mri_paths:
        console.print(f"[red]No NIfTI files found in {mri_dir}[/red]")
        sys.exit(1)

    cfg = PipelineConfig(
        patient_id=args.patient_id,
        n_simulations=args.n_simulations,
        output_dir=args.output or "outputs",
    )

    with console.status("Running pipeline..."):
        pipeline = BrainSurgicalPlannerPipeline(cfg)
        result = pipeline.run(mri_paths)

    console.print(result.surgical_report)


def run_mock(args):
    """Run with synthetic data — no MRI needed."""
    from src.pipeline import BrainSurgicalPlannerPipeline, PipelineConfig

    cfg = PipelineConfig(
        patient_id=args.patient_id or "MOCK_001",
        n_simulations=args.n_simulations,
        output_dir=args.output or "outputs",
    )

    # Empty paths → triggers mock segmentation
    mri_paths = {"image": "mock"}

    with console.status("Running mock pipeline..."):
        pipeline = BrainSurgicalPlannerPipeline(cfg)
        result = pipeline.run(mri_paths)

    console.print(result.surgical_report)


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Surgical Planner CLI")
    parser.add_argument("--mri", type=str, help="Path to MRI directory")
    parser.add_argument("--patient-id", type=str, default="PATIENT_001")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic data")
    parser.add_argument("--causal-only", action="store_true", help="Skip imaging")
    parser.add_argument("--tumor-size", type=float, default=0.3, help="For causal-only mode")
    parser.add_argument("--n-simulations", type=int, default=200)
    parser.add_argument("--output", type=str, help="Output file/directory")
    args = parser.parse_args()

    if args.causal_only:
        run_causal_only(args)
    elif args.mock:
        run_mock(args)
    elif args.mri:
        run_full_pipeline(args)
    else:
        # Default: causal-only demo
        console.print("[yellow]No MRI specified — running causal demo[/yellow]")
        args.causal_only = True
        run_causal_only(args)


if __name__ == "__main__":
    main()
