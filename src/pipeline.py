"""
src/pipeline.py
Master pipeline — end-to-end: MRI → Surgical Plan.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .imaging.segmentation import BrainTumorSegmenter
from .imaging.reconstruction import BrainReconstructionPipeline
from .graph.anatomical_graph import AnatomicalGraphBuilder
from .graph.gnn_model import AnatomicalGNNInference
from .causal.scm import BrainTumorSCM
from .agents.surgical_planner import SurgicalPlannerOrchestrator


@dataclass
class PipelineConfig:
    patient_id:     str   = "PATIENT_001"
    device:         str   = "auto"
    n_simulations:  int   = 200
    model_path:     Optional[str] = None
    gnn_path:       Optional[str] = None
    output_dir:     str   = "outputs"
    export_meshes:  bool  = True
    export_graph:   bool  = True


@dataclass
class PipelineResult:
    patient_id:       str
    twin_summary:     Dict
    graph_json:       str
    gnn_prediction:   Dict
    top_plans:        List[Dict]
    surgical_report:  str
    elapsed_seconds:  float
    errors:           List[str] = field(default_factory=list)


class BrainSurgicalPlannerPipeline:
    """
    Full end-to-end pipeline:

        MRI (NIfTI)
           ↓  BrainTumorSegmenter
        Segmentation mask
           ↓  BrainReconstructionPipeline
        BrainDigitalTwin (3D meshes)
           ↓  AnatomicalGraphBuilder
        Anatomical Graph (NetworkX)
           ↓  AnatomicalGNNInference
        Surgical Risk Prediction
           ↓  BrainTumorSCM + DoCalculusEngine
        Causal Intervention Results
           ↓  CounterfactualEngine (Monte-Carlo)
        Top 5 Surgical Plans
           ↓  SurgicalPlannerOrchestrator
        Surgical Report
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._init_components()

    def _init_components(self):
        cfg = self.config
        self.segmenter    = BrainTumorSegmenter(model_path=cfg.model_path, device=cfg.device)
        self.reconstructor = BrainReconstructionPipeline()
        self.graph_builder = AnatomicalGraphBuilder()
        self.gnn           = AnatomicalGNNInference(model_path=cfg.gnn_path, device=cfg.device)
        logger.info(f"Pipeline initialized for patient: {cfg.patient_id}")

    def run(
        self,
        mri_paths: Dict[str, str],
        patient_params: Optional[Dict] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            mri_paths: {"t1": path, "t1ce": path, "t2": path, "flair": path}
            patient_params: optional override of SCM parameters

        Returns:
            PipelineResult with plans and surgical report
        """
        t0 = time.time()
        cfg = self.config
        errors = []

        logger.info(f"=== Starting pipeline for {cfg.patient_id} ===")

        # Step 1: Segmentation
        logger.info("Step 1/5: MRI Segmentation")
        seg = self.segmenter.segment(mri_paths)

        # Step 2: 3D Reconstruction
        logger.info("Step 2/5: 3D Reconstruction")
        twin = self.reconstructor.reconstruct(seg, patient_id=cfg.patient_id)
        twin_summary = twin.summary()

        if cfg.export_meshes:
            out_dir = Path(cfg.output_dir) / cfg.patient_id / "meshes"
            self.reconstructor.export_obj(twin, str(out_dir))

        # Step 3: Anatomical Graph
        logger.info("Step 3/5: Building Anatomical Graph")
        graph = self.graph_builder.build(twin_summary, patient_id=cfg.patient_id)
        feature_matrix, node_names = self.graph_builder.get_node_feature_matrix()
        edge_index = self.graph_builder.get_edge_index()

        graph_json = self.graph_builder.to_json()
        if cfg.export_graph:
            graph_path = Path(cfg.output_dir) / cfg.patient_id / "graph.json"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text(graph_json)

        # Step 4: GNN Risk Prediction
        logger.info("Step 4/5: GNN Risk Prediction")
        gnn_pred = self.gnn.predict(feature_matrix, edge_index, node_names)
        gnn_dict = gnn_pred.to_dict()

        # Step 5: Causal Planning
        logger.info("Step 5/5: Causal Surgical Planning")

        # Initialize SCM with patient-specific parameters
        scm_params = self._extract_scm_params(seg, gnn_dict, patient_params)
        scm = BrainTumorSCM(patient_params=scm_params)

        orchestrator = SurgicalPlannerOrchestrator(
            scm=scm,
            n_simulations=cfg.n_simulations,
        )
        state = orchestrator.run(
            patient_id=cfg.patient_id,
            twin_summary=twin_summary,
            gnn_prediction=gnn_dict,
        )

        if state.get("error"):
            errors.append(state["error"])

        # Write report
        report = state.get("surgical_report", "")
        if report:
            report_path = Path(cfg.output_dir) / cfg.patient_id / "surgical_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report)
            logger.info(f"Report written to {report_path}")

        elapsed = time.time() - t0
        logger.info(f"=== Pipeline complete in {elapsed:.1f}s ===")

        return PipelineResult(
            patient_id=cfg.patient_id,
            twin_summary=twin_summary,
            graph_json=graph_json,
            gnn_prediction=gnn_dict,
            top_plans=state.get("top_plans", []),
            surgical_report=report,
            elapsed_seconds=elapsed,
            errors=errors,
        )

    def _extract_scm_params(
        self,
        seg: Dict,
        gnn: Dict,
        patient_params: Optional[Dict],
    ) -> Dict:
        """Derive initial SCM variable values from segmentation + GNN outputs."""
        structures = seg.get("structures", [])
        tumor_voxels = sum(
            s["voxel_count"] for s in structures if s.get("is_tumor")
        )
        total_voxels = max(1, sum(s["voxel_count"] for s in structures))

        tumor_fraction = tumor_voxels / total_voxels

        params = {
            "tumor_size": float(min(tumor_fraction * 3, 1.0)),
            "surgical_risk": gnn.get("mortality_risk", 0.1),
        }

        if patient_params:
            params.update(patient_params)

        return params
