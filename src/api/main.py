"""
src/api/main.py
FastAPI REST API for the Brain Surgical Planner.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from loguru import logger
import uvicorn
import time
import json

from ..pipeline import BrainSurgicalPlannerPipeline, PipelineConfig
from ..causal.scm import BrainTumorSCM
from ..causal.do_calculus import DoCalculusEngine, SurgicalAction
from ..causal.counterfactual import CounterfactualEngine, CounterfactualQuery
from ..simulation.snn_physiology import IntraoperativeMonitor, PHYSIO_CHANNELS
from ..graph.knowledge_graph import AnatomicalKnowledgeGraph
from ..database.operations import PatientDB


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Surgical Planner",
    description="Causal AI surgical planning via Do-Calculus and Counterfactual Simulation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy init)
_pipeline: Optional[BrainSurgicalPlannerPipeline] = None
_jobs: Dict[str, Dict] = {}


def get_pipeline(patient_id: str = "API_PATIENT") -> BrainSurgicalPlannerPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BrainSurgicalPlannerPipeline(
            PipelineConfig(patient_id=patient_id, n_simulations=100)
        )
    return _pipeline


# ─── Schemas ──────────────────────────────────────────────────────────────────
class SCMParams(BaseModel):
    tumor_size:            float = Field(0.3, ge=0, le=1)
    blood_flow:            float = Field(0.7, ge=0, le=1)
    oxygen_saturation:     float = Field(0.95, ge=0, le=1)
    intracranial_pressure: float = Field(0.2, ge=0, le=1)
    edema_volume:          float = Field(0.2, ge=0, le=1)
    inflammatory_response: float = Field(0.3, ge=0, le=1)
    mass_effect:           float = Field(0.25, ge=0, le=1)


class InterventionRequest(BaseModel):
    action: str = Field(..., description="SurgicalAction enum value")
    patient_params: SCMParams = SCMParams()


class CounterfactualRequest(BaseModel):
    factual_action:        Optional[str] = None
    counterfactual_action: str
    observed_outcome:      Dict[str, float] = {}
    patient_params:        SCMParams = SCMParams()


class PlanSearchRequest(BaseModel):
    patient_params: SCMParams = SCMParams()
    n_simulations:  int       = Field(200, ge=10, le=1000)
    top_k:          int       = Field(5, ge=1, le=10)


class VitalsSnapshot(BaseModel):
    vitals: Dict[str, float] = Field(..., description="Physiology values keyed by channel name")


class KnowledgeGraphReasoningRequest(BaseModel):
    structures_present: List[str] = Field(..., description="Anatomical structures present")
    planned_action: str = Field(..., description="Surgical action to evaluate")


class PatientCreateRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., description="Patient name")
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    tumor_type: Optional[str] = None
    tumor_location: Optional[str] = None
    tumor_size: Optional[float] = None
    grade: Optional[str] = None


class PatientUpdateRequest(BaseModel):
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    tumor_type: Optional[str] = None
    tumor_location: Optional[str] = None
    tumor_size: Optional[float] = None
    grade: Optional[str] = None


class MRIStudyCreateRequest(BaseModel):
    patient_id: str
    study_id: str
    study_date: Optional[str] = None
    modality: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    metadata_json: Optional[Dict] = None


class SurgicalPlanCreateRequest(BaseModel):
    patient_id: str
    study_id: Optional[str] = None
    plan_id: str
    actions: List[str]
    expected_recovery: Optional[float] = None
    expected_risk: Optional[float] = None
    blood_loss_ml: Optional[float] = None
    nerve_damage_prob: Optional[float] = None
    icu_days: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    notes: Optional[str] = None


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "brain-surgical-planner"}


@app.get("/actions")
async def list_actions():
    """List all available surgical actions."""
    from ..causal.do_calculus import ACTION_REGISTRY
    return {
        "actions": [
            {
                "action": action.value,
                "description": params.get("description", ""),
                "risk_factor": params.get("risk_factor", 0.0),
                "target_variable": params.get("target_variable", ""),
            }
            for action, params in ACTION_REGISTRY.items()
        ]
    }


@app.get("/scm/variables")
async def scm_variables(params: SCMParams = None):
    """Return current SCM variable definitions."""
    scm = BrainTumorSCM()
    return {
        "variables": {
            name: {
                "value":       var.value,
                "min":         var.min_val,
                "max":         var.max_val,
                "description": var.description,
            }
            for name, var in scm.variables.items()
        },
        "dag_edges": list(scm.dag.edges()),
    }


@app.post("/intervene")
async def intervene(req: InterventionRequest):
    """Apply a do() intervention and return downstream causal effects."""
    try:
        action = SurgicalAction(req.action)
    except ValueError:
        raise HTTPException(400, f"Unknown action: {req.action}. See GET /actions")

    scm = BrainTumorSCM(patient_params=req.patient_params.model_dump())
    engine = DoCalculusEngine(scm)
    result = engine.intervene(action, noise=False)

    return {
        "action":            result.action.value,
        "intervention":      str(result.intervention),
        "pre_recovery":      round(result.pre_state.get("recovery_score", 0), 4),
        "post_recovery":     round(result.post_state.get("recovery_score", 0), 4),
        "recovery_gain":     round(result.recovery_gain, 4),
        "risk_increase":     round(result.risk_increase, 4),
        "net_utility":       round(result.net_utility, 4),
        "downstream_effects": result.downstream_effects,
        "pre_state":         result.pre_state,
        "post_state":        result.post_state,
    }


@app.post("/counterfactual")
async def counterfactual(req: CounterfactualRequest):
    """Answer: what would have happened if we had done X instead?"""
    try:
        cf_action = SurgicalAction(req.counterfactual_action)
    except ValueError:
        raise HTTPException(400, f"Unknown action: {req.counterfactual_action}")

    factual_action = None
    if req.factual_action:
        try:
            factual_action = SurgicalAction(req.factual_action)
        except ValueError:
            raise HTTPException(400, f"Unknown factual action: {req.factual_action}")

    scm = BrainTumorSCM(patient_params=req.patient_params.model_dump())
    engine = CounterfactualEngine(scm, n_simulations=50)

    query = CounterfactualQuery(
        factual_action=factual_action,
        counterfactual_action=cf_action,
        observed_outcome=req.observed_outcome or scm.evaluate(),
    )

    result = engine.run_counterfactual(query)

    return {
        "query":                str(result.query),
        "factual_recovery":     round(result.factual_state.get("recovery_score", 0), 4),
        "counterfactual_recovery": round(result.counterfactual_state.get("recovery_score", 0), 4),
        "recovery_delta":       round(result.recovery_delta, 4),
        "was_better":           result.was_better,
        "explanation":          result.explanation,
        "factual_state":        result.factual_state,
        "counterfactual_state": result.counterfactual_state,
    }


@app.post("/search/plans")
async def search_plans(req: PlanSearchRequest):
    """Run Monte-Carlo counterfactual search over all surgical plans."""
    scm = BrainTumorSCM(patient_params=req.patient_params.model_dump())
    engine = CounterfactualEngine(scm, n_simulations=req.n_simulations)
    plans = engine.monte_carlo_search(top_k=req.top_k)

    return {
        "top_plans": [p.to_dict() for p in plans],
        "n_plans_evaluated": len(engine._generate_candidate_plans()),
        "n_simulations_per_plan": req.n_simulations,
    }


@app.post("/pipeline/upload")
async def run_pipeline_upload(
    background_tasks: BackgroundTasks,
    t1: Optional[UploadFile] = File(None),
    t1ce: Optional[UploadFile] = File(None),
    t2: Optional[UploadFile] = File(None),
    flair: Optional[UploadFile] = File(None),
    patient_id: str = "UPLOAD_PATIENT",
    n_simulations: int = 100,
):
    """Upload MRI files and run the full pipeline (async background job)."""
    import uuid
    job_id = str(uuid.uuid4())[:8]
    tmp_dir = tempfile.mkdtemp()
    paths = {}

    for modality, upload in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)]:
        if upload:
            path = Path(tmp_dir) / f"{modality}.nii.gz"
            path.write_bytes(await upload.read())
            paths[modality] = str(path)

    _jobs[job_id] = {"status": "queued", "patient_id": patient_id}

    async def _run():
        _jobs[job_id]["status"] = "running"
        try:
            cfg = PipelineConfig(patient_id=patient_id, n_simulations=n_simulations)
            pipeline = BrainSurgicalPlannerPipeline(cfg)
            result = pipeline.run(paths)
            _jobs[job_id].update({
                "status": "done",
                "top_plans": result.top_plans,
                "report_preview": result.surgical_report[:500],
            })
        except Exception as e:
            _jobs[job_id].update({"status": "error", "error": str(e)})

    background_tasks.add_task(_run)

    return {"job_id": job_id, "status": "queued"}


@app.get("/pipeline/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


# ─── Simulation Endpoints ─────────────────────────────────────────────────────
_monitor: Optional[IntraoperativeMonitor] = None


def get_monitor() -> IntraoperativeMonitor:
    global _monitor
    if _monitor is None:
        _monitor = IntraoperativeMonitor()
    return _monitor


@app.get("/simulation/channels")
async def simulation_channels():
    """List available physiology monitoring channels."""
    return {
        "channels": PHYSIO_CHANNELS,
        "n_channels": len(PHYSIO_CHANNELS),
    }


@app.post("/simulation/vitals")
async def simulation_vitals(req: VitalsSnapshot):
    """Process a vital signs reading through the SNN monitor."""
    monitor = get_monitor()
    state = monitor.update(req.vitals, timestamp_ms=0.0)
    return state.to_dict()


@app.post("/simulation/simulate")
async def simulation_simulate(
    duration_ms: int = 10000,
    dt_ms: int = 100,
    tumor_removal_at_ms: Optional[int] = 5000,
):
    """Run a full surgery simulation and return the physiology timeline."""
    monitor = IntraoperativeMonitor()
    states = monitor.simulate_surgery(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        tumor_removal_at_ms=tumor_removal_at_ms,
    )
    return {
        "states": [s.to_dict() for s in states],
        "n_timesteps": len(states),
        "final_outcome": states[-1].predicted_outcome if states else 0.0,
        "alerts_summary": {
            "total": sum(len(s.alerts) for s in states),
            "critical": sum(1 for s in states if s.alert_level == "CRITICAL"),
            "warning": sum(1 for s in states if s.alert_level == "WARNING"),
        },
    }


# ─── Knowledge Graph Endpoints ────────────────────────────────────────────────
_kg: Optional[AnatomicalKnowledgeGraph] = None


def get_kg() -> AnatomicalKnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = AnatomicalKnowledgeGraph()
        _kg.populate()
    return _kg


@app.get("/knowledge-graph/blood-supply/{structure}")
async def kg_blood_supply(structure: str):
    """Query all structures that supply blood to the given structure."""
    kg = get_kg()
    suppliers = kg.get_blood_supply_chain(structure)
    return {
        "structure": structure,
        "suppliers": suppliers,
        "n_suppliers": len(suppliers),
    }


@app.get("/knowledge-graph/compression/{tumor}")
async def kg_compression(tumor: str):
    """Query all structures compressed by a given tumor."""
    kg = get_kg()
    compressed = kg.get_compression_chain(tumor)
    return {
        "tumor": tumor,
        "compressed_structures": compressed,
        "n_compressed": len(compressed),
    }


@app.get("/knowledge-graph/action-risks/{action}")
async def kg_action_risks(action: str):
    """Query all structures at risk from a surgical action."""
    kg = get_kg()
    risks = kg.get_action_risks(action)
    return {
        "action": action,
        "risks": risks,
        "n_risks": len(risks),
    }


@app.post("/knowledge-graph/reasoning")
async def kg_reasoning(req: KnowledgeGraphReasoningRequest):
    """Run symbolic reasoning combining structures present with planned action."""
    kg = get_kg()
    result = kg.symbolic_reasoning(
        structures_present=req.structures_present,
        planned_action=req.planned_action,
    )
    return result


@app.get("/knowledge-graph/stats")
async def kg_stats():
    """Get knowledge graph statistics."""
    kg = get_kg()
    return {
        "n_nodes": kg._nx_fallback.number_of_nodes() if kg._nx_fallback else 0,
        "n_edges": kg._nx_fallback.number_of_edges() if kg._nx_fallback else 0,
        "neo4j_connected": kg.driver is not None,
    }


# ─── Patient Management Endpoints ─────────────────────────────────────────────
_patient_db: Optional[PatientDB] = None


def get_patient_db() -> PatientDB:
    global _patient_db
    if _patient_db is None:
        _patient_db = PatientDB()
    return _patient_db


@app.post("/patients")
async def create_patient(req: PatientCreateRequest):
    """Create a new patient record."""
    db = get_patient_db()
    try:
        patient_data = req.model_dump()
        if patient_data.get("date_of_birth"):
            from datetime import datetime
            patient_data["date_of_birth"] = datetime.fromisoformat(patient_data["date_of_birth"])
        patient = db.create_patient(patient_data)
        return patient.to_dict()
    except Exception as e:
        raise HTTPException(400, f"Failed to create patient: {str(e)}")


@app.get("/patients")
async def list_patients(limit: int = 100, offset: int = 0):
    """List all patients with pagination."""
    db = get_patient_db()
    patients = db.list_patients(limit=limit, offset=offset)
    return {"patients": [p.to_dict() for p in patients], "count": len(patients)}


@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    """Get a patient by ID."""
    db = get_patient_db()
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, f"Patient {patient_id} not found")
    return patient.to_dict()


@app.put("/patients/{patient_id}")
async def update_patient(patient_id: str, req: PatientUpdateRequest):
    """Update a patient record."""
    db = get_patient_db()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if updates.get("date_of_birth"):
        from datetime import datetime
        updates["date_of_birth"] = datetime.fromisoformat(updates["date_of_birth"])
    patient = db.update_patient(patient_id, updates)
    if not patient:
        raise HTTPException(404, f"Patient {patient_id} not found")
    return patient.to_dict()


@app.delete("/patients/{patient_id}")
async def delete_patient(patient_id: str):
    """Delete a patient and all related records."""
    db = get_patient_db()
    success = db.delete_patient(patient_id)
    if not success:
        raise HTTPException(404, f"Patient {patient_id} not found")
    return {"status": "deleted", "patient_id": patient_id}


@app.post("/patients/{patient_id}/studies")
async def create_mri_study(patient_id: str, req: MRIStudyCreateRequest):
    """Create an MRI study for a patient."""
    db = get_patient_db()
    try:
        study_data = req.model_dump()
        if study_data.get("study_date"):
            from datetime import datetime
            study_data["study_date"] = datetime.fromisoformat(study_data["study_date"])
        study = db.create_study(study_data)
        return study.to_dict()
    except Exception as e:
        raise HTTPException(400, f"Failed to create study: {str(e)}")


@app.get("/patients/{patient_id}/studies")
async def list_patient_studies(patient_id: str):
    """List all MRI studies for a patient."""
    db = get_patient_db()
    studies = db.get_studies_for_patient(patient_id)
    return {"studies": [s.to_dict() for s in studies], "count": len(studies)}


@app.post("/patients/{patient_id}/plans")
async def create_surgical_plan(patient_id: str, req: SurgicalPlanCreateRequest):
    """Create a surgical plan for a patient."""
    db = get_patient_db()
    try:
        plan_data = req.model_dump()
        plan = db.create_plan(plan_data)
        return plan.to_dict()
    except Exception as e:
        raise HTTPException(400, f"Failed to create plan: {str(e)}")


@app.get("/patients/{patient_id}/plans")
async def list_patient_plans(patient_id: str):
    """List all surgical plans for a patient."""
    db = get_patient_db()
    plans = db.get_plans_for_patient(patient_id)
    return {"plans": [p.to_dict() for p in plans], "count": len(plans)}


@app.put("/plans/{plan_id}/status")
async def update_plan_status(plan_id: str, status: str):
    """Update the status of a surgical plan."""
    db = get_patient_db()
    plan = db.update_plan_status(plan_id, status)
    if not plan:
        raise HTTPException(404, f"Plan {plan_id} not found")
    return plan.to_dict()


# ─── GNN Training Endpoints ───────────────────────────────────────────────────
_training_jobs: Dict[str, Dict] = {}

# ─── Monitoring & Metrics ─────────────────────────────────────────────────────
_request_times: List[float] = []
_request_counts: Dict[str, int] = {}
_start_time: float = time.time()


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Log request metrics."""
    global _request_times
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    _request_times.append(elapsed)
    route = request.url.path
    _request_counts[route] = _request_counts.get(route, 0) + 1

    logger.info(f"{request.method} {route} → {response.status_code} in {elapsed*1000:.1f}ms")
    return response


@app.get("/monitor/metrics")
async def metrics():
    """Get API performance metrics."""
    uptime = time.time() - _start_time
    recent = _request_times[-100:] if len(_request_times) > 100 else _request_times
    return {
        "uptime_seconds": round(uptime, 1),
        "total_requests": len(_request_times),
        "requests_per_route": dict(sorted(_request_counts.items(), key=lambda x: -x[1])),
        "avg_latency_ms": round(sum(recent) / len(recent) * 1000, 2) if recent else 0,
        "max_latency_ms": round(max(recent) * 1000, 2) if recent else 0,
        "p99_latency_ms": round(sorted(recent)[int(len(recent) * 0.99)] * 1000, 2) if len(recent) >= 100 else 0,
    }


@app.get("/monitor/health/detailed")
async def health_detailed():
    """Detailed health check with service status."""
    db_status = "unknown"
    try:
        db = get_patient_db()
        db.list_patients(limit=1)
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"

    kg_status = "unknown"
    try:
        kg = get_kg()
        kg_status = "ok" if kg._nx_fallback is not None else "error"
    except Exception as e:
        kg_status = f"error: {str(e)[:50]}"

    return {
        "status": "ok",
        "service": "brain-surgical-planner",
        "version": "0.1.0",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "services": {
            "api": "ok",
            "database": db_status,
            "knowledge_graph": kg_status,
        },
        "endpoints_available": len(_request_counts),
    }


@app.post("/gnn/generate-data")
async def gnn_generate_data(n_patients: int = 50, output_dir: str = "data/training"):
    """Generate synthetic training data for GNN."""
    import uuid
    job_id = str(uuid.uuid4())[:8]

    _training_jobs[job_id] = {"status": "running", "type": "data_generation"}

    async def _run():
        try:
            import sys
            sys.path.insert(0, "/app")
            from scripts.train_gnn import generate_synthetic_data
            generate_synthetic_data(output_dir, n_patients)
            _training_jobs[job_id]["status"] = "done"
            _training_jobs[job_id]["n_patients"] = n_patients
        except Exception as e:
            _training_jobs[job_id]["status"] = "error"
            _training_jobs[job_id]["error"] = str(e)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(_run)

    return {"job_id": job_id, "status": "started", "n_patients": n_patients}


@app.post("/gnn/train")
async def gnn_train(
    background_tasks: BackgroundTasks,
    data_dir: str = "data/training",
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_channels: int = 128,
    output_dir: str = "models/gnn",
):
    """Train the GNN model."""
    import uuid
    job_id = str(uuid.uuid4())[:8]

    _training_jobs[job_id] = {"status": "running", "type": "training", "epochs": epochs}

    async def _run():
        try:
            import sys
            sys.path.insert(0, "/app")
            from scripts.train_gnn import train
            train(
                data_dir=data_dir,
                epochs=epochs,
                lr=lr,
                hidden_channels=hidden_channels,
                output=f"{output_dir}/model.pt",
            )
            _training_jobs[job_id]["status"] = "done"
            _training_jobs[job_id]["model_path"] = f"{output_dir}/model.pt"
        except Exception as e:
            _training_jobs[job_id]["status"] = "error"
            _training_jobs[job_id]["error"] = str(e)

    background_tasks.add_task(_run)

    return {"job_id": job_id, "status": "started", "epochs": epochs}


@app.get("/gnn/jobs/{job_id}")
async def gnn_job_status(job_id: str):
    """Get status of a GNN training job."""
    if job_id not in _training_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _training_jobs[job_id]


@app.get("/gnn/predict")
async def gnn_predict(
    patient_id: str = "API_PATIENT",
    model_path: Optional[str] = None,
):
    """Run GNN prediction for a patient."""
    try:
        from ..graph.gnn_model import AnatomicalGNNInference
        from ..graph.anatomical_graph import AnatomicalGraphBuilder
        import numpy as np

        builder = AnatomicalGraphBuilder()
        mock_structures = [
            {"name": "brainstem", "volume_mm3": 5000, "centroid_mm": [0, 0, -20], "is_tumor": False},
            {"name": "enhancing_tumor", "volume_mm3": 15000, "centroid_mm": [10, 5, 10], "is_tumor": True},
            {"name": "white_matter", "volume_mm3": 200000, "centroid_mm": [0, 0, 0], "is_tumor": False},
            {"name": "gray_matter", "volume_mm3": 100000, "centroid_mm": [0, 0, 15], "is_tumor": False},
            {"name": "middle_cerebral_artery", "volume_mm3": 500, "centroid_mm": [-20, 0, 5], "is_tumor": False},
        ]
        for s in mock_structures:
            builder._add_node(s)
        builder._add_anatomical_priors()

        # Extract feature matrix and edge index
        node_names = list(builder.graph.nodes())
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        feature_matrix = np.stack([
            builder.graph.nodes[n]["features"].to_vector()
            for n in node_names
        ])
        edges = list(builder.graph.edges())
        edge_index = np.array([[name_to_idx[u], name_to_idx[v]] for u, v in edges], dtype=np.int64).T

        inference = AnatomicalGNNInference(model_path=model_path)
        prediction = inference.predict(feature_matrix, edge_index, node_names)

        return {
            "patient_id": patient_id,
            "prediction": {
                "blood_loss_ml": prediction.blood_loss_ml,
                "nerve_damage_prob": prediction.nerve_damage_prob,
                "recovery_score": prediction.recovery_score,
                "mortality_risk": prediction.mortality_risk,
                "icu_days_estimate": prediction.icu_days_estimate,
                "confidence": prediction.confidence,
                "node_risks": prediction.node_risks,
            }
        }
    except Exception as e:
        raise HTTPException(500, f"GNN prediction failed: {str(e)}")


@app.get("/gnn/model-info")
async def gnn_model_info():
    """Get information about the GNN model architecture."""
    return {
        "architecture": "Heterogeneous Graph Transformer (GATv2)",
        "input_features": 14,
        "hidden_channels": 128,
        "num_layers": 4,
        "attention_heads": 4,
        "output_heads": ["blood_loss", "nerve_damage", "recovery", "mortality", "icu_days"],
        "description": "Multi-task GNN for surgical risk prediction from anatomical brain graphs",
    }


# ─── Output / Artifact Serving Endpoints ──────────────────────────────────────
OUTPUT_DIR = Path("/app/outputs")


def _list_output_patients() -> List[str]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted([
        d.name for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / "surgical_report.txt").exists()
    ])


def _patient_output_dir(patient_id: str) -> Path:
    d = OUTPUT_DIR / patient_id
    if not d.exists():
        raise HTTPException(404, f"No outputs for patient {patient_id}")
    return d


@app.get("/outputs")
async def list_outputs():
    """List all patient IDs with pipeline output directories."""
    patients = _list_output_patients()
    return {
        "patients": patients,
        "count": len(patients),
    }


@app.get("/outputs/{patient_id}")
async def get_patient_outputs(patient_id: str):
    """List all output artifacts for a patient."""
    out_dir = _patient_output_dir(patient_id)
    meshes_dir = out_dir / "meshes"
    artifacts = {
        "report": (out_dir / "surgical_report.txt").exists(),
        "graph": (out_dir / "graph.json").exists(),
        "meshes": [],
    }
    if meshes_dir.exists():
        artifacts["meshes"] = sorted(f.name for f in meshes_dir.iterdir() if f.suffix == ".obj")
    return {
        "patient_id": patient_id,
        "artifacts": artifacts,
    }


@app.get("/outputs/{patient_id}/meshes")
async def list_patient_meshes(patient_id: str):
    """List available mesh files for a patient."""
    out_dir = _patient_output_dir(patient_id)
    meshes_dir = out_dir / "meshes"
    if not meshes_dir.exists():
        return {"patient_id": patient_id, "meshes": []}
    meshes = sorted(f.name for f in meshes_dir.iterdir() if f.suffix == ".obj")
    return {
        "patient_id": patient_id,
        "meshes": meshes,
    }


@app.get("/outputs/{patient_id}/meshes/{mesh_file}")
async def serve_patient_mesh(patient_id: str, mesh_file: str):
    """Serve a mesh .obj file for download."""
    out_dir = _patient_output_dir(patient_id)
    mesh_path = out_dir / "meshes" / mesh_file
    if not mesh_path.exists() or not mesh_path.suffix == ".obj":
        raise HTTPException(404, f"Mesh {mesh_file} not found for {patient_id}")
    return FileResponse(
        str(mesh_path),
        media_type="text/plain",
        filename=mesh_file,
    )


@app.get("/outputs/{patient_id}/report")
async def get_patient_report(patient_id: str):
    """Get the surgical report text for a patient."""
    out_dir = _patient_output_dir(patient_id)
    report_path = out_dir / "surgical_report.txt"
    if not report_path.exists():
        raise HTTPException(404, f"No report for patient {patient_id}")
    return PlainTextResponse(
        report_path.read_text(),
        headers={"Content-Disposition": f"inline; filename=report_{patient_id}.txt"},
    )


@app.get("/outputs/{patient_id}/graph")
async def get_patient_graph(patient_id: str):
    """Get the anatomical graph JSON for a patient."""
    out_dir = _patient_output_dir(patient_id)
    graph_path = out_dir / "graph.json"
    if not graph_path.exists():
        raise HTTPException(404, f"No graph for patient {patient_id}")
    return JSONResponse(
        json.loads(graph_path.read_text()),
        headers={"Content-Disposition": f"inline; filename=graph_{patient_id}.json"},
    )


class PipelineRunRequest(BaseModel):
    patient_id: str = Field(..., description="Patient ID for output directory")
    mri_dir: str = Field(..., description="Path to adapted MRI directory on disk (inside container)")
    n_simulations: int = Field(100, ge=10, le=1000)


@app.post("/pipeline/run")
async def run_pipeline(
    req: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    """Run full pipeline using adapted MRI data on disk (background job)."""
    import uuid

    mri_dir = Path(req.mri_dir)
    if not mri_dir.exists():
        raise HTTPException(400, f"MRI directory not found: {req.mri_dir}")

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "queued", "patient_id": req.patient_id}

    async def _run():
        _jobs[job_id]["status"] = "running"
        try:
            cfg = PipelineConfig(
                patient_id=req.patient_id,
                n_simulations=req.n_simulations,
                output_dir="/app/outputs",
                export_meshes=True,
                export_graph=True,
            )
            pipeline = BrainSurgicalPlannerPipeline(cfg)
            mri_paths = {}
            for modality in ["t1", "t1ce", "t2", "flair"]:
                candidates = list(mri_dir.glob(f"*{modality}*.nii*"))
                if candidates:
                    mri_paths[modality] = str(candidates[0])
            result = pipeline.run(mri_paths)
            _jobs[job_id].update({
                "status": "done",
                "elapsed_seconds": result.elapsed_seconds,
                "top_plans": result.top_plans,
                "report_preview": result.surgical_report[:500],
            })
        except Exception as e:
            logger.error(f"Pipeline job {job_id} failed: {e}")
            _jobs[job_id].update({"status": "error", "error": str(e)})

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued", "patient_id": req.patient_id}


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
