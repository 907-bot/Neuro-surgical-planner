"""
src/graph/anatomical_graph.py
Convert BrainDigitalTwin → Anatomical Graph (nodes=structures, edges=relationships).
Each node carries physiological features; edges encode spatial + functional relationships.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from loguru import logger


# ─── Edge Relationship Types ──────────────────────────────────────────────────
class RelationType(str, Enum):
    BLOOD_SUPPLY    = "blood_supply"      # artery → structure
    VENOUS_DRAINAGE = "venous_drainage"   # structure → vein
    SPATIAL_CONTACT = "spatial_contact"  # structures touching
    COMPRESSION     = "compression"       # tumor → structure
    NEURAL_PATHWAY  = "neural_pathway"    # nerve → structure
    FUNCTIONAL_DEP  = "functional_dep"    # A depends on B physiologically
    LYMPHATIC       = "lymphatic"


# ─── Prior anatomical knowledge ───────────────────────────────────────────────
# Defines expected edges in a healthy brain.
# Format: (source, target, RelationType, weight)
BRAIN_ANATOMICAL_PRIORS: List[Tuple[str, str, RelationType, float]] = [
    # Arterial supply
    ("middle_cerebral_artery",   "white_matter",       RelationType.BLOOD_SUPPLY, 0.9),
    ("middle_cerebral_artery",   "gray_matter",        RelationType.BLOOD_SUPPLY, 0.9),
    ("anterior_cerebral_artery", "white_matter",       RelationType.BLOOD_SUPPLY, 0.7),
    ("posterior_cerebral_artery","cerebellum",         RelationType.BLOOD_SUPPLY, 0.8),
    ("basilar_artery",           "brainstem",          RelationType.BLOOD_SUPPLY, 0.95),
    ("internal_carotid_artery",  "middle_cerebral_artery", RelationType.BLOOD_SUPPLY, 1.0),

    # Venous drainage
    ("white_matter",    "dural_sinus",   RelationType.VENOUS_DRAINAGE, 0.8),
    ("gray_matter",     "dural_sinus",   RelationType.VENOUS_DRAINAGE, 0.8),
    ("cerebellum",      "dural_sinus",   RelationType.VENOUS_DRAINAGE, 0.6),

    # Functional dependencies
    ("brainstem",  "gray_matter",   RelationType.FUNCTIONAL_DEP, 0.7),
    ("ventricles", "white_matter",  RelationType.FUNCTIONAL_DEP, 0.5),
    ("gray_matter","white_matter",  RelationType.FUNCTIONAL_DEP, 0.6),
]

# Tumor-specific edges (added when tumor is detected)
TUMOR_EDGE_TEMPLATES = [
    ("enhancing_tumor",      "necrotic_tumor_core",   RelationType.SPATIAL_CONTACT, 0.95),
    ("peritumoral_edema",    "enhancing_tumor",       RelationType.SPATIAL_CONTACT, 0.95),
    ("peritumoral_edema",    "white_matter",          RelationType.COMPRESSION,     0.8),
]


# ─── Node feature schema ──────────────────────────────────────────────────────
@dataclass
class NodeFeatures:
    """
    Physiological and geometric features for each anatomical node.
    These become the node feature vector x in the GNN.
    """
    # Geometry (from mesh)
    volume_mm3:       float = 0.0
    centroid_x:       float = 0.0
    centroid_y:       float = 0.0
    centroid_z:       float = 0.0
    surface_area_mm2: float = 0.0

    # Physiology (from literature / estimated)
    blood_flow_ml_per_min:  float = 50.0
    oxygen_saturation:      float = 0.98
    metabolic_rate:         float = 1.0   # relative
    intracranial_pressure:  float = 10.0  # mmHg baseline

    # Pathology
    is_tumor:         bool  = False
    is_critical:      bool  = False  # eloquent cortex, brainstem etc.
    edema_grade:      float = 0.0    # 0–3
    enhancement_ratio:float = 0.0   # from T1ce/T1

    # Surgical accessibility (0 = inaccessible, 1 = easy)
    surgical_accessibility: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Convert to fixed-length feature vector for GNN input."""
        return np.array([
            self.volume_mm3 / 1e4,        # normalize
            self.centroid_x / 128.0,
            self.centroid_y / 128.0,
            self.centroid_z / 128.0,
            self.surface_area_mm2 / 1e4,
            self.blood_flow_ml_per_min / 100.0,
            self.oxygen_saturation,
            self.metabolic_rate,
            self.intracranial_pressure / 20.0,
            float(self.is_tumor),
            float(self.is_critical),
            self.edema_grade / 3.0,
            self.enhancement_ratio,
            self.surgical_accessibility,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "volume_norm", "cx", "cy", "cz", "surface_area_norm",
            "blood_flow_norm", "oxygen_sat", "metabolic_rate", "icp_norm",
            "is_tumor", "is_critical", "edema_grade_norm", "enhancement_ratio",
            "surgical_accessibility",
        ]


# ─── Builder ──────────────────────────────────────────────────────────────────
class AnatomicalGraphBuilder:
    """
    Constructs a NetworkX DiGraph representing the patient's brain anatomy.

    Node = anatomical structure
    Edge = physiological / spatial relationship
    """

    CRITICAL_STRUCTURES = {
        "brainstem", "internal_carotid_artery", "basilar_artery",
        "middle_cerebral_artery", "motor_cortex", "speech_area",
    }

    SPATIAL_CONTACT_THRESHOLD_MM = 5.0  # distance < this → spatial contact edge

    def __init__(self):
        self.graph = nx.DiGraph()

    def build(
        self,
        twin_summary: Dict,
        patient_id: str = "unknown",
    ) -> nx.DiGraph:
        """
        Build anatomical graph from BrainDigitalTwin summary.

        Args:
            twin_summary: output of BrainDigitalTwin.summary()
            patient_id:   patient identifier

        Returns:
            DiGraph with node features and typed, weighted edges
        """
        self.graph = nx.DiGraph(patient_id=patient_id)

        structures = twin_summary.get("structures", [])
        logger.info(f"Building anatomical graph: {len(structures)} structures")

        # Step 1: Add nodes
        for s in structures:
            self._add_node(s)

        # Step 2: Add prior anatomical edges
        self._add_anatomical_priors()

        # Step 3: Add spatial contact edges from centroids
        self._add_spatial_edges(structures)

        # Step 4: Add tumor-specific edges
        if any(s["is_tumor"] for s in structures):
            self._add_tumor_edges(structures)

        logger.info(
            f"Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def _add_node(self, structure: Dict):
        name = structure["name"]
        features = NodeFeatures(
            volume_mm3=structure.get("volume_mm3", 0.0),
            centroid_x=structure.get("centroid_mm", [0, 0, 0])[0] if "centroid_mm" in structure
                        else structure.get("centroid_voxel", [0, 0, 0])[0],
            centroid_y=structure.get("centroid_mm", [0, 0, 0])[1] if "centroid_mm" in structure
                        else structure.get("centroid_voxel", [0, 0, 0])[1],
            centroid_z=structure.get("centroid_mm", [0, 0, 0])[2] if "centroid_mm" in structure
                        else structure.get("centroid_voxel", [0, 0, 0])[2],
            is_tumor=structure.get("is_tumor", False),
            is_critical=name in self.CRITICAL_STRUCTURES,
        )

        # Adjust physiology for tumor nodes
        if features.is_tumor:
            features.blood_flow_ml_per_min = 120.0  # hypervascular
            features.metabolic_rate = 2.5
            features.surgical_accessibility = 0.3
            if name == "necrotic_tumor_core":
                features.blood_flow_ml_per_min = 5.0  # necrotic = low flow
                features.oxygen_saturation = 0.5

        self.graph.add_node(
            name,
            label_id=structure.get("label_id", -1),
            features=features,
            feature_vector=features.to_vector(),
            **{k: v for k, v in structure.items() if k not in ("centroid_mm", "centroid_voxel", "label_id")},
        )

    def _add_anatomical_priors(self):
        """Add edges from the hard-coded anatomical prior knowledge base."""
        for src, dst, rel_type, weight in BRAIN_ANATOMICAL_PRIORS:
            if src in self.graph and dst in self.graph:
                self.graph.add_edge(
                    src, dst,
                    relation=rel_type,
                    weight=weight,
                    causal=True,   # this edge participates in SCM
                    prior=True,
                )

    def _add_spatial_edges(self, structures: List[Dict]):
        """Add spatial contact edges between nearby structures."""
        # Build centroid lookup
        centroids = {}
        for s in structures:
            name = s["name"]
            if name in self.graph:
                c = s.get("centroid_mm") or s.get("centroid_voxel", [0, 0, 0])
                centroids[name] = np.array(c)

        names = list(centroids.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                dist = np.linalg.norm(centroids[a] - centroids[b])
                if dist < self.SPATIAL_CONTACT_THRESHOLD_MM:
                    # Bidirectional spatial contact
                    for src, dst in [(a, b), (b, a)]:
                        if not self.graph.has_edge(src, dst):
                            self.graph.add_edge(
                                src, dst,
                                relation=RelationType.SPATIAL_CONTACT,
                                weight=max(0.1, 1.0 - dist / self.SPATIAL_CONTACT_THRESHOLD_MM),
                                distance_mm=round(float(dist), 2),
                                causal=False,
                                prior=False,
                            )

    def _add_tumor_edges(self, structures: List[Dict]):
        """Add tumor-specific edges based on templates."""
        for src, dst, rel_type, weight in TUMOR_EDGE_TEMPLATES:
            if src in self.graph and dst in self.graph:
                self.graph.add_edge(
                    src, dst,
                    relation=rel_type,
                    weight=weight,
                    causal=True,
                    prior=False,
                )

        # Dynamic compression edges: tumor nodes → nearby critical structures
        tumor_nodes = [n for n, d in self.graph.nodes(data=True)
                       if d.get("is_tumor", False)]
        critical_nodes = [n for n in self.graph.nodes()
                          if n in self.CRITICAL_STRUCTURES and n in self.graph]

        for tumor in tumor_nodes:
            tumor_feat: NodeFeatures = self.graph.nodes[tumor]["features"]
            tumor_centroid = np.array([tumor_feat.centroid_x, tumor_feat.centroid_y, tumor_feat.centroid_z])

            for critical in critical_nodes:
                c_feat: NodeFeatures = self.graph.nodes[critical]["features"]
                c_centroid = np.array([c_feat.centroid_x, c_feat.centroid_y, c_feat.centroid_z])
                dist = np.linalg.norm(tumor_centroid - c_centroid)

                if dist < 20.0:  # within 20mm → compression risk
                    self.graph.add_edge(
                        tumor, critical,
                        relation=RelationType.COMPRESSION,
                        weight=max(0.0, 1.0 - dist / 20.0),
                        distance_mm=round(float(dist), 2),
                        causal=True,
                        prior=False,
                    )
                    logger.warning(
                        f"⚠  Compression risk: {tumor} → {critical} "
                        f"(dist={dist:.1f}mm, weight={1.0 - dist/20.0:.2f})"
                    )

    def get_node_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return (N, F) feature matrix and list of node names."""
        names = list(self.graph.nodes())
        vectors = [self.graph.nodes[n]["feature_vector"] for n in names]
        return np.stack(vectors, axis=0), names

    def get_edge_index(self) -> np.ndarray:
        """Return (2, E) edge index array for PyG."""
        names = list(self.graph.nodes())
        name_to_idx = {n: i for i, n in enumerate(names)}
        edges = list(self.graph.edges())
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        src = [name_to_idx[u] for u, v in edges]
        dst = [name_to_idx[v] for u, v in edges]
        return np.array([src, dst], dtype=np.int64)

    def to_json(self, path: Optional[str] = None) -> str:
        data = nx.node_link_data(self.graph)
        # Make features serializable
        for node in data["nodes"]:
            if "features" in node:
                features: NodeFeatures = node["features"]
                node["features"] = features.__dict__
                node["features"]["is_tumor"] = bool(node["features"]["is_tumor"])
                node["features"]["is_critical"] = bool(node["features"]["is_critical"])
            if "feature_vector" in node:
                node["feature_vector"] = node["feature_vector"].tolist()

        json_str = json.dumps(data, indent=2, default=str)
        if path:
            Path(path).write_text(json_str)
        return json_str
