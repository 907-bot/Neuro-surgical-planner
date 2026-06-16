"""
tests/test_graph.py
Unit tests for anatomical graph construction and GNN inference.
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.imaging.segmentation import BrainTumorSegmenter, BRAIN_LABELS
from src.imaging.reconstruction import BrainReconstructionPipeline, AnatomicalMesh
from src.graph.anatomical_graph import AnatomicalGraphBuilder, NodeFeatures, RelationType
from src.graph.gnn_model import AnatomicalGNNInference


MOCK_TWIN_SUMMARY = {
    "patient_id": "TEST_001",
    "structures": [
        {
            "name": "enhancing_tumor",
            "label_id": 3,
            "is_tumor": True,
            "voxel_count": 1024,
            "volume_mm3": 8192.0,
            "centroid_mm": [64.0, 64.0, 64.0],
        },
        {
            "name": "necrotic_tumor_core",
            "label_id": 1,
            "is_tumor": True,
            "voxel_count": 512,
            "volume_mm3": 4096.0,
            "centroid_mm": [64.0, 64.0, 64.0],
        },
        {
            "name": "peritumoral_edema",
            "label_id": 2,
            "is_tumor": True,
            "voxel_count": 2048,
            "volume_mm3": 16384.0,
            "centroid_mm": [64.0, 64.0, 64.0],
        },
        {
            "name": "white_matter",
            "label_id": 4,
            "is_tumor": False,
            "voxel_count": 50000,
            "volume_mm3": 400000.0,
            "centroid_mm": [60.0, 60.0, 60.0],
        },
        {
            "name": "brainstem",
            "label_id": 7,
            "is_tumor": False,
            "voxel_count": 4096,
            "volume_mm3": 32768.0,
            "centroid_mm": [64.0, 50.0, 30.0],
        },
    ],
    "tumor_count": 3,
    "total_tumor_volume_mm3": 28672.0,
    "voxel_spacing": (1.0, 1.0, 1.0),
}


# ─── Segmentation Tests ───────────────────────────────────────────────────────
class TestBrainTumorSegmenter:

    def test_mock_segmentation_runs(self):
        segmenter = BrainTumorSegmenter()
        result = segmenter._mock_segment({})
        assert "mask" in result
        assert "structures" in result
        assert result["mask"].shape == (128, 128, 128)

    def test_mock_contains_tumor(self):
        segmenter = BrainTumorSegmenter()
        result = segmenter._mock_segment({})
        tumor_structures = [s for s in result["structures"] if s["is_tumor"]]
        assert len(tumor_structures) > 0

    def test_mock_label_values_valid(self):
        segmenter = BrainTumorSegmenter()
        result = segmenter._mock_segment({})
        unique_labels = np.unique(result["mask"])
        for label in unique_labels:
            assert label in result["labels"], f"Label {label} not in labels dict"

    def test_segment_mock_path(self):
        """segment() with a non-existent path should fall back to mock."""
        segmenter = BrainTumorSegmenter()
        result = segmenter.segment({"image": "mock"})
        assert "structures" in result


# ─── Reconstruction Tests ─────────────────────────────────────────────────────
class TestBrainReconstructionPipeline:

    def _mock_seg_result(self):
        segmenter = BrainTumorSegmenter()
        return segmenter._mock_segment({})

    def test_reconstruct_returns_twin(self):
        pipeline = BrainReconstructionPipeline()
        seg = self._mock_seg_result()
        twin = pipeline.reconstruct(seg, patient_id="TEST")
        assert twin.patient_id == "TEST"
        assert isinstance(twin.meshes, dict)

    def test_twin_has_tumor_meshes(self):
        pipeline = BrainReconstructionPipeline()
        seg = self._mock_seg_result()
        twin = pipeline.reconstruct(seg)
        assert len(twin.tumor_meshes) > 0

    def test_mesh_vertices_float32(self):
        pipeline = BrainReconstructionPipeline()
        seg = self._mock_seg_result()
        twin = pipeline.reconstruct(seg)
        for name, mesh in twin.meshes.items():
            assert mesh.vertices.dtype == np.float32, f"{name} vertices not float32"

    def test_mesh_volume_positive(self):
        pipeline = BrainReconstructionPipeline()
        seg = self._mock_seg_result()
        twin = pipeline.reconstruct(seg)
        for name, mesh in twin.meshes.items():
            assert mesh.volume_mm3 >= 0, f"{name} has negative volume"

    def test_summary_serializable(self):
        import json
        pipeline = BrainReconstructionPipeline()
        seg = self._mock_seg_result()
        twin = pipeline.reconstruct(seg)
        summary = twin.summary()
        json.dumps(summary)  # should not raise


# ─── Anatomical Graph Tests ───────────────────────────────────────────────────
class TestAnatomicalGraphBuilder:

    def test_build_returns_graph(self):
        builder = AnatomicalGraphBuilder()
        G = builder.build(MOCK_TWIN_SUMMARY)
        import networkx as nx
        assert isinstance(G, nx.DiGraph)

    def test_all_structures_are_nodes(self):
        builder = AnatomicalGraphBuilder()
        G = builder.build(MOCK_TWIN_SUMMARY)
        for s in MOCK_TWIN_SUMMARY["structures"]:
            assert s["name"] in G.nodes, f"{s['name']} not in graph"

    def test_nodes_have_feature_vectors(self):
        builder = AnatomicalGraphBuilder()
        G = builder.build(MOCK_TWIN_SUMMARY)
        for name in G.nodes:
            assert "feature_vector" in G.nodes[name]
            assert G.nodes[name]["feature_vector"] is not None

    def test_feature_vector_dimension(self):
        builder = AnatomicalGraphBuilder()
        G = builder.build(MOCK_TWIN_SUMMARY)
        expected_dim = len(NodeFeatures.feature_names())
        for name in G.nodes:
            vec = G.nodes[name]["feature_vector"]
            assert len(vec) == expected_dim, f"{name} feature dim mismatch"

    def test_tumor_nodes_flagged(self):
        builder = AnatomicalGraphBuilder()
        G = builder.build(MOCK_TWIN_SUMMARY)
        for s in MOCK_TWIN_SUMMARY["structures"]:
            if s["is_tumor"]:
                assert G.nodes[s["name"]].get("is_tumor") is True

    def test_prior_edges_added(self):
        builder = AnatomicalGraphBuilder()
        # Add a vascular structure to the summary
        summary = dict(MOCK_TWIN_SUMMARY)
        summary["structures"] = list(MOCK_TWIN_SUMMARY["structures"]) + [{
            "name": "middle_cerebral_artery",
            "label_id": 10,
            "is_tumor": False,
            "voxel_count": 200,
            "volume_mm3": 1600.0,
            "centroid_mm": [64.0, 64.0, 64.0],
        }]
        G = builder.build(summary)
        # Prior: middle_cerebral_artery → white_matter should exist
        if "white_matter" in G.nodes and "middle_cerebral_artery" in G.nodes:
            assert G.has_edge("middle_cerebral_artery", "white_matter")

    def test_get_feature_matrix_shape(self):
        builder = AnatomicalGraphBuilder()
        builder.build(MOCK_TWIN_SUMMARY)
        matrix, names = builder.get_node_feature_matrix()
        n_nodes = len(MOCK_TWIN_SUMMARY["structures"])
        assert matrix.shape == (n_nodes, len(NodeFeatures.feature_names()))
        assert len(names) == n_nodes

    def test_edge_index_shape(self):
        builder = AnatomicalGraphBuilder()
        builder.build(MOCK_TWIN_SUMMARY)
        ei = builder.get_edge_index()
        assert ei.ndim == 2
        assert ei.shape[0] == 2

    def test_json_serializable(self):
        import json
        builder = AnatomicalGraphBuilder()
        builder.build(MOCK_TWIN_SUMMARY)
        json_str = builder.to_json()
        data = json.loads(json_str)
        assert "nodes" in data


# ─── GNN Inference Tests ──────────────────────────────────────────────────────
class TestAnatomicalGNNInference:

    def test_mock_prediction_runs(self):
        gnn = AnatomicalGNNInference()
        builder = AnatomicalGraphBuilder()
        builder.build(MOCK_TWIN_SUMMARY)
        matrix, names = builder.get_node_feature_matrix()
        ei = builder.get_edge_index()
        pred = gnn.predict(matrix, ei, names)
        assert pred is not None

    def test_node_risks_all_present(self):
        gnn = AnatomicalGNNInference()
        builder = AnatomicalGraphBuilder()
        builder.build(MOCK_TWIN_SUMMARY)
        matrix, names = builder.get_node_feature_matrix()
        ei = builder.get_edge_index()
        pred = gnn.predict(matrix, ei, names)
        for name in names:
            assert name in pred.node_risks

    def test_risk_values_in_range(self):
        gnn = AnatomicalGNNInference()
        builder = AnatomicalGraphBuilder()
        builder.build(MOCK_TWIN_SUMMARY)
        matrix, names = builder.get_node_feature_matrix()
        ei = builder.get_edge_index()
        pred = gnn.predict(matrix, ei, names)
        for name, risk in pred.node_risks.items():
            assert 0.0 <= risk <= 1.0, f"{name} risk {risk} out of range"

    def test_tumor_nodes_higher_risk(self):
        gnn = AnatomicalGNNInference()
        builder = AnatomicalGraphBuilder()
        builder.build(MOCK_TWIN_SUMMARY)
        matrix, names = builder.get_node_feature_matrix()
        ei = builder.get_edge_index()
        pred = gnn._mock_predict(names)

        tumor_risks = [v for k, v in pred.node_risks.items() if "tumor" in k]
        non_tumor = [v for k, v in pred.node_risks.items() if "tumor" not in k]
        if tumor_risks and non_tumor:
            assert np.mean(tumor_risks) > np.mean(non_tumor)

    def test_prediction_to_dict(self):
        import json
        gnn = AnatomicalGNNInference()
        builder = AnatomicalGraphBuilder()
        builder.build(MOCK_TWIN_SUMMARY)
        matrix, names = builder.get_node_feature_matrix()
        ei = builder.get_edge_index()
        pred = gnn.predict(matrix, ei, names)
        json.dumps(pred.to_dict())  # should not raise
