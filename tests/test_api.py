"""
tests/test_api.py
Basic tests for the Brain Surgical Planner API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "brain-surgical-planner"


def test_list_actions():
    """Test list actions endpoint."""
    response = client.get("/actions")
    assert response.status_code == 200
    data = response.json()
    assert "actions" in data
    assert len(data["actions"]) > 0


def test_scm_variables():
    """Test SCM variables endpoint."""
    response = client.get("/scm/variables")
    assert response.status_code == 200
    data = response.json()
    assert "variables" in data
    assert "dag_edges" in data


def test_intervene():
    """Test intervene endpoint."""
    response = client.post(
        "/intervene",
        json={
            "action": "remove_tumor_full",
            "patient_params": {
                "tumor_size": 0.3,
                "blood_flow": 0.7,
                "oxygen_saturation": 0.95,
                "intracranial_pressure": 0.2,
                "edema_volume": 0.2,
                "inflammatory_response": 0.3,
                "mass_effect": 0.25,
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "recovery_gain" in data
    assert "net_utility" in data


def test_counterfactual():
    """Test counterfactual endpoint."""
    response = client.post(
        "/counterfactual",
        json={
            "factual_action": "remove_tumor_full",
            "counterfactual_action": "drain_csf",
            "patient_params": {
                "tumor_size": 0.3,
                "blood_flow": 0.7,
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "was_better" in data
    assert "recovery_delta" in data


def test_search_plans():
    """Test search plans endpoint."""
    response = client.post(
        "/search/plans",
        json={
            "patient_params": {"tumor_size": 0.3},
            "n_simulations": 50,
            "top_k": 3,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "top_plans" in data
    assert len(data["top_plans"]) <= 3


def test_simulation_channels():
    """Test simulation channels endpoint."""
    response = client.get("/simulation/channels")
    assert response.status_code == 200
    data = response.json()
    assert "channels" in data
    assert data["n_channels"] > 0


def test_simulation_vitals():
    """Test simulation vitals endpoint."""
    response = client.post(
        "/simulation/vitals",
        json={
            "vitals": {
                "blood_pressure_systolic": 120,
                "heart_rate": 72,
                "spo2": 98,
                "intracranial_pressure": 15,
                "cerebral_blood_flow": 45,
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "alert_level" in data
    assert "predicted_outcome" in data


def test_knowledge_graph_stats():
    """Test knowledge graph stats endpoint."""
    response = client.get("/knowledge-graph/stats")
    assert response.status_code == 200
    data = response.json()
    assert "n_nodes" in data
    assert "n_edges" in data


def test_knowledge_graph_blood_supply():
    """Test knowledge graph blood supply endpoint."""
    response = client.get("/knowledge-graph/blood-supply/white_matter")
    assert response.status_code == 200
    data = response.json()
    assert "suppliers" in data


def test_create_patient():
    """Test create patient endpoint."""
    response = client.post(
        "/patients",
        json={
            "patient_id": "TEST001",
            "name": "Test Patient",
            "diagnosis": "Glioblastoma",
            "tumor_type": "glioblastoma",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "TEST001"


def test_list_patients():
    """Test list patients endpoint."""
    response = client.get("/patients")
    assert response.status_code == 200
    data = response.json()
    assert "patients" in data
    assert "count" in data


def test_get_patient():
    """Test get patient endpoint."""
    # First create a patient
    client.post(
        "/patients",
        json={"patient_id": "TEST002", "name": "Test Patient 2"}
    )
    
    response = client.get("/patients/TEST002")
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "TEST002"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
