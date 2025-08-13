"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
import tempfile
import json
from pathlib import Path

from openrubricrl.api.server import create_app, rubric_manager
from openrubricrl.core.rubric import Rubric, Criterion, Scale


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_rubric_file():
    """Create a temporary rubric file."""
    rubric_data = {
        "name": "test_rubric",
        "version": "1.0.0",
        "scale": {"min": 0.0, "max": 10.0},
        "criteria": [
            {"name": "quality", "description": "Overall quality", "weight": 0.6},
            {"name": "accuracy", "description": "Accuracy", "weight": 0.4}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(rubric_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_rubrics_empty(client):
    """Test listing rubrics when none are loaded."""
    # Clear any existing rubrics
    rubric_manager.rubrics.clear()
    rubric_manager.scorers.clear()
    
    response = client.get("/rubrics")
    assert response.status_code == 200
    assert response.json() == []


def test_load_rubric(client, sample_rubric_file):
    """Test loading a rubric via API."""
    response = client.post(
        "/load-rubric",
        params={
            "file_path": sample_rubric_file,
            "provider": "openai"
        }
    )
    assert response.status_code == 200
    assert "loaded successfully" in response.json()["message"]


def test_list_rubrics_after_load(client, sample_rubric_file):
    """Test listing rubrics after loading one."""
    # Load rubric first
    client.post("/load-rubric", params={"file_path": sample_rubric_file})
    
    response = client.get("/rubrics")
    assert response.status_code == 200
    
    rubrics = response.json()
    assert len(rubrics) == 1
    assert rubrics[0]["name"] == "test_rubric"
    assert rubrics[0]["version"] == "1.0.0"
    assert rubrics[0]["criteria_count"] == 2


def test_get_rubric_details(client, sample_rubric_file):
    """Test getting detailed rubric information."""
    # Load rubric first
    client.post("/load-rubric", params={"file_path": sample_rubric_file})
    
    response = client.get("/rubrics/test_rubric")
    assert response.status_code == 200
    
    rubric_data = response.json()
    assert rubric_data["name"] == "test_rubric"
    assert rubric_data["version"] == "1.0.0"
    assert len(rubric_data["criteria"]) == 2


def test_get_nonexistent_rubric(client):
    """Test getting a rubric that doesn't exist."""
    response = client.get("/rubrics/nonexistent")
    assert response.status_code == 404


def test_score_without_api_key(client, sample_rubric_file):
    """Test scoring without API key (should fail gracefully)."""
    # Load rubric first
    client.post("/load-rubric", params={"file_path": sample_rubric_file})
    
    # Try to score without API key
    response = client.post(
        "/score/test_rubric",
        json={
            "task_input": "Write a hello world program",
            "model_output": "print('Hello, World!')"
        }
    )
    
    # Should fail due to missing API key
    assert response.status_code == 500


def test_score_with_invalid_rubric(client):
    """Test scoring with non-existent rubric."""
    response = client.post(
        "/score/invalid_rubric",
        json={
            "task_input": "Test input",
            "model_output": "Test output"
        }
    )
    assert response.status_code == 404


def test_api_models_validation():
    """Test that API models validate correctly."""
    from openrubricrl.api.models import ScoreRequest, ScoreResponse
    
    # Valid request
    request = ScoreRequest(
        task_input="Test input",
        model_output="Test output"
    )
    assert request.task_input == "Test input"
    assert request.model_output == "Test output"
    
    # Valid response
    response = ScoreResponse(
        overall_score=8.5,
        overall_explanation="Good output",
        criterion_scores={"quality": 8.0, "accuracy": 9.0},
        criterion_explanations={"quality": "High quality", "accuracy": "Very accurate"},
        rubric_name="test",
        rubric_version="1.0.0"
    )
    assert response.overall_score == 8.5
    assert response.rubric_name == "test"


def test_load_invalid_file(client):
    """Test loading an invalid file."""
    response = client.post(
        "/load-rubric",
        params={"file_path": "/nonexistent/file.json"}
    )
    assert response.status_code == 404