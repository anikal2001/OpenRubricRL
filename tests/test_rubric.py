"""Tests for rubric data structures."""

import pytest
from pathlib import Path
import tempfile
import json

from openrubricrl.core.rubric import Rubric, Criterion, Scale, Example


def test_rubric_creation():
    """Test basic rubric creation."""
    scale = Scale(min=0.0, max=10.0)
    criteria = [
        Criterion(name="quality", description="Overall quality", weight=0.6),
        Criterion(name="accuracy", description="Accuracy", weight=0.4)
    ]
    
    rubric = Rubric(
        name="test_rubric",
        version="1.0.0",
        scale=scale,
        criteria=criteria
    )
    
    assert rubric.name == "test_rubric"
    assert rubric.version == "1.0.0"
    assert len(rubric.criteria) == 2
    assert rubric.scale.min == 0.0
    assert rubric.scale.max == 10.0


def test_weight_validation():
    """Test that criterion weights must sum to 1.0."""
    scale = Scale(min=0.0, max=10.0)
    
    # Should fail - weights don't sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        criteria = [
            Criterion(name="quality", description="Test", weight=0.5),
            Criterion(name="accuracy", description="Test", weight=0.3)  # Total = 0.8
        ]
        Rubric(name="test", version="1.0.0", scale=scale, criteria=criteria)
    
    # Should pass - weights sum to 1.0
    criteria = [
        Criterion(name="quality", description="Test", weight=0.6),
        Criterion(name="accuracy", description="Test", weight=0.4)
    ]
    rubric = Rubric(name="test", version="1.0.0", scale=scale, criteria=criteria)
    assert len(rubric.criteria) == 2


def test_rubric_from_file():
    """Test loading rubric from file."""
    rubric_data = {
        "name": "file_test",
        "version": "1.0.0",
        "scale": {"min": 0.0, "max": 10.0},
        "criteria": [
            {"name": "quality", "description": "Test", "weight": 1.0}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(rubric_data, f)
        temp_path = f.name
    
    try:
        rubric = Rubric.from_file(temp_path)
        assert rubric.name == "file_test"
        assert rubric.version == "1.0.0"
        assert len(rubric.criteria) == 1
    finally:
        Path(temp_path).unlink()


def test_rubric_to_dict():
    """Test converting rubric to dictionary."""
    scale = Scale(min=0.0, max=10.0)
    criteria = [
        Criterion(name="quality", description="Test", weight=1.0)
    ]
    
    rubric = Rubric(
        name="dict_test",
        version="1.0.0",
        scale=scale,
        criteria=criteria
    )
    
    data = rubric.to_dict()
    assert data["name"] == "dict_test"
    assert data["version"] == "1.0.0"
    assert data["scale"]["min"] == 0.0
    assert len(data["criteria"]) == 1


def test_criterion_with_examples():
    """Test criterion with examples."""
    example = Example(
        input="Test input",
        output="Test output",
        score=8.5,
        explanation="Good example"
    )
    
    criterion = Criterion(
        name="quality",
        description="Test criterion",
        weight=1.0,
        examples={"excellent": [example]}
    )
    
    assert criterion.examples["excellent"][0].score == 8.5
    assert criterion.examples["excellent"][0].explanation == "Good example"


def test_version_format_validation():
    """Test semantic version validation."""
    scale = Scale(min=0.0, max=10.0)
    criteria = [Criterion(name="test", description="Test", weight=1.0)]
    
    # Valid version
    rubric = Rubric(name="test", version="1.2.3", scale=scale, criteria=criteria)
    assert rubric.version == "1.2.3"
    
    # Invalid version should raise validation error
    with pytest.raises(ValueError):
        Rubric(name="test", version="1.2", scale=scale, criteria=criteria)