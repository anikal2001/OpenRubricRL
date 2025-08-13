"""Tests for prompt builder."""

import pytest
from openrubricrl.core.rubric import Rubric, Criterion, Scale, Example
from openrubricrl.core.prompt_builder import PromptBuilder, build_prompt_for_rubric


@pytest.fixture
def sample_rubric():
    """Create a sample rubric for testing."""
    scale = Scale(min=0.0, max=10.0)
    
    example = Example(
        input="Write a function to add two numbers",
        output="def add(a, b): return a + b",
        score=9.0,
        explanation="Simple and correct implementation"
    )
    
    criteria = [
        Criterion(
            name="correctness",
            description="Does the code work correctly?",
            weight=0.6,
            examples={"excellent": [example]}
        ),
        Criterion(
            name="readability",
            description="Is the code readable?",
            weight=0.4
        )
    ]
    
    return Rubric(
        name="code_quality",
        version="1.0.0",
        description="Basic code quality rubric",
        domain="code",
        scale=scale,
        criteria=criteria
    )


def test_prompt_builder_creation(sample_rubric):
    """Test creating a prompt builder."""
    builder = PromptBuilder(sample_rubric)
    assert builder.rubric.name == "code_quality"


def test_build_scoring_prompt(sample_rubric):
    """Test building a complete scoring prompt."""
    builder = PromptBuilder(sample_rubric)
    
    prompt = builder.build_scoring_prompt(
        task_input="Write a function to multiply two numbers",
        model_output="def multiply(x, y): return x * y",
        include_examples=True
    )
    
    # Check that essential components are in the prompt
    assert "code_quality" in prompt
    assert "correctness" in prompt
    assert "readability" in prompt
    assert "Write a function to multiply two numbers" in prompt
    assert "def multiply(x, y): return x * y" in prompt
    assert "JSON format" in prompt
    assert "0.0" in prompt and "10.0" in prompt  # Scale


def test_prompt_without_examples(sample_rubric):
    """Test building prompt without examples."""
    builder = PromptBuilder(sample_rubric)
    
    prompt = builder.build_scoring_prompt(
        task_input="Test input",
        model_output="Test output",
        include_examples=False
    )
    
    # Should not contain example content
    assert "def add(a, b): return a + b" not in prompt
    assert "Simple and correct implementation" not in prompt
    
    # But should still contain main components
    assert "correctness" in prompt
    assert "readability" in prompt


def test_build_prompt_for_rubric_convenience_function(sample_rubric):
    """Test the convenience function."""
    prompt = build_prompt_for_rubric(
        rubric=sample_rubric,
        task_input="Test input",
        model_output="Test output"
    )
    
    assert "code_quality" in prompt
    assert "Test input" in prompt
    assert "Test output" in prompt


def test_prompt_includes_weights(sample_rubric):
    """Test that prompt includes criterion weights."""
    builder = PromptBuilder(sample_rubric)
    
    prompt = builder.build_scoring_prompt(
        task_input="Test",
        model_output="Test"
    )
    
    # Should include weight information
    assert "60%" in prompt or "0.6" in prompt  # correctness weight
    assert "40%" in prompt or "0.4" in prompt  # readability weight


def test_prompt_output_format(sample_rubric):
    """Test that prompt specifies correct output format."""
    builder = PromptBuilder(sample_rubric)
    
    prompt = builder.build_scoring_prompt(
        task_input="Test",
        model_output="Test"
    )
    
    # Should specify the expected JSON structure
    assert "correctness_score" in prompt
    assert "correctness_explanation" in prompt
    assert "readability_score" in prompt
    assert "readability_explanation" in prompt
    assert "overall_score" in prompt
    assert "overall_explanation" in prompt