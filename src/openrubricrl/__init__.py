"""OpenRubricRL - Convert rubrics into LLM-based reward functions."""

__version__ = "0.1.0"
__author__ = "OpenRubricRL Team"

from .core.rubric import Rubric
from .core.scorer import RubricScorer
from .api.server import create_app

__all__ = ["Rubric", "RubricScorer", "create_app"]