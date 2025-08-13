"""Rubric data structures and validation."""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

import jsonschema
from pydantic import BaseModel, Field, validator


class Example(BaseModel):
    """Example of input/output pair with score and explanation."""
    input: str
    output: str
    score: float
    explanation: str


class SubCriterion(BaseModel):
    """Sub-criterion within a main criterion."""
    name: str
    description: str


class Criterion(BaseModel):
    """A single evaluation criterion."""
    name: str
    description: str
    weight: float = Field(..., ge=0, le=1)
    examples: Optional[Dict[str, List[Example]]] = None
    subcriteria: Optional[List[SubCriterion]] = None


class Scale(BaseModel):
    """Scoring scale definition."""
    min: float
    max: float
    type: str = Field(default="continuous", regex="^(continuous|discrete)$")


class HybridMetric(BaseModel):
    """Automated metric to combine with LLM scoring."""
    name: str
    type: str = Field(..., regex="^(bleu|rouge|accuracy|perplexity|custom)$")
    weight: float = Field(..., ge=0, le=1)
    config: Optional[Dict[str, Any]] = None


class Metadata(BaseModel):
    """Rubric metadata."""
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: Optional[List[str]] = None
    license: str = "MIT"


class Rubric(BaseModel):
    """Main rubric class."""
    name: str
    version: str = Field(..., regex=r"^\d+\.\d+\.\d+$")
    description: Optional[str] = None
    domain: Optional[str] = Field(None, regex="^(code|dialogue|creative_writing|reasoning|general)$")
    scale: Scale
    criteria: List[Criterion] = Field(..., min_items=1)
    hybrid_metrics: Optional[List[HybridMetric]] = None
    metadata: Optional[Metadata] = None

    @validator('criteria')
    def validate_weights_sum_to_one(cls, v):
        """Ensure criterion weights sum to 1.0."""
        total_weight = sum(c.weight for c in v)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Criterion weights must sum to 1.0, got {total_weight}")
        return v

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "Rubric":
        """Load rubric from JSON or YAML file."""
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rubric":
        """Create rubric from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rubric to dictionary."""
        return self.dict(exclude_none=True)

    def to_file(self, file_path: Union[str, Path], format: str = "auto") -> None:
        """Save rubric to file."""
        file_path = Path(file_path)
        
        if format == "auto":
            format = "yaml" if file_path.suffix.lower() in ['.yaml', '.yml'] else "json"
        
        data = self.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format == "yaml":
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def validate_schema(self, schema_path: Optional[Union[str, Path]] = None) -> bool:
        """Validate rubric against JSON schema."""
        if schema_path is None:
            # Use default schema from package
            schema_path = Path(__file__).parent.parent.parent.parent / "rubric_schema.json"
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        try:
            jsonschema.validate(self.to_dict(), schema)
            return True
        except jsonschema.ValidationError:
            return False