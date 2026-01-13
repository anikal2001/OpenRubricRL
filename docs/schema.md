# Rubric Schema Reference

This document provides a comprehensive reference for the OpenRubricRL rubric schema. Rubrics define how model outputs should be evaluated and are converted into LLM-based scoring prompts.

## 📋 Table of Contents

- [Overview](#overview)
- [Schema Structure](#schema-structure)
- [Field Reference](#field-reference)
- [Examples](#examples)
- [Validation](#validation)
- [Best Practices](#best-practices)

## Overview

OpenRubricRL uses a structured JSON/YAML schema to define evaluation rubrics. Each rubric contains:

- **Metadata**: Name, version, description, and domain
- **Scale**: Scoring range and type
- **Criteria**: Individual evaluation dimensions with weights
- **Examples**: Sample inputs/outputs with scores (optional)
- **Hybrid Metrics**: Automated metrics to combine with LLM scoring (optional)

## Schema Structure

```json
{
  "name": "string",
  "version": "string (semver)",
  "description": "string (optional)",
  "domain": "enum (optional)",
  "scale": {
    "min": "number",
    "max": "number", 
    "type": "enum"
  },
  "criteria": [
    {
      "name": "string",
      "description": "string",
      "weight": "number (0-1)",
      "examples": "object (optional)",
      "subcriteria": "array (optional)"
    }
  ],
  "hybrid_metrics": "array (optional)",
  "metadata": "object (optional)"
}
```

## Field Reference

### Root Level Fields

#### `name` (required)
- **Type**: `string`
- **Description**: Unique identifier for the rubric
- **Example**: `"code_quality_basic"`

#### `version` (required)
- **Type**: `string`
- **Pattern**: `^\d+\.\d+\.\d+$` (semantic versioning)
- **Description**: Version of the rubric following semver format
- **Example**: `"1.0.0"`

#### `description` (optional)
- **Type**: `string`
- **Description**: Human-readable description of what this rubric evaluates
- **Example**: `"Basic code quality evaluation rubric for Python functions"`

#### `domain` (optional)
- **Type**: `string`
- **Enum**: `["code", "dialogue", "creative_writing", "reasoning", "general"]`
- **Description**: Domain this rubric is designed for
- **Example**: `"code"`

### Scale Object

#### `scale` (required)
Defines the scoring range and type.

```json
{
  "min": 0.0,
  "max": 10.0,
  "type": "continuous"
}
```

##### `scale.min` (required)
- **Type**: `number`
- **Description**: Minimum possible score
- **Example**: `0.0`

##### `scale.max` (required)
- **Type**: `number`
- **Description**: Maximum possible score
- **Example**: `10.0`

##### `scale.type` (optional)
- **Type**: `string`
- **Enum**: `["continuous", "discrete"]`
- **Default**: `"continuous"`
- **Description**: Whether scores can be any value in range (continuous) or only integers (discrete)

### Criteria Array

#### `criteria` (required)
Array of evaluation criteria. Must contain at least one criterion.

```json
{
  "name": "correctness",
  "description": "Does the code solve the problem correctly?",
  "weight": 0.4,
  "examples": {
    "excellent": [...],
    "good": [...],
    "poor": [...]
  },
  "subcriteria": [...]
}
```

##### `criteria[].name` (required)
- **Type**: `string`
- **Description**: Unique name for this criterion
- **Example**: `"correctness"`

##### `criteria[].description` (required)
- **Type**: `string`
- **Description**: Detailed description of what this criterion evaluates
- **Example**: `"Does the code solve the problem correctly and handle edge cases?"`

##### `criteria[].weight` (required)
- **Type**: `number`
- **Range**: `0.0` to `1.0`
- **Constraint**: All criterion weights must sum to `1.0`
- **Description**: Relative importance of this criterion
- **Example**: `0.4`

##### `criteria[].examples` (optional)
Object containing example inputs/outputs with scores and explanations.

```json
{
  "excellent": [
    {
      "input": "Write a function to reverse a string",
      "output": "def reverse_string(s):\n    return s[::-1]",
      "score": 9.5,
      "explanation": "Correct and efficient implementation"
    }
  ],
  "good": [...],
  "poor": [...]
}
```

**Example Object Structure:**
- `input` (string): The task or prompt given
- `output` (string): The model's response
- `score` (number): Score for this example
- `explanation` (string): Why this score was given

##### `criteria[].subcriteria` (optional)
Array of sub-criteria for more granular evaluation.

```json
[
  {
    "name": "naming",
    "description": "Variable and function names are descriptive"
  },
  {
    "name": "structure", 
    "description": "Code is well-organized and follows conventions"
  }
]
```

### Hybrid Metrics (Optional)

#### `hybrid_metrics` (optional)
Array of automated metrics to combine with LLM scoring.

```json
[
  {
    "name": "bleu_score",
    "type": "bleu",
    "weight": 0.2,
    "config": {
      "reference_key": "expected_output"
    }
  }
]
```

##### `hybrid_metrics[].name` (required)
- **Type**: `string`
- **Description**: Name of the metric

##### `hybrid_metrics[].type` (required)
- **Type**: `string`
- **Enum**: `["bleu", "rouge", "accuracy", "perplexity", "custom"]`
- **Description**: Type of automated metric

##### `hybrid_metrics[].weight` (required)
- **Type**: `number`
- **Range**: `0.0` to `1.0`
- **Description**: Weight of this metric in final score

##### `hybrid_metrics[].config` (optional)
- **Type**: `object`
- **Description**: Configuration specific to the metric type

### Metadata (Optional)

#### `metadata` (optional)
Additional information about the rubric.

```json
{
  "author": "OpenRubricRL Team",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "tags": ["python", "code-quality"],
  "license": "MIT"
}
```

## Examples

### Basic Code Quality Rubric

```json
{
  "name": "code_quality_basic",
  "version": "1.0.0",
  "description": "Basic code quality evaluation for Python functions",
  "domain": "code",
  "scale": {
    "min": 0.0,
    "max": 10.0,
    "type": "continuous"
  },
  "criteria": [
    {
      "name": "correctness",
      "description": "Does the code solve the problem correctly?",
      "weight": 0.5,
      "examples": {
        "excellent": [
          {
            "input": "Write a function to find the maximum of two numbers",
            "output": "def max_two(a, b):\n    return a if a > b else b",
            "score": 9.0,
            "explanation": "Correct, concise, and readable"
          }
        ]
      }
    },
    {
      "name": "style",
      "description": "Does the code follow Python style conventions?",
      "weight": 0.3,
      "subcriteria": [
        {
          "name": "naming",
          "description": "Uses descriptive variable names"
        },
        {
          "name": "formatting",
          "description": "Follows PEP 8 formatting"
        }
      ]
    },
    {
      "name": "efficiency",
      "description": "Is the solution efficient?",
      "weight": 0.2
    }
  ],
  "metadata": {
    "author": "OpenRubricRL Team",
    "tags": ["python", "code-quality"],
    "license": "MIT"
  }
}
```

### Dialogue Quality Rubric

```yaml
name: dialogue_quality
version: 2.1.0
description: Evaluates quality of conversational AI responses
domain: dialogue
scale:
  min: 1
  max: 5
  type: discrete
criteria:
  - name: relevance
    description: How well does the response address the user's question?
    weight: 0.4
  - name: helpfulness
    description: Does the response provide useful information?
    weight: 0.3
  - name: tone
    description: Is the tone appropriate and friendly?
    weight: 0.2
  - name: safety
    description: Is the response safe and appropriate?
    weight: 0.1
```

### Creative Writing Rubric with Hybrid Metrics

```json
{
  "name": "creative_writing_advanced",
  "version": "1.2.0",
  "description": "Advanced creative writing evaluation with automated metrics",
  "domain": "creative_writing",
  "scale": {
    "min": 0.0,
    "max": 100.0,
    "type": "continuous"
  },
  "criteria": [
    {
      "name": "creativity",
      "description": "Originality and creative expression",
      "weight": 0.4
    },
    {
      "name": "coherence",
      "description": "Logical flow and structure",
      "weight": 0.3
    },
    {
      "name": "engagement",
      "description": "How engaging and interesting is the writing?",
      "weight": 0.3
    }
  ],
  "hybrid_metrics": [
    {
      "name": "readability",
      "type": "custom",
      "weight": 0.1,
      "config": {
        "metric_function": "flesch_reading_ease"
      }
    }
  ]
}
```

## Validation

### Automatic Validation

OpenRubricRL automatically validates rubrics when loading:

```python
from openrubricrl.core.rubric import Rubric

# Load and validate from file
rubric = Rubric.from_file("my_rubric.json")

# Validate against JSON schema
is_valid = rubric.validate_schema()
```

### Common Validation Errors

1. **Weight Sum Error**: Criterion weights don't sum to 1.0
   ```
   ValueError: Criterion weights must sum to 1.0, got 0.9
   ```

2. **Invalid Version**: Version doesn't follow semver format
   ```
   ValidationError: version must match pattern ^\d+\.\d+\.\d+$
   ```

3. **Invalid Domain**: Domain not in allowed values
   ```
   ValidationError: domain must be one of: code, dialogue, creative_writing, reasoning, general
   ```

### CLI Validation

Use the CLI to validate rubrics:

```bash
# Validate only
openrubricrl validate my_rubric.json --validate-only

# Validate and pretty-print
openrubricrl validate my_rubric.json --output-format yaml
```

## Best Practices

### 1. Criterion Design

- **Use 3-7 criteria**: Too few lacks nuance, too many becomes unwieldy
- **Make criteria independent**: Avoid overlap between criteria
- **Weight by importance**: Critical aspects should have higher weights
- **Provide clear descriptions**: Be specific about what each criterion evaluates

### 2. Examples

- **Include diverse examples**: Cover excellent, good, and poor cases
- **Provide explanations**: Help the LLM understand your reasoning
- **Use realistic scenarios**: Examples should reflect actual use cases
- **Keep examples concise**: Focus on the key aspects being evaluated

### 3. Scale Selection

- **Continuous vs Discrete**: Use continuous for nuanced scoring, discrete for clear categories
- **Appropriate range**: 0-10 for detailed scoring, 1-5 for simple ratings
- **Consider your use case**: Match scale to how scores will be used

### 4. Version Management

- **Semantic versioning**: Use major.minor.patch format
- **Document changes**: Keep track of what changed between versions
- **Backward compatibility**: Consider impact of changes on existing evaluations

### 5. Domain Specificity

- **Choose appropriate domain**: Helps with prompt optimization
- **Domain-specific criteria**: Tailor criteria to your specific domain
- **Use domain examples**: Examples should be relevant to the domain

### 6. Testing and Iteration

- **Test with real data**: Validate rubrics on actual model outputs
- **Iterate based on results**: Refine criteria and weights based on performance
- **Get human feedback**: Have domain experts review your rubrics

## Schema Validation

The complete JSON schema is available at `rubric_schema.json` in the repository root. You can use it to validate rubrics in any JSON schema validator:

```bash
# Using ajv-cli
ajv validate -s rubric_schema.json -d my_rubric.json

# Using jsonschema (Python)
python -m jsonschema rubric_schema.json my_rubric.json
```

## Migration Guide

### From v1.0 to v2.0

If you have rubrics from an older version, here are the key changes:

1. **Required fields**: `version` is now required
2. **Weight validation**: Weights must sum exactly to 1.0
3. **Domain enum**: Limited to specific domains
4. **Example structure**: Examples now require `explanation` field

### Automated Migration

Use the CLI to help migrate old rubrics:

```bash
openrubricrl migrate old_rubric.json --to-version 2.0.0
```

---

For more information, see the [API Documentation](api.md) and [Integration Guide](integrations.md).