# API Documentation

This document provides comprehensive documentation for the OpenRubricRL REST API. The API allows you to score model outputs using rubrics via HTTP endpoints.

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Request/Response Models](#requestresponse-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

The OpenRubricRL API is built with FastAPI and provides endpoints for:

- **Scoring**: Evaluate model outputs using loaded rubrics
- **Batch Scoring**: Score multiple outputs efficiently
- **Rubric Management**: Load, list, and inspect rubrics
- **Health Monitoring**: Check API status

### Base URL

```
http://localhost:8000  # Default local development
```

### API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Getting Started

### Starting the Server

#### Using CLI
```bash
# Start with default settings
openrubricrl serve

# Custom host and port
openrubricrl serve --host 0.0.0.0 --port 8080

# Auto-load rubrics from directory
openrubricrl serve --rubrics-dir ./rubrics/

# Development mode with auto-reload
openrubricrl serve --reload
```

#### Using Python
```python
import uvicorn
from openrubricrl.api.server import app

uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Loading Rubrics

Before scoring, you need to load rubrics into the server:

```bash
# Load via API
curl -X POST "http://localhost:8000/load-rubric" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./my_rubric.json"}'

# Or auto-load when starting server
openrubricrl serve --rubrics-dir ./rubrics/
```

## Authentication

The API currently supports API key authentication for LLM providers:

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Request Headers
```bash
# Pass API keys in request body
curl -X POST "http://localhost:8000/score/my_rubric" \
  -H "Content-Type: application/json" \
  -d '{
    "task_input": "...",
    "model_output": "...",
    "llm_config": {
      "api_key": "your-api-key",
      "provider": "openai"
    }
  }'
```

## Endpoints

### Health Check

#### `GET /health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Rubric Management

#### `GET /rubrics`

List all loaded rubrics.

**Response:**
```json
[
  {
    "name": "code_quality_basic",
    "version": "1.0.0",
    "description": "Basic code quality evaluation",
    "domain": "code",
    "criteria_count": 3,
    "scale_min": 0.0,
    "scale_max": 10.0
  }
]
```

#### `GET /rubrics/{rubric_name}`

Get detailed information about a specific rubric.

**Parameters:**
- `rubric_name` (path): Name of the rubric

**Response:**
```json
{
  "name": "code_quality_basic",
  "version": "1.0.0",
  "description": "Basic code quality evaluation",
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
      "weight": 0.4
    }
  ]
}
```

#### `POST /load-rubric`

Load a rubric from file.

**Request Body:**
```json
{
  "file_path": "/path/to/rubric.json",
  "name": "custom_name",
  "provider": "openai",
  "model": "gpt-4"
}
```

**Response:**
```json
{
  "message": "Rubric code_quality_basic loaded successfully"
}
```

### Scoring

#### `POST /score/{rubric_name}`

Score a model output using a specific rubric.

**Parameters:**
- `rubric_name` (path): Name of the rubric to use

**Request Body:**
```json
{
  "task_input": "Write a function to reverse a string",
  "model_output": "def reverse_string(s):\n    return s[::-1]",
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.1,
    "api_key": "your-api-key"
  }
}
```

**Response:**
```json
{
  "overall_score": 8.5,
  "overall_explanation": "Good implementation with correct logic and clean code style.",
  "criterion_scores": {
    "correctness": 9.0,
    "readability": 8.0,
    "efficiency": 8.5
  },
  "criterion_explanations": {
    "correctness": "Correctly reverses the string using slicing",
    "readability": "Clean and readable code with good naming",
    "efficiency": "Efficient O(n) solution using Python slicing"
  },
  "rubric_name": "code_quality_basic",
  "rubric_version": "1.0.0"
}
```

#### `POST /score`

Score using the default rubric or specify rubric in request body.

**Request Body:**
```json
{
  "task_input": "Write a function to reverse a string",
  "model_output": "def reverse_string(s):\n    return s[::-1]",
  "rubric_name": "code_quality_basic",
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4"
  }
}
```

### Batch Scoring

#### `POST /batch-score/{rubric_name}`

Score multiple outputs using a specific rubric.

**Parameters:**
- `rubric_name` (path): Name of the rubric to use

**Request Body:**
```json
{
  "items": [
    {
      "task_input": "Write a function to reverse a string",
      "model_output": "def reverse_string(s):\n    return s[::-1]"
    },
    {
      "task_input": "Write a function to find maximum",
      "model_output": "def find_max(a, b):\n    return max(a, b)"
    }
  ],
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.1
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "overall_score": 8.5,
      "overall_explanation": "Good implementation...",
      "criterion_scores": {...},
      "criterion_explanations": {...},
      "rubric_name": "code_quality_basic",
      "rubric_version": "1.0.0"
    }
  ],
  "summary": {
    "total_items": 2,
    "average_score": 8.2,
    "min_score": 7.8,
    "max_score": 8.5
  }
}
```

## Request/Response Models

### ScoreRequest

```json
{
  "task_input": "string (required)",
  "model_output": "string (required)", 
  "rubric_name": "string (optional)",
  "llm_config": {
    "provider": "openai|anthropic (optional)",
    "model": "string (optional)",
    "api_key": "string (optional)",
    "temperature": "number (optional)",
    "max_tokens": "number (optional)"
  }
}
```

### BatchScoreRequest

```json
{
  "items": [
    {
      "task_input": "string",
      "model_output": "string"
    }
  ],
  "rubric_name": "string (optional)",
  "llm_config": "object (optional)"
}
```

### ScoreResponse

```json
{
  "overall_score": "number",
  "overall_explanation": "string",
  "criterion_scores": {
    "criterion_name": "number"
  },
  "criterion_explanations": {
    "criterion_name": "string"
  },
  "rubric_name": "string",
  "rubric_version": "string"
}
```

### BatchScoreResponse

```json
{
  "results": ["ScoreResponse[]"],
  "summary": {
    "total_items": "number",
    "average_score": "number",
    "min_score": "number",
    "max_score": "number"
  }
}
```

### RubricSummary

```json
{
  "name": "string",
  "version": "string",
  "description": "string",
  "domain": "string",
  "criteria_count": "number",
  "scale_min": "number",
  "scale_max": "number"
}
```

### ErrorResponse

```json
{
  "error": "string",
  "detail": "string (optional)",
  "error_code": "string (optional)"
}
```

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (rubric not found)
- `422`: Validation Error (invalid request body)
- `500`: Internal Server Error

### Error Response Format

```json
{
  "error": "Rubric not found",
  "detail": "Rubric 'nonexistent_rubric' not found",
  "error_code": "RUBRIC_NOT_FOUND"
}
```

### Common Errors

#### Rubric Not Found
```json
{
  "error": "Rubric not found",
  "detail": "Rubric 'my_rubric' not found"
}
```

#### Invalid API Key
```json
{
  "error": "Authentication failed",
  "detail": "Invalid OpenAI API key"
}
```

#### Validation Error
```json
{
  "error": "Validation error",
  "detail": "task_input is required"
}
```

## Rate Limiting

### Default Limits

- **Scoring endpoints**: 100 requests/minute per IP
- **Batch scoring**: 10 requests/minute per IP
- **Management endpoints**: 1000 requests/minute per IP

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

### Rate Limit Exceeded

```json
{
  "error": "Rate limit exceeded",
  "detail": "Too many requests. Try again in 60 seconds.",
  "error_code": "RATE_LIMIT_EXCEEDED"
}
```

## Examples

### Python Client

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Load a rubric
def load_rubric(file_path, name=None):
    response = requests.post(f"{BASE_URL}/load-rubric", json={
        "file_path": file_path,
        "name": name,
        "provider": "openai"
    })
    return response.json()

# Score a single output
def score_output(rubric_name, task_input, model_output):
    response = requests.post(f"{BASE_URL}/score/{rubric_name}", json={
        "task_input": task_input,
        "model_output": model_output,
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.1
        }
    })
    return response.json()

# Batch scoring
def batch_score(rubric_name, items):
    response = requests.post(f"{BASE_URL}/batch-score/{rubric_name}", json={
        "items": items,
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4"
        }
    })
    return response.json()

# Example usage
if __name__ == "__main__":
    # Load rubric
    load_result = load_rubric("./my_rubric.json", "code_quality")
    print(f"Loaded: {load_result}")
    
    # Score single output
    result = score_output(
        rubric_name="code_quality",
        task_input="Write a function to reverse a string",
        model_output="def reverse_string(s):\n    return s[::-1]"
    )
    print(f"Score: {result['overall_score']}")
    print(f"Explanation: {result['overall_explanation']}")
    
    # Batch scoring
    items = [
        {
            "task_input": "Write a function to reverse a string",
            "model_output": "def reverse_string(s):\n    return s[::-1]"
        },
        {
            "task_input": "Write a function to find maximum",
            "model_output": "def find_max(a, b):\n    return max(a, b)"
        }
    ]
    
    batch_result = batch_score("code_quality", items)
    print(f"Batch average: {batch_result['summary']['average_score']}")
```

### JavaScript Client

```javascript
class OpenRubricRLClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async loadRubric(filePath, name = null, provider = 'openai') {
        const response = await fetch(`${this.baseUrl}/load-rubric`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: filePath,
                name: name,
                provider: provider
            })
        });
        return await response.json();
    }
    
    async score(rubricName, taskInput, modelOutput, llmConfig = {}) {
        const response = await fetch(`${this.baseUrl}/score/${rubricName}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_input: taskInput,
                model_output: modelOutput,
                llm_config: {
                    provider: 'openai',
                    model: 'gpt-4',
                    temperature: 0.1,
                    ...llmConfig
                }
            })
        });
        return await response.json();
    }
    
    async batchScore(rubricName, items, llmConfig = {}) {
        const response = await fetch(`${this.baseUrl}/batch-score/${rubricName}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                items: items,
                llm_config: {
                    provider: 'openai',
                    model: 'gpt-4',
                    ...llmConfig
                }
            })
        });
        return await response.json();
    }
    
    async listRubrics() {
        const response = await fetch(`${this.baseUrl}/rubrics`);
        return await response.json();
    }
}

// Example usage
const client = new OpenRubricRLClient();

async function example() {
    // Load rubric
    await client.loadRubric('./my_rubric.json', 'code_quality');
    
    // Score output
    const result = await client.score(
        'code_quality',
        'Write a function to reverse a string',
        'def reverse_string(s):\n    return s[::-1]'
    );
    
    console.log(`Score: ${result.overall_score}`);
    console.log(`Explanation: ${result.overall_explanation}`);
}
```

### cURL Examples

```bash
# Load a rubric
curl -X POST "http://localhost:8000/load-rubric" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "./my_rubric.json",
    "name": "code_quality",
    "provider": "openai"
  }'

# List rubrics
curl -X GET "http://localhost:8000/rubrics"

# Score a single output
curl -X POST "http://localhost:8000/score/code_quality" \
  -H "Content-Type: application/json" \
  -d '{
    "task_input": "Write a function to reverse a string",
    "model_output": "def reverse_string(s):\n    return s[::-1]",
    "llm_config": {
      "provider": "openai",
      "model": "gpt-4",
      "temperature": 0.1
    }
  }'

# Batch scoring
curl -X POST "http://localhost:8000/batch-score/code_quality" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "task_input": "Write a function to reverse a string",
        "model_output": "def reverse_string(s):\n    return s[::-1]"
      }
    ],
    "llm_config": {
      "provider": "openai",
      "model": "gpt-4"
    }
  }'
```

## Configuration

### Environment Variables

```bash
# LLM Provider API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Server Configuration
OPENRUBRICRL_HOST=0.0.0.0
OPENRUBRICRL_PORT=8000
OPENRUBRICRL_LOG_LEVEL=INFO

# Rate Limiting
OPENRUBRICRL_RATE_LIMIT_ENABLED=true
OPENRUBRICRL_RATE_LIMIT_REQUESTS=100
OPENRUBRICRL_RATE_LIMIT_WINDOW=60
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["openrubricrl", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t openrubricrl-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key openrubricrl-api
```

---

For more information, see the [Rubric Schema Reference](schema.md) and [Integration Guide](integrations.md).
