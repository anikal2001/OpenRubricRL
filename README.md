# OpenRubricRL

**An open-source pipeline that converts human-written rubrics into LLM-based reward functions for RL and RLHF training.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Problem It Solves

Current RLHF pipelines require expensive human labelers to score outputs. Labs want LLM-based reward models to scale scoring — but they need high-quality rubrics to make that work. **No open standard exists for turning a rubric into a reusable, consistent reward function.**

OpenRubricRL fills this gap by providing:
- 📋 A standard JSON/YAML schema for defining evaluation rubrics
- 🤖 Automatic conversion of rubrics into LLM scoring prompts  
- 🔌 Ready-to-use API and CLI tools for scoring model outputs
- 🧪 Integration with popular RL libraries (RLlib, TRL, CleanRL)

## 🚀 Quick Start

### Installation

```bash
pip install openrubricrl
```

For development with all features:
```bash
pip install openrubricrl[all]
```

### Basic Usage

#### 1. Create a Rubric

```bash
openrubricrl create-template my_rubric --domain code
```

This creates `my_rubric.json` with a basic template. Edit it to define your criteria:

```json
{
  "name": "code_quality_basic",
  "version": "1.0.0",
  "description": "Basic code quality evaluation",
  "domain": "code",
  "scale": {"min": 0.0, "max": 10.0},
  "criteria": [
    {
      "name": "correctness",
      "description": "Does the code solve the problem correctly?",
      "weight": 0.4,
      "examples": {
        "excellent": [
          {
            "input": "Write a function to reverse a string",
            "output": "def reverse_string(s): return s[::-1]",
            "score": 9.0,
            "explanation": "Correct and efficient implementation"
          }
        ]
      }
    },
    {
      "name": "readability", 
      "description": "Is the code clean and readable?",
      "weight": 0.6
    }
  ]
}
```

#### 2. Score Model Outputs

**Command Line:**
```bash
export OPENAI_API_KEY="your-key-here"

openrubricrl score my_rubric.json \
  "Write a function to add two numbers" \
  "def add(a, b): return a + b"
```

**Python API:**
```python
from openrubricrl import Rubric, create_openai_scorer

# Load rubric
rubric = Rubric.from_file("my_rubric.json")

# Create scorer
scorer = create_openai_scorer(rubric, api_key="your-key")

# Score an output
result = await scorer.score(
    task_input="Write a function to add two numbers",
    model_output="def add(a, b): return a + b"
)

print(f"Score: {result.overall_score}/10")
print(f"Explanation: {result.overall_explanation}")
```

**REST API:**
```bash
# Start server
openrubricrl serve --rubrics-dir ./rubrics

# Score via HTTP
curl -X POST "http://localhost:8000/score/my_rubric" \
  -H "Content-Type: application/json" \
  -d '{
    "task_input": "Write a function to add two numbers",
    "model_output": "def add(a, b): return a + b"
  }'
```

## 🏗️ Architecture

### Core Components

1. **Rubric Schema** (`rubric_schema.json`): JSON schema defining the standard format
2. **Prompt Builder** (`prompt_builder.py`): Converts rubrics into LLM prompts
3. **Scorer** (`scorer.py`): Handles LLM API calls and response parsing
4. **API Server** (`server.py`): FastAPI-based REST API
5. **CLI** (`cli.py`): Command-line interface

### Supported LLM Providers

- ✅ OpenAI (GPT-5, GPT-o3)
- ✅ Anthropic (Claude)
- 🔄 Local models via vLLM (coming soon)

## 📖 Examples

See the [`examples/`](examples/) directory for complete examples:

- [`code_evaluation.py`](examples/code_evaluation.py) - Scoring code generation
- [`dialogue_quality.py`](examples/dialogue_quality.py) - Evaluating chatbot responses  
- [`creative_writing.py`](examples/creative_writing.py) - Scoring creative content
- [`batch_scoring.py`](examples/batch_scoring.py) - Processing multiple outputs

## 🔗 Integrations

### Reinforcement Learning Libraries

```python
# RLlib integration example
from openrubricrl.integrations.rllib import RubricRewardFunction

reward_fn = RubricRewardFunction(
    rubric_path="my_rubric.json",
    provider="openai"
)

# Use in your RL training loop
reward = reward_fn(state, action, context)
```

### Hugging Face Transformers

```python
from openrubricrl.integrations.transformers import RubricCallback

trainer = Trainer(
    model=model,
    callbacks=[RubricCallback(rubric_path="my_rubric.json")],
    # ... other args
)
```

## 🧪 Development

### Setup

```bash
git clone https://github.com/openrubricrl/openrubricrl.git
cd openrubricrl
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## 📚 Documentation

- [Rubric Schema Reference](docs/schema.md)
- [API Documentation](docs/api.md)
- [Integration Guide](docs/integrations.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🗓️ Roadmap

### Phase 1 - Foundation ✅
- [x] JSON/YAML schema for rubrics
- [x] Rubric → prompt converter  
- [x] Minimal scoring API with OpenAI/Anthropic
- [x] CLI tool for local scoring

### Phase 2 - Community & Repository (Q2 2025)
- [ ] Open Rubric Hub (Git repo with curated rubrics)
- [ ] Templates for common domains (code, dialogue, writing)
- [ ] Contribution guidelines and review process

### Phase 3 - Integrations & Scaling (Q3 2025)  
- [ ] RLlib / TRL integration examples
- [ ] Hybrid reward module (LLM + automated metrics)
- [ ] Bias/drift detection module
- [ ] Local model support via vLLM

### Phase 4 - Sustainability (Q4 2025)
- [ ] Hosted API service (optional paid tier)
- [ ] Enterprise features and support
- [ ] Dataset hosting for scoring logs

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Early Adopters & Target Users

- 🔬 Small RL research teams without budget for large-scale human feedback
- 🏆 AI hackathon participants who want reward shaping quickly  
- 🚀 Startups doing RLHF in niche domains (customer service bots, educational tutors)
- 🎓 Academics studying automated evaluation methods

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by the need for standardized evaluation in RLHF
- Built on top of excellent libraries: FastAPI, Pydantic, Click
- Thanks to the open-source RL and NLP communities

---

**🔗 Links:**
- [GitHub Repository](https://github.com/openrubricrl/openrubricrl)
- [Documentation](https://github.com/anikal2001/OpenRubricRL/tree/main/docs)
- [PyPI Package](https://pypi.org/project/openrubricrl/)
