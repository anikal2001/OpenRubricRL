# OpenRubricRL Examples

This directory contains comprehensive examples demonstrating how to use OpenRubricRL with different RL libraries and use cases.

## 🚀 Quick Start

1. **Set up API keys:**
```bash
export OPENAI_API_KEY="your-openai-key"
# or
export ANTHROPIC_API_KEY="your-anthropic-key"
```

2. **Install dependencies:**
```bash
pip install openrubricrl[all]
```

3. **Run basic examples:**
```bash
python basic_usage.py
```

## 📁 Example Files

### Core Examples

- **`basic_usage.py`** - Comprehensive overview of core OpenRubricRL functionality
  - Rubric creation and validation
  - Basic scoring and batch scoring
  - Reward processing with normalization
  - Prompt building
  - Logging and statistics
  - Provider comparison (OpenAI vs Anthropic)

### RL Integration Examples

- **`rllib_example.py`** - Ray RLlib integration for policy learning
  - Custom environment with rubric rewards
  - PPO training with reward function wrapping
  - Episode logging and checkpointing
  - Sample code generation task

- **`trl_example.py`** - Transformers Reinforcement Learning (TRL) integration
  - PPO fine-tuning with rubric rewards
  - Supervised fine-tuning with rubric evaluation
  - Dialogue quality assessment
  - Model comparison and evaluation

- **`cleanrl_example.py`** - CleanRL integration for text-based RL
  - Text adventure environment
  - Custom reward functions
  - Language model environments
  - Comprehensive logging and statistics

## 🎯 Use Cases Covered

### 1. Code Generation (`rllib_example.py`)
- **Task:** Train agents to generate Python code
- **Rubric:** Code correctness, readability, efficiency
- **RL Library:** Ray RLlib with PPO
- **Environment:** Custom code generation tasks

### 2. Dialogue Systems (`trl_example.py`)
- **Task:** Fine-tune language models for better conversations
- **Rubric:** Response relevance, helpfulness, clarity
- **RL Library:** TRL with PPO
- **Model:** DialoGPT or similar conversational models

### 3. Text Adventures (`cleanrl_example.py`)
- **Task:** Learn to take good actions in text-based games
- **Rubric:** Action creativity, relevance, safety
- **RL Library:** CleanRL-compatible environments
- **Environment:** Custom text adventure scenarios

### 4. Question Answering (`cleanrl_example.py`)
- **Task:** Generate high-quality answers to questions
- **Rubric:** Answer accuracy, completeness, clarity
- **RL Library:** Custom gymnasium environments
- **Dataset:** Custom Q&A pairs

## 🛠️ Running Examples

### Basic Usage
```bash
# Run all core functionality examples
python basic_usage.py

# This will create:
# - Sample rubrics (code_quality.json, creative_writing.json)
# - Generated prompts (sample_prompt.txt)
# - Log files (openrubricrl_demo.log)
```

### RLlib Integration
```bash
# Install RLlib first
pip install ray[rllib]

# Run RLlib example
python rllib_example.py

# Outputs:
# - Training logs (rllib_training.log)
# - Model checkpoints
# - Sample generated code
```

### TRL Integration
```bash
# Install TRL first
pip install trl torch transformers datasets

# Run TRL example
python trl_example.py

# Outputs:
# - Fine-tuned models
# - Training logs
# - Evaluation results
```

### CleanRL Integration
```bash
# Install gymnasium
pip install gymnasium

# Run CleanRL example
python cleanrl_example.py

# Outputs:
# - Reward logs (cleanrl_rewards.csv)
# - Training statistics
# - Sample game interactions
```

## 📊 Expected Results

### Code Generation (RLlib)
- **Initial Performance:** Random code snippets with low rubric scores
- **After Training:** Syntactically correct functions that solve basic problems
- **Metrics:** Rubric scores improve from ~2/10 to ~7/10 over training

### Dialogue Fine-tuning (TRL)
- **Initial Performance:** Generic or irrelevant responses
- **After Training:** More helpful, relevant, and clear responses
- **Metrics:** Response quality scores improve significantly

### Text Adventures (CleanRL)
- **Initial Performance:** Random actions with poor outcomes
- **After Training:** Strategic, creative, and safe actions
- **Metrics:** Action quality and safety scores improve

## 🔧 Customization

### Creating Custom Rubrics

Each example includes rubric creation. Customize criteria and weights:

```python
custom_rubric = {
    "name": "my_custom_rubric",
    "version": "1.0.0",
    "domain": "my_domain",
    "criteria": [
        {
            "name": "my_criterion",
            "description": "What this measures",
            "weight": 0.5,  # Must sum to 1.0 across all criteria
            "examples": {
                "excellent": [...],
                "poor": [...]
            }
        }
    ]
}
```

### Custom Environments

Extend the base environments for your use case:

```python
from openrubricrl.integrations.cleanrl import RubricWrapper

class MyCustomEnv(gym.Env):
    # Your environment implementation
    pass

# Wrap with rubric rewards
wrapped_env = RubricWrapper(
    env=MyCustomEnv(),
    reward_function=my_reward_function,
    reward_weight=1.0
)
```

### Custom Reward Functions

Create domain-specific reward functions:

```python
from openrubricrl.integrations.base import BaseRewardFunction

class MyRewardFunction(BaseRewardFunction):
    def extract_input_output(self, *args, **kwargs):
        # Extract task input and model output from your data format
        return task_input, model_output
```

## 🐛 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Use smaller batch sizes
   - Add delays between requests
   - Cache results when possible

2. **Memory Issues**
   - Use smaller models for testing
   - Reduce batch sizes
   - Clear caches periodically

3. **Scoring Failures**
   - Check API key validity
   - Verify rubric format
   - Handle network errors gracefully

### Debug Mode

Enable debug logging:

```python
from openrubricrl.logging import OpenRubricLogger

logger = OpenRubricLogger(log_level="DEBUG")
```

### Performance Optimization

- **Cache Results:** Enable reward caching for repeated evaluations
- **Batch Scoring:** Use batch APIs when available
- **Async Processing:** Use async scoring for parallel processing
- **Local Models:** Consider local LLMs for high-volume applications

## 📚 Next Steps

1. **Adapt Examples:** Modify the examples for your specific use case
2. **Create Rubrics:** Design rubrics for your domain and evaluation criteria
3. **Integrate:** Add OpenRubricRL to your existing RL training pipeline
4. **Scale:** Move to production with proper error handling and monitoring
5. **Contribute:** Share your rubrics and examples with the community

## 🤝 Contributing

Found an issue or want to add more examples? Please:

1. Fork the repository
2. Create your feature branch
3. Add your example with documentation
4. Submit a pull request

---

For more information, see the main [OpenRubricRL documentation](../README.md).