# Integration Guide

This document provides comprehensive guidance for integrating OpenRubricRL with popular reinforcement learning libraries and frameworks. Learn how to use rubric-based scoring in your RL training pipelines.

## 📋 Table of Contents

- [Overview](#overview)
- [Supported Integrations](#supported-integrations)
- [RLlib Integration](#rllib-integration)
- [TRL Integration](#trl-integration)
- [CleanRL Integration](#cleanrl-integration)
- [Custom Integration](#custom-integration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

OpenRubricRL provides seamless integration with popular RL libraries, allowing you to use rubric-based reward functions in your training pipelines. The integrations handle:

- **Reward Function Creation**: Convert rubrics into callable reward functions
- **Batch Processing**: Efficient scoring of multiple outputs
- **Caching**: Avoid redundant LLM calls for identical inputs
- **Error Handling**: Graceful fallbacks when scoring fails
- **Logging**: Detailed metrics and debugging information

## Supported Integrations

| Library | Status | Features |
|---------|--------|----------|
| **RLlib** | ✅ Full Support | Custom reward functions, multi-agent, distributed training |
| **TRL** | ✅ Full Support | RLHF pipelines, PPO training, reward modeling |
| **CleanRL** | ✅ Full Support | Simple integration, custom environments |
| **Custom** | ✅ Base Classes | Build your own integration using base classes |

## RLlib Integration

RLlib is Ray's scalable reinforcement learning library. OpenRubricRL integrates as custom reward functions and environments.

### Installation

```bash
pip install openrubricrl[rllib]
# or
pip install "openrubricrl[all]"
```

### Basic Usage

```python
from openrubricrl.integrations.rllib import RubricRewardFunction
from openrubricrl.core.rubric import Rubric
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

# Initialize Ray
ray.init()

# Load your rubric
rubric = Rubric.from_file("code_quality_rubric.json")

# Create reward function
reward_fn = RubricRewardFunction(
    rubric=rubric,
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Configure PPO with custom reward
config = {
    "env": "YourCustomEnv",
    "reward_fn": reward_fn,
    "num_workers": 4,
    "train_batch_size": 4000,
}

# Train the agent
trainer = PPO(config=config)
for i in range(100):
    result = trainer.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']}")
```

### Custom Environment with Rubric Rewards

```python
import gym
from gym import spaces
from openrubricrl.integrations.rllib import RubricEnvironment

class CodeGenerationEnv(RubricEnvironment):
    def __init__(self, config):
        super().__init__(config)
        
        # Define observation and action spaces
        self.observation_space = spaces.Text(max_length=1000)
        self.action_space = spaces.Text(max_length=2000)
        
        # Load rubric
        self.setup_rubric("code_quality_rubric.json")
    
    def reset(self):
        # Generate a new coding task
        self.current_task = self.generate_task()
        return {"task": self.current_task}
    
    def step(self, action):
        # Action is the generated code
        generated_code = action
        
        # Score using rubric
        reward = self.score_output(
            task_input=self.current_task,
            model_output=generated_code
        )
        
        # Check if episode is done
        done = True  # Single-step episodes for code generation
        
        return {
            "task": self.current_task,
            "code": generated_code
        }, reward, done, {}
    
    def generate_task(self):
        tasks = [
            "Write a function to reverse a string",
            "Implement binary search",
            "Create a class for a binary tree"
        ]
        return random.choice(tasks)

# Register environment
from ray.tune.registry import register_env
register_env("CodeGeneration", lambda config: CodeGenerationEnv(config))

# Train with the environment
config = {
    "env": "CodeGeneration",
    "env_config": {
        "rubric_path": "code_quality_rubric.json",
        "provider": "openai"
    }
}
```

### Multi-Agent Training

```python
from openrubricrl.integrations.rllib import MultiAgentRubricEnvironment

class MultiAgentCodeReview(MultiAgentRubricEnvironment):
    def __init__(self, config):
        super().__init__(config)
        
        # Define agents: code_generator and code_reviewer
        self.agents = ["code_generator", "code_reviewer"]
        
        # Load different rubrics for each agent
        self.setup_agent_rubrics({
            "code_generator": "code_generation_rubric.json",
            "code_reviewer": "code_review_rubric.json"
        })
    
    def step(self, action_dict):
        rewards = {}
        observations = {}
        dones = {"__all__": False}
        
        # Code generator produces code
        if "code_generator" in action_dict:
            code = action_dict["code_generator"]
            rewards["code_generator"] = self.score_agent_output(
                agent="code_generator",
                task_input=self.current_task,
                model_output=code
            )
        
        # Code reviewer reviews the code
        if "code_reviewer" in action_dict:
            review = action_dict["code_reviewer"]
            rewards["code_reviewer"] = self.score_agent_output(
                agent="code_reviewer", 
                task_input=f"Review this code: {code}",
                model_output=review
            )
        
        return observations, rewards, dones, {}
```

## TRL Integration

TRL (Transformer Reinforcement Learning) is designed for training language models with RL. Perfect for RLHF pipelines.

### Installation

```bash
pip install openrubricrl[trl]
```

### RLHF Pipeline

```python
from openrubricrl.integrations.trl import RubricRewardModel
from openrubricrl.core.rubric import Rubric
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load rubric
rubric = Rubric.from_file("dialogue_quality_rubric.json")

# Create rubric-based reward model
reward_model = RubricRewardModel(
    rubric=rubric,
    provider="openai",
    model="gpt-4",
    tokenizer=tokenizer
)

# Configure PPO training
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    reward_model=reward_model
)

# Training loop
for epoch in range(10):
    # Generate responses
    prompts = ["Hello, how are you?", "What's the weather like?"]
    
    # Get model responses
    responses = ppo_trainer.generate(prompts, max_length=50)
    
    # Calculate rewards using rubric
    rewards = reward_model.score_batch(prompts, responses)
    
    # Update model with PPO
    stats = ppo_trainer.step(prompts, responses, rewards)
    
    print(f"Epoch {epoch}: mean_reward = {stats['ppo/mean_reward']}")
```

### Custom Reward Model

```python
from openrubricrl.integrations.trl import BaseRubricRewardModel
import torch

class CustomDialogueRewardModel(BaseRubricRewardModel):
    def __init__(self, rubric, **kwargs):
        super().__init__(rubric, **kwargs)
        
        # Add custom preprocessing
        self.safety_keywords = ["harmful", "inappropriate", "offensive"]
    
    def preprocess_input(self, prompt, response):
        """Custom preprocessing before scoring."""
        # Add safety check
        if any(keyword in response.lower() for keyword in self.safety_keywords):
            return None  # Skip scoring, return penalty
        
        return super().preprocess_input(prompt, response)
    
    def postprocess_score(self, score, prompt, response):
        """Custom postprocessing after scoring."""
        # Apply length penalty
        if len(response.split()) < 5:
            score *= 0.5  # Penalty for too short responses
        
        return score
    
    def compute_reward(self, prompts, responses):
        """Compute rewards for a batch of prompt-response pairs."""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # Preprocess
            processed = self.preprocess_input(prompt, response)
            if processed is None:
                rewards.append(-1.0)  # Safety penalty
                continue
            
            # Score with rubric
            score = self.score_output(prompt, response)
            
            # Postprocess
            final_score = self.postprocess_score(score, prompt, response)
            rewards.append(final_score)
        
        return torch.tensor(rewards, dtype=torch.float32)
```

## CleanRL Integration

CleanRL provides clean, single-file implementations of RL algorithms. OpenRubricRL integrates as reward functions.

### Installation

```bash
pip install openrubricrl[cleanrl]
```

### Basic Integration

```python
from openrubricrl.integrations.cleanrl import RubricRewardWrapper
from openrubricrl.core.rubric import Rubric
import gym

# Create base environment
env = gym.make("YourEnvironment")

# Load rubric
rubric = Rubric.from_file("task_completion_rubric.json")

# Wrap environment with rubric-based rewards
env = RubricRewardWrapper(
    env=env,
    rubric=rubric,
    provider="anthropic",
    model="claude-3-sonnet-20240229"
)

# Use in training loop
obs = env.reset()
for step in range(1000):
    action = policy(obs)  # Your policy
    obs, reward, done, info = env.step(action)
    
    # Reward is now computed using the rubric
    print(f"Step {step}: reward = {reward}")
    
    if done:
        obs = env.reset()
```

### Custom Reward Wrapper

```python
from openrubricrl.integrations.cleanrl import BaseRubricWrapper
import numpy as np

class CodeExecutionWrapper(BaseRubricWrapper):
    def __init__(self, env, rubric, **kwargs):
        super().__init__(env, rubric, **kwargs)
        
        # Track episode data
        self.episode_code = []
        self.episode_task = None
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Extract task from observation
        self.episode_task = obs.get("task", "")
        self.episode_code = []
        
        return obs
    
    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        
        # Accumulate code generation
        if "code" in obs:
            self.episode_code.append(obs["code"])
        
        # Compute rubric reward at episode end
        if done and self.episode_task and self.episode_code:
            full_code = "\n".join(self.episode_code)
            
            # Score the complete code
            rubric_reward = self.score_output(
                task_input=self.episode_task,
                model_output=full_code
            )
            
            # Combine with original reward
            combined_reward = 0.7 * rubric_reward + 0.3 * original_reward
            
            # Add detailed info
            info["rubric_score"] = rubric_reward
            info["original_reward"] = original_reward
            info["combined_reward"] = combined_reward
            
            return obs, combined_reward, done, info
        
        return obs, original_reward, done, info
```

## Custom Integration

Build your own integration using OpenRubricRL's base classes.

### Base Reward Function

```python
from openrubricrl.integrations.base import BaseRewardFunction
from openrubricrl.core.rubric import Rubric
import asyncio

class MyCustomRewardFunction(BaseRewardFunction):
    def __init__(self, rubric_path, provider="openai", **kwargs):
        # Load rubric
        self.rubric = Rubric.from_file(rubric_path)
        
        # Initialize scorer
        super().__init__(
            rubric=self.rubric,
            provider=provider,
            **kwargs
        )
        
        # Custom initialization
        self.cache = {}
        self.call_count = 0
    
    def __call__(self, task_input, model_output):
        """Make the reward function callable."""
        self.call_count += 1
        
        # Check cache
        cache_key = hash(task_input + model_output)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Score with rubric
        try:
            score = asyncio.run(self.score_async(task_input, model_output))
            self.cache[cache_key] = score
            return score
        except Exception as e:
            print(f"Scoring failed: {e}")
            return 0.0  # Fallback score
    
    async def score_async(self, task_input, model_output):
        """Async scoring for better performance."""
        result = await self.scorer.score(
            task_input=task_input,
            model_output=model_output
        )
        return result.overall_score
    
    def get_stats(self):
        """Get usage statistics."""
        return {
            "total_calls": self.call_count,
            "cache_size": len(self.cache),
            "cache_hit_rate": 1 - (self.call_count / max(len(self.cache), 1))
        }
```

### Environment Wrapper

```python
from openrubricrl.integrations.base import BaseEnvironmentWrapper
import gym

class RubricEnvironmentWrapper(BaseEnvironmentWrapper):
    def __init__(self, env, rubric_config):
        super().__init__(env, rubric_config)
        
        # Custom configuration
        self.reward_scale = rubric_config.get("reward_scale", 1.0)
        self.min_reward = rubric_config.get("min_reward", -1.0)
        self.max_reward = rubric_config.get("max_reward", 1.0)
    
    def compute_reward(self, observation, action, info):
        """Compute reward based on current state."""
        # Extract relevant information
        task = info.get("task", "")
        output = info.get("output", str(action))
        
        if not task or not output:
            return 0.0
        
        # Score with rubric
        raw_score = self.score_output(task, output)
        
        # Scale and clip reward
        scaled_score = raw_score * self.reward_scale
        clipped_score = np.clip(scaled_score, self.min_reward, self.max_reward)
        
        return clipped_score
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Compute rubric-based reward
        rubric_reward = self.compute_reward(obs, action, info)
        
        # Add to info for debugging
        info["original_reward"] = reward
        info["rubric_reward"] = rubric_reward
        
        return obs, rubric_reward, done, info
```

## Best Practices

### 1. Performance Optimization

```python
# Use batch scoring when possible
rewards = reward_model.score_batch(prompts, responses)

# Enable caching to avoid redundant calls
reward_fn = RubricRewardFunction(
    rubric=rubric,
    cache_size=1000,  # Cache up to 1000 scores
    cache_ttl=3600    # Cache for 1 hour
)

# Use async scoring for better throughput
async def score_multiple(inputs):
    tasks = [reward_fn.score_async(inp, out) for inp, out in inputs]
    return await asyncio.gather(*tasks)
```

### 2. Error Handling

```python
class RobustRewardFunction(BaseRewardFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_score = 0.0
        self.error_count = 0
    
    def __call__(self, task_input, model_output):
        try:
            return self.score_output(task_input, model_output)
        except Exception as e:
            self.error_count += 1
            print(f"Scoring error ({self.error_count}): {e}")
            return self.fallback_score
```

### 3. Monitoring and Logging

```python
import logging
from openrubricrl.integrations.base import BaseRewardFunction

class MonitoredRewardFunction(BaseRewardFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "total_calls": 0,
            "total_score": 0.0,
            "error_count": 0
        }
    
    def __call__(self, task_input, model_output):
        self.metrics["total_calls"] += 1
        
        try:
            score = self.score_output(task_input, model_output)
            self.metrics["total_score"] += score
            
            self.logger.info(f"Scored output: {score:.2f}")
            return score
            
        except Exception as e:
            self.metrics["error_count"] += 1
            self.logger.error(f"Scoring failed: {e}")
            return 0.0
    
    def get_average_score(self):
        if self.metrics["total_calls"] == 0:
            return 0.0
        return self.metrics["total_score"] / self.metrics["total_calls"]
```

### 4. Rubric Selection

```python
class AdaptiveRewardFunction(BaseRewardFunction):
    def __init__(self, rubric_configs):
        self.rubrics = {}
        self.scorers = {}
        
        # Load multiple rubrics
        for name, config in rubric_configs.items():
            rubric = Rubric.from_file(config["path"])
            self.rubrics[name] = rubric
            self.scorers[name] = self.create_scorer(rubric, config)
    
    def select_rubric(self, task_input):
        """Select appropriate rubric based on task."""
        if "code" in task_input.lower():
            return "code_quality"
        elif "dialogue" in task_input.lower():
            return "dialogue_quality"
        else:
            return "general"
    
    def __call__(self, task_input, model_output):
        rubric_name = self.select_rubric(task_input)
        scorer = self.scorers[rubric_name]
        
        result = scorer.score(task_input, model_output)
        return result.overall_score
```

## Troubleshooting

### Common Issues

#### 1. API Rate Limits

```python
from openrubricrl.integrations.base import RateLimitedRewardFunction

reward_fn = RateLimitedRewardFunction(
    rubric=rubric,
    requests_per_minute=60,  # Respect API limits
    retry_attempts=3,
    backoff_factor=2.0
)
```

#### 2. Memory Issues with Large Batches

```python
def score_large_batch(reward_fn, inputs, batch_size=10):
    """Score large batches in chunks."""
    results = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        batch_scores = reward_fn.score_batch(batch)
        results.extend(batch_scores)
        
        # Optional: add delay between batches
        time.sleep(1)
    
    return results
```

#### 3. Network Connectivity Issues

```python
class ReliableRewardFunction(BaseRewardFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offline_mode = False
        self.cached_scores = {}
    
    def __call__(self, task_input, model_output):
        if self.offline_mode:
            # Use cached scores or simple heuristics
            return self.get_offline_score(task_input, model_output)
        
        try:
            return self.score_output(task_input, model_output)
        except ConnectionError:
            print("Network error, switching to offline mode")
            self.offline_mode = True
            return self.get_offline_score(task_input, model_output)
    
    def get_offline_score(self, task_input, model_output):
        # Simple heuristic scoring when offline
        return len(model_output) / 100.0  # Example: length-based score
```

### Debugging Tips

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Monitor API Usage**:
   ```python
   reward_fn = RubricRewardFunction(rubric, monitor_usage=True)
   print(reward_fn.get_usage_stats())
   ```

3. **Test with Simple Examples**:
   ```python
   # Test with known good examples
   test_score = reward_fn("Write hello world", "print('Hello, World!')")
   assert test_score > 0, "Basic scoring should work"
   ```

4. **Validate Rubric Loading**:
   ```python
   try:
       rubric = Rubric.from_file("my_rubric.json")
       print(f"Loaded rubric: {rubric.name} v{rubric.version}")
   except Exception as e:
       print(f"Rubric loading failed: {e}")
   ```

---

For more information, see the [Rubric Schema Reference](schema.md) and [API Documentation](api.md).
