"""Example: Using OpenRubricRL with Ray RLlib for code generation training."""

import os
import json
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
import gymnasium as gym
from gymnasium import spaces

from openrubricrl.integrations.rllib import create_rllib_reward_function, RubricRewardWrapper
from openrubricrl.logging import OpenRubricLogger, TrainingLogger


class CodeGenerationEnv(gym.Env):
    """Simple code generation environment for demonstration."""
    
    def __init__(self, config: EnvContext):
        super().__init__()
        
        # Load coding problems
        self.problems = [
            {
                "prompt": "Write a function to add two numbers",
                "test_cases": [("add(2, 3)", "5"), ("add(-1, 1)", "0")],
                "description": "Create a function that takes two numbers and returns their sum"
            },
            {
                "prompt": "Write a function to find the maximum of two numbers",
                "test_cases": [("max_num(5, 3)", "5"), ("max_num(-1, -5)", "-1")],
                "description": "Create a function that returns the larger of two numbers"
            },
            {
                "prompt": "Write a function to reverse a string",
                "test_cases": [("reverse('hello')", "'olleh'"), ("reverse('abc')", "'cba'")],
                "description": "Create a function that returns the reverse of an input string"
            }
        ]
        
        self.current_problem_idx = 0
        self.max_steps = 1
        self.step_count = 0
        
        # Define observation and action spaces
        # Observation: problem description as text
        self.observation_space = spaces.Dict({
            "task_input": spaces.Text(max_length=1000),
            "model_output": spaces.Text(max_length=1000)
        })
        
        # Action: generated code as text
        self.action_space = spaces.Text(max_length=500)
    
    def reset(self, **kwargs):
        """Reset environment to a new coding problem."""
        self.current_problem_idx = (self.current_problem_idx + 1) % len(self.problems)
        self.step_count = 0
        
        problem = self.problems[self.current_problem_idx]
        
        observation = {
            "task_input": problem["prompt"],
            "model_output": ""  # Will be filled by agent's action
        }
        
        info = {
            "problem": problem,
            "problem_idx": self.current_problem_idx
        }
        
        return observation, info
    
    def step(self, action):
        """Execute one step - agent provides code solution."""
        self.step_count += 1
        
        problem = self.problems[self.current_problem_idx]
        generated_code = action if isinstance(action, str) else str(action)
        
        # Update observation with the generated code
        observation = {
            "task_input": problem["prompt"],
            "model_output": generated_code
        }
        
        # Basic reward based on code execution (simplified)
        base_reward = self._evaluate_code(generated_code, problem["test_cases"])
        
        # Episode ends after one step
        terminated = True
        truncated = False
        
        info = {
            "problem": problem,
            "generated_code": generated_code,
            "base_reward": base_reward
        }
        
        return observation, base_reward, terminated, truncated, info
    
    def _evaluate_code(self, code: str, test_cases) -> float:
        """Simple code evaluation (in practice, use safe execution)."""
        if not code or len(code.strip()) < 10:
            return 0.0
        
        # Simple heuristics (replace with safe code execution)
        score = 0.0
        
        # Check for function definition
        if "def " in code:
            score += 0.3
        
        # Check for return statement
        if "return " in code:
            score += 0.2
        
        # Check for reasonable length
        if 20 <= len(code) <= 200:
            score += 0.2
        
        # Check for basic syntax patterns
        if "(" in code and ")" in code:
            score += 0.1
        
        # Random variation for demonstration
        import random
        score += random.uniform(-0.1, 0.2)
        
        return max(0.0, min(1.0, score))


def train_with_rubric():
    """Main training function using RLlib with rubric rewards."""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Set up logging
    logger = OpenRubricLogger(log_file="rllib_training.log")
    training_logger = TrainingLogger("rllib_code_generation", logger)
    
    try:
        # Create rubric reward function
        rubric_reward = create_rllib_reward_function(
            rubric_path="code_quality_basic.json",  # Assuming this exists
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            input_key="task_input",
            output_key="model_output"
        )
        
        # Register environment with rubric wrapper
        def env_creator(config):
            base_env = CodeGenerationEnv(config)
            return RubricRewardWrapper(
                env=base_env,
                reward_function=rubric_reward,
                reward_weight=1.0  # Use rubric reward entirely
            )
        
        tune.register_env("CodeGenWithRubric", env_creator)
        
        # Configure PPO
        config = (
            PPOConfig()
            .environment("CodeGenWithRubric")
            .framework("torch")
            .training(
                lr=3e-4,
                num_sgd_iter=10,
                train_batch_size=1000,
                model={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                }
            )
            .rollouts(
                num_rollout_workers=2,
                rollout_fragment_length=50
            )
            .evaluation(
                evaluation_interval=10,
                evaluation_num_episodes=5
            )
        )
        
        # Custom callback for logging
        class RubricLoggingCallback:
            def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
                """Log episode results."""
                episode_reward = episode.total_reward
                episode_length = episode.length
                
                # Get detailed reward info from last step
                last_info = episode.last_info_for()
                if last_info and "detailed_reward" in last_info:
                    detailed_reward = last_info["detailed_reward"]
                    
                    # Log to our training logger
                    training_logger.log_episode_end(
                        total_reward=episode_reward,
                        episode_length=episode_length,
                        additional_info={
                            "rubric_reward": last_info.get("rubric_reward"),
                            "llm_explanation": detailed_reward.explanation,
                            "generated_code": last_info.get("generated_code", "")
                        }
                    )
        
        # Add callback to config
        config.callbacks(RubricLoggingCallback)
        
        # Build and train
        algo = config.build()
        
        logger.log_info("Starting RLlib training with rubric rewards")
        
        for i in range(50):  # Train for 50 iterations
            result = algo.train()
            
            # Log training progress
            episode_reward_mean = result["episode_reward_mean"]
            episodes_this_iter = result["episodes_this_iter"]
            
            training_logger.log_step(
                reward=episode_reward_mean,
                additional_metrics={
                    "episodes": episodes_this_iter,
                    "timesteps_total": result["timesteps_total"]
                }
            )
            
            if i % 10 == 0:
                logger.log_info(f"Iteration {i}: avg reward = {episode_reward_mean:.3f}")
                
                # Save checkpoint
                checkpoint_path = algo.save()
                logger.log_info(f"Checkpoint saved to: {checkpoint_path}")
        
        # End training
        training_logger.end_session()
        
        # Save final model
        final_checkpoint = algo.save()
        logger.log_info(f"Final model saved to: {final_checkpoint}")
        
        # Generate some sample outputs
        logger.log_info("Generating sample outputs...")
        
        env = env_creator({})
        obs, info = env.reset()
        
        for i in range(3):
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            logger.log_info(f"Sample {i+1}:")
            logger.log_info(f"  Problem: {info.get('problem', {}).get('prompt', 'N/A')}")
            logger.log_info(f"  Generated: {info.get('generated_code', 'N/A')}")
            logger.log_info(f"  Reward: {reward:.3f}")
            
            if "detailed_reward" in info:
                detailed = info["detailed_reward"]
                logger.log_info(f"  Explanation: {detailed.explanation}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        algo.stop()
        
    except Exception as e:
        logger.log_error(e, "Training failed")
        raise
    
    finally:
        ray.shutdown()


def create_sample_rubric():
    """Create a sample rubric for code generation if it doesn't exist."""
    rubric_path = "code_quality_basic.json"
    
    if not os.path.exists(rubric_path):
        rubric_data = {
            "name": "code_quality_basic",
            "version": "1.0.0",
            "description": "Basic code quality evaluation for simple functions",
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
                                "input": "Write a function to add two numbers",
                                "output": "def add(a, b):\n    return a + b",
                                "score": 9.0,
                                "explanation": "Correct and clean implementation"
                            }
                        ],
                        "poor": [
                            {
                                "input": "Write a function to add two numbers",
                                "output": "def add():\n    print('hello')",
                                "score": 2.0,
                                "explanation": "Does not solve the problem"
                            }
                        ]
                    }
                },
                {
                    "name": "readability",
                    "description": "Is the code clean and readable?",
                    "weight": 0.3
                },
                {
                    "name": "efficiency",
                    "description": "Is the solution efficient?",
                    "weight": 0.3
                }
            ]
        }
        
        with open(rubric_path, 'w') as f:
            json.dump(rubric_data, f, indent=2)
        
        print(f"Created sample rubric: {rubric_path}")


if __name__ == "__main__":
    # Create sample rubric if needed
    create_sample_rubric()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Set it before running:")
        print("export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run training
    train_with_rubric()