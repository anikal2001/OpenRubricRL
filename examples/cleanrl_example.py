"""Example: Using OpenRubricRL with CleanRL for text-based RL."""

import os
import json
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim

from openrubricrl.integrations.cleanrl import (
    create_cleanrl_reward_function,
    RubricWrapper,
    LanguageModelRewardFunction,
    RewardLogger
)
from openrubricrl.logging import OpenRubricLogger


class TextAdventureEnv(gym.Env):
    """Simple text adventure environment for demonstration."""
    
    def __init__(self):
        super().__init__()
        
        # Define scenarios
        self.scenarios = [
            {
                "situation": "You are in a dark forest. You see a path to the north and east.",
                "good_actions": ["go north", "head north", "walk north", "go east", "head east"],
                "bad_actions": ["go south", "stay here", "sleep"]
            },
            {
                "situation": "You found a treasure chest. What do you do?",
                "good_actions": ["open chest", "examine chest", "check for traps"],
                "bad_actions": ["ignore chest", "walk away", "break chest"]
            },
            {
                "situation": "A dragon appears! It looks angry.",
                "good_actions": ["run away", "hide", "speak to dragon", "offer treasure"],
                "bad_actions": ["attack dragon", "provoke dragon", "stand still"]
            },
            {
                "situation": "You reach a river. How do you cross?",
                "good_actions": ["look for bridge", "find shallow area", "make raft"],
                "bad_actions": ["swim across", "give up", "go back"]
            }
        ]
        
        self.current_scenario = 0
        self.step_count = 0
        self.max_steps = 1
        
        # Observation: current situation text
        self.observation_space = spaces.Text(max_length=500)
        
        # Action: text action
        self.action_space = spaces.Text(max_length=100)
    
    def reset(self, **kwargs):
        """Reset to a new scenario."""
        self.current_scenario = random.randint(0, len(self.scenarios) - 1)
        self.step_count = 0
        
        scenario = self.scenarios[self.current_scenario]
        observation = scenario["situation"]
        
        info = {
            "scenario": scenario,
            "scenario_id": self.current_scenario
        }
        
        return observation, info
    
    def step(self, action):
        """Execute action in the text adventure."""
        self.step_count += 1
        
        scenario = self.scenarios[self.current_scenario]
        action_text = action if isinstance(action, str) else str(action)
        
        # Simple reward based on action quality
        action_lower = action_text.lower().strip()
        
        base_reward = 0.0
        if any(good_action in action_lower for good_action in scenario["good_actions"]):
            base_reward = 0.8 + random.uniform(0, 0.2)
        elif any(bad_action in action_lower for bad_action in scenario["bad_actions"]):
            base_reward = random.uniform(0, 0.3)
        else:
            base_reward = random.uniform(0.3, 0.7)  # Neutral action
        
        # Episode ends after one step for simplicity
        terminated = True
        truncated = False
        
        next_observation = f"You decided to: {action_text}"
        
        info = {
            "scenario": scenario,
            "action_taken": action_text,
            "base_reward": base_reward
        }
        
        return next_observation, base_reward, terminated, truncated, info


def create_text_adventure_rubric():
    """Create a rubric for text adventure actions."""
    rubric_path = "text_adventure_rubric.json"
    
    if not os.path.exists(rubric_path):
        rubric_data = {
            "name": "text_adventure_actions",
            "version": "1.0.0",
            "description": "Evaluate text adventure game actions",
            "domain": "general",
            "scale": {"min": 0.0, "max": 10.0},
            "criteria": [
                {
                    "name": "creativity",
                    "description": "Is the action creative and interesting?",
                    "weight": 0.3,
                    "examples": {
                        "excellent": [
                            {
                                "input": "You see a locked door.",
                                "output": "examine the door frame for hidden switches",
                                "score": 9.0,
                                "explanation": "Creative problem-solving approach"
                            }
                        ],
                        "poor": [
                            {
                                "input": "You see a locked door.",
                                "output": "do nothing",
                                "score": 2.0,
                                "explanation": "No creativity or effort"
                            }
                        ]
                    }
                },
                {
                    "name": "relevance",
                    "description": "Is the action relevant to the current situation?",
                    "weight": 0.4
                },
                {
                    "name": "safety",
                    "description": "Is the action reasonably safe and prudent?",
                    "weight": 0.3
                }
            ]
        }
        
        with open(rubric_path, 'w') as f:
            json.dump(rubric_data, f, indent=2)
        
        print(f"Created text adventure rubric: {rubric_path}")
    
    return rubric_path


def observation_to_text(observation) -> str:
    """Convert observation to text for rubric evaluation."""
    if isinstance(observation, str):
        return observation
    else:
        return str(observation)


def action_to_text(action) -> str:
    """Convert action to text for rubric evaluation."""
    if isinstance(action, str):
        return action
    else:
        return str(action)


class SimpleTextAgent:
    """Simple agent for text adventure."""
    
    def __init__(self, action_templates: List[str]):
        self.action_templates = action_templates
    
    def get_action(self, observation: str) -> str:
        """Generate action based on observation."""
        # Simple rule-based action generation
        obs_lower = observation.lower()
        
        if "forest" in obs_lower or "path" in obs_lower:
            return random.choice(["go north", "go east", "explore north"])
        elif "chest" in obs_lower or "treasure" in obs_lower:
            return random.choice(["open chest", "examine chest carefully", "check for traps first"])
        elif "dragon" in obs_lower:
            return random.choice(["run away quickly", "hide behind rocks", "try to negotiate"])
        elif "river" in obs_lower:
            return random.choice(["look for bridge", "find shallow crossing", "build a raft"])
        else:
            return random.choice(self.action_templates)


def train_with_cleanrl():
    """Training example using CleanRL-style environment with rubric rewards."""
    
    # Set up logging
    logger = OpenRubricLogger(log_file="cleanrl_training.log")
    reward_logger = RewardLogger("cleanrl_rewards.csv")
    
    try:
        # Create rubric
        rubric_path = create_text_adventure_rubric()
        
        # Create reward function
        reward_function = create_cleanrl_reward_function(
            rubric_path=rubric_path,
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            observation_to_text=observation_to_text,
            action_to_text=action_to_text
        )
        
        # Create environment with rubric wrapper
        base_env = TextAdventureEnv()
        env = RubricWrapper(
            env=base_env,
            reward_function=reward_function,
            reward_weight=1.0,  # Use rubric reward entirely
            replace_reward=True
        )
        
        # Create simple agent
        action_templates = [
            "explore the area",
            "look around carefully",
            "proceed cautiously",
            "examine surroundings",
            "move forward slowly"
        ]
        agent = SimpleTextAgent(action_templates)
        
        logger.log_info("Starting CleanRL-style training with rubric rewards")
        
        # Training loop
        num_episodes = 10
        
        for episode in range(num_episodes):
            observation, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            logger.log_info(f"Episode {episode + 1}/{num_episodes}")
            logger.log_info(f"Scenario: {observation}")
            
            # Episode loop
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # Agent chooses action
                action = agent.get_action(observation)
                
                # Environment step
                next_observation, reward, terminated, truncated, step_info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Log step details
                logger.log_info(f"  Action: {action}")
                logger.log_info(f"  Reward: {reward:.3f}")
                
                if "detailed_reward" in step_info:
                    detailed = step_info["detailed_reward"]
                    logger.log_info(f"  LLM Explanation: {detailed.explanation}")
                    
                    # Log to reward logger
                    reward_logger.log_step(episode_length, step_info)
                
                observation = next_observation
            
            # Log episode summary
            logger.log_info(f"Episode {episode + 1} completed: reward={episode_reward:.3f}, length={episode_length}")
            reward_logger.log_episode_end()
        
        # Get final statistics
        stats = reward_logger.get_statistics()
        logger.log_info(f"Training statistics: {stats}")
        
        logger.log_info("CleanRL training completed successfully")
        
    except Exception as e:
        logger.log_error(e, "CleanRL training failed")
        raise


def test_language_model_environment():
    """Test specialized language model environment."""
    
    logger = OpenRubricLogger(log_file="lm_env_test.log")
    
    try:
        # Create a simple language model environment
        class SimpleQAEnv(gym.Env):
            """Simple Q&A environment."""
            
            def __init__(self):
                super().__init__()
                
                self.questions = [
                    "What is the capital of France?",
                    "How do you make coffee?",
                    "What is photosynthesis?",
                    "How do computers work?"
                ]
                
                self.current_question_idx = 0
                
                self.observation_space = spaces.Text(max_length=200)
                self.action_space = spaces.Text(max_length=500)
            
            def reset(self, **kwargs):
                self.current_question_idx = random.randint(0, len(self.questions) - 1)
                question = self.questions[self.current_question_idx]
                
                return question, {"question_id": self.current_question_idx}
            
            def step(self, action):
                answer = action if isinstance(action, str) else str(action)
                
                # Simple evaluation (in practice, use more sophisticated methods)
                if len(answer) > 10:
                    base_reward = 0.5 + random.uniform(0, 0.5)
                else:
                    base_reward = random.uniform(0, 0.4)
                
                return answer, base_reward, True, False, {"answer": answer}
        
        # Create rubric for Q&A
        qa_rubric = {
            "name": "qa_quality",
            "version": "1.0.0",
            "description": "Evaluate question answering quality",
            "domain": "general",
            "scale": {"min": 0.0, "max": 10.0},
            "criteria": [
                {
                    "name": "accuracy",
                    "description": "Is the answer factually correct?",
                    "weight": 0.5
                },
                {
                    "name": "completeness",
                    "description": "Does the answer fully address the question?",
                    "weight": 0.3
                },
                {
                    "name": "clarity",
                    "description": "Is the answer clear and well-written?",
                    "weight": 0.2
                }
            ]
        }
        
        # Save rubric
        with open("qa_rubric.json", 'w') as f:
            json.dump(qa_rubric, f, indent=2)
        
        # Create language model reward function
        lm_reward_function = LanguageModelRewardFunction(
            rubric="qa_rubric.json",
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create environment
        base_env = SimpleQAEnv()
        env = RubricWrapper(
            env=base_env,
            reward_function=lm_reward_function,
            reward_weight=1.0
        )
        
        logger.log_info("Testing language model environment")
        
        # Test responses
        test_responses = [
            "Paris is the capital of France.",
            "To make coffee: 1) Boil water, 2) Add coffee grounds to filter, 3) Pour hot water over grounds, 4) Serve.",
            "Photosynthesis is the process plants use to convert sunlight into energy.",
            "I don't know."
        ]
        
        for i in range(4):
            observation, info = env.reset()
            logger.log_info(f"Question: {observation}")
            
            response = test_responses[i % len(test_responses)]
            next_obs, reward, terminated, truncated, step_info = env.step(response)
            
            logger.log_info(f"Response: {response}")
            logger.log_info(f"Reward: {reward:.3f}")
            
            if "detailed_reward" in step_info:
                detailed = step_info["detailed_reward"]
                logger.log_info(f"Explanation: {detailed.explanation}")
            
            logger.log_info("---")
        
        logger.log_info("Language model environment test completed")
        
    except Exception as e:
        logger.log_error(e, "Language model environment test failed")
        raise


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Set it before running:")
        print("export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    print("Running CleanRL examples...")
    
    # Run text adventure training
    print("\n1. Text adventure training with rubric rewards")
    train_with_cleanrl()
    
    # Test language model environment
    print("\n2. Language model environment test")
    test_language_model_environment()
    
    print("\nAll CleanRL examples completed!")