"""RLlib integration for OpenRubricRL."""

from typing import Dict, Any, Optional, Union
import numpy as np

from .base import BaseRewardFunction
from ..core.rubric import Rubric
from ..core.scorer import RubricScorer, create_openai_scorer, create_anthropic_scorer


class RLlibRewardFunction(BaseRewardFunction):
    """Reward function for Ray RLlib integration."""
    
    def __init__(
        self,
        rubric: Union[str, Rubric],
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        input_key: str = "task_input",
        output_key: str = "model_output",
        **kwargs
    ):
        # Create scorer
        if isinstance(rubric, str):
            rubric_obj = Rubric.from_file(rubric)
        else:
            rubric_obj = rubric
        
        if provider == "openai":
            scorer = create_openai_scorer(rubric_obj, api_key=api_key, model=model)
        elif provider == "anthropic":
            scorer = create_anthropic_scorer(rubric_obj, api_key=api_key, model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        super().__init__(rubric=rubric_obj, scorer=scorer, **kwargs)
        
        self.input_key = input_key
        self.output_key = output_key
    
    def extract_input_output(self, observation: Dict[str, Any], **kwargs) -> tuple[str, str]:
        """Extract input/output from RLlib observation."""
        task_input = observation.get(self.input_key, "")
        model_output = observation.get(self.output_key, "")
        
        if not task_input or not model_output:
            raise ValueError(f"Missing required keys: {self.input_key}, {self.output_key}")
        
        return task_input, model_output


def create_rllib_reward_function(
    rubric_path: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> RLlibRewardFunction:
    """Convenience function to create RLlib reward function."""
    return RLlibRewardFunction(
        rubric=rubric_path,
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )


class RLlibCallback:
    """RLlib callback for logging reward details."""
    
    def __init__(self, reward_function: RLlibRewardFunction, log_frequency: int = 100):
        self.reward_function = reward_function
        self.log_frequency = log_frequency
        self.episode_count = 0
    
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """Called at the end of each episode."""
        self.episode_count += 1
        
        if self.episode_count % self.log_frequency == 0:
            # Log detailed reward information
            last_info = episode.last_info_for()
            if last_info and "detailed_reward" in last_info:
                detailed_reward = last_info["detailed_reward"]
                
                print(f"Episode {self.episode_count} Reward Details:")
                print(f"  Final Reward: {detailed_reward.reward:.3f}")
                print(f"  LLM Score: {detailed_reward.raw_llm_score:.3f}")
                print(f"  Explanation: {detailed_reward.explanation}")
                
                if detailed_reward.hybrid_components:
                    print("  Hybrid Metrics:")
                    for metric, score in detailed_reward.hybrid_components.items():
                        print(f"    {metric}: {score:.3f}")


# Example RLlib environment wrapper
class RubricRewardWrapper:
    """Environment wrapper that adds rubric-based rewards."""
    
    def __init__(self, env, reward_function: RLlibRewardFunction, reward_weight: float = 1.0):
        self.env = env
        self.reward_function = reward_function
        self.reward_weight = reward_weight
        
        # Delegate most methods to wrapped env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step the environment and add rubric reward."""
        obs, reward, done, info = self.env.step(action)
        
        # Calculate rubric-based reward
        try:
            rubric_reward = self.reward_function(obs)
            
            # Blend with original reward
            total_reward = reward + (rubric_reward * self.reward_weight)
            
            # Add detailed reward info
            detailed_reward = self.reward_function.get_detailed_reward(obs)
            info["rubric_reward"] = rubric_reward
            info["detailed_reward"] = detailed_reward
            
        except Exception as e:
            print(f"Warning: Rubric reward calculation failed: {e}")
            total_reward = reward
        
        return obs, total_reward, done, info
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def render(self, **kwargs):
        """Render the environment."""
        return self.env.render(**kwargs)