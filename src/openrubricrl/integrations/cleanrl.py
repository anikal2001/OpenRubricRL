"""CleanRL integration for OpenRubricRL."""

from typing import Dict, Any, Optional, Union, Callable
import numpy as np
import gymnasium as gym

from .base import BaseRewardFunction
from ..core.rubric import Rubric
from ..core.scorer import create_openai_scorer, create_anthropic_scorer


class CleanRLRewardFunction(BaseRewardFunction):
    """Reward function for CleanRL integration."""
    
    def __init__(
        self,
        rubric: Union[str, Rubric],
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        observation_to_text: Optional[Callable] = None,
        action_to_text: Optional[Callable] = None,
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
        
        # Functions to convert observations/actions to text
        self.observation_to_text = observation_to_text or self._default_obs_to_text
        self.action_to_text = action_to_text or self._default_action_to_text
    
    def _default_obs_to_text(self, observation) -> str:
        """Default observation to text conversion."""
        if isinstance(observation, (int, float)):
            return f"State: {observation}"
        elif isinstance(observation, np.ndarray):
            return f"State: {observation.tolist()}"
        elif isinstance(observation, dict):
            return f"State: {observation}"
        else:
            return str(observation)
    
    def _default_action_to_text(self, action) -> str:
        """Default action to text conversion."""
        if isinstance(action, (int, float)):
            return f"Action: {action}"
        elif isinstance(action, np.ndarray):
            return f"Action: {action.tolist()}"
        else:
            return str(action)
    
    def extract_input_output(self, observation, action, **kwargs) -> tuple[str, str]:
        """Extract input/output from CleanRL observation and action."""
        task_input = self.observation_to_text(observation)
        model_output = self.action_to_text(action)
        
        return task_input, model_output


class RubricWrapper(gym.Wrapper):
    """Gymnasium environment wrapper that adds rubric-based rewards."""
    
    def __init__(
        self,
        env: gym.Env,
        reward_function: CleanRLRewardFunction,
        reward_weight: float = 1.0,
        replace_reward: bool = False
    ):
        super().__init__(env)
        self.reward_function = reward_function
        self.reward_weight = reward_weight
        self.replace_reward = replace_reward
        
        self.last_observation = None
    
    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        self.last_observation = obs
        return obs, info
    
    def step(self, action):
        """Step the environment and add rubric reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate rubric-based reward using last observation and current action
        try:
            rubric_reward = self.reward_function(self.last_observation, action)
            
            if self.replace_reward:
                # Replace original reward entirely
                total_reward = rubric_reward
            else:
                # Add to original reward with weight
                total_reward = reward + (rubric_reward * self.reward_weight)
            
            # Add detailed reward info
            detailed_reward = self.reward_function.get_detailed_reward(self.last_observation, action)
            info["rubric_reward"] = rubric_reward
            info["original_reward"] = reward
            info["detailed_reward"] = detailed_reward
            
        except Exception as e:
            print(f"Warning: Rubric reward calculation failed: {e}")
            total_reward = reward
        
        self.last_observation = obs
        return obs, total_reward, terminated, truncated, info


def create_cleanrl_reward_function(
    rubric_path: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    observation_to_text: Optional[Callable] = None,
    action_to_text: Optional[Callable] = None,
    **kwargs
) -> CleanRLRewardFunction:
    """Convenience function to create CleanRL reward function."""
    return CleanRLRewardFunction(
        rubric=rubric_path,
        provider=provider,
        model=model,
        api_key=api_key,
        observation_to_text=observation_to_text,
        action_to_text=action_to_text,
        **kwargs
    )


# Text-based environment adapters for language model training
class TextObservationWrapper(gym.ObservationWrapper):
    """Wrapper that converts observations to text format."""
    
    def __init__(self, env: gym.Env, obs_to_text_fn: Callable):
        super().__init__(env)
        self.obs_to_text_fn = obs_to_text_fn
        
        # Update observation space to text
        self.observation_space = gym.spaces.Text(max_length=1000)
    
    def observation(self, observation):
        """Convert observation to text."""
        return self.obs_to_text_fn(observation)


class TextActionWrapper(gym.ActionWrapper):
    """Wrapper that converts text actions to environment actions."""
    
    def __init__(self, env: gym.Env, text_to_action_fn: Callable):
        super().__init__(env)
        self.text_to_action_fn = text_to_action_fn
        
        # Update action space to text
        self.action_space = gym.spaces.Text(max_length=1000)
    
    def action(self, action):
        """Convert text action to environment action."""
        return self.text_to_action_fn(action)


class LanguageModelRewardFunction(CleanRLRewardFunction):
    """Specialized reward function for language model environments."""
    
    def extract_input_output(self, prompt: str, response: str, **kwargs) -> tuple[str, str]:
        """Extract input/output from language model prompt/response."""
        return prompt, response


def create_language_model_env(
    base_env: gym.Env,
    rubric_path: str,
    provider: str = "openai",
    prompt_key: str = "prompt",
    response_key: str = "response",
    **kwargs
) -> gym.Env:
    """Create a language model environment with rubric rewards."""
    
    # Create reward function
    reward_function = LanguageModelRewardFunction(
        rubric=rubric_path,
        provider=provider,
        **kwargs
    )
    
    # Wrap environment
    env = RubricWrapper(base_env, reward_function)
    
    return env


# Logging and monitoring utilities
class RewardLogger:
    """Logger for tracking rubric reward statistics during training."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.episode_rewards = []
        self.step_rewards = []
        self.episode_count = 0
        
        if log_file:
            with open(log_file, 'w') as f:
                f.write("episode,step,rubric_reward,original_reward,total_reward,explanation\n")
    
    def log_step(self, step: int, info: Dict[str, Any]):
        """Log reward information for a single step."""
        if "detailed_reward" in info:
            detailed_reward = info["detailed_reward"]
            
            self.step_rewards.append(detailed_reward.reward)
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"{self.episode_count},{step},{detailed_reward.reward},"
                           f"{info.get('original_reward', 0)},{info.get('rubric_reward', 0)},"
                           f'"{detailed_reward.explanation}"\n')
    
    def log_episode_end(self):
        """Log episode summary statistics."""
        if self.step_rewards:
            episode_total = sum(self.step_rewards)
            episode_mean = episode_total / len(self.step_rewards)
            
            self.episode_rewards.append(episode_total)
            self.episode_count += 1
            
            print(f"Episode {self.episode_count}: "
                  f"Mean reward = {episode_mean:.3f}, "
                  f"Total reward = {episode_total:.3f}")
            
            self.step_rewards = []
    
    def get_statistics(self) -> Dict[str, float]:
        """Get reward statistics."""
        if not self.episode_rewards:
            return {}
        
        rewards = np.array(self.episode_rewards)
        return {
            "mean_episode_reward": rewards.mean(),
            "std_episode_reward": rewards.std(),
            "min_episode_reward": rewards.min(),
            "max_episode_reward": rewards.max(),
            "total_episodes": len(rewards)
        }