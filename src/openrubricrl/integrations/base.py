"""Base classes for RL integrations."""

import asyncio
from typing import Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod

from ..core.rubric import Rubric
from ..core.scorer import RubricScorer
from ..core.reward_processor import RewardProcessor, ProcessedReward


class BaseRewardFunction(ABC):
    """Abstract base class for RL reward functions."""
    
    def __init__(
        self,
        rubric: Union[str, Rubric],
        scorer: Optional[RubricScorer] = None,
        processor: Optional[RewardProcessor] = None,
        cache_rewards: bool = True,
        async_mode: bool = False
    ):
        # Load rubric if path provided
        if isinstance(rubric, str):
            self.rubric = Rubric.from_file(rubric)
        else:
            self.rubric = rubric
        
        self.scorer = scorer
        self.processor = processor or RewardProcessor(self.rubric)
        self.cache_rewards = cache_rewards
        self.async_mode = async_mode
        
        # Initialize cache if enabled
        if cache_rewards:
            from ..core.reward_processor import get_reward_cache
            self.cache = get_reward_cache()
        else:
            self.cache = None
    
    @abstractmethod
    def extract_input_output(self, *args, **kwargs) -> tuple[str, str]:
        """Extract task input and model output from RL-specific arguments."""
        pass
    
    def __call__(self, *args, **kwargs) -> float:
        """Main reward function call."""
        if self.async_mode:
            # Run async version in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.async_call(*args, **kwargs))
            finally:
                loop.close()
        else:
            return self.sync_call(*args, **kwargs)
    
    def sync_call(self, *args, **kwargs) -> float:
        """Synchronous reward calculation (uses async internally)."""
        return asyncio.run(self.async_call(*args, **kwargs))
    
    async def async_call(self, *args, **kwargs) -> float:
        """Asynchronous reward calculation."""
        try:
            # Extract input/output from RL-specific arguments
            task_input, model_output = self.extract_input_output(*args, **kwargs)
            
            # Check cache first
            if self.cache:
                cached_reward = self.cache.get(task_input, model_output, self.rubric.name)
                if cached_reward:
                    return cached_reward.reward
            
            # Score using LLM
            if not self.scorer:
                raise ValueError("No scorer configured for reward function")
            
            scoring_result = await self.scorer.score(
                task_input=task_input,
                model_output=model_output
            )
            
            # Process reward
            processed_reward = self.processor.process_reward(
                scoring_result=scoring_result,
                input_text=task_input,
                output_text=model_output,
                **kwargs
            )
            
            # Cache result
            if self.cache:
                self.cache.put(task_input, model_output, self.rubric.name, processed_reward)
            
            return processed_reward.reward
            
        except Exception as e:
            # Return neutral reward on error to avoid breaking training
            print(f"Warning: Reward calculation failed: {e}")
            return 0.0
    
    def get_detailed_reward(self, *args, **kwargs) -> ProcessedReward:
        """Get detailed reward information (not just the scalar)."""
        if self.async_mode:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.async_get_detailed_reward(*args, **kwargs))
            finally:
                loop.close()
        else:
            return asyncio.run(self.async_get_detailed_reward(*args, **kwargs))
    
    async def async_get_detailed_reward(self, *args, **kwargs) -> ProcessedReward:
        """Get detailed reward information asynchronously."""
        task_input, model_output = self.extract_input_output(*args, **kwargs)
        
        # Check cache first
        if self.cache:
            cached_reward = self.cache.get(task_input, model_output, self.rubric.name)
            if cached_reward:
                return cached_reward
        
        # Score and process
        scoring_result = await self.scorer.score(
            task_input=task_input,
            model_output=model_output
        )
        
        processed_reward = self.processor.process_reward(
            scoring_result=scoring_result,
            input_text=task_input,
            output_text=model_output,
            **kwargs
        )
        
        # Cache result
        if self.cache:
            self.cache.put(task_input, model_output, self.rubric.name, processed_reward)
        
        return processed_reward