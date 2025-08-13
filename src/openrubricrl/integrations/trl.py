"""TRL (Transformers Reinforcement Learning) integration."""

from typing import Dict, Any, Optional, Union, List
import torch

from .base import BaseRewardFunction
from ..core.rubric import Rubric
from ..core.scorer import create_openai_scorer, create_anthropic_scorer


class TRLRewardFunction(BaseRewardFunction):
    """Reward function for TRL (Transformers RL) integration."""
    
    def __init__(
        self,
        rubric: Union[str, Rubric],
        provider: str = "openai", 
        model: Optional[str] = None,
        api_key: Optional[str] = None,
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
    
    def extract_input_output(self, query: str, response: str, **kwargs) -> tuple[str, str]:
        """Extract input/output from TRL query/response format."""
        return query, response
    
    def __call__(self, query: str, response: str, **kwargs) -> float:
        """TRL-compatible reward function call."""
        return super().__call__(query, response, **kwargs)


class TRLBatchRewardFunction:
    """Batch reward function for TRL training."""
    
    def __init__(self, reward_function: TRLRewardFunction):
        self.reward_function = reward_function
    
    async def __call__(self, queries: List[str], responses: List[str]) -> List[float]:
        """Calculate rewards for a batch of query/response pairs."""
        import asyncio
        
        # Calculate rewards concurrently
        tasks = [
            self.reward_function.async_call(query, response)
            for query, response in zip(queries, responses)
        ]
        
        rewards = await asyncio.gather(*tasks)
        return rewards
    
    def sync_call(self, queries: List[str], responses: List[str]) -> List[float]:
        """Synchronous batch reward calculation."""
        import asyncio
        return asyncio.run(self(queries, responses))


def create_trl_reward_function(
    rubric_path: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> TRLRewardFunction:
    """Convenience function to create TRL reward function."""
    return TRLRewardFunction(
        rubric=rubric_path,
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )


class PPOTrainerWithRubric:
    """PPO Trainer wrapper with rubric-based rewards."""
    
    def __init__(
        self,
        model,
        tokenizer,
        reward_function: TRLRewardFunction,
        **ppo_config
    ):
        try:
            from trl import PPOTrainer, PPOConfig
        except ImportError:
            raise ImportError("Install TRL: pip install trl")
        
        self.reward_function = reward_function
        self.batch_reward_function = TRLBatchRewardFunction(reward_function)
        
        # Create PPO config
        config = PPOConfig(**ppo_config)
        
        # Initialize PPO trainer
        self.trainer = PPOTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
        )
    
    def step(self, queries: List[str], responses: List[str], **kwargs):
        """Perform one training step with rubric rewards."""
        # Calculate rewards using rubric
        rewards = self.batch_reward_function.sync_call(queries, responses)
        
        # Convert to tensors
        reward_tensors = [torch.tensor(r) for r in rewards]
        
        # Prepare for PPO step
        query_tensors = [self.trainer.tokenizer.encode(q, return_tensors="pt")[0] for q in queries]
        response_tensors = [self.trainer.tokenizer.encode(r, return_tensors="pt")[0] for r in responses]
        
        # Perform PPO step
        stats = self.trainer.step(query_tensors, response_tensors, reward_tensors)
        
        # Add reward statistics
        stats["reward/mean"] = sum(rewards) / len(rewards)
        stats["reward/std"] = torch.tensor(rewards).std().item()
        stats["reward/min"] = min(rewards)
        stats["reward/max"] = max(rewards)
        
        return stats
    
    def save_model(self, path: str):
        """Save the trained model."""
        self.trainer.save_model(path)


# Integration with Hugging Face Transformers
class RubricCallback:
    """Transformers callback for rubric-based evaluation during training."""
    
    def __init__(
        self,
        reward_function: TRLRewardFunction,
        eval_queries: List[str],
        eval_interval: int = 100
    ):
        self.reward_function = reward_function
        self.eval_queries = eval_queries
        self.eval_interval = eval_interval
        self.step_count = 0
    
    def on_step_end(self, trainer, **kwargs):
        """Called at the end of each training step."""
        self.step_count += 1
        
        if self.step_count % self.eval_interval == 0:
            self._evaluate_model(trainer)
    
    def _evaluate_model(self, trainer):
        """Evaluate the model using rubric rewards."""
        model = trainer.model
        tokenizer = trainer.tokenizer
        
        total_reward = 0.0
        
        for query in self.eval_queries:
            # Generate response
            inputs = tokenizer.encode(query, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Calculate reward
            reward = self.reward_function(query, response)
            total_reward += reward
        
        avg_reward = total_reward / len(self.eval_queries)
        
        # Log to trainer
        trainer.log({"eval/rubric_reward": avg_reward})
        print(f"Step {self.step_count}: Average rubric reward = {avg_reward:.3f}")


def create_ppo_trainer_with_rubric(
    model,
    tokenizer, 
    rubric_path: str,
    provider: str = "openai",
    **ppo_config
) -> PPOTrainerWithRubric:
    """Create a PPO trainer with rubric-based rewards."""
    reward_function = create_trl_reward_function(
        rubric_path=rubric_path,
        provider=provider
    )
    
    return PPOTrainerWithRubric(
        model=model,
        tokenizer=tokenizer,
        reward_function=reward_function,
        **ppo_config
    )