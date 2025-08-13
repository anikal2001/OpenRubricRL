"""LLM-based scoring using rubrics."""

import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod

import openai
import anthropic
from pydantic import BaseModel

from .rubric import Rubric
from .prompt_builder import PromptBuilder


class ScoringResult(BaseModel):
    """Result of scoring a model output."""
    overall_score: float
    overall_explanation: str
    criterion_scores: Dict[str, float]
    criterion_explanations: Dict[str, str]
    raw_response: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 2000),
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", 0.1),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class RubricScorer:
    """Score model outputs using LLM-based rubric evaluation."""
    
    def __init__(
        self,
        rubric: Rubric,
        llm_provider: LLMProvider,
        include_examples: bool = True,
        max_examples_per_criterion: int = 2
    ):
        self.rubric = rubric
        self.llm_provider = llm_provider
        self.prompt_builder = PromptBuilder(rubric)
        self.include_examples = include_examples
        self.max_examples_per_criterion = max_examples_per_criterion
    
    async def score(
        self,
        task_input: str,
        model_output: str,
        **llm_kwargs
    ) -> ScoringResult:
        """Score a model output using the rubric."""
        
        # Build the scoring prompt
        prompt = self.prompt_builder.build_scoring_prompt(
            task_input=task_input,
            model_output=model_output,
            include_examples=self.include_examples,
            max_examples_per_criterion=self.max_examples_per_criterion
        )
        
        # Get LLM response
        raw_response = await self.llm_provider.generate(prompt, **llm_kwargs)
        
        # Parse the response
        try:
            parsed_result = self._parse_scoring_response(raw_response)
            return ScoringResult(
                raw_response=raw_response,
                **parsed_result
            )
        except Exception as e:
            # Fallback: try to extract what we can
            return self._fallback_parse(raw_response, str(e))
    
    def _parse_scoring_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured scoring result."""
        
        # Try to find JSON in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response[json_start:json_end]
        result = json.loads(json_str)
        
        # Extract criterion scores and explanations
        criterion_scores = {}
        criterion_explanations = {}
        
        for criterion in self.rubric.criteria:
            score_key = f"{criterion.name}_score"
            explanation_key = f"{criterion.name}_explanation"
            
            if score_key in result:
                criterion_scores[criterion.name] = float(result[score_key])
            if explanation_key in result:
                criterion_explanations[criterion.name] = str(result[explanation_key])
        
        return {
            "overall_score": float(result.get("overall_score", 0)),
            "overall_explanation": str(result.get("overall_explanation", "")),
            "criterion_scores": criterion_scores,
            "criterion_explanations": criterion_explanations
        }
    
    def _fallback_parse(self, response: str, error_msg: str) -> ScoringResult:
        """Fallback parsing when structured parsing fails."""
        return ScoringResult(
            overall_score=0.0,
            overall_explanation=f"Failed to parse response: {error_msg}",
            criterion_scores={},
            criterion_explanations={},
            raw_response=response
        )
    
    async def score_batch(
        self,
        inputs: List[Dict[str, str]],
        **llm_kwargs
    ) -> List[ScoringResult]:
        """Score multiple inputs in batch."""
        tasks = [
            self.score(
                task_input=item["task_input"],
                model_output=item["model_output"],
                **llm_kwargs
            )
            for item in inputs
        ]
        
        return await asyncio.gather(*tasks)


# Convenience functions
def create_openai_scorer(
    rubric: Rubric,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    **scorer_kwargs
) -> RubricScorer:
    """Create a scorer using OpenAI."""
    provider = OpenAIProvider(api_key=api_key, model=model)
    return RubricScorer(rubric=rubric, llm_provider=provider, **scorer_kwargs)


def create_anthropic_scorer(
    rubric: Rubric,
    api_key: Optional[str] = None,
    model: str = "claude-3-sonnet-20240229",
    **scorer_kwargs
) -> RubricScorer:
    """Create a scorer using Anthropic."""
    provider = AnthropicProvider(api_key=api_key, model=model)
    return RubricScorer(rubric=rubric, llm_provider=provider, **scorer_kwargs)