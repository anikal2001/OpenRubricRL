"""Reward processing, normalization, and hybrid metric blending."""

import math
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from .rubric import Rubric
from .scorer import ScoringResult


@dataclass
class ProcessedReward:
    """Final processed reward with all components."""
    reward: float
    raw_llm_score: float
    normalized_score: float
    hybrid_components: Dict[str, float]
    explanation: str
    metadata: Dict[str, Any]


class MetricCalculator(ABC):
    """Abstract base class for automated metrics."""
    
    @abstractmethod
    def calculate(self, input_text: str, output_text: str, **kwargs) -> float:
        """Calculate the metric score."""
        pass


class BLEUCalculator(MetricCalculator):
    """BLEU score calculator."""
    
    def __init__(self, reference_texts: Optional[List[str]] = None):
        self.reference_texts = reference_texts or []
    
    def calculate(self, input_text: str, output_text: str, **kwargs) -> float:
        """Calculate BLEU score against references."""
        try:
            from sacrebleu import sentence_bleu
            
            references = kwargs.get('references', self.reference_texts)
            if not references:
                return 0.0
            
            score = sentence_bleu(output_text, references)
            return score.score / 100.0  # Normalize to 0-1
            
        except ImportError:
            raise ImportError("Install sacrebleu: pip install sacrebleu")


class ROUGECalculator(MetricCalculator):
    """ROUGE score calculator."""
    
    def __init__(self, rouge_type: str = "rouge1"):
        self.rouge_type = rouge_type
    
    def calculate(self, input_text: str, output_text: str, **kwargs) -> float:
        """Calculate ROUGE score."""
        try:
            from rouge_score import rouge_scorer
            
            references = kwargs.get('references', [])
            if not references:
                return 0.0
            
            scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
            
            # Calculate against all references and take max
            scores = []
            for ref in references:
                score = scorer.score(ref, output_text)
                scores.append(score[self.rouge_type].fmeasure)
            
            return max(scores) if scores else 0.0
            
        except ImportError:
            raise ImportError("Install rouge-score: pip install rouge-score")


class AccuracyCalculator(MetricCalculator):
    """Simple accuracy calculator."""
    
    def calculate(self, input_text: str, output_text: str, **kwargs) -> float:
        """Calculate accuracy based on expected output."""
        expected = kwargs.get('expected_output', '').strip().lower()
        actual = output_text.strip().lower()
        
        if not expected:
            return 1.0  # No ground truth available
        
        return 1.0 if expected == actual else 0.0


class PerplexityCalculator(MetricCalculator):
    """Perplexity calculator (placeholder - requires language model)."""
    
    def calculate(self, input_text: str, output_text: str, **kwargs) -> float:
        """Calculate perplexity (simplified placeholder)."""
        # This is a simplified version - real implementation would use a language model
        # For now, return inverse of length-normalized log probability estimate
        text_length = len(output_text.split())
        if text_length == 0:
            return 1.0
        
        # Simple heuristic: shorter, common words = lower perplexity
        avg_word_length = len(output_text.replace(' ', '')) / text_length
        estimated_perplexity = math.exp(avg_word_length / 10.0)
        
        # Normalize to 0-1 (lower perplexity = higher score)
        return 1.0 / (1.0 + estimated_perplexity)


class RewardProcessor:
    """Processes and normalizes rewards, blends with automated metrics."""
    
    def __init__(
        self,
        rubric: Rubric,
        normalization_method: str = "min_max",
        hybrid_blend_mode: str = "weighted_average"
    ):
        self.rubric = rubric
        self.normalization_method = normalization_method
        self.hybrid_blend_mode = hybrid_blend_mode
        self.metric_calculators = self._setup_metric_calculators()
    
    def _setup_metric_calculators(self) -> Dict[str, MetricCalculator]:
        """Setup automated metric calculators based on rubric."""
        calculators = {}
        
        if not self.rubric.hybrid_metrics:
            return calculators
        
        for metric in self.rubric.hybrid_metrics:
            if metric.type == "bleu":
                calculators[metric.name] = BLEUCalculator()
            elif metric.type == "rouge":
                rouge_type = metric.config.get("rouge_type", "rouge1") if metric.config else "rouge1"
                calculators[metric.name] = ROUGECalculator(rouge_type)
            elif metric.type == "accuracy":
                calculators[metric.name] = AccuracyCalculator()
            elif metric.type == "perplexity":
                calculators[metric.name] = PerplexityCalculator()
            # Custom metrics would be loaded here
        
        return calculators
    
    def normalize_score(self, score: float, method: Optional[str] = None) -> float:
        """Normalize LLM score to 0-1 range."""
        method = method or self.normalization_method
        
        scale_min = self.rubric.scale.min
        scale_max = self.rubric.scale.max
        
        if method == "min_max":
            # Linear normalization to 0-1
            return (score - scale_min) / (scale_max - scale_min)
        
        elif method == "sigmoid":
            # Sigmoid normalization (handles outliers better)
            midpoint = (scale_min + scale_max) / 2
            steepness = 4.0 / (scale_max - scale_min)  # Adjustable steepness
            return 1.0 / (1.0 + math.exp(-steepness * (score - midpoint)))
        
        elif method == "z_score":
            # Z-score normalization (requires historical data)
            # Simplified version assuming mean = midpoint, std = range/4
            mean = (scale_min + scale_max) / 2
            std = (scale_max - scale_min) / 4
            z_score = (score - mean) / std
            # Convert to 0-1 using sigmoid
            return 1.0 / (1.0 + math.exp(-z_score))
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def calculate_hybrid_metrics(
        self, 
        input_text: str, 
        output_text: str, 
        **metric_kwargs
    ) -> Dict[str, float]:
        """Calculate all automated metrics."""
        hybrid_scores = {}
        
        for metric_name, calculator in self.metric_calculators.items():
            try:
                score = calculator.calculate(input_text, output_text, **metric_kwargs)
                hybrid_scores[metric_name] = score
            except Exception as e:
                # Log error but don't fail the entire process
                hybrid_scores[metric_name] = 0.0
                print(f"Warning: Failed to calculate {metric_name}: {e}")
        
        return hybrid_scores
    
    def blend_scores(
        self, 
        llm_score: float, 
        hybrid_scores: Dict[str, float],
        method: Optional[str] = None
    ) -> float:
        """Blend LLM score with automated metrics."""
        method = method or self.hybrid_blend_mode
        
        if not self.rubric.hybrid_metrics or not hybrid_scores:
            return llm_score
        
        if method == "weighted_average":
            # Calculate LLM weight (1.0 minus sum of hybrid weights)
            hybrid_weight_sum = sum(metric.weight for metric in self.rubric.hybrid_metrics)
            llm_weight = max(0.0, 1.0 - hybrid_weight_sum)
            
            # Weighted combination
            total_score = llm_score * llm_weight
            
            for metric in self.rubric.hybrid_metrics:
                metric_score = hybrid_scores.get(metric.name, 0.0)
                total_score += metric_score * metric.weight
            
            return total_score
        
        elif method == "multiplicative":
            # Multiplicative blending (all scores must be good)
            combined_score = llm_score
            
            for metric in self.rubric.hybrid_metrics:
                metric_score = hybrid_scores.get(metric.name, 1.0)
                weight = metric.weight
                # Apply weighted geometric mean
                combined_score *= metric_score ** weight
            
            return combined_score
        
        elif method == "max":
            # Take maximum of LLM and hybrid scores
            all_scores = [llm_score] + list(hybrid_scores.values())
            return max(all_scores)
        
        elif method == "min":
            # Take minimum (conservative approach)
            all_scores = [llm_score] + list(hybrid_scores.values())
            return min(all_scores)
        
        else:
            raise ValueError(f"Unknown blend method: {method}")
    
    def process_reward(
        self,
        scoring_result: ScoringResult,
        input_text: str,
        output_text: str,
        **metric_kwargs
    ) -> ProcessedReward:
        """Process a scoring result into a final reward."""
        
        # Normalize LLM score
        normalized_score = self.normalize_score(scoring_result.overall_score)
        
        # Calculate hybrid metrics
        hybrid_scores = self.calculate_hybrid_metrics(
            input_text, output_text, **metric_kwargs
        )
        
        # Blend scores
        final_reward = self.blend_scores(normalized_score, hybrid_scores)
        
        # Prepare metadata
        metadata = {
            "rubric_name": self.rubric.name,
            "rubric_version": self.rubric.version,
            "normalization_method": self.normalization_method,
            "blend_method": self.hybrid_blend_mode,
            "criterion_scores": scoring_result.criterion_scores,
            "criterion_explanations": scoring_result.criterion_explanations,
        }
        
        return ProcessedReward(
            reward=final_reward,
            raw_llm_score=scoring_result.overall_score,
            normalized_score=normalized_score,
            hybrid_components=hybrid_scores,
            explanation=scoring_result.overall_explanation,
            metadata=metadata
        )


class RewardCache:
    """Simple in-memory cache for processed rewards."""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, ProcessedReward] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def _make_key(self, input_text: str, output_text: str, rubric_name: str) -> str:
        """Create cache key from inputs."""
        import hashlib
        combined = f"{rubric_name}:{input_text}:{output_text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, input_text: str, output_text: str, rubric_name: str) -> Optional[ProcessedReward]:
        """Get cached reward if available."""
        key = self._make_key(input_text, output_text, rubric_name)
        
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        return None
    
    def put(self, input_text: str, output_text: str, rubric_name: str, reward: ProcessedReward) -> None:
        """Cache a processed reward."""
        key = self._make_key(input_text, output_text, rubric_name)
        
        # Evict least recently used if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = reward
        
        if key not in self.access_order:
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


# Global cache instance
_reward_cache = RewardCache()


def get_reward_cache() -> RewardCache:
    """Get the global reward cache."""
    return _reward_cache