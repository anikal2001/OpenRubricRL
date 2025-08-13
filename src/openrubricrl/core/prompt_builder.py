"""Convert rubrics into LLM scoring prompts."""

from typing import Dict, List, Optional
from .rubric import Rubric, Criterion, Example


class PromptBuilder:
    """Converts rubrics into LLM scoring prompts."""
    
    def __init__(self, rubric: Rubric):
        self.rubric = rubric
    
    def build_scoring_prompt(
        self, 
        task_input: str, 
        model_output: str,
        include_examples: bool = True,
        max_examples_per_criterion: int = 2
    ) -> str:
        """Build a complete scoring prompt for the given input/output pair."""
        
        prompt_parts = [
            self._build_header(),
            self._build_rubric_description(),
            self._build_criteria_section(include_examples, max_examples_per_criterion),
            self._build_scoring_instructions(),
            self._build_task_section(task_input, model_output),
            self._build_output_format()
        ]
        
        return "\n\n".join(prompt_parts)
    
    def _build_header(self) -> str:
        """Build the prompt header."""
        return f"""You are an expert evaluator tasked with scoring model outputs using the "{self.rubric.name}" rubric.
Your goal is to provide accurate, consistent, and fair scores based on the defined criteria."""
    
    def _build_rubric_description(self) -> str:
        """Build the rubric description section."""
        parts = [f"# Rubric: {self.rubric.name}"]
        
        if self.rubric.description:
            parts.append(f"Description: {self.rubric.description}")
        
        if self.rubric.domain:
            parts.append(f"Domain: {self.rubric.domain}")
        
        parts.append(f"Score Range: {self.rubric.scale.min} to {self.rubric.scale.max}")
        
        return "\n".join(parts)
    
    def _build_criteria_section(
        self, 
        include_examples: bool = True, 
        max_examples_per_criterion: int = 2
    ) -> str:
        """Build the criteria section with optional examples."""
        parts = ["# Evaluation Criteria"]
        
        for i, criterion in enumerate(self.rubric.criteria, 1):
            criterion_section = [
                f"## {i}. {criterion.name.title()} (Weight: {criterion.weight:.1%})",
                f"{criterion.description}"
            ]
            
            if criterion.subcriteria:
                criterion_section.append("Sub-criteria:")
                for sub in criterion.subcriteria:
                    criterion_section.append(f"- {sub.name}: {sub.description}")
            
            if include_examples and criterion.examples:
                criterion_section.append(self._build_examples_section(criterion, max_examples_per_criterion))
            
            parts.append("\n".join(criterion_section))
        
        return "\n\n".join(parts)
    
    def _build_examples_section(self, criterion: Criterion, max_examples: int) -> str:
        """Build examples section for a criterion."""
        if not criterion.examples:
            return ""
        
        examples_parts = ["### Examples:"]
        
        for quality_level in ["excellent", "good", "poor"]:
            if quality_level in criterion.examples:
                examples = criterion.examples[quality_level][:max_examples]
                if examples:
                    examples_parts.append(f"**{quality_level.title()} ({examples[0].score:.1f}/{self.rubric.scale.max}):**")
                    for example in examples:
                        examples_parts.append(f"- Input: {example.input}")
                        examples_parts.append(f"  Output: {example.output}")
                        examples_parts.append(f"  Score: {example.score}")
                        examples_parts.append(f"  Explanation: {example.explanation}")
        
        return "\n".join(examples_parts)
    
    def _build_scoring_instructions(self) -> str:
        """Build scoring instructions."""
        return f"""# Scoring Instructions

1. Evaluate the model output against each criterion carefully
2. Consider the weight of each criterion in your final score
3. Provide specific explanations for your scores
4. Be consistent and fair in your evaluation
5. Use the full scale ({self.rubric.scale.min}-{self.rubric.scale.max}) appropriately
6. Base your evaluation on objective qualities, not personal preferences"""
    
    def _build_task_section(self, task_input: str, model_output: str) -> str:
        """Build the task-specific section."""
        return f"""# Task to Evaluate

**Input:**
{task_input}

**Model Output:**
{model_output}"""
    
    def _build_output_format(self) -> str:
        """Build the expected output format section."""
        criterion_format = []
        for criterion in self.rubric.criteria:
            criterion_format.append(f'  "{criterion.name}_score": <score>,')
            criterion_format.append(f'  "{criterion.name}_explanation": "<detailed explanation>",')
        
        return f"""# Required Output Format

Provide your evaluation in the following JSON format:

{{
{chr(10).join(criterion_format)}
  "overall_score": <weighted average of criterion scores>,
  "overall_explanation": "<summary of the evaluation>"
}}

Ensure all scores are between {self.rubric.scale.min} and {self.rubric.scale.max}."""


def build_prompt_for_rubric(
    rubric: Rubric,
    task_input: str,
    model_output: str,
    **kwargs
) -> str:
    """Convenience function to build a prompt for a rubric."""
    builder = PromptBuilder(rubric)
    return builder.build_scoring_prompt(task_input, model_output, **kwargs)