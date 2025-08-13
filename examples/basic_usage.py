"""Basic usage examples for OpenRubricRL."""

import os
import json
import asyncio
from pathlib import Path

from openrubricrl import Rubric, RubricScorer, create_openai_scorer, create_anthropic_scorer
from openrubricrl.core.prompt_builder import build_prompt_for_rubric
from openrubricrl.core.reward_processor import RewardProcessor
from openrubricrl.logging import OpenRubricLogger, get_statistics


def create_sample_rubrics():
    """Create sample rubrics for demonstration."""
    
    # Code quality rubric
    code_rubric = {
        "name": "code_quality_comprehensive",
        "version": "1.0.0", 
        "description": "Comprehensive code quality evaluation",
        "domain": "code",
        "scale": {"min": 0.0, "max": 10.0},
        "criteria": [
            {
                "name": "correctness",
                "description": "Does the code solve the problem correctly and handle edge cases?",
                "weight": 0.4,
                "examples": {
                    "excellent": [
                        {
                            "input": "Write a function to check if a string is a palindrome",
                            "output": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
                            "score": 9.0,
                            "explanation": "Handles case insensitivity and spaces correctly"
                        }
                    ],
                    "good": [
                        {
                            "input": "Write a function to check if a string is a palindrome",
                            "output": "def is_palindrome(s):\n    return s == s[::-1]",
                            "score": 7.0,
                            "explanation": "Correct logic but doesn't handle edge cases"
                        }
                    ],
                    "poor": [
                        {
                            "input": "Write a function to check if a string is a palindrome",
                            "output": "def check_palindrome():\n    print('hello')",
                            "score": 2.0,
                            "explanation": "Doesn't solve the problem at all"
                        }
                    ]
                }
            },
            {
                "name": "readability",
                "description": "Is the code clean, well-named, and easy to understand?",
                "weight": 0.3,
                "subcriteria": [
                    {"name": "naming", "description": "Variables and functions have descriptive names"},
                    {"name": "structure", "description": "Code is well-organized and formatted"},
                    {"name": "comments", "description": "Complex logic is explained with comments"}
                ]
            },
            {
                "name": "efficiency",
                "description": "Is the solution computationally efficient?",
                "weight": 0.3
            }
        ],
        "hybrid_metrics": [
            {
                "name": "code_length",
                "type": "custom",
                "weight": 0.1,
                "config": {"penalty_threshold": 100}
            }
        ]
    }
    
    # Creative writing rubric
    writing_rubric = {
        "name": "creative_writing",
        "version": "1.0.0",
        "description": "Evaluate creative writing quality",
        "domain": "creative_writing",
        "scale": {"min": 0.0, "max": 10.0},
        "criteria": [
            {
                "name": "creativity",
                "description": "Is the writing original and imaginative?",
                "weight": 0.3
            },
            {
                "name": "coherence",
                "description": "Does the writing flow logically and make sense?",
                "weight": 0.3
            },
            {
                "name": "style",
                "description": "Is the writing style engaging and appropriate?",
                "weight": 0.2
            },
            {
                "name": "grammar",
                "description": "Is the writing grammatically correct?",
                "weight": 0.2
            }
        ]
    }
    
    # Save rubrics
    with open("code_quality.json", 'w') as f:
        json.dump(code_rubric, f, indent=2)
    
    with open("creative_writing.json", 'w') as f:
        json.dump(writing_rubric, f, indent=2)
    
    print("Created sample rubrics: code_quality.json, creative_writing.json")


async def basic_scoring_example():
    """Demonstrate basic scoring functionality."""
    
    print("\n=== Basic Scoring Example ===")
    
    # Load rubric
    rubric = Rubric.from_file("code_quality.json")
    print(f"Loaded rubric: {rubric.name} v{rubric.version}")
    
    # Create scorer
    scorer = create_openai_scorer(
        rubric=rubric,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )
    
    # Test cases
    test_cases = [
        {
            "input": "Write a function to calculate factorial",
            "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "input": "Write a function to reverse a list",
            "output": "def reverse_list(lst):\n    return lst[::-1]"
        },
        {
            "input": "Write a function to find maximum in list",
            "output": "def find_max(numbers):\n    max_val = numbers[0]\n    for num in numbers:\n        if num > max_val:\n            max_val = num\n    return max_val"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Task: {test_case['input']}")
        print(f"Code: {test_case['output']}")
        
        # Score the output
        result = await scorer.score(
            task_input=test_case['input'],
            model_output=test_case['output']
        )
        
        results.append(result)
        
        print(f"Overall Score: {result.overall_score:.2f}/10")
        print(f"Explanation: {result.overall_explanation}")
        
        print("Criterion Breakdown:")
        for criterion in rubric.criteria:
            name = criterion.name
            score = result.criterion_scores.get(name, 0)
            explanation = result.criterion_explanations.get(name, "No explanation")
            print(f"  {name}: {score:.2f} (weight: {criterion.weight:.1%})")
            print(f"    {explanation}")
    
    # Summary statistics
    scores = [r.overall_score for r in results]
    print(f"\nSummary:")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    print(f"Best score: {max(scores):.2f}")
    print(f"Worst score: {min(scores):.2f}")


async def reward_processing_example():
    """Demonstrate reward processing with normalization and hybrid metrics."""
    
    print("\n=== Reward Processing Example ===")
    
    # Load rubric
    rubric = Rubric.from_file("code_quality.json")
    
    # Create scorer and processor
    scorer = create_openai_scorer(rubric, api_key=os.getenv("OPENAI_API_KEY"))
    processor = RewardProcessor(
        rubric=rubric,
        normalization_method="min_max",
        hybrid_blend_mode="weighted_average"
    )
    
    # Test different normalization methods
    test_input = "Write a function to sort a list"
    test_output = "def sort_list(lst):\n    return sorted(lst)"
    
    # Get scoring result
    scoring_result = await scorer.score(test_input, test_output)
    
    print(f"Raw LLM Score: {scoring_result.overall_score:.2f}")
    
    # Test different normalization methods
    normalization_methods = ["min_max", "sigmoid", "z_score"]
    
    for method in normalization_methods:
        normalized = processor.normalize_score(scoring_result.overall_score, method)
        print(f"Normalized ({method}): {normalized:.3f}")
    
    # Process with hybrid metrics
    processed_reward = processor.process_reward(
        scoring_result=scoring_result,
        input_text=test_input,
        output_text=test_output
    )
    
    print(f"\nProcessed Reward: {processed_reward.reward:.3f}")
    print(f"Hybrid Components: {processed_reward.hybrid_components}")
    print(f"Explanation: {processed_reward.explanation}")


async def batch_scoring_example():
    """Demonstrate batch scoring for multiple outputs."""
    
    print("\n=== Batch Scoring Example ===")
    
    # Load rubric
    rubric = Rubric.from_file("creative_writing.json")
    
    # Create scorer
    scorer = create_openai_scorer(rubric, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Batch of writing samples
    writing_samples = [
        {
            "task_input": "Write a short story about a time traveler",
            "model_output": "Sarah stepped into the gleaming machine, her heart pounding. The world dissolved into swirling colors, and suddenly she stood in ancient Rome, watching gladiators prepare for battle."
        },
        {
            "task_input": "Write a poem about the ocean",
            "model_output": "Waves crash against the rocky shore,\nEndless blue stretching evermore,\nSeagulls dance in salty air,\nOcean's song beyond compare."
        },
        {
            "task_input": "Describe a magical forest",
            "model_output": "The forest sparkled with ethereal light. Ancient trees whispered secrets while luminescent flowers bloomed in impossible colors. Fairy lights danced between the branches."
        }
    ]
    
    print(f"Scoring {len(writing_samples)} writing samples...")
    
    # Batch score
    results = await scorer.score_batch(writing_samples)
    
    # Display results
    for i, (sample, result) in enumerate(zip(writing_samples, results), 1):
        print(f"\nSample {i}:")
        print(f"Prompt: {sample['task_input']}")
        print(f"Writing: {sample['model_output']}")
        print(f"Score: {result.overall_score:.2f}/10")
        print(f"Explanation: {result.overall_explanation}")


def prompt_building_example():
    """Demonstrate prompt building functionality."""
    
    print("\n=== Prompt Building Example ===")
    
    # Load rubric
    rubric = Rubric.from_file("code_quality.json")
    
    # Build a scoring prompt
    prompt = build_prompt_for_rubric(
        rubric=rubric,
        task_input="Write a function to calculate the area of a circle",
        model_output="def circle_area(radius):\n    return 3.14159 * radius ** 2",
        include_examples=True,
        max_examples_per_criterion=1
    )
    
    print("Generated Scoring Prompt:")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
    # Save prompt to file for inspection
    with open("sample_prompt.txt", 'w') as f:
        f.write(prompt)
    
    print("\nPrompt saved to: sample_prompt.txt")


def logging_example():
    """Demonstrate logging functionality."""
    
    print("\n=== Logging Example ===")
    
    # Set up logger
    logger = OpenRubricLogger(
        log_file="openrubricrl_demo.log",
        log_level="INFO"
    )
    
    # Log some sample events
    logger.log_info("Starting demonstration session")
    logger.log_info("This is an informational message")
    logger.log_warning("This is a warning message")
    logger.log_debug("This is a debug message (won't show unless log level is DEBUG)")
    
    # Simulate error logging
    try:
        raise ValueError("This is a demo error")
    except Exception as e:
        logger.log_error(e, "Demo error context")
    
    # Log training metrics
    logger.log_training_step(
        step=100,
        episode=10,
        rewards=[0.75, 0.82, 0.68, 0.91],
        additional_metrics={"loss": 0.15, "accuracy": 0.87}
    )
    
    logger.log_episode_summary(
        episode=10,
        total_reward=3.16,
        episode_length=4,
        additional_info={"exploration_rate": 0.1}
    )
    
    # Get statistics (if any results exist)
    stats = get_statistics()
    if stats:
        print(f"Statistics: {stats}")
    else:
        print("No scoring statistics available yet")
    
    print("Logging demo completed. Check openrubricrl_demo.log for output.")


def rubric_validation_example():
    """Demonstrate rubric validation."""
    
    print("\n=== Rubric Validation Example ===")
    
    # Test valid rubric
    try:
        rubric = Rubric.from_file("code_quality.json")
        print(f"✅ Valid rubric: {rubric.name}")
        
        # Test validation
        is_valid = rubric.validate_schema()
        print(f"Schema validation: {'✅ Passed' if is_valid else '❌ Failed'}")
        
    except Exception as e:
        print(f"❌ Rubric validation failed: {e}")
    
    # Test invalid rubric (weights don't sum to 1)
    invalid_rubric_data = {
        "name": "invalid_test",
        "version": "1.0.0",
        "scale": {"min": 0.0, "max": 10.0},
        "criteria": [
            {"name": "test1", "description": "Test", "weight": 0.3},
            {"name": "test2", "description": "Test", "weight": 0.3}  # Total = 0.6, not 1.0
        ]
    }
    
    try:
        invalid_rubric = Rubric(**invalid_rubric_data)
        print("❌ Should have failed validation")
    except ValueError as e:
        print(f"✅ Correctly caught invalid rubric: {e}")


async def comparison_example():
    """Compare different LLM providers."""
    
    print("\n=== Provider Comparison Example ===")
    
    # Check if both API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("No API keys available for comparison")
        return
    
    # Load rubric
    rubric = Rubric.from_file("code_quality.json")
    
    # Test case
    test_input = "Write a function to find the greatest common divisor"
    test_output = """def gcd(a, b):
    while b:
        a, b = b, a % b
    return a"""
    
    providers = []
    
    if openai_key:
        providers.append(("OpenAI", create_openai_scorer(rubric, api_key=openai_key)))
    
    if anthropic_key:
        providers.append(("Anthropic", create_anthropic_scorer(rubric, api_key=anthropic_key)))
    
    print(f"Comparing {len(providers)} providers...")
    print(f"Task: {test_input}")
    print(f"Code: {test_output}")
    
    for provider_name, scorer in providers:
        try:
            result = await scorer.score(test_input, test_output)
            
            print(f"\n{provider_name} Results:")
            print(f"  Score: {result.overall_score:.2f}/10")
            print(f"  Explanation: {result.overall_explanation}")
            
        except Exception as e:
            print(f"\n{provider_name} failed: {e}")


async def main():
    """Run all examples."""
    
    print("OpenRubricRL Basic Usage Examples")
    print("=" * 40)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run examples.")
        print("Some examples will be skipped.")
    
    # Create sample rubrics
    create_sample_rubrics()
    
    # Run examples
    examples = [
        ("Rubric Validation", rubric_validation_example),
        ("Prompt Building", prompt_building_example),
        ("Logging", logging_example),
    ]
    
    # Add async examples if API keys are available
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        async_examples = [
            ("Basic Scoring", basic_scoring_example),
            ("Reward Processing", reward_processing_example),
            ("Batch Scoring", batch_scoring_example),
            ("Provider Comparison", comparison_example),
        ]
        
        for name, func in async_examples:
            print(f"\nRunning: {name}")
            try:
                await func()
            except Exception as e:
                print(f"Example failed: {e}")
    
    # Run sync examples
    for name, func in examples:
        print(f"\nRunning: {name}")
        try:
            func()
        except Exception as e:
            print(f"Example failed: {e}")
    
    print("\n" + "=" * 40)
    print("All examples completed!")
    print("\nGenerated files:")
    print("- code_quality.json (sample rubric)")
    print("- creative_writing.json (sample rubric)")
    print("- sample_prompt.txt (generated prompt)")
    print("- openrubricrl_demo.log (log file)")


if __name__ == "__main__":
    asyncio.run(main())