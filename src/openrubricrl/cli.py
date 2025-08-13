"""Command-line interface for OpenRubricRL."""

import os
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any

import click
import yaml

from .core.rubric import Rubric
from .core.scorer import create_openai_scorer, create_anthropic_scorer


@click.group()
@click.version_option()
def main():
    """OpenRubricRL - Convert rubrics into LLM-based reward functions."""
    pass


@main.command()
@click.argument('rubric_file', type=click.Path(exists=True))
@click.option('--validate-only', is_flag=True, help='Only validate the rubric schema')
@click.option('--output-format', type=click.Choice(['json', 'yaml']), default='json', help='Output format')
def validate(rubric_file: str, validate_only: bool, output_format: str):
    """Validate a rubric file."""
    try:
        rubric = Rubric.from_file(rubric_file)
        
        if validate_only:
            click.echo(f"✅ Rubric '{rubric.name}' is valid")
        else:
            # Pretty print the parsed rubric
            data = rubric.to_dict()
            if output_format == 'yaml':
                click.echo(yaml.dump(data, default_flow_style=False, indent=2))
            else:
                click.echo(json.dumps(data, indent=2, ensure_ascii=False))
                
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        exit(1)


@main.command()
@click.argument('rubric_file', type=click.Path(exists=True))
@click.argument('task_input')
@click.argument('model_output')
@click.option('--provider', type=click.Choice(['openai', 'anthropic']), default='openai', help='LLM provider')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
@click.option('--temperature', type=float, default=0.1, help='Temperature for LLM generation')
@click.option('--include-examples/--no-examples', default=True, help='Include examples in prompt')
@click.option('--max-examples', type=int, default=2, help='Max examples per criterion')
@click.option('--output-format', type=click.Choice(['json', 'table']), default='table', help='Output format')
@click.option('--save-prompt', type=click.Path(), help='Save the generated prompt to file')
def score(
    rubric_file: str,
    task_input: str,
    model_output: str,
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
    temperature: float,
    include_examples: bool,
    max_examples: int,
    output_format: str,
    save_prompt: Optional[str]
):
    """Score a model output using a rubric."""
    
    async def async_score():
        try:
            # Load rubric
            rubric = Rubric.from_file(rubric_file)
            click.echo(f"Loaded rubric: {rubric.name} v{rubric.version}")
            
            # Create scorer
            if provider == 'openai':
                api_key_final = api_key or os.getenv('OPENAI_API_KEY')
                if not api_key_final:
                    raise click.ClickException("OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
                
                scorer = create_openai_scorer(
                    rubric=rubric,
                    api_key=api_key_final,
                    model=model or "gpt-4",
                    include_examples=include_examples,
                    max_examples_per_criterion=max_examples
                )
            else:  # anthropic
                api_key_final = api_key or os.getenv('ANTHROPIC_API_KEY')
                if not api_key_final:
                    raise click.ClickException("Anthropic API key required. Set ANTHROPIC_API_KEY env var or use --api-key")
                
                scorer = create_anthropic_scorer(
                    rubric=rubric,
                    api_key=api_key_final,
                    model=model or "claude-3-sonnet-20240229",
                    include_examples=include_examples,
                    max_examples_per_criterion=max_examples
                )
            
            # Save prompt if requested
            if save_prompt:
                prompt = scorer.prompt_builder.build_scoring_prompt(
                    task_input=task_input,
                    model_output=model_output,
                    include_examples=include_examples,
                    max_examples_per_criterion=max_examples
                )
                with open(save_prompt, 'w') as f:
                    f.write(prompt)
                click.echo(f"Prompt saved to: {save_prompt}")
            
            # Score the output
            with click.progressbar(length=1, label='Scoring') as bar:
                result = await scorer.score(
                    task_input=task_input,
                    model_output=model_output,
                    temperature=temperature
                )
                bar.update(1)
            
            # Display results
            if output_format == 'json':
                click.echo(json.dumps({
                    'overall_score': result.overall_score,
                    'overall_explanation': result.overall_explanation,
                    'criterion_scores': result.criterion_scores,
                    'criterion_explanations': result.criterion_explanations
                }, indent=2, ensure_ascii=False))
            else:  # table
                click.echo(f"\n📊 Overall Score: {result.overall_score:.2f}/{rubric.scale.max}")
                click.echo(f"📝 Explanation: {result.overall_explanation}\n")
                
                click.echo("📋 Criterion Breakdown:")
                for criterion in rubric.criteria:
                    name = criterion.name
                    score = result.criterion_scores.get(name, 0)
                    explanation = result.criterion_explanations.get(name, "No explanation")
                    click.echo(f"  • {name.title()}: {score:.2f} (weight: {criterion.weight:.1%})")
                    click.echo(f"    {explanation}\n")
                    
        except Exception as e:
            click.echo(f"❌ Scoring failed: {e}", err=True)
            exit(1)
    
    asyncio.run(async_score())


@main.command()
@click.argument('rubric_file', type=click.Path(exists=True))
@click.argument('inputs_file', type=click.Path(exists=True))
@click.option('--provider', type=click.Choice(['openai', 'anthropic']), default='openai', help='LLM provider')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
@click.option('--output', type=click.Path(), help='Output file for results')
@click.option('--concurrent', type=int, default=5, help='Number of concurrent requests')
def batch_score(
    rubric_file: str,
    inputs_file: str,
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
    output: Optional[str],
    concurrent: int
):
    """Score multiple inputs from a JSON/YAML file."""
    
    async def async_batch_score():
        try:
            # Load rubric
            rubric = Rubric.from_file(rubric_file)
            click.echo(f"Loaded rubric: {rubric.name} v{rubric.version}")
            
            # Load inputs
            with open(inputs_file, 'r') as f:
                if inputs_file.endswith(('.yaml', '.yml')):
                    inputs = yaml.safe_load(f)
                else:
                    inputs = json.load(f)
            
            if not isinstance(inputs, list):
                raise click.ClickException("Input file must contain a list of {task_input, model_output} objects")
            
            click.echo(f"Loaded {len(inputs)} items to score")
            
            # Create scorer
            if provider == 'openai':
                api_key_final = api_key or os.getenv('OPENAI_API_KEY')
                if not api_key_final:
                    raise click.ClickException("OpenAI API key required")
                scorer = create_openai_scorer(rubric=rubric, api_key=api_key_final, model=model or "gpt-4")
            else:
                api_key_final = api_key or os.getenv('ANTHROPIC_API_KEY')
                if not api_key_final:
                    raise click.ClickException("Anthropic API key required")
                scorer = create_anthropic_scorer(rubric=rubric, api_key=api_key_final, model=model or "claude-3-sonnet-20240229")
            
            # Score in batches
            results = []
            
            with click.progressbar(inputs, label='Scoring') as items:
                # Process in chunks to respect rate limits
                for i in range(0, len(inputs), concurrent):
                    chunk = inputs[i:i+concurrent]
                    chunk_results = await scorer.score_batch(chunk)
                    results.extend(chunk_results)
                    
                    # Update progress bar
                    for _ in chunk:
                        items.update(1)
            
            # Prepare output
            output_data = {
                'rubric': {
                    'name': rubric.name,
                    'version': rubric.version
                },
                'summary': {
                    'total_items': len(results),
                    'average_score': sum(r.overall_score for r in results) / len(results),
                    'min_score': min(r.overall_score for r in results),
                    'max_score': max(r.overall_score for r in results),
                },
                'results': [
                    {
                        'overall_score': r.overall_score,
                        'overall_explanation': r.overall_explanation,
                        'criterion_scores': r.criterion_scores,
                        'criterion_explanations': r.criterion_explanations
                    }
                    for r in results
                ]
            }
            
            # Save or display results
            if output:
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                click.echo(f"Results saved to: {output}")
            else:
                click.echo(json.dumps(output_data, indent=2, ensure_ascii=False))
                
        except Exception as e:
            click.echo(f"❌ Batch scoring failed: {e}", err=True)
            exit(1)
    
    asyncio.run(async_batch_score())


@main.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--rubrics-dir', type=click.Path(exists=True), help='Directory to auto-load rubrics from')
def serve(host: str, port: int, reload: bool, rubrics_dir: Optional[str]):
    """Start the OpenRubricRL API server."""
    try:
        import uvicorn
        from .api.server import app, rubric_manager
        
        # Auto-load rubrics if directory specified
        if rubrics_dir:
            rubrics_path = Path(rubrics_dir)
            click.echo(f"Loading rubrics from: {rubrics_path}")
            
            for file_path in rubrics_path.glob("*.json"):
                try:
                    rubric_manager.load_rubric(str(file_path))
                    click.echo(f"  ✅ Loaded: {file_path.name}")
                except Exception as e:
                    click.echo(f"  ❌ Failed to load {file_path.name}: {e}")
            
            for file_path in rubrics_path.glob("*.yaml"):
                try:
                    rubric_manager.load_rubric(str(file_path))
                    click.echo(f"  ✅ Loaded: {file_path.name}")
                except Exception as e:
                    click.echo(f"  ❌ Failed to load {file_path.name}: {e}")
        
        click.echo(f"🚀 Starting server at http://{host}:{port}")
        click.echo(f"📚 API docs available at http://{host}:{port}/docs")
        
        uvicorn.run(app, host=host, port=port, reload=reload)
        
    except ImportError:
        click.echo("❌ uvicorn not installed. Install with: pip install openrubricrl[dev]", err=True)
        exit(1)


@main.command()
@click.argument('name')
@click.option('--domain', type=click.Choice(['code', 'dialogue', 'creative_writing', 'reasoning', 'general']), default='general')
@click.option('--scale-min', type=float, default=0.0)
@click.option('--scale-max', type=float, default=10.0)
@click.option('--output', type=click.Path(), help='Output file (default: {name}.json)')
def create_template(name: str, domain: str, scale_min: float, scale_max: float, output: Optional[str]):
    """Create a new rubric template."""
    
    template = {
        "name": name,
        "version": "1.0.0",
        "description": f"Template rubric for {domain} evaluation",
        "domain": domain,
        "scale": {
            "min": scale_min,
            "max": scale_max,
            "type": "continuous"
        },
        "criteria": [
            {
                "name": "quality",
                "description": "Overall quality of the output",
                "weight": 0.5,
                "examples": {
                    "excellent": [],
                    "good": [],
                    "poor": []
                }
            },
            {
                "name": "accuracy",
                "description": "Accuracy and correctness of the output",
                "weight": 0.3
            },
            {
                "name": "clarity",
                "description": "Clarity and understandability of the output",
                "weight": 0.2
            }
        ],
        "metadata": {
            "author": "OpenRubricRL User",
            "created_at": "2024-01-01T00:00:00Z",
            "tags": [domain],
            "license": "MIT"
        }
    }
    
    output_file = output or f"{name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    click.echo(f"✅ Template rubric created: {output_file}")
    click.echo("Edit the file to customize criteria, weights, and examples.")


if __name__ == '__main__':
    main()