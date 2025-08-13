"""FastAPI server for OpenRubricRL scoring API."""

import os
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.rubric import Rubric
from ..core.scorer import RubricScorer, OpenAIProvider, AnthropicProvider
from .models import (
    ScoreRequest, 
    BatchScoreRequest, 
    ScoreResponse, 
    BatchScoreResponse,
    RubricSummary,
    ErrorResponse
)


class RubricManager:
    """Manages loaded rubrics and scorers."""
    
    def __init__(self):
        self.rubrics: Dict[str, Rubric] = {}
        self.scorers: Dict[str, RubricScorer] = {}
    
    def load_rubric(self, file_path: str, name: Optional[str] = None) -> str:
        """Load a rubric from file."""
        rubric = Rubric.from_file(file_path)
        rubric_name = name or rubric.name
        
        self.rubrics[rubric_name] = rubric
        
        # Create default scorer
        self.create_scorer(rubric_name)
        
        return rubric_name
    
    def create_scorer(
        self, 
        rubric_name: str, 
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        """Create a scorer for a rubric."""
        if rubric_name not in self.rubrics:
            raise ValueError(f"Rubric {rubric_name} not found")
        
        rubric = self.rubrics[rubric_name]
        
        if provider == "openai":
            from ..core.scorer import create_openai_scorer
            model = model or "gpt-4"
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            scorer = create_openai_scorer(rubric, api_key=api_key, model=model)
        elif provider == "anthropic":
            from ..core.scorer import create_anthropic_scorer
            model = model or "claude-3-sonnet-20240229"
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            scorer = create_anthropic_scorer(rubric, api_key=api_key, model=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.scorers[rubric_name] = scorer
    
    def get_rubric(self, name: str) -> Rubric:
        """Get a rubric by name."""
        if name not in self.rubrics:
            raise ValueError(f"Rubric {name} not found")
        return self.rubrics[name]
    
    def get_scorer(self, name: str) -> RubricScorer:
        """Get a scorer by rubric name."""
        if name not in self.scorers:
            raise ValueError(f"Scorer for rubric {name} not found")
        return self.scorers[name]
    
    def list_rubrics(self) -> List[RubricSummary]:
        """List all loaded rubrics."""
        summaries = []
        for name, rubric in self.rubrics.items():
            summaries.append(RubricSummary(
                name=name,
                version=rubric.version,
                description=rubric.description,
                domain=rubric.domain,
                criteria_count=len(rubric.criteria),
                scale_min=rubric.scale.min,
                scale_max=rubric.scale.max
            ))
        return summaries


# Global rubric manager
rubric_manager = RubricManager()


def get_rubric_manager() -> RubricManager:
    """Dependency to get rubric manager."""
    return rubric_manager


def create_app(
    title: str = "OpenRubricRL API",
    description: str = "API for LLM-based rubric scoring",
    version: str = "0.1.0"
) -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc)
            ).dict()
        )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": version}
    
    @app.get("/rubrics", response_model=List[RubricSummary])
    async def list_rubrics(manager: RubricManager = Depends(get_rubric_manager)):
        """List all available rubrics."""
        return manager.list_rubrics()
    
    @app.get("/rubrics/{rubric_name}")
    async def get_rubric(
        rubric_name: str,
        manager: RubricManager = Depends(get_rubric_manager)
    ):
        """Get detailed information about a rubric."""
        try:
            rubric = manager.get_rubric(rubric_name)
            return rubric.to_dict()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @app.post("/score/{rubric_name}", response_model=ScoreResponse)
    async def score_with_rubric(
        rubric_name: str,
        request: ScoreRequest,
        manager: RubricManager = Depends(get_rubric_manager)
    ):
        """Score a model output using a specific rubric."""
        try:
            scorer = manager.get_scorer(rubric_name)
            rubric = manager.get_rubric(rubric_name)
            
            result = await scorer.score(
                task_input=request.task_input,
                model_output=request.model_output,
                **request.llm_config
            )
            
            return ScoreResponse(
                overall_score=result.overall_score,
                overall_explanation=result.overall_explanation,
                criterion_scores=result.criterion_scores,
                criterion_explanations=result.criterion_explanations,
                rubric_name=rubric.name,
                rubric_version=rubric.version
            )
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
    
    @app.post("/score", response_model=ScoreResponse)
    async def score_with_default_rubric(
        request: ScoreRequest,
        manager: RubricManager = Depends(get_rubric_manager)
    ):
        """Score using the default or specified rubric."""
        if not request.rubric_name:
            rubrics = manager.list_rubrics()
            if not rubrics:
                raise HTTPException(status_code=400, detail="No rubrics available and none specified")
            request.rubric_name = rubrics[0].name
        
        return await score_with_rubric(request.rubric_name, request, manager)
    
    @app.post("/batch-score/{rubric_name}", response_model=BatchScoreResponse)
    async def batch_score_with_rubric(
        rubric_name: str,
        request: BatchScoreRequest,
        manager: RubricManager = Depends(get_rubric_manager)
    ):
        """Score multiple outputs using a specific rubric."""
        try:
            scorer = manager.get_scorer(rubric_name)
            rubric = manager.get_rubric(rubric_name)
            
            results = await scorer.score_batch(
                inputs=request.items,
                **request.llm_config
            )
            
            responses = [
                ScoreResponse(
                    overall_score=result.overall_score,
                    overall_explanation=result.overall_explanation,
                    criterion_scores=result.criterion_scores,
                    criterion_explanations=result.criterion_explanations,
                    rubric_name=rubric.name,
                    rubric_version=rubric.version
                )
                for result in results
            ]
            
            # Calculate summary statistics
            scores = [r.overall_score for r in responses]
            summary = {
                "total_items": len(responses),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0
            }
            
            return BatchScoreResponse(results=responses, summary=summary)
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")
    
    @app.post("/load-rubric")
    async def load_rubric_endpoint(
        file_path: str,
        name: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        manager: RubricManager = Depends(get_rubric_manager)
    ):
        """Load a rubric from file."""
        try:
            if not Path(file_path).exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
            rubric_name = manager.load_rubric(file_path, name)
            manager.create_scorer(rubric_name, provider=provider, model=model)
            
            return {"message": f"Rubric {rubric_name} loaded successfully"}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load rubric: {str(e)}")
    
    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)