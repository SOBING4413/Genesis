from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel, Field
from app.hypothesis_engine.engine import hypothesis_engine

router = APIRouter()


class HypothesisGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500, description="Research topic")
    domains: List[str] = Field(
        default=["biology", "medicine", "chemistry", "computer_science"],
        description="Scientific domains to explore",
    )
    depth: str = Field(default="medium", pattern="^(shallow|medium|deep)$")
    creativity: float = Field(default=0.7, ge=0.0, le=1.0)
    strictness: float = Field(default=0.8, ge=0.0, le=1.0)


class HypothesisStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(generated|validated|accepted|rejected)$")


@router.post("/generate")
async def generate_hypothesis(request: HypothesisGenerateRequest):
    """Generate novel scientific hypotheses for a research topic."""
    result = await hypothesis_engine.generate(
        topic=request.topic,
        domains=request.domains,
        depth=request.depth,
        creativity=request.creativity,
        strictness=request.strictness,
    )
    return result


@router.get("/")
async def list_hypotheses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, pattern="^(generated|validated|accepted|rejected)$"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
):
    """List all hypotheses with optional filtering."""
    return await hypothesis_engine.list_hypotheses(
        limit=limit,
        offset=offset,
        status=status,
        min_confidence=min_confidence,
    )


@router.get("/{hypothesis_id}")
async def get_hypothesis(hypothesis_id: str):
    """Get a single hypothesis by ID."""
    result = await hypothesis_engine.get_hypothesis(hypothesis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return result


@router.patch("/{hypothesis_id}/status")
async def update_hypothesis_status(hypothesis_id: str, body: HypothesisStatusUpdate):
    """Update the status of a hypothesis (accept/reject/validate)."""
    try:
        result = await hypothesis_engine.update_hypothesis_status(hypothesis_id, body.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not result:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return result
