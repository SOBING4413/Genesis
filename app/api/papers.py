from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

from app.database.session import get_db
from app.database.models import Paper, PaperStatus
from app.ingestion.literature_miner import LiteratureMiner
from app.tasks.paper_tasks import process_paper_task
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
literature_miner = LiteratureMiner()


class PaperResponse(BaseModel):
    id: str
    title: str
    authors: List[str]
    abstract: str
    source: str
    source_id: Optional[str] = None
    url: Optional[str] = None
    published_date: Optional[str] = None
    keywords: List[str]
    status: str
    citation_count: int
    created_at: str

    model_config = {"from_attributes": True}


class PaperImportRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    sources: List[str] = Field(default=["arxiv", "pubmed"])
    max_per_source: int = Field(default=10, ge=1, le=50)


@router.get("/", response_model=List[PaperResponse])
async def list_papers(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    source: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List papers with optional source and status filters."""
    query = select(Paper)
    if source:
        query = query.where(Paper.source == source)
    if status:
        try:
            query = query.where(Paper.status == PaperStatus(status))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status '{status}'")
    query = query.order_by(Paper.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    papers = result.scalars().all()

    return [
        PaperResponse(
            id=str(p.id),
            title=p.title,
            authors=p.authors or [],
            abstract=p.abstract or "",
            source=p.source,
            source_id=p.source_id,
            url=p.url,
            published_date=p.published_date.isoformat() if p.published_date else None,
            keywords=p.keywords or [],
            status=p.status.value if p.status else "pending",
            citation_count=p.citation_count or 0,
            created_at=p.created_at.isoformat() if p.created_at else "",
        )
        for p in papers
    ]


@router.get("/count")
async def count_papers(
    source: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get total paper count with optional filters."""
    query = select(func.count(Paper.id))
    if source:
        query = query.where(Paper.source == source)
    if status:
        try:
            query = query.where(Paper.status == PaperStatus(status))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status '{status}'")
    result = await db.execute(query)
    return {"count": result.scalar() or 0}


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single paper by ID."""
    try:
        uid = UUID(paper_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid paper ID format")

    result = await db.execute(select(Paper).where(Paper.id == uid))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    return PaperResponse(
        id=str(p.id),
        title=p.title,
        authors=p.authors or [],
        abstract=p.abstract or "",
        source=p.source,
        source_id=p.source_id,
        url=p.url,
        published_date=p.published_date.isoformat() if p.published_date else None,
        keywords=p.keywords or [],
        status=p.status.value if p.status else "pending",
        citation_count=p.citation_count or 0,
        created_at=p.created_at.isoformat() if p.created_at else "",
    )


@router.post("/import")
async def import_papers(
    request: PaperImportRequest,
    background_tasks: BackgroundTasks,
):
    """Import papers from scientific sources, then queue processing via Celery."""
    imported_ids = await literature_miner.search_and_import_with_ids(
        query=request.query,
        sources=request.sources,
        max_per_source=request.max_per_source,
    )
    # Queue background processing for each imported paper
    for paper_id in imported_ids:
        background_tasks.add_task(_enqueue_paper_processing, paper_id)

    return {
        "imported": len(imported_ids),
        "queued_for_processing": len(imported_ids),
        "query": request.query,
        "sources": request.sources,
    }


async def _enqueue_paper_processing(paper_id: str):
    """Enqueue paper for async processing via Celery."""
    try:
        process_paper_task.delay(paper_id)
    except Exception as e:
        # Celery may be unavailable (e.g. no Redis); process inline as fallback
        from loguru import logger
        logger.warning(f"Celery unavailable, processing inline: {e}")
        await literature_miner.process_paper(paper_id)
