from fastapi import APIRouter, Query
from typing import Optional
from app.knowledge_graph.builder import kg_builder
from app.database.session import async_session
from app.database.models import Paper, Hypothesis, ResearchAgentLog, PaperStatus
from sqlalchemy import select, func, and_
from datetime import datetime, timedelta

router = APIRouter()


@router.get("/")
async def get_analytics():
    """Full analytics dashboard data."""
    async with async_session() as session:
        # Paper counts
        total_papers = (await session.execute(select(func.count(Paper.id)))).scalar() or 0

        papers_by_source = dict(
            (await session.execute(
                select(Paper.source, func.count(Paper.id)).group_by(Paper.source)
            )).all()
        )

        papers_by_status = dict(
            (await session.execute(
                select(Paper.status, func.count(Paper.id)).group_by(Paper.status)
            )).all()
        )
        # Convert enum keys to string
        papers_by_status = {
            (k.value if hasattr(k, "value") else str(k)): v
            for k, v in papers_by_status.items()
        }

        # Hypothesis stats
        total_hypotheses = (
            await session.execute(select(func.count(Hypothesis.id)))
        ).scalar() or 0

        avg_confidence = (
            await session.execute(select(func.avg(Hypothesis.confidence_score)))
        ).scalar() or 0.0

        hyp_by_status = dict(
            (await session.execute(
                select(Hypothesis.status, func.count(Hypothesis.id)).group_by(Hypothesis.status)
            )).all()
        )

        hyp_by_type = dict(
            (await session.execute(
                select(Hypothesis.hypothesis_type, func.count(Hypothesis.id))
                .group_by(Hypothesis.hypothesis_type)
            )).all()
        )

        # Recent activity (last 7 days)
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent_papers = (
            await session.execute(
                select(func.count(Paper.id)).where(Paper.created_at >= cutoff)
            )
        ).scalar() or 0
        recent_hypotheses = (
            await session.execute(
                select(func.count(Hypothesis.id)).where(Hypothesis.created_at >= cutoff)
            )
        ).scalar() or 0

        # Agent activity
        agent_activity = dict(
            (await session.execute(
                select(ResearchAgentLog.agent_type, func.count(ResearchAgentLog.id))
                .group_by(ResearchAgentLog.agent_type)
            )).all()
        )
        avg_duration = (
            await session.execute(select(func.avg(ResearchAgentLog.duration_ms)))
        ).scalar() or 0.0

    # Graph stats
    try:
        graph_stats = await kg_builder.get_graph_statistics()
    except Exception:
        graph_stats = {"error": "Knowledge graph unavailable"}

    return {
        "papers": {
            "total": total_papers,
            "by_source": papers_by_source,
            "by_status": papers_by_status,
            "recent_7d": recent_papers,
        },
        "hypotheses": {
            "total": total_hypotheses,
            "average_confidence": round(float(avg_confidence), 3),
            "by_status": hyp_by_status,
            "by_type": hyp_by_type,
            "recent_7d": recent_hypotheses,
        },
        "agents": {
            "executions_by_type": agent_activity,
            "average_duration_ms": round(float(avg_duration), 1),
        },
        "knowledge_graph": graph_stats,
    }


@router.get("/timeline")
async def get_timeline(days: int = Query(30, ge=1, le=365)):
    """Get paper and hypothesis ingestion timeline."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    async with async_session() as session:
        paper_timeline = (
            await session.execute(
                select(
                    func.date_trunc("day", Paper.created_at).label("day"),
                    func.count(Paper.id).label("count"),
                )
                .where(Paper.created_at >= cutoff)
                .group_by("day")
                .order_by("day")
            )
        ).all()

        hyp_timeline = (
            await session.execute(
                select(
                    func.date_trunc("day", Hypothesis.created_at).label("day"),
                    func.count(Hypothesis.id).label("count"),
                )
                .where(Hypothesis.created_at >= cutoff)
                .group_by("day")
                .order_by("day")
            )
        ).all()

    return {
        "days": days,
        "papers": [{"day": str(r.day)[:10], "count": r.count} for r in paper_timeline],
        "hypotheses": [{"day": str(r.day)[:10], "count": r.count} for r in hyp_timeline],
    }
