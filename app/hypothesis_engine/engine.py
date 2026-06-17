from typing import Dict, Any, Optional, List
from loguru import logger
import uuid
import time

from app.agents.coordinator_agent import CoordinatorAgent
from app.database.session import async_session
from app.database.models import Hypothesis
from sqlalchemy import select, func


class HypothesisEngine:
    """Main engine for generating and managing scientific hypotheses."""

    def __init__(self):
        self.coordinator = CoordinatorAgent()

    async def generate(
        self,
        topic: str,
        domains: List[str] = None,
        depth: str = "medium",
        creativity: float = 0.7,
        strictness: float = 0.8,
    ) -> Dict[str, Any]:
        """Run full hypothesis generation pipeline."""
        if domains is None:
            domains = ["biology", "medicine", "chemistry", "computer_science"]

        # FIX: Use UUID for session_id to avoid collision on short/similar topics
        session_id = f"hyp_{uuid.uuid4().hex[:12]}_{int(time.time())}"

        logger.info(f"HypothesisEngine generating for topic: {topic} [session={session_id}]")

        result = await self.coordinator.run({
            "topic": topic,
            "domains": domains,
            "depth": depth,
            "creativity": creativity,
            "strictness": strictness,
            "session_id": session_id,
        })

        return result

    async def get_hypothesis(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """Get a hypothesis by ID."""
        try:
            uid = uuid.UUID(hypothesis_id)
        except ValueError:
            return None

        async with async_session() as session:
            result = await session.execute(
                select(Hypothesis).where(Hypothesis.id == uid)
            )
            h = result.scalar_one_or_none()
            if not h:
                return None

            return {
                "id": str(h.id),
                "title": h.title,
                "description": h.description,
                "type": h.hypothesis_type,
                "confidence": h.confidence_score,
                "evidence_count": h.evidence_count,
                "reasoning_chain": h.reasoning_chain,
                "supporting_papers": h.supporting_papers,
                "source_concepts": h.source_concepts,
                "target_concepts": h.target_concepts,
                "status": h.status,
                "agent_id": h.agent_id,
                "created_at": h.created_at.isoformat() if h.created_at else None,
                "updated_at": h.updated_at.isoformat() if h.updated_at else None,
            }

    async def list_hypotheses(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """List hypotheses with optional filters and total count."""
        async with async_session() as session:
            base_query = select(Hypothesis)
            if status:
                base_query = base_query.where(Hypothesis.status == status)
            if min_confidence > 0:
                base_query = base_query.where(Hypothesis.confidence_score >= min_confidence)

            # Count total
            count_result = await session.execute(
                select(func.count()).select_from(base_query.subquery())
            )
            total = count_result.scalar() or 0

            # Fetch page
            result = await session.execute(
                base_query
                .order_by(Hypothesis.confidence_score.desc(), Hypothesis.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            hypotheses = result.scalars().all()

            return {
                "total": total,
                "offset": offset,
                "limit": limit,
                "hypotheses": [
                    {
                        "id": str(h.id),
                        "title": h.title,
                        "type": h.hypothesis_type,
                        "confidence": h.confidence_score,
                        "status": h.status,
                        "created_at": h.created_at.isoformat() if h.created_at else None,
                    }
                    for h in hypotheses
                ],
            }

    async def update_hypothesis_status(
        self, hypothesis_id: str, status: str
    ) -> Optional[Dict[str, Any]]:
        """Update hypothesis status (accept/reject/validate)."""
        valid_statuses = {"generated", "validated", "accepted", "rejected"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of {valid_statuses}")

        try:
            uid = uuid.UUID(hypothesis_id)
        except ValueError:
            return None

        async with async_session() as session:
            result = await session.execute(select(Hypothesis).where(Hypothesis.id == uid))
            h = result.scalar_one_or_none()
            if not h:
                return None
            h.status = status
            await session.commit()
            await session.refresh(h)
            return {"id": str(h.id), "status": h.status}


hypothesis_engine = HypothesisEngine()
