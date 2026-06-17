from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
import asyncio
import time
import uuid

from app.agents.base_agent import BaseAgent
from app.agents.researcher_agent import ResearcherAgent
from app.agents.inventor_agent import InventorAgent
from app.agents.critic_agent import CriticAgent
from app.agents.evaluator_agent import EvaluatorAgent
from app.database.session import async_session
from app.database.models import Hypothesis


class CoordinatorAgent(BaseAgent):
    """Agent that orchestrates the entire research workflow."""

    def __init__(self):
        super().__init__(agent_type="coordinator")
        self.researcher = ResearcherAgent()
        self.inventor = InventorAgent()
        self.critic = CriticAgent()
        self.evaluator = EvaluatorAgent()

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full research pipeline: Research -> Invent -> Critique -> Evaluate.
        """
        topic = input_data.get("topic", "")
        domains = input_data.get("domains", ["biology", "medicine"])
        depth = input_data.get("depth", "medium")
        creativity = input_data.get("creativity", 0.7)
        strictness = input_data.get("strictness", 0.8)
        session_id = input_data.get("session_id", f"session_{uuid.uuid4().hex[:8]}")

        if not topic:
            return {"error": "Topic is required", "pipeline_complete": False}

        logger.info(f"CoordinatorAgent starting research pipeline on: {topic}")
        start_time = time.time()
        pipeline_errors = []

        # Phase 1: Research
        logger.info("Phase 1: Researching...")
        try:
            research_result = await self.researcher.run({
                "topic": topic,
                "constraints": [],
                "depth": depth,
                "session_id": session_id,
            })
        except Exception as e:
            logger.error(f"Research phase failed: {e}")
            research_result = {"evidence_count": 0, "related_concepts": [], "kg_connections": 0}
            pipeline_errors.append(f"research: {str(e)}")

        # Phase 2: Invent
        logger.info("Phase 2: Generating hypotheses...")
        try:
            invention_result = await self.inventor.run({
                "research_findings": research_result,
                "domains": domains,
                "creativity_level": creativity,
                "session_id": session_id,
            })
        except Exception as e:
            logger.error(f"Invention phase failed: {e}")
            invention_result = {"hypotheses": [], "cross_domain_connections_found": 0}
            pipeline_errors.append(f"invention: {str(e)}")

        hypotheses = invention_result.get("hypotheses", [])

        if not hypotheses:
            logger.warning("No hypotheses generated, returning early")
            return {
                "topic": topic,
                "session_id": session_id,
                "pipeline_complete": False,
                "total_hypotheses_generated": 0,
                "top_hypotheses": [],
                "errors": pipeline_errors,
                "pipeline_duration_ms": int((time.time() - start_time) * 1000),
            }

        # Phase 3: Critique
        logger.info(f"Phase 3: Critiquing {len(hypotheses)} hypotheses...")
        try:
            critique_result = await self.critic.run({
                "hypotheses": hypotheses,
                "criteria": [
                    "logical_consistency",
                    "empirical_support",
                    "falsifiability",
                    "novelty",
                ],
                "strictness": strictness,
                "session_id": session_id,
            })
        except Exception as e:
            logger.error(f"Critique phase failed: {e}")
            critique_result = {"critiques": []}
            pipeline_errors.append(f"critique: {str(e)}")

        # Phase 4: Evaluate
        logger.info("Phase 4: Evaluating and scoring...")
        combined = []
        critique_map = {
            c.get("hypothesis_title"): c
            for c in critique_result.get("critiques", [])
        }
        for h in hypotheses:
            combined.append({
                "hypothesis": h,
                "critiques": critique_map.get(h.get("title"), {}),
            })

        try:
            evaluation_result = await self.evaluator.run({
                "hypotheses_with_critiques": combined,
                "weighting_scheme": {
                    "logical_consistency": 0.25,
                    "empirical_support": 0.30,
                    "novelty": 0.20,
                    "falsifiability": 0.15,
                    "parsimony": 0.10,
                },
                "session_id": session_id,
            })
        except Exception as e:
            logger.error(f"Evaluation phase failed: {e}")
            evaluation_result = {"ranked_hypotheses": []}
            pipeline_errors.append(f"evaluation: {str(e)}")

        # Save top hypotheses to database
        await self._save_hypotheses(topic, evaluation_result, session_id)

        total_time = int((time.time() - start_time) * 1000)

        output = {
            "topic": topic,
            "session_id": session_id,
            "pipeline_complete": len(pipeline_errors) == 0,
            "total_hypotheses_generated": len(hypotheses),
            "top_hypotheses": evaluation_result.get("ranked_hypotheses", [])[:5],
            "research_summary": {
                "evidence_found": research_result.get("evidence_count", 0),
                "concepts_discovered": len(research_result.get("related_concepts", [])),
                "cross_domain_connections": invention_result.get("cross_domain_connections_found", 0),
            },
            "average_hypothesis_score": evaluation_result.get("average_score", 0),
            "pipeline_duration_ms": total_time,
            "errors": pipeline_errors if pipeline_errors else None,
        }

        await self.log_execution(session_id, input_data, output, duration_ms=total_time)
        return output

    async def _save_hypotheses(self, topic: str, evaluation: dict, session_id: str):
        """Save evaluated hypotheses to database."""
        ranked = evaluation.get("ranked_hypotheses", [])
        if not ranked:
            return
        try:
            async with async_session() as session:
                for h in ranked:
                    hypothesis = Hypothesis(
                        title=h.get("hypothesis_title", f"Hypothesis on {topic}"),
                        description=h.get("description", ""),
                        hypothesis_type="cross_domain",
                        confidence_score=float(h.get("final_score", 0.5)),
                        evidence_count=0,
                        reasoning_chain=h.get("recommended_next_steps", []),
                        agent_id=session_id,
                        status="generated",
                    )
                    session.add(hypothesis)
                await session.commit()
                logger.info(f"Saved {len(ranked)} hypotheses for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save hypotheses: {e}")
