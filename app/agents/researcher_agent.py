from typing import Dict, Any, List
from loguru import logger
from app.agents.base_agent import BaseAgent
from app.agents.llm_service import llm_service
from app.knowledge_graph.neo4j_client import neo4j_client
from app.database.session import async_session
from app.database.models import Paper
from sqlalchemy import select


class ResearcherAgent(BaseAgent):
    """Agent that searches for new information and gathers evidence."""

    def __init__(self):
        super().__init__(agent_type="researcher")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Research a topic by gathering evidence from knowledge graph and papers.

        Args:
            input_data: {
                "topic": str,
                "constraints": List[str],
                "depth": str (shallow/medium/deep),
                "session_id": str
            }
        """
        topic = input_data.get("topic", "")
        constraints = input_data.get("constraints", [])
        depth = input_data.get("depth", "medium")
        session_id = input_data.get("session_id", "")

        logger.info(f"ResearcherAgent researching: {topic}")

        # 1. Search knowledge graph for related concepts
        kg_results = await neo4j_client.search_concepts(topic, limit=20)

        # 2. Get top related concepts from KG
        related_concepts = [r["name"] for r in kg_results if r.get("score", 0) > 0.5]

        # 3. Fetch supporting papers from database
        async with async_session() as session:
            query = select(Paper).where(
                Paper.abstract.ilike(f"%{topic}%"),
                Paper.status == "completed",
            ).limit(10)
            result = await session.execute(query)
            papers = result.scalars().all()

        paper_evidence = [
            {
                "id": str(p.id),
                "title": p.title,
                "abstract": p.abstract[:300],
                "source": p.source,
                "source_id": p.source_id,
                "relevance": 0.8,
            }
            for p in papers
        ]

        # 4. Use LLM to synthesize findings
        prompt = f"""You are a scientific researcher. Analyze the following information about "{topic}".

Related concepts found: {related_concepts[:10]}
Supporting papers found: {len(paper_evidence)}

Constraints: {constraints}
Research depth: {depth}

Provide a structured research summary including:
1. Key findings and known relationships
2. Gaps in current knowledge
3. Promising research directions
4. Confidence assessment for each finding

Return your analysis in JSON format."""

        llm_response = await llm_service.chat_json([
            {"role": "system", "content": "You are a thorough scientific researcher."},
            {"role": "user", "content": prompt},
        ])

        output = {
            "topic": topic,
            "related_concepts": related_concepts,
            "evidence_count": len(paper_evidence),
            "supporting_papers": paper_evidence,
            "kg_connections": len(kg_results),
            "analysis": llm_response.get("analysis", llm_response),
            "research_gaps": llm_response.get("research_gaps", []),
            "confidence": llm_response.get("confidence", 0.5),
        }

        await self.log_execution(session_id, input_data, output)
        return output