from typing import Dict, Any, List
from loguru import logger
from app.agents.base_agent import BaseAgent
from app.agents.llm_service import llm_service
from app.knowledge_graph.neo4j_client import neo4j_client


class InventorAgent(BaseAgent):
    """Agent that generates novel ideas and theories by connecting disparate concepts."""

    def __init__(self):
        super().__init__(agent_type="inventor")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate novel hypotheses by connecting concepts.

        Args:
            input_data: {
                "research_findings": Dict,
                "domains": List[str],
                "creativity_level": float (0.0-1.0),
                "session_id": str
            }
        """
        findings = input_data.get("research_findings", {})
        domains = input_data.get("domains", ["biology", "medicine", "chemistry"])
        creativity = input_data.get("creativity_level", 0.7)
        session_id = input_data.get("session_id", "")

        logger.info(f"InventorAgent generating hypotheses across domains: {domains}")

        # 1. Get high-impact concepts from KG
        top_concepts = await neo4j_client.get_highest_impact_concepts(limit=30)

        # 2. Find cross-domain connections
        cross_domain_connections = []
        for i, c1 in enumerate(top_concepts):
            for j, c2 in enumerate(top_concepts):
                if i >= j:
                    continue
                if c1.get("type") != c2.get("type"):
                    paths = await neo4j_client.find_path_between_concepts(
                        c1["name"], c2["name"], max_depth=3
                    )
                    if paths:
                        cross_domain_connections.append({
                            "concept_a": c1["name"],
                            "concept_b": c2["name"],
                            "type_a": c1.get("type"),
                            "type_b": c2.get("type"),
                            "path_length": len(paths),
                        })

        # 3. Use LLM to generate hypotheses
        prompt = f"""You are a highly creative scientific inventor. Your task is to generate NOVEL research hypotheses.

Cross-domain concept connections found: {cross_domain_connections[:15]}
Research findings: {str(findings)[:1000]}
Domains to explore: {domains}
Creativity level: {creativity}

Generate 3-5 novel hypotheses that:
1. Connect concepts across different scientific domains
2. Are grounded in existing evidence but propose NEW relationships
3. Are specific and testable
4. Include a confidence estimate based on supporting evidence

For each hypothesis provide:
- Title
- Description
- Source concepts involved
- Target concepts involved
- Reasoning chain (step by step)
- Initial confidence score (0-1)
- Type (cross_domain, novel_connection, analogical, or emergent)

Return as JSON array."""

        llm_response = await llm_service.chat_json([
            {"role": "system", "content": "You are a brilliant scientific inventor who sees connections others miss."},
            {"role": "user", "content": prompt},
        ])

        hypotheses = llm_response.get("hypotheses", [])
        if isinstance(hypotheses, dict):
            hypotheses = [hypotheses]

        output = {
            "hypotheses_generated": len(hypotheses),
            "hypotheses": hypotheses,
            "cross_domain_connections_found": len(cross_domain_connections),
            "creativity_level": creativity,
        }

        await self.log_execution(session_id, input_data, output)
        return output