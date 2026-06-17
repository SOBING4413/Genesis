from typing import Dict, Any, List
from loguru import logger
from app.agents.base_agent import BaseAgent
from app.agents.llm_service import llm_service


class CriticAgent(BaseAgent):
    """Agent that tests hypotheses for weaknesses, flaws, and gaps."""

    def __init__(self):
        super().__init__(agent_type="critic")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critically evaluate hypotheses for weaknesses.

        Args:
            input_data: {
                "hypotheses": List[Dict],
                "criteria": List[str],
                "strictness": float (0.0-1.0),
                "session_id": str
            }
        """
        hypotheses = input_data.get("hypotheses", [])
        criteria = input_data.get("criteria", [
            "logical_consistency",
            "empirical_support",
            "falsifiability",
            "parsimony",
            "novelty",
        ])
        strictness = input_data.get("strictness", 0.8)
        session_id = input_data.get("session_id", "")

        logger.info(f"CriticAgent evaluating {len(hypotheses)} hypotheses")

        critiques = []

        for hypothesis in hypotheses:
            title = hypothesis.get("title", "Unknown")
            desc = hypothesis.get("description", "")
            reasoning = hypothesis.get("reasoning_chain", [])

            prompt = f"""You are a rigorous scientific critic. Evaluate this hypothesis critically.

HYPOTHESIS: {title}
DESCRIPTION: {desc}
REASONING: {reasoning}

Evaluation criteria: {criteria}
Strictness level: {strictness}/1.0

For each criterion in the list, provide a score (0.0-1.0), a specific critique, potential counterarguments, and missing evidence.

Also identify logical fallacies, alternative explanations, and methodological concerns.

Return ONLY valid JSON with this structure:
{{
  "scores": {{"criterion_name": 0.0}},
  "critiques": ["critique text"],
  "logical_fallacies": ["fallacy"],
  "missing_evidence": ["evidence needed"],
  "alternative_explanations": ["alternative"],
  "overall_assessment": "text",
  "recommendation": "accept|revise|reject"
}}"""

            llm_response = await llm_service.chat_json([
                {"role": "system", "content": "You are a relentless but fair scientific critic. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ])

            critiques.append({
                "hypothesis_title": title,
                "scores": llm_response.get("scores", {}),
                "critiques": llm_response.get("critiques", []),
                "logical_fallacies": llm_response.get("logical_fallacies", []),
                "missing_evidence": llm_response.get("missing_evidence", []),
                "alternative_explanations": llm_response.get("alternative_explanations", []),
                "overall_assessment": llm_response.get("overall_assessment", ""),
                "recommendation": llm_response.get("recommendation", "revise"),
            })

        output = {
            "hypotheses_evaluated": len(critiques),
            "critiques": critiques,
            "average_scores": self._compute_average_scores(critiques),
            "recommendations_summary": self._summarize_recommendations(critiques),
        }

        await self.log_execution(session_id, input_data, output)
        return output

    def _compute_average_scores(self, critiques: List[Dict]) -> Dict[str, float]:
        """
        FIX: Compute average per criterion correctly.
        Original bug: divided by len(critiques) for each criterion but counted
        across all keys, leading to wrong averages.
        """
        per_criterion: Dict[str, List[float]] = {}
        for c in critiques:
            for key, val in c.get("scores", {}).items():
                if isinstance(val, (int, float)):
                    per_criterion.setdefault(key, []).append(float(val))
        return {k: round(sum(v) / len(v), 3) for k, v in per_criterion.items()}

    def _summarize_recommendations(self, critiques: List[Dict]) -> Dict[str, int]:
        recs = {"accept": 0, "revise": 0, "reject": 0}
        for c in critiques:
            rec = c.get("recommendation", "revise").lower()
            if rec in recs:
                recs[rec] += 1
            else:
                recs["revise"] += 1  # Default unknown to revise
        return recs
