from typing import Dict, Any, List
from loguru import logger
from app.agents.base_agent import BaseAgent
from app.agents.llm_service import llm_service


class EvaluatorAgent(BaseAgent):
    """Agent that scores and validates hypotheses."""

    def __init__(self):
        super().__init__(agent_type="evaluator")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assign final validity scores to hypotheses.

        Args:
            input_data: {
                "hypotheses_with_critiques": List[Dict],
                "weighting_scheme": Dict,
                "session_id": str
            }
        """
        hypotheses = input_data.get("hypotheses_with_critiques", [])
        weights = input_data.get("weighting_scheme", {
            "logical_consistency": 0.25,
            "empirical_support": 0.30,
            "novelty": 0.20,
            "falsifiability": 0.15,
            "parsimony": 0.10,
        })
        session_id = input_data.get("session_id", "")

        logger.info(f"EvaluatorAgent scoring {len(hypotheses)} hypotheses")

        evaluated = []

        for item in hypotheses:
            hypothesis = item.get("hypothesis", item)
            critiques = item.get("critiques", item.get("critique", {}))

            scores = critiques.get("scores", {})
            weighted_score = sum(
                scores.get(criterion, 0) * weight
                for criterion, weight in weights.items()
                if criterion in scores
            )

            # Normalize
            total_weight = sum(
                weight for criterion, weight in weights.items()
                if criterion in scores
            )
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.5

            # Use LLM for final assessment
            prompt = f"""As a senior scientific evaluator, provide a final assessment.

HYPOTHESIS: {hypothesis.get('title', 'Unknown')}
WEIGHTED SCORE: {final_score:.3f}
CRITIQUE SCORES: {scores}
CRITIQUES: {critiques.get('critiques', [])}

Provide:
1. Final validity score (0-1)
2. Confidence in this score (0-1)
3. Recommended next steps
4. Priority level (low/medium/high/critical)

Return JSON."""

            llm_response = await llm_service.chat_json([
                {"role": "system", "content": "You are a senior scientific evaluator making final assessments."},
                {"role": "user", "content": prompt},
            ])

            evaluated.append({
                "hypothesis_title": hypothesis.get("title", "Unknown"),
                "final_score": llm_response.get("final_score", final_score),
                "confidence": llm_response.get("confidence", 0.5),
                "component_scores": scores,
                "weighted_score": final_score,
                "recommended_next_steps": llm_response.get("recommended_next_steps", []),
                "priority": llm_response.get("priority", "medium"),
            })

        # Rank by score
        evaluated.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        output = {
            "hypotheses_evaluated": len(evaluated),
            "ranked_hypotheses": evaluated,
            "top_hypothesis": evaluated[0] if evaluated else None,
            "average_score": (
                sum(e["final_score"] for e in evaluated) / len(evaluated)
                if evaluated else 0
            ),
        }

        await self.log_execution(session_id, input_data, output)
        return output