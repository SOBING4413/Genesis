from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from app.config import get_settings
from loguru import logger
import json
import re


class LLMService:
    _instance: Optional["LLMService"] = None
    _client: Optional[AsyncOpenAI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self):
        if self._client is None:
            settings = get_settings()
            if settings.openai_api_key and not settings.openai_api_key.startswith("sk-your"):
                self._client = AsyncOpenAI(
                    api_key=settings.openai_api_key,
                    timeout=60.0,
                    max_retries=3,
                )
                logger.info("OpenAI client initialized")
            else:
                logger.warning("No valid OpenAI API key found. LLM features will be simulated.")
                self._client = None
        return self

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Send a chat completion request."""
        if not self._client:
            return self._simulate_response(messages)

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._simulate_response(messages)

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Get structured JSON response from LLM with robust parsing."""
        content = await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return self._parse_json_safe(content)

    def _parse_json_safe(self, content: str) -> Dict[str, Any]:
        """Parse JSON with multiple fallback strategies."""
        if not content:
            return {}
        # Direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Extract first JSON object/array
        match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from LLM response (len={len(content)})")
        return {"raw_response": content}

    def _simulate_response(self, messages: List[Dict[str, str]]) -> str:
        """Simulate LLM response when no API key is available — returns valid JSON."""
        last_msg = messages[-1]["content"] if messages else ""

        # Detect what kind of JSON is expected and return appropriate mock
        if "hypothesis" in last_msg.lower() or "hypotheses" in last_msg.lower():
            return json.dumps({
                "hypotheses": [
                    {
                        "title": "Simulated Cross-Domain Hypothesis",
                        "description": "A simulated hypothesis connecting biological and computational concepts.",
                        "source_concepts": ["protein", "gene"],
                        "target_concepts": ["algorithm", "network"],
                        "reasoning_chain": ["Step 1: Identify pattern", "Step 2: Apply analogy"],
                        "initial_confidence": 0.6,
                        "type": "cross_domain",
                    }
                ]
            })
        if "scores" in last_msg.lower() or "criterion" in last_msg.lower():
            return json.dumps({
                "scores": {
                    "logical_consistency": 0.7,
                    "empirical_support": 0.6,
                    "falsifiability": 0.75,
                    "novelty": 0.65,
                    "parsimony": 0.7,
                },
                "critiques": ["Simulated critique: hypothesis lacks direct empirical support."],
                "logical_fallacies": [],
                "missing_evidence": ["Controlled experiments needed"],
                "alternative_explanations": ["Null hypothesis explanation"],
                "overall_assessment": "Moderate quality hypothesis requiring revision.",
                "recommendation": "revise",
            })
        if "final_score" in last_msg.lower() or "validity" in last_msg.lower():
            return json.dumps({
                "final_score": 0.65,
                "confidence": 0.7,
                "recommended_next_steps": [
                    "Design in vitro experiments",
                    "Literature review on related mechanisms",
                ],
                "priority": "medium",
            })
        # Generic research summary
        return json.dumps({
            "analysis": {
                "key_findings": ["Simulated finding 1", "Simulated finding 2"],
                "summary": "Simulated research analysis. Connect your OpenAI API key for real results.",
            },
            "research_gaps": ["Gap 1: Limited cross-domain studies"],
            "confidence": 0.5,
        })


llm_service = LLMService()
