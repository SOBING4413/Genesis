from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger
from app.database.session import async_session
from app.database.models import ResearchAgentLog


class BaseAgent(ABC):
    """Base class for all research agents."""

    def __init__(self, agent_type: str, model: str = "gpt-4o"):
        self.agent_type = agent_type
        self.model = model

    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main function."""
        pass

    async def log_execution(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        reasoning: str = "",
        duration_ms: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """Log agent execution to database. Never raises — logging failures are non-critical."""
        try:
            # Sanitize large payloads before storing
            safe_input = self._truncate_json(input_data)
            safe_output = self._truncate_json(output_data)

            async with async_session() as session:
                log_entry = ResearchAgentLog(
                    agent_type=self.agent_type,
                    session_id=session_id,
                    input_data=safe_input,
                    output_data=safe_output,
                    reasoning=reasoning[:5000] if reasoning else "",
                    duration_ms=duration_ms,
                    success=success,
                    error_message=error_message,
                )
                session.add(log_entry)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to log agent execution for {self.agent_type}: {e}")

    def _truncate_json(self, data: Any, max_str_len: int = 2000) -> Any:
        """Recursively truncate long strings in a JSON-serializable structure."""
        if isinstance(data, str):
            return data[:max_str_len] + "..." if len(data) > max_str_len else data
        if isinstance(data, dict):
            return {k: self._truncate_json(v, max_str_len) for k, v in data.items()}
        if isinstance(data, list):
            return [self._truncate_json(item, max_str_len) for item in data[:50]]
        return data
