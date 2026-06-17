from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class BaseSource(ABC):
    """Base class for scientific literature sources."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers by query."""
        pass

    @abstractmethod
    async def fetch_paper(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single paper by its source ID."""
        pass

    @abstractmethod
    async def fetch_batch(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple papers by source IDs."""
        pass

    @abstractmethod
    def parse_response(self, raw_data: Any) -> Dict[str, Any]:
        """Parse raw response into standardized paper format."""
        pass