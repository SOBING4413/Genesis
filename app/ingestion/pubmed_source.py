from pymed import PubMed
from typing import List, Dict, Any, Optional
from app.ingestion.base_source import BaseSource
from loguru import logger


class PubMedSource(BaseSource):
    def __init__(self):
        self.pubmed = PubMed(tool="GenesisAI", email="genesis@example.com")

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search PubMed articles."""
        logger.info(f"Searching PubMed for: {query}")
        results = self.pubmed.query(query, max_results=max_results)
        papers = []
        for article in results:
            paper = self.parse_response(article)
            papers.append(paper)
        return papers

    async def fetch_paper(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single PubMed article by PMID."""
        try:
            results = self.pubmed.query(f"PMID: {source_id}", max_results=1)
            for article in results:
                return self.parse_response(article)
            return None
        except Exception as e:
            logger.error(f"Failed to fetch PubMed paper {source_id}: {e}")
            return None

    async def fetch_batch(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple PubMed articles."""
        papers = []
        for sid in source_ids:
            paper = await self.fetch_paper(sid)
            if paper:
                papers.append(paper)
        return papers

    def parse_response(self, raw_data: Any) -> Dict[str, Any]:
        """Parse PubMed article into standardized format."""
        return {
            "title": raw_data.title or "",
            "authors": [str(a) for a in (raw_data.authors or [])],
            "abstract": raw_data.abstract or "",
            "source": "pubmed",
            "source_id": raw_data.pubmed_id or "",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{raw_data.pubmed_id}/" if raw_data.pubmed_id else "",
            "published_date": str(raw_data.publication_date) if raw_data.publication_date else None,
            "keywords": raw_data.keywords or [],
            "doi": raw_data.doi or "",
            "journal": raw_data.journal or "",
            "methods": raw_data.methods or "",
            "conclusions": raw_data.conclusions or "",
        }