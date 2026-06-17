import arxiv
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.ingestion.base_source import BaseSource
from loguru import logger


class ArxivSource(BaseSource):
    def __init__(self):
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
            num_retries=3,
        )

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv papers."""
        logger.info(f"Searching arXiv for: {query}")
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in self.client.results(search):
            paper = self.parse_response(result)
            papers.append(paper)
        return papers

    async def fetch_paper(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single arXiv paper."""
        try:
            search = arxiv.Search(id_list=[source_id])
            result = next(self.client.results(search))
            return self.parse_response(result)
        except (StopIteration, Exception) as e:
            logger.error(f"Failed to fetch arXiv paper {source_id}: {e}")
            return None

    async def fetch_batch(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple arXiv papers."""
        papers = []
        for sid in source_ids:
            paper = await self.fetch_paper(sid)
            if paper:
                papers.append(paper)
        return papers

    def parse_response(self, raw_data: Any) -> Dict[str, Any]:
        """Parse arXiv API result into standardized format."""
        return {
            "title": raw_data.title.replace("\n", " ").strip(),
            "authors": [str(a) for a in raw_data.authors],
            "abstract": raw_data.summary.replace("\n", " ").strip(),
            "source": "arxiv",
            "source_id": raw_data.entry_id.split("/")[-1],
            "url": raw_data.entry_id,
            "pdf_url": raw_data.pdf_url,
            "published_date": raw_data.published.isoformat() if raw_data.published else None,
            "updated_date": raw_data.updated.isoformat() if raw_data.updated else None,
            "categories": raw_data.categories,
            "doi": raw_data.doi,
            "comment": raw_data.comment,
            "journal_ref": raw_data.journal_ref,
            "primary_category": raw_data.primary_category,
            "links": [link.href for link in raw_data.links],
        }