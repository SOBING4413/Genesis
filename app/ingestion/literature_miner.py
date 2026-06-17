from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from app.ingestion.arxiv_source import ArxivSource
from app.ingestion.pubmed_source import PubMedSource
from app.database.models import Paper, PaperStatus, ConceptExtraction
from app.database.session import async_session
from app.nlp.concept_extractor import concept_extractor
from app.nlp.embeddings import embedding_service
from app.knowledge_graph.builder import kg_builder
from sqlalchemy import select


class LiteratureMiner:
    def __init__(self):
        self.sources = {
            "arxiv": ArxivSource(),
            "pubmed": PubMedSource(),
        }

    async def search_and_import(
        self, query: str, sources: List[str] = None, max_per_source: int = 10
    ) -> int:
        """Search multiple sources and import papers. Returns total count imported."""
        ids = await self.search_and_import_with_ids(query, sources, max_per_source)
        return len(ids)

    async def search_and_import_with_ids(
        self, query: str, sources: List[str] = None, max_per_source: int = 10
    ) -> List[str]:
        """Search multiple sources and import papers. Returns list of imported paper IDs."""
        if sources is None:
            sources = list(self.sources.keys())

        imported_ids: List[str] = []

        for source_name in sources:
            source = self.sources.get(source_name)
            if not source:
                logger.warning(f"Unknown source: {source_name}")
                continue

            try:
                papers_data = await source.search(query, max_results=max_per_source)
                for paper_data in papers_data:
                    paper = await self.import_paper(paper_data)
                    if paper:
                        imported_ids.append(str(paper.id))
            except Exception as e:
                logger.error(f"Error searching {source_name}: {e}")

        logger.info(f"Imported {len(imported_ids)} papers for query: {query}")
        return imported_ids

    async def import_paper(self, paper_data: Dict[str, Any]) -> Optional[Paper]:
        """Import a single paper into the database. Returns Paper or None if duplicate."""
        try:
            async with async_session() as session:
                # Check if paper already exists (by source_id)
                source_id = paper_data.get("source_id")
                if source_id:
                    existing = await session.execute(
                        select(Paper).where(Paper.source_id == source_id)
                    )
                    if existing.scalar_one_or_none():
                        logger.debug(f"Paper already exists: {source_id}")
                        return None

                paper = Paper(
                    title=paper_data.get("title", "Untitled"),
                    authors=paper_data.get("authors", []),
                    abstract=paper_data.get("abstract", ""),
                    source=paper_data.get("source", ""),
                    source_id=source_id,
                    url=paper_data.get("url", ""),
                    published_date=self._parse_date(paper_data.get("published_date")),
                    keywords=paper_data.get("keywords", []),
                    categories=paper_data.get("categories", []),
                    status=PaperStatus.PENDING,
                    metadata_json={
                        k: v for k, v in paper_data.items()
                        if k not in {"title", "authors", "abstract", "source", "source_id",
                                     "url", "published_date", "keywords", "categories"}
                    },
                )

                session.add(paper)
                await session.commit()
                await session.refresh(paper)

                logger.info(f"Imported paper: {paper.title[:60]}")
                return paper

        except Exception as e:
            logger.error(f"Failed to import paper: {e}")
            return None

    async def process_paper(self, paper_id: str) -> bool:
        """
        Process a paper: extract concepts, generate embeddings, build KG.
        Returns True on success, False on failure.
        """
        async with async_session() as session:
            result = await session.execute(select(Paper).where(Paper.id == paper_id))
            paper = result.scalar_one_or_none()
            if not paper:
                logger.warning(f"Paper not found: {paper_id}")
                return False

            paper.status = PaperStatus.PROCESSING
            await session.commit()

            try:
                text = f"{paper.title}\n\n{paper.abstract}"
                if paper.full_text:
                    # Limit full text to avoid OOM
                    text += f"\n\n{paper.full_text[:5000]}"

                # 1. Extract concepts
                extraction_result = concept_extractor.extract(text)

                extraction = ConceptExtraction(
                    paper_id=paper.id,
                    concepts=extraction_result["concepts"],
                    relationships=extraction_result["relationships"],
                    confidence_scores={
                        c["name"]: c.get("confidence", 0.5)
                        for c in extraction_result["concepts"]
                        if isinstance(c, dict) and "name" in c
                    },
                    extraction_method=extraction_result["method"],
                )
                session.add(extraction)

                # 2. Generate embedding (if model available)
                try:
                    embedding_service.embed_text(text)
                    paper.embedding_id = f"emb_{paper.id}"
                except RuntimeError:
                    logger.debug("Embedding model not ready, skipping")

                # 3. Build knowledge graph
                try:
                    await kg_builder.build_from_paper(paper, extraction)
                except Exception as e:
                    logger.warning(f"KG build failed for paper {paper.id}: {e}")

                paper.status = PaperStatus.COMPLETED
                await session.commit()
                logger.info(f"Processed paper: {paper.title[:50]}")
                return True

            except Exception as e:
                paper.status = PaperStatus.FAILED
                await session.commit()
                logger.error(f"Failed to process paper {paper.id}: {e}", exc_info=True)
                return False

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
            try:
                return datetime.strptime(date_str.replace("Z", "+00:00").split("+")[0], fmt.split("%z")[0])
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
