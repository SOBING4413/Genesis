from loguru import logger
from app.tasks.celery_app import celery_app
import asyncio


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_paper_task(self, paper_id: str):
    """
    Process a paper asynchronously.
    FIX: asyncio.run() is correct here since Celery workers are sync.
    However we need to create a fresh event loop each time to avoid
    "Event loop is closed" errors across task retries.
    """
    logger.info(f"Task: processing paper {paper_id}")
    try:
        from app.ingestion.literature_miner import LiteratureMiner
        miner = LiteratureMiner()
        # Use asyncio.run() which creates a new event loop each call — safe for Celery
        success = asyncio.run(miner.process_paper(paper_id))
        if not success:
            raise RuntimeError(f"Processing returned False for paper {paper_id}")
        return {"status": "completed", "paper_id": paper_id}
    except Exception as exc:
        logger.error(f"Task failed for paper {paper_id}: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def search_and_import_task(self, query: str, sources: list, max_per_source: int = 10):
    """Search and import papers asynchronously."""
    logger.info(f"Task: searching '{query}' from {sources}")
    try:
        from app.ingestion.literature_miner import LiteratureMiner
        miner = LiteratureMiner()
        imported_ids = asyncio.run(
            miner.search_and_import_with_ids(query, sources, max_per_source)
        )
        # Queue individual processing for each imported paper
        for paper_id in imported_ids:
            process_paper_task.delay(paper_id)
        return {"imported": len(imported_ids), "query": query, "paper_ids": imported_ids}
    except Exception as exc:
        logger.error(f"Search task failed: {exc}")
        raise self.retry(exc=exc)
