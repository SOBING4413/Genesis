from app.tasks.celery_app import celery_app
from app.tasks.paper_tasks import process_paper_task

__all__ = ["celery_app", "process_paper_task"]