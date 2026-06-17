from fastapi import APIRouter
from app.api.papers import router as papers_router
from app.api.graph import router as graph_router
from app.api.hypothesis import router as hypothesis_router
from app.api.analytics import router as analytics_router

api_router = APIRouter(prefix="/api")

api_router.include_router(papers_router, prefix="/papers", tags=["Papers"])
api_router.include_router(graph_router, prefix="/graph", tags=["Knowledge Graph"])
api_router.include_router(hypothesis_router, prefix="/hypothesis", tags=["Hypothesis"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
