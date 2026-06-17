import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import time

from app.config import get_settings
from app.api.router import api_router
from app.database.session import init_db, engine
from app.knowledge_graph.neo4j_client import neo4j_client
from app.nlp.embeddings import embedding_service
from app.agents.llm_service import llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize resources on start, clean up on shutdown."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name}...")

    # Initialize database
    await init_db()
    logger.info("Database tables created")

    # Initialize Neo4j
    try:
        await neo4j_client.initialize()
        logger.info("Neo4j connected")
    except Exception as e:
        logger.error(f"Neo4j initialization failed (continuing without graph): {e}")

    # Initialize embedding model
    try:
        await embedding_service.initialize()
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.error(f"Embedding model failed to load (continuing without embeddings): {e}")

    # Initialize LLM
    await llm_service.initialize()
    logger.info("LLM service initialized")

    logger.info(f"{settings.app_name} is ready!")
    yield

    # Cleanup
    await neo4j_client.close()
    await engine.dispose()
    logger.info("Resources cleaned up")


settings = get_settings()

app = FastAPI(
    title="Genesis AI",
    description="Autonomous Scientific Discovery Engine",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# FIX: CORS — can't use allow_origins=["*"] with allow_credentials=True (browser rejects).
# Either enumerate origins or disable credentials.
allowed_origins = settings.allowed_origins if hasattr(settings, "allowed_origins") else [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url)},
    )


# Routes
app.include_router(api_router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "Genesis AI",
        "version": "1.0.0",
        "status": "operational",
        "description": "Autonomous Scientific Discovery Engine",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check with dependency status."""
    status = {"status": "healthy", "services": {}}

    # Check Neo4j
    try:
        await neo4j_client.run_query("RETURN 1 AS ok")
        status["services"]["neo4j"] = "connected"
    except Exception:
        status["services"]["neo4j"] = "unavailable"
        status["status"] = "degraded"

    # Check LLM
    status["services"]["llm"] = "ready" if llm_service._client else "simulated"

    # Check embeddings
    status["services"]["embeddings"] = "ready" if embedding_service._model else "unavailable"

    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
