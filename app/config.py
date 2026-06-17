from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Genesis AI"
    debug: bool = True
    secret_key: str = "default-secret-key-change-in-production"
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]

    # Database
    postgres_user: str = "genesis"
    postgres_password: str = "genesis_pass"
    postgres_db: str = "genesis_ai"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_pass"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # AI
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Rate limiting / pagination defaults
    max_papers_per_import: int = 50
    max_hypothesis_depth: int = 5

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # FIX: pydantic-settings v2 uses model_config, not inner class Config
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
