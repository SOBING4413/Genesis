import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all modules can be imported without errors."""
    from app.config import get_settings
    from app.database.models import Paper, Hypothesis, ConceptExtraction, ResearchAgentLog
    from app.nlp.concept_extractor import concept_extractor
    from app.nlp.embeddings import embedding_service
    from app.knowledge_graph.builder import kg_builder
    from app.agents.researcher_agent import ResearcherAgent
    from app.agents.inventor_agent import InventorAgent
    from app.agents.critic_agent import CriticAgent
    from app.agents.evaluator_agent import EvaluatorAgent
    from app.agents.coordinator_agent import CoordinatorAgent
    from app.hypothesis_engine.engine import hypothesis_engine
    from app.ingestion.literature_miner import LiteratureMiner
    assert True


def test_concept_extraction_basic():
    """Test concept extraction with sample text."""
    from app.nlp.concept_extractor import concept_extractor

    text = """
    The TP53 gene encodes the p53 protein, which regulates the cell cycle.
    Mutations in TP53 are associated with various cancers including breast cancer.
    The MDM2 protein inhibits p53 activity through ubiquitination.
    Therapeutic agents such as nutlin-3 target the MDM2-p53 interaction.
    """
    result = concept_extractor.extract(text)
    assert "concepts" in result
    assert "relationships" in result
    assert "method" in result
    assert isinstance(result["concepts"], list)
    assert isinstance(result["relationships"], list)
    assert result["method"] in ("spacy", "regex", "none")


def test_concept_extraction_empty():
    """Test concept extraction handles empty input gracefully."""
    from app.nlp.concept_extractor import concept_extractor

    result = concept_extractor.extract("")
    assert result["concepts"] == []
    assert result["relationships"] == []
    assert result["method"] == "none"

    result2 = concept_extractor.extract("   ")
    assert result2["method"] == "none"


def test_config_defaults():
    """Test settings have sensible defaults."""
    from app.config import get_settings
    settings = get_settings()
    assert settings.app_name == "Genesis AI"
    assert settings.postgres_port == 5432
    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.sync_database_url.startswith("postgresql://")


def test_llm_service_simulate():
    """Test LLM simulation returns valid JSON."""
    import json
    from app.agents.llm_service import LLMService

    svc = LLMService()
    # Simulate hypothesis response
    response = svc._simulate_response([
        {"role": "user", "content": "Generate hypotheses about protein folding"}
    ])
    parsed = json.loads(response)
    assert "hypotheses" in parsed

    # Simulate critique response
    response2 = svc._simulate_response([
        {"role": "user", "content": "Evaluate scores and criterion for hypothesis"}
    ])
    parsed2 = json.loads(response2)
    assert "scores" in parsed2
    assert "recommendation" in parsed2


def test_critic_average_scores():
    """Test the bug-fixed average score computation."""
    from app.agents.critic_agent import CriticAgent
    critic = CriticAgent()

    critiques = [
        {"scores": {"logical_consistency": 0.8, "novelty": 0.6}},
        {"scores": {"logical_consistency": 0.6, "novelty": 0.8}},
        {"scores": {"logical_consistency": 0.7}},
    ]
    averages = critic._compute_average_scores(critiques)
    # logical_consistency avg = (0.8 + 0.6 + 0.7) / 3 = 0.7
    assert abs(averages["logical_consistency"] - 0.7) < 0.001
    # novelty avg = (0.6 + 0.8) / 2 = 0.7
    assert abs(averages["novelty"] - 0.7) < 0.001


def test_literature_miner_date_parse():
    """Test date parsing with various formats."""
    from app.ingestion.literature_miner import LiteratureMiner
    miner = LiteratureMiner()

    assert miner._parse_date(None) is None
    assert miner._parse_date("") is None
    assert miner._parse_date("2024-01-15") is not None
    assert miner._parse_date("2024-01-15T10:30:00") is not None
    assert miner._parse_date("invalid-date") is None
