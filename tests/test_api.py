import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.main import app


# FIX: fixture must be async (async_generator) to work with @pytest.mark.asyncio
@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Genesis AI"
    assert data["version"] == "1.0.0"
    assert "docs" in data


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")  # degraded when services unavailable
    assert "services" in data


@pytest.mark.asyncio
async def test_graph_search_no_query(client):
    response = await client.get("/api/graph/search")
    assert response.status_code == 422  # Missing required query param


@pytest.mark.asyncio
async def test_graph_search_valid(client):
    response = await client.get("/api/graph/search?query=protein&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_papers_list(client):
    response = await client.get("/api/papers/?limit=5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_papers_invalid_status(client):
    response = await client.get("/api/papers/?status=invalid_status")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_hypothesis_list(client):
    response = await client.get("/api/hypothesis/")
    assert response.status_code == 200
    data = response.json()
    assert "hypotheses" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_hypothesis_not_found(client):
    response = await client.get("/api/hypothesis/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_hypothesis_invalid_id(client):
    response = await client.get("/api/hypothesis/not-a-uuid")
    # Should return 404 (invalid UUID treated as not found)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_analytics_endpoint(client):
    response = await client.get("/api/analytics/")
    assert response.status_code == 200
    data = response.json()
    assert "papers" in data
    assert "hypotheses" in data
    assert "knowledge_graph" in data


@pytest.mark.asyncio
async def test_analytics_timeline(client):
    response = await client.get("/api/analytics/timeline?days=7")
    assert response.status_code == 200
    data = response.json()
    assert "papers" in data
    assert "hypotheses" in data


@pytest.mark.asyncio
async def test_find_path_same_concept(client):
    response = await client.get("/api/graph/path?concept_a=protein&concept_b=protein")
    assert response.status_code == 400  # Same concept should return error
