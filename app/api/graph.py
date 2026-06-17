from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from app.knowledge_graph.neo4j_client import neo4j_client
from app.knowledge_graph.builder import kg_builder

router = APIRouter()


@router.get("/")
async def get_graph_overview():
    """Get overall knowledge graph statistics."""
    return await kg_builder.get_graph_statistics()


@router.get("/search")
async def search_graph(
    query: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=100),
):
    """Search concepts in the knowledge graph."""
    results = await neo4j_client.search_concepts(query, limit=limit)
    return {"query": query, "results": results, "total": len(results)}


@router.get("/path")
async def find_path(
    concept_a: str = Query(..., min_length=1, max_length=200),
    concept_b: str = Query(..., min_length=1, max_length=200),
    max_depth: int = Query(4, ge=1, le=6),
):
    """Find shortest path between two concepts."""
    if concept_a == concept_b:
        raise HTTPException(status_code=400, detail="concept_a and concept_b must be different")
    paths = await neo4j_client.find_path_between_concepts(concept_a, concept_b, max_depth)
    return {
        "concept_a": concept_a,
        "concept_b": concept_b,
        "paths_found": len(paths),
        "paths": paths,
    }


@router.get("/top-concepts")
async def top_concepts(limit: int = Query(50, ge=1, le=200)):
    """Get highest impact concepts ranked by paper mentions."""
    concepts = await neo4j_client.get_highest_impact_concepts(limit=limit)
    return {"concepts": concepts, "total": len(concepts)}


@router.get("/concept/{concept_name}/network")
async def concept_network(
    concept_name: str,
    depth: int = Query(2, ge=1, le=4),
):
    """Get the full concept network around a given concept."""
    network = await neo4j_client.get_concept_network(concept_name, depth)
    if network["node_count"] == 0:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found or has no connections")
    return network


@router.get("/concept/{concept_name}/neighbors")
async def concept_neighbors(
    concept_name: str,
    limit: int = Query(20, ge=1, le=100),
):
    """Get direct neighbors of a concept."""
    neighbors = await neo4j_client.get_concept_neighbors(concept_name, limit=limit)
    return {"concept": concept_name, "neighbors": neighbors, "total": len(neighbors)}
