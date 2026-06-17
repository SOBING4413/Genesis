from app.knowledge_graph.neo4j_client import neo4j_client
from app.database.models import Paper, ConceptExtraction
from loguru import logger
from typing import List, Dict, Any


class KnowledgeGraphBuilder:
    def __init__(self):
        self.client = neo4j_client

    async def build_from_paper(self, paper: Paper, extraction: ConceptExtraction) -> int:
        """Build knowledge graph nodes and relationships from a paper's concept extraction."""
        relations_created = 0

        # 1. Create paper node
        await self.client.create_paper_node({
            "source_id": paper.source_id or str(paper.id),
            "title": paper.title,
            "abstract": paper.abstract,
            "source": paper.source,
            "url": paper.url or "",
            "published_date": paper.published_date.isoformat() if paper.published_date else None,
        })

        # 2. Create concept nodes and link to paper
        concepts = extraction.concepts or []
        for concept in concepts:
            concept_name = concept.get("name", concept.get("text", "")).strip()
            concept_type = concept.get("type", "generic")

            if not concept_name or len(concept_name) < 2:
                continue

            await self.client.create_concept_node(concept_name, concept_type)

            source_id = paper.source_id or str(paper.id)
            await self.client.create_relationship(
                source_id=source_id,
                target_id=concept_name,
                rel_type="MENTIONS",
                properties={"confidence": concept.get("confidence", 1.0)},
                source_label="Paper",
                target_label="Concept",
            )
            relations_created += 1

        # 3. Create concept-to-concept relationships
        relationships = extraction.relationships or []
        for rel in relationships:
            source_concept = rel.get("source", "").strip()
            target_concept = rel.get("target", "").strip()
            rel_type = rel.get("type", "RELATED_TO").upper().replace(" ", "_")

            if not source_concept or not target_concept:
                continue
            if source_concept == target_concept:
                continue  # Skip self-loops

            await self.client.create_concept_node(source_concept)
            await self.client.create_concept_node(target_concept)

            await self.client.create_relationship(
                source_id=source_concept,
                target_id=target_concept,
                rel_type=rel_type,
                properties={
                    "confidence": rel.get("confidence", 0.5),
                    "evidence": paper.source_id or str(paper.id),
                },
                source_label="Concept",
                target_label="Concept",
            )
            relations_created += 1

        logger.info(
            f"Built KG for paper '{paper.title[:50]}...': {relations_created} relations"
        )
        return relations_created

    async def get_concept_network(self, concept_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get the network of concepts surrounding a given concept."""
        return await self.client.get_concept_network(concept_name, depth)

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        query_totals = """
        MATCH (n)
        WITH labels(n)[0] AS lbl, count(n) AS cnt
        WITH collect({label: lbl, count: cnt}) AS label_counts,
             sum(cnt) AS total_nodes
        RETURN total_nodes, label_counts
        """
        query_rels = """
        MATCH ()-[r]->()
        WITH type(r) AS rel_type, count(r) AS cnt
        WITH collect({type: rel_type, count: cnt}) AS rel_counts,
             sum(cnt) AS total_rels
        RETURN total_rels, rel_counts
        """
        query_concepts = """
        MATCH (c:Concept)
        WITH c.type AS type, count(c) AS count
        WHERE type IS NOT NULL
        RETURN type, count
        ORDER BY count DESC
        """

        totals = await self.client.run_query_safe(query_totals)
        rels = await self.client.run_query_safe(query_rels)
        concept_types = await self.client.run_query_safe(query_concepts)

        total_nodes = totals[0]["total_nodes"] if totals else 0
        label_counts = {item["label"]: item["count"] for item in (totals[0].get("label_counts", []) if totals else [])}
        total_rels = rels[0]["total_rels"] if rels else 0
        rel_type_counts = {item["type"]: item["count"] for item in (rels[0].get("rel_counts", []) if rels else [])}

        return {
            "total_nodes": total_nodes,
            "node_labels": label_counts,
            "total_relationships": total_rels,
            "relationship_types": rel_type_counts,
            "concept_type_distribution": {item["type"]: item["count"] for item in concept_types},
        }


kg_builder = KnowledgeGraphBuilder()
