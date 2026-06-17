from neo4j import AsyncGraphDatabase, AsyncDriver
from app.config import get_settings
from loguru import logger
from typing import Optional, Any, List, Dict
import asyncio


class Neo4jClient:
    _instance: Optional["Neo4jClient"] = None
    _driver: Optional[AsyncDriver] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self):
        if self._driver is None:
            settings = get_settings()
            self._driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_pool_size=50,
                connection_timeout=30.0,
            )
            # Verify connectivity
            await self._driver.verify_connectivity()
            logger.info("Neo4j driver initialized and connected")
            await self._create_constraints()
            await self._create_indexes()
            self._initialized = True
        return self

    async def _create_constraints(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.source_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        ]
        for constraint in constraints:
            try:
                await self.run_query(constraint)
            except Exception as e:
                logger.warning(f"Constraint warning (may already exist): {e}")

    async def _create_indexes(self):
        """Create full-text and composite indexes for search performance."""
        indexes = [
            # Full-text index for concept search
            "CREATE FULLTEXT INDEX concept_index IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.type]",
            # Full-text index for paper search
            "CREATE FULLTEXT INDEX paper_index IF NOT EXISTS FOR (p:Paper) ON EACH [p.title, p.abstract]",
            # Range index for ordering
            "CREATE INDEX paper_source IF NOT EXISTS FOR (p:Paper) ON (p.source)",
        ]
        for idx in indexes:
            try:
                await self.run_query(idx)
            except Exception as e:
                logger.warning(f"Index warning (may already exist): {e}")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call initialize() first.")
        return self._driver

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
            logger.info("Neo4j driver closed")

    async def run_query(self, query: str, params: dict = None) -> List[Dict]:
        """Run a Cypher query and return results as list of dicts."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call initialize() first.")
        async with self._driver.session(database="neo4j") as session:
            result = await session.run(query, params or {})
            records = await result.data()
            return records

    async def run_query_safe(self, query: str, params: dict = None) -> List[Dict]:
        """Run query with error handling — returns empty list on failure."""
        try:
            return await self.run_query(query, params)
        except Exception as e:
            logger.error(f"Query failed: {e}\nQuery: {query[:200]}")
            return []

    async def create_paper_node(self, paper_data: dict) -> dict:
        query = """
        MERGE (p:Paper {source_id: $source_id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.source = $source,
            p.url = $url,
            p.published_date = $published_date,
            p.updated_at = datetime()
        RETURN p
        """
        params = {
            "source_id": paper_data["source_id"],
            "title": paper_data.get("title", ""),
            "abstract": paper_data.get("abstract", ""),
            "source": paper_data.get("source", ""),
            "url": paper_data.get("url", ""),
            "published_date": paper_data.get("published_date"),
        }
        records = await self.run_query(query, params)
        return records[0] if records else {}

    async def create_concept_node(self, concept_name: str, concept_type: str = "generic") -> dict:
        query = """
        MERGE (c:Concept {name: $name})
        ON CREATE SET c.type = $type, c.created_at = datetime()
        ON MATCH SET c.type = CASE WHEN c.type = 'generic' THEN $type ELSE c.type END
        RETURN c
        """
        records = await self.run_query(query, {"name": concept_name, "type": concept_type})
        return records[0] if records else {}

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict = None,
        source_label: str = "Paper",
        target_label: str = "Concept",
    ) -> dict:
        """
        FIX: Concept nodes use 'name' as key, Paper nodes use 'source_id'.
        Build correct MATCH clause per label.
        """
        # Determine lookup key per label
        source_key = "source_id" if source_label == "Paper" else "name"
        target_key = "source_id" if target_label == "Paper" else "name"

        # Sanitize relationship type (must be valid Cypher identifier)
        safe_rel_type = rel_type.upper().replace(" ", "_").replace("-", "_")

        query = f"""
        MATCH (a:{source_label} {{{source_key}: $source_id}})
        MATCH (b:{target_label} {{{target_key}: $target_id}})
        MERGE (a)-[r:{safe_rel_type}]->(b)
        SET r += $properties
        RETURN r
        """
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "properties": properties or {},
        }
        records = await self.run_query_safe(query, params)
        return records[0] if records else {}

    async def search_concepts(self, query_text: str, limit: int = 20) -> List[Dict]:
        """Search concepts using full-text index with fallback to CONTAINS."""
        # Escape special chars for full-text search
        safe_query = query_text.replace('"', '\\"').replace("'", "\\'")
        ft_query = """
        CALL db.index.fulltext.queryNodes('concept_index', $query)
        YIELD node, score
        RETURN node.name AS name, node.type AS type, score
        ORDER BY score DESC
        LIMIT $limit
        """
        try:
            records = await self.run_query(ft_query, {"query": safe_query, "limit": limit})
            if records:
                return records
        except Exception as e:
            logger.debug(f"Full-text search unavailable, using fallback: {e}")

        # Fallback: case-insensitive CONTAINS
        fallback_query = """
        MATCH (c:Concept)
        WHERE toLower(c.name) CONTAINS toLower($query)
        WITH c, 
             CASE WHEN toLower(c.name) = toLower($query) THEN 2.0
                  WHEN toLower(c.name) STARTS WITH toLower($query) THEN 1.5
                  ELSE 1.0 END AS score
        RETURN c.name AS name, c.type AS type, score
        ORDER BY score DESC
        LIMIT $limit
        """
        return await self.run_query_safe(fallback_query, {"query": query_text, "limit": limit})

    async def get_paper_graph(self, paper_source_id: str, depth: int = 2) -> List[Dict]:
        """Get a paper's concept network."""
        query = """
        MATCH path = (p:Paper {source_id: $source_id})-[*1..$depth]-(related)
        WITH path, [node IN nodes(path) | {id: coalesce(node.source_id, node.name), 
                                           label: labels(node)[0], 
                                           name: coalesce(node.title, node.name)}] AS node_list,
             [rel IN relationships(path) | type(rel)] AS rel_types
        RETURN node_list, rel_types
        LIMIT 100
        """
        return await self.run_query_safe(query, {"source_id": paper_source_id, "depth": depth})

    async def find_path_between_concepts(
        self, concept_a: str, concept_b: str, max_depth: int = 4
    ) -> List[Dict]:
        """
        Find shortest path between two concepts.
        FIX: Added guard for same-concept lookup (shortestPath with same start/end
        node and minimum 1 hop can cause issues).
        """
        if concept_a == concept_b:
            return []
        query = """
        MATCH (a:Concept {name: $concept_a}), (b:Concept {name: $concept_b})
        MATCH path = shortestPath((a)-[*1..$max_depth]-(b))
        WITH path,
             [node IN nodes(path) | node.name] AS node_names,
             [rel IN relationships(path) | type(rel)] AS rel_types,
             length(path) AS path_length
        RETURN node_names, rel_types, path_length
        ORDER BY path_length
        LIMIT 5
        """
        return await self.run_query_safe(
            query,
            {"concept_a": concept_a, "concept_b": concept_b, "max_depth": max_depth},
        )

    async def get_highest_impact_concepts(self, limit: int = 50) -> List[Dict]:
        """Get concepts ranked by paper mention count."""
        query = """
        MATCH (c:Concept)<-[r:MENTIONS]-(p:Paper)
        WITH c, count(r) AS mention_count, count(DISTINCT p) AS paper_count
        RETURN c.name AS name, c.type AS type, mention_count, paper_count,
               (paper_count * 2 + mention_count) AS impact_score
        ORDER BY impact_score DESC, paper_count DESC
        LIMIT $limit
        """
        return await self.run_query_safe(query, {"limit": limit})

    async def get_concept_neighbors(self, concept_name: str, limit: int = 20) -> List[Dict]:
        """Get direct neighbors of a concept node."""
        query = """
        MATCH (c:Concept {name: $name})-[r]-(neighbor:Concept)
        RETURN neighbor.name AS name, neighbor.type AS type, type(r) AS relationship,
               count(r) AS strength
        ORDER BY strength DESC
        LIMIT $limit
        """
        return await self.run_query_safe(query, {"name": concept_name, "limit": limit})

    async def get_concept_network(self, concept_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get concept network as nodes+edges for visualization."""
        query = """
        MATCH path = (c:Concept {name: $name})-[*1..$depth]-(neighbor:Concept)
        WITH nodes(path) AS path_nodes, relationships(path) AS path_rels
        UNWIND range(0, size(path_rels)-1) AS idx
        WITH path_nodes[idx] AS src, path_rels[idx] AS rel, path_nodes[idx+1] AS tgt
        RETURN DISTINCT
            src.name AS source, src.type AS source_type,
            type(rel) AS relationship,
            tgt.name AS target, tgt.type AS target_type
        LIMIT 200
        """
        records = await self.run_query_safe(query, {"name": concept_name, "depth": depth})

        nodes = {}
        edges = []
        for r in records:
            nodes[r["source"]] = {"name": r["source"], "type": r["source_type"]}
            nodes[r["target"]] = {"name": r["target"], "type": r["target_type"]}
            edges.append({
                "source": r["source"],
                "target": r["target"],
                "type": r["relationship"],
            })

        return {
            "center_concept": concept_name,
            "nodes": list(nodes.values()),
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }


# Singleton instance
neo4j_client = Neo4jClient()
