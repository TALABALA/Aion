"""
AION SQLite Graph Store

High-performance SQLite-based graph storage with:
- Adjacency list model with JSON properties
- Full-text search with FTS5
- Vector embeddings with efficient similarity search
- Optimized indices for graph traversal
- Connection pooling and WAL mode
"""

from __future__ import annotations

import json
import struct
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import structlog

from aion.systems.knowledge.types import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    Path as GraphPath,
    Subgraph,
    GraphStatistics,
    Provenance,
    GraphEmbedding,
)
from aion.systems.knowledge.store.base import GraphStore

logger = structlog.get_logger(__name__)


class SQLiteGraphStore(GraphStore):
    """
    SQLite-based graph storage with advanced features.

    Schema:
    - entities: Node storage with JSON properties
    - relationships: Edge storage with JSON properties
    - entity_aliases: Alias lookup table
    - entity_embeddings: Vector embeddings (stored as BLOB)
    - entity_fts: Full-text search index

    Performance optimizations:
    - WAL mode for concurrent reads
    - Prepared statement caching
    - Batch operations
    - Covering indices for common queries
    """

    def __init__(
        self,
        db_path: str = "./data/knowledge_graph.db",
        enable_fts: bool = True,
        enable_wal: bool = True,
    ):
        self.db_path = Path(db_path)
        self.enable_fts = enable_fts
        self.enable_wal = enable_wal

        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(
            str(self.db_path),
            isolation_level=None,  # Autocommit mode for explicit transactions
        )
        self._connection.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrency
        if self.enable_wal:
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA synchronous=NORMAL")

        # Performance settings
        await self._connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
        await self._connection.execute("PRAGMA temp_store=MEMORY")
        await self._connection.execute("PRAGMA mmap_size=268435456")  # 256MB mmap

        await self._create_schema()

        self._initialized = True
        logger.info(f"SQLite graph store initialized: {self.db_path}")

    async def shutdown(self) -> None:
        """Close the database connection."""
        if self._connection:
            # Checkpoint WAL before closing
            if self.enable_wal:
                await self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await self._connection.close()
            self._connection = None
        self._initialized = False
        logger.info("SQLite graph store shutdown")

    async def health_check(self) -> Dict[str, Any]:
        """Check store health."""
        if not self._connection:
            return {"status": "error", "message": "Not connected"}

        try:
            cursor = await self._connection.execute("SELECT COUNT(*) FROM entities")
            count = (await cursor.fetchone())[0]
            return {
                "status": "ok",
                "db_path": str(self.db_path),
                "entity_count": count,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _create_schema(self) -> None:
        """Create database schema with optimized indices."""
        await self._connection.executescript("""
            -- Entities table
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                custom_type TEXT,
                description TEXT DEFAULT '',
                aliases_json TEXT DEFAULT '[]',
                properties_json TEXT DEFAULT '{}',
                provenance_json TEXT,
                confidence REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,
                degree_centrality REAL DEFAULT 0.0,
                betweenness_centrality REAL DEFAULT 0.0,
                pagerank REAL DEFAULT 0.0,
                version INTEGER DEFAULT 1,
                previous_version_id TEXT,
                content_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );

            -- Entity aliases for fast lookup
            CREATE TABLE IF NOT EXISTS entity_aliases (
                entity_id TEXT NOT NULL,
                alias TEXT NOT NULL COLLATE NOCASE,
                PRIMARY KEY (entity_id, alias),
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );

            -- Entity embeddings (multiple types per entity)
            CREATE TABLE IF NOT EXISTS entity_embeddings (
                entity_id TEXT NOT NULL,
                embedding_type TEXT NOT NULL DEFAULT 'text',
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                model_id TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (entity_id, embedding_type),
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );

            -- Relationships table
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                custom_type TEXT,
                properties_json TEXT DEFAULT '{}',
                weight REAL DEFAULT 1.0,
                confidence REAL DEFAULT 1.0,
                bidirectional INTEGER DEFAULT 0,
                transitive INTEGER DEFAULT 0,
                provenance_json TEXT,
                valid_from TEXT,
                valid_until TEXT,
                version INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
            );

            -- Optimized indices
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type) WHERE deleted_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE) WHERE deleted_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_entities_hash ON entities(content_hash) WHERE deleted_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_entities_importance ON entities(importance DESC) WHERE deleted_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_entities_pagerank ON entities(pagerank DESC) WHERE deleted_at IS NULL;

            CREATE INDEX IF NOT EXISTS idx_aliases_alias ON entity_aliases(alias COLLATE NOCASE);

            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);
            CREATE INDEX IF NOT EXISTS idx_rel_source_type ON relationships(source_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_rel_target_type ON relationships(target_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_rel_endpoints ON relationships(source_id, target_id);
            CREATE INDEX IF NOT EXISTS idx_rel_valid ON relationships(valid_from, valid_until)
                WHERE valid_from IS NOT NULL OR valid_until IS NOT NULL;
        """)

        # Create FTS5 virtual table for full-text search
        if self.enable_fts:
            await self._connection.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(
                    name,
                    description,
                    aliases,
                    content=entities,
                    content_rowid=rowid
                );

                -- Triggers to keep FTS in sync
                CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
                    INSERT INTO entity_fts(rowid, name, description, aliases)
                    VALUES (new.rowid, new.name, new.description, new.aliases_json);
                END;

                CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
                    INSERT INTO entity_fts(entity_fts, rowid, name, description, aliases)
                    VALUES ('delete', old.rowid, old.name, old.description, old.aliases_json);
                END;

                CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
                    INSERT INTO entity_fts(entity_fts, rowid, name, description, aliases)
                    VALUES ('delete', old.rowid, old.name, old.description, old.aliases_json);
                    INSERT INTO entity_fts(rowid, name, description, aliases)
                    VALUES (new.rowid, new.name, new.description, new.aliases_json);
                END;
            """)

    # ==========================================================================
    # Entity Operations
    # ==========================================================================

    async def create_entity(self, entity: Entity) -> str:
        """Create a new entity."""
        now = datetime.now().isoformat()

        await self._connection.execute("""
            INSERT INTO entities (
                id, name, entity_type, custom_type, description,
                aliases_json, properties_json, provenance_json,
                confidence, importance, version, content_hash,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.id,
            entity.name,
            entity.entity_type.value,
            entity.custom_type,
            entity.description,
            json.dumps(entity.aliases),
            json.dumps(entity.properties),
            json.dumps(entity.provenance.to_dict()) if entity.provenance else None,
            entity.confidence,
            entity.importance,
            entity.version,
            entity.content_hash,
            entity.created_at.isoformat() if isinstance(entity.created_at, datetime) else now,
            now,
        ))

        # Add aliases
        for alias in entity.aliases:
            await self._connection.execute(
                "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                (entity.id, alias.lower()),
            )

        # Add name as alias too
        await self._connection.execute(
            "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
            (entity.id, entity.name.lower()),
        )

        # Add embeddings
        if entity.text_embedding and entity.text_embedding.vector:
            await self._save_embedding(
                entity.id,
                entity.text_embedding.vector,
                "text",
                entity.text_embedding.model_id,
            )

        if entity.graph_embedding and entity.graph_embedding.vector:
            await self._save_embedding(
                entity.id,
                entity.graph_embedding.vector,
                "graph",
                entity.graph_embedding.model_id,
            )

        await self._connection.commit()

        logger.debug(f"Created entity: {entity.name} ({entity.id})")
        return entity.id

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM entities WHERE id = ? AND deleted_at IS NULL",
            (entity_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        entity = self._row_to_entity(row)

        # Load embeddings
        await self._load_entity_embeddings(entity)

        return entity

    async def get_entity_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
    ) -> Optional[Entity]:
        """Get an entity by name or alias."""
        # Try exact name match first
        query = "SELECT * FROM entities WHERE name = ? COLLATE NOCASE AND deleted_at IS NULL"
        params: list = [name]

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type.value)

        cursor = await self._connection.execute(query, params)
        row = await cursor.fetchone()

        if row:
            entity = self._row_to_entity(row)
            await self._load_entity_embeddings(entity)
            return entity

        # Try alias match
        query = """
            SELECT e.* FROM entities e
            JOIN entity_aliases a ON e.id = a.entity_id
            WHERE a.alias = ? COLLATE NOCASE AND e.deleted_at IS NULL
        """
        params = [name.lower()]

        if entity_type:
            query += " AND e.entity_type = ?"
            params.append(entity_type.value)

        cursor = await self._connection.execute(query, params)
        row = await cursor.fetchone()

        if row:
            entity = self._row_to_entity(row)
            await self._load_entity_embeddings(entity)
            return entity

        return None

    async def update_entity(self, entity: Entity) -> bool:
        """Update an entity."""
        entity.updated_at = datetime.now()
        entity.version += 1

        result = await self._connection.execute("""
            UPDATE entities SET
                name = ?, entity_type = ?, custom_type = ?,
                description = ?, aliases_json = ?, properties_json = ?,
                provenance_json = ?, confidence = ?, importance = ?,
                degree_centrality = ?, betweenness_centrality = ?, pagerank = ?,
                version = ?, content_hash = ?, updated_at = ?
            WHERE id = ? AND deleted_at IS NULL
        """, (
            entity.name,
            entity.entity_type.value,
            entity.custom_type,
            entity.description,
            json.dumps(entity.aliases),
            json.dumps(entity.properties),
            json.dumps(entity.provenance.to_dict()) if entity.provenance else None,
            entity.confidence,
            entity.importance,
            entity.degree_centrality,
            entity.betweenness_centrality,
            entity.pagerank,
            entity.version,
            entity.content_hash,
            entity.updated_at.isoformat(),
            entity.id,
        ))

        # Update aliases
        await self._connection.execute(
            "DELETE FROM entity_aliases WHERE entity_id = ?",
            (entity.id,),
        )
        for alias in entity.aliases:
            await self._connection.execute(
                "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                (entity.id, alias.lower()),
            )
        await self._connection.execute(
            "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
            (entity.id, entity.name.lower()),
        )

        await self._connection.commit()

        return result.rowcount > 0

    async def delete_entity(self, entity_id: str, soft: bool = True) -> bool:
        """Delete an entity."""
        if soft:
            result = await self._connection.execute(
                "UPDATE entities SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), entity_id),
            )
        else:
            # Hard delete - cascades to aliases, embeddings, relationships
            await self._connection.execute(
                "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            )
            result = await self._connection.execute(
                "DELETE FROM entities WHERE id = ?",
                (entity_id,),
            )

        await self._connection.commit()

        return result.rowcount > 0

    async def search_entities(
        self,
        query: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entity]:
        """Search for entities with optional full-text search."""
        if query and self.enable_fts:
            # Use FTS5 for text search
            sql = """
                SELECT e.* FROM entities e
                JOIN entity_fts f ON e.rowid = f.rowid
                WHERE entity_fts MATCH ? AND e.deleted_at IS NULL
            """
            # Escape FTS query
            fts_query = query.replace('"', '""')
            params: list = [f'"{fts_query}"*']

            if entity_type:
                sql += " AND e.entity_type = ?"
                params.append(entity_type.value)

            sql += " ORDER BY rank LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        else:
            sql = "SELECT * FROM entities WHERE deleted_at IS NULL"
            params = []

            if query:
                sql += " AND (name LIKE ? OR description LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])

            if entity_type:
                sql += " AND entity_type = ?"
                params.append(entity_type.value)

            if properties:
                for key, value in properties.items():
                    sql += " AND json_extract(properties_json, ?) = ?"
                    params.extend([f"$.{key}", json.dumps(value)])

            sql += " ORDER BY importance DESC, name LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()

        entities = [self._row_to_entity(row) for row in rows]
        return entities

    async def count_entities(self, entity_type: Optional[EntityType] = None) -> int:
        """Count entities efficiently."""
        sql = "SELECT COUNT(*) FROM entities WHERE deleted_at IS NULL"
        params = []

        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type.value)

        cursor = await self._connection.execute(sql, params)
        row = await cursor.fetchone()
        return row[0] if row else 0

    # ==========================================================================
    # Relationship Operations
    # ==========================================================================

    async def create_relationship(self, rel: Relationship) -> str:
        """Create a new relationship."""
        now = datetime.now().isoformat()

        await self._connection.execute("""
            INSERT INTO relationships (
                id, source_id, target_id, relation_type, custom_type,
                properties_json, weight, confidence, bidirectional, transitive,
                provenance_json, valid_from, valid_until, version,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rel.id,
            rel.source_id,
            rel.target_id,
            rel.relation_type.value,
            rel.custom_type,
            json.dumps(rel.properties),
            rel.weight,
            rel.confidence,
            1 if rel.bidirectional else 0,
            1 if rel.transitive else 0,
            json.dumps(rel.provenance.to_dict()) if rel.provenance else None,
            rel.valid_from.isoformat() if rel.valid_from else None,
            rel.valid_until.isoformat() if rel.valid_until else None,
            rel.version,
            rel.created_at.isoformat() if isinstance(rel.created_at, datetime) else now,
            now,
        ))

        await self._connection.commit()

        logger.debug(f"Created relationship: {rel.source_id} -[{rel.relation_type.value}]-> {rel.target_id}")
        return rel.id

    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM relationships WHERE id = ?",
            (rel_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_relationship(row)

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[RelationType] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        relationships = []

        if direction in ("outgoing", "both"):
            sql = "SELECT * FROM relationships WHERE source_id = ?"
            params: list = [entity_id]

            if relation_type:
                sql += " AND relation_type = ?"
                params.append(relation_type.value)

            cursor = await self._connection.execute(sql, params)
            rows = await cursor.fetchall()
            relationships.extend([self._row_to_relationship(row) for row in rows])

        if direction in ("incoming", "both"):
            sql = "SELECT * FROM relationships WHERE target_id = ?"
            params = [entity_id]

            if relation_type:
                sql += " AND relation_type = ?"
                params.append(relation_type.value)

            cursor = await self._connection.execute(sql, params)
            rows = await cursor.fetchall()

            # Don't add duplicates for bidirectional
            seen = {r.id for r in relationships}
            for row in rows:
                rel = self._row_to_relationship(row)
                if rel.id not in seen:
                    relationships.append(rel)

        return relationships

    async def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship."""
        result = await self._connection.execute(
            "DELETE FROM relationships WHERE id = ?",
            (rel_id,),
        )
        await self._connection.commit()

        return result.rowcount > 0

    async def get_relationship_between(
        self,
        source_id: str,
        target_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> Optional[Relationship]:
        """Get relationship between two specific entities."""
        sql = "SELECT * FROM relationships WHERE source_id = ? AND target_id = ?"
        params: list = [source_id, target_id]

        if relation_type:
            sql += " AND relation_type = ?"
            params.append(relation_type.value)

        cursor = await self._connection.execute(sql, params)
        row = await cursor.fetchone()

        if row:
            return self._row_to_relationship(row)

        # Check reverse for bidirectional
        sql = """
            SELECT * FROM relationships
            WHERE target_id = ? AND source_id = ? AND bidirectional = 1
        """
        params = [source_id, target_id]

        if relation_type:
            sql += " AND relation_type = ?"
            params.append(relation_type.value)

        cursor = await self._connection.execute(sql, params)
        row = await cursor.fetchone()

        return self._row_to_relationship(row) if row else None

    # ==========================================================================
    # Graph Traversal
    # ==========================================================================

    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        depth: int = 1,
    ) -> Subgraph:
        """Get neighboring entities using BFS."""
        subgraph = Subgraph()
        visited = set()

        async def traverse(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited:
                return

            visited.add(current_id)

            entity = await self.get_entity(current_id)
            if entity:
                subgraph.add_entity(entity)

            # Get relationships
            relationships = await self.get_relationships(current_id, direction="both")

            for rel in relationships:
                # Filter by type
                if relation_types and rel.relation_type not in relation_types:
                    continue

                subgraph.add_relationship(rel)

                # Traverse to neighbor
                next_id = rel.target_id if rel.source_id == current_id else rel.source_id
                await traverse(next_id, current_depth + 1)

        await traverse(entity_id, 0)

        return subgraph

    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Optional[GraphPath]:
        """Find shortest path using bidirectional BFS."""
        if start_id == end_id:
            entity = await self.get_entity(start_id)
            return GraphPath(entities=[entity] if entity else [], relationships=[])

        # Bidirectional BFS for efficiency
        forward_visited = {start_id: (None, None)}  # id -> (prev_id, rel)
        backward_visited = {end_id: (None, None)}

        forward_queue = deque([start_id])
        backward_queue = deque([end_id])

        forward_depth = 0
        backward_depth = 0

        meeting_point = None

        while forward_queue or backward_queue:
            # Expand forward
            if forward_queue and forward_depth <= max_depth // 2:
                level_size = len(forward_queue)
                for _ in range(level_size):
                    current = forward_queue.popleft()

                    rels = await self.get_relationships(current, direction="both")

                    for rel in rels:
                        if relation_types and rel.relation_type not in relation_types:
                            continue

                        next_id = rel.target_id if rel.source_id == current else rel.source_id

                        if next_id in backward_visited:
                            meeting_point = next_id
                            forward_visited[next_id] = (current, rel)
                            break

                        if next_id not in forward_visited:
                            forward_visited[next_id] = (current, rel)
                            forward_queue.append(next_id)

                    if meeting_point:
                        break

                forward_depth += 1

            if meeting_point:
                break

            # Expand backward
            if backward_queue and backward_depth <= max_depth // 2:
                level_size = len(backward_queue)
                for _ in range(level_size):
                    current = backward_queue.popleft()

                    rels = await self.get_relationships(current, direction="both")

                    for rel in rels:
                        if relation_types and rel.relation_type not in relation_types:
                            continue

                        next_id = rel.target_id if rel.source_id == current else rel.source_id

                        if next_id in forward_visited:
                            meeting_point = next_id
                            backward_visited[next_id] = (current, rel)
                            break

                        if next_id not in backward_visited:
                            backward_visited[next_id] = (current, rel)
                            backward_queue.append(next_id)

                    if meeting_point:
                        break

                backward_depth += 1

            if meeting_point:
                break

            if forward_depth + backward_depth > max_depth:
                break

        if not meeting_point:
            return None

        # Reconstruct path
        # Forward path: start -> meeting_point
        forward_path_ids = []
        forward_rels = []
        current = meeting_point
        while current != start_id:
            prev, rel = forward_visited[current]
            forward_path_ids.append(current)
            if rel:
                forward_rels.append(rel)
            current = prev

        forward_path_ids.append(start_id)
        forward_path_ids.reverse()
        forward_rels.reverse()

        # Backward path: meeting_point -> end
        backward_path_ids = []
        backward_rels = []
        current = meeting_point
        while current != end_id:
            prev, rel = backward_visited[current]
            if prev:
                backward_path_ids.append(prev)
                if rel:
                    backward_rels.append(rel)
            current = prev

        # Combine paths
        all_entity_ids = forward_path_ids + backward_path_ids
        all_rels = forward_rels + backward_rels

        # Load entities
        entities = []
        for eid in all_entity_ids:
            entity = await self.get_entity(eid)
            if entity:
                entities.append(entity)

        path = GraphPath(entities=entities, relationships=all_rels)
        path.compute_metrics()

        return path

    # ==========================================================================
    # Embedding Operations
    # ==========================================================================

    async def _save_embedding(
        self,
        entity_id: str,
        embedding: List[float],
        embedding_type: str,
        model_id: Optional[str],
    ) -> None:
        """Save entity embedding as BLOB."""
        embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)

        await self._connection.execute("""
            INSERT OR REPLACE INTO entity_embeddings
            (entity_id, embedding_type, embedding, dimension, model_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entity_id,
            embedding_type,
            embedding_bytes,
            len(embedding),
            model_id,
            datetime.now().isoformat(),
        ))

    async def _load_entity_embeddings(self, entity: Entity) -> None:
        """Load embeddings for an entity."""
        cursor = await self._connection.execute(
            "SELECT * FROM entity_embeddings WHERE entity_id = ?",
            (entity.id,),
        )
        rows = await cursor.fetchall()

        for row in rows:
            embedding_type = row["embedding_type"]
            dimension = row["dimension"]
            embedding_bytes = row["embedding"]
            model_id = row["model_id"]

            vector = list(struct.unpack(f'{dimension}f', embedding_bytes))

            graph_emb = GraphEmbedding(
                embedding_type=embedding_type,
                vector=vector,
                dimension=dimension,
                model_id=model_id,
            )

            if embedding_type == "text":
                entity.text_embedding = graph_emb
            elif embedding_type == "graph":
                entity.graph_embedding = graph_emb

    async def save_embedding(
        self,
        entity_id: str,
        embedding: List[float],
        embedding_type: str = "text",
        model_id: Optional[str] = None,
    ) -> None:
        """Public method to save an embedding."""
        await self._save_embedding(entity_id, embedding, embedding_type, model_id)
        await self._connection.commit()

    async def get_embedding(
        self,
        entity_id: str,
        embedding_type: str = "text",
    ) -> Optional[List[float]]:
        """Get an embedding for an entity."""
        cursor = await self._connection.execute(
            "SELECT embedding, dimension FROM entity_embeddings WHERE entity_id = ? AND embedding_type = ?",
            (entity_id, embedding_type),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        embedding_bytes = row["embedding"]
        dimension = row["dimension"]

        return list(struct.unpack(f'{dimension}f', embedding_bytes))

    async def find_similar_by_embedding(
        self,
        embedding: List[float],
        embedding_type: str = "text",
        limit: int = 10,
        threshold: float = 0.7,
    ) -> List[Tuple[Entity, float]]:
        """Find entities similar to a given embedding using cosine similarity."""
        import numpy as np

        query_vec = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        # Get all embeddings of the specified type
        cursor = await self._connection.execute(
            "SELECT entity_id, embedding, dimension FROM entity_embeddings WHERE embedding_type = ?",
            (embedding_type,),
        )
        rows = await cursor.fetchall()

        results = []

        for row in rows:
            entity_id = row["entity_id"]
            dimension = row["dimension"]
            embedding_bytes = row["embedding"]

            entity_vec = np.array(
                struct.unpack(f'{dimension}f', embedding_bytes),
                dtype=np.float32
            )
            entity_norm = np.linalg.norm(entity_vec)

            if entity_norm > 0:
                similarity = float(np.dot(query_vec, entity_vec) / (query_norm * entity_norm))
                if similarity >= threshold:
                    entity = await self.get_entity(entity_id)
                    if entity:
                        results.append((entity, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_stats(self) -> GraphStatistics:
        """Get comprehensive graph statistics."""
        stats = GraphStatistics()

        # Entity counts
        cursor = await self._connection.execute(
            "SELECT COUNT(*) FROM entities WHERE deleted_at IS NULL"
        )
        stats.total_entities = (await cursor.fetchone())[0]

        cursor = await self._connection.execute(
            "SELECT COUNT(*) FROM relationships"
        )
        stats.total_relationships = (await cursor.fetchone())[0]

        # By type
        cursor = await self._connection.execute("""
            SELECT entity_type, COUNT(*) as count FROM entities
            WHERE deleted_at IS NULL GROUP BY entity_type
        """)
        stats.entities_by_type = {row["entity_type"]: row["count"] for row in await cursor.fetchall()}

        cursor = await self._connection.execute("""
            SELECT relation_type, COUNT(*) as count FROM relationships
            GROUP BY relation_type
        """)
        stats.relationships_by_type = {row["relation_type"]: row["count"] for row in await cursor.fetchall()}

        # Degree statistics
        if stats.total_entities > 0:
            cursor = await self._connection.execute("""
                SELECT e.id, COUNT(r.id) as degree
                FROM entities e
                LEFT JOIN relationships r ON e.id = r.source_id OR e.id = r.target_id
                WHERE e.deleted_at IS NULL
                GROUP BY e.id
            """)
            degrees = [row["degree"] for row in await cursor.fetchall()]

            if degrees:
                stats.avg_degree = sum(degrees) / len(degrees)
                stats.max_degree = max(degrees)

            # Graph density
            n = stats.total_entities
            if n > 1:
                max_edges = n * (n - 1)
                stats.density = stats.total_relationships / max_edges

        # Temporal bounds
        cursor = await self._connection.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM entities WHERE deleted_at IS NULL"
        )
        row = await cursor.fetchone()
        if row[0]:
            stats.oldest_entity = datetime.fromisoformat(row[0])
        if row[1]:
            stats.newest_entity = datetime.fromisoformat(row[1])

        stats.computed_at = datetime.now()

        return stats

    # ==========================================================================
    # Bulk Operations
    # ==========================================================================

    async def bulk_create_entities(self, entities: List[Entity]) -> List[str]:
        """Create multiple entities efficiently in a single transaction."""
        ids = []
        now = datetime.now().isoformat()

        await self._connection.execute("BEGIN TRANSACTION")
        try:
            for entity in entities:
                await self._connection.execute("""
                    INSERT INTO entities (
                        id, name, entity_type, custom_type, description,
                        aliases_json, properties_json, confidence, importance,
                        version, content_hash, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.id,
                    entity.name,
                    entity.entity_type.value,
                    entity.custom_type,
                    entity.description,
                    json.dumps(entity.aliases),
                    json.dumps(entity.properties),
                    entity.confidence,
                    entity.importance,
                    entity.version,
                    entity.content_hash,
                    now,
                    now,
                ))
                ids.append(entity.id)

                # Add aliases
                for alias in entity.aliases + [entity.name]:
                    await self._connection.execute(
                        "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                        (entity.id, alias.lower()),
                    )

            await self._connection.execute("COMMIT")
        except Exception as e:
            await self._connection.execute("ROLLBACK")
            raise e

        return ids

    async def bulk_create_relationships(self, relationships: List[Relationship]) -> List[str]:
        """Create multiple relationships efficiently."""
        ids = []
        now = datetime.now().isoformat()

        await self._connection.execute("BEGIN TRANSACTION")
        try:
            for rel in relationships:
                await self._connection.execute("""
                    INSERT INTO relationships (
                        id, source_id, target_id, relation_type, custom_type,
                        properties_json, weight, confidence, bidirectional, transitive,
                        valid_from, valid_until, version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel.id,
                    rel.source_id,
                    rel.target_id,
                    rel.relation_type.value,
                    rel.custom_type,
                    json.dumps(rel.properties),
                    rel.weight,
                    rel.confidence,
                    1 if rel.bidirectional else 0,
                    1 if rel.transitive else 0,
                    rel.valid_from.isoformat() if rel.valid_from else None,
                    rel.valid_until.isoformat() if rel.valid_until else None,
                    rel.version,
                    now,
                    now,
                ))
                ids.append(rel.id)

            await self._connection.execute("COMMIT")
        except Exception as e:
            await self._connection.execute("ROLLBACK")
            raise e

        return ids

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _row_to_entity(self, row) -> Entity:
        """Convert database row to Entity."""
        aliases = json.loads(row["aliases_json"]) if row["aliases_json"] else []
        properties = json.loads(row["properties_json"]) if row["properties_json"] else {}

        provenance = None
        if row["provenance_json"]:
            prov_data = json.loads(row["provenance_json"])
            provenance = Provenance(
                source_type=prov_data.get("source_type", "manual"),
                source_id=prov_data.get("source_id"),
                extraction_confidence=prov_data.get("extraction_confidence", 1.0),
            )

        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            custom_type=row["custom_type"],
            description=row["description"] or "",
            aliases=aliases,
            properties=properties,
            provenance=provenance,
            confidence=row["confidence"],
            importance=row["importance"],
            degree_centrality=row["degree_centrality"],
            betweenness_centrality=row["betweenness_centrality"],
            pagerank=row["pagerank"],
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now(),
            deleted_at=datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
        )

    def _row_to_relationship(self, row) -> Relationship:
        """Convert database row to Relationship."""
        properties = json.loads(row["properties_json"]) if row["properties_json"] else {}

        provenance = None
        if row["provenance_json"]:
            prov_data = json.loads(row["provenance_json"])
            provenance = Provenance(
                source_type=prov_data.get("source_type", "manual"),
                source_id=prov_data.get("source_id"),
                extraction_confidence=prov_data.get("extraction_confidence", 1.0),
            )

        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=RelationType(row["relation_type"]),
            custom_type=row["custom_type"],
            properties=properties,
            weight=row["weight"],
            confidence=row["confidence"],
            bidirectional=bool(row["bidirectional"]),
            transitive=bool(row["transitive"]),
            provenance=provenance,
            valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now(),
        )
