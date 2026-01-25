"""
AION Initial Schema Migration

Creates all tables required for the AION persistence layer:
- Memories with embeddings support
- Plans with graph structure
- Processes and tasks
- Events for replay
- Evolution checkpoints
- Tool executions
- Configuration storage
- Sessions tracking
"""

from aion.persistence.migrations.runner import Migration


class InitialSchema(Migration):
    """Initial database schema for AION."""

    version = "001"
    name = "initial_schema"
    description = "Create all AION persistence tables"

    async def up(self, connection) -> None:
        """Create the initial schema."""

        # =================================================================
        # MEMORIES
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                emotional_valence REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                decay_rate REAL DEFAULT 0.01,
                source TEXT,
                context TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP,
                version INTEGER DEFAULT 1
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted_at)
        """)

        # Embeddings stored separately for efficiency
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                embedding_model TEXT DEFAULT 'default',
                dimensions INTEGER NOT NULL,
                compressed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Memory relations for graph structure
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_relations (
                source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, target_id, relation_type)
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_relations_target
            ON memory_relations(target_id)
        """)

        # FAISS index storage
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS faiss_indices (
                name TEXT PRIMARY KEY,
                index_data BLOB NOT NULL,
                id_mapping TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                index_type TEXT DEFAULT 'flat',
                num_vectors INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # =================================================================
        # PLANS
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                nodes TEXT NOT NULL,
                edges TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                deleted_at TIMESTAMP,
                version INTEGER DEFAULT 1
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_plans_priority ON plans(priority DESC)
        """)

        # Plan checkpoints for rollback
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS plan_checkpoints (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
                checkpoint_name TEXT NOT NULL,
                nodes TEXT NOT NULL,
                edges TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_plan_checkpoints_plan
            ON plan_checkpoints(plan_id, created_at DESC)
        """)

        # =================================================================
        # PROCESSES
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS processes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                process_type TEXT NOT NULL,
                state TEXT DEFAULT 'created',
                priority INTEGER DEFAULT 0,
                cpu_affinity TEXT,
                memory_limit INTEGER,
                config TEXT,
                result TEXT,
                error TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP,
                version INTEGER DEFAULT 1
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_processes_state ON processes(state)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_processes_type ON processes(process_type)
        """)

        # =================================================================
        # TASKS
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                handler TEXT NOT NULL,
                schedule TEXT,
                cron_expression TEXT,
                interval_seconds INTEGER,
                next_run_at TIMESTAMP,
                last_run_at TIMESTAMP,
                last_run_status TEXT,
                run_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                timeout_seconds INTEGER DEFAULT 300,
                enabled INTEGER DEFAULT 1,
                config TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP,
                version INTEGER DEFAULT 1
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON tasks(next_run_at)
            WHERE enabled = 1 AND deleted_at IS NULL
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_enabled ON tasks(enabled)
        """)

        # =================================================================
        # EVENTS
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT NOT NULL,
                correlation_id TEXT,
                causation_id TEXT,
                sequence_number INTEGER,
                processed INTEGER DEFAULT 0,
                processed_at TIMESTAMP,
                error TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_source ON events(source)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_unprocessed ON events(processed)
            WHERE processed = 0
        """)

        # =================================================================
        # EVOLUTION
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS evolution_checkpoints (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                generation INTEGER NOT NULL,
                fitness_score REAL,
                performance_snapshot TEXT,
                active_hypotheses TEXT,
                successful_adaptations TEXT,
                configuration TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_at TIMESTAMP,
                version INTEGER DEFAULT 1
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_evolution_generation
            ON evolution_checkpoints(generation DESC)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_evolution_fitness
            ON evolution_checkpoints(fitness_score DESC)
        """)

        # Hypothesis tracking
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                checkpoint_id TEXT REFERENCES evolution_checkpoints(id) ON DELETE CASCADE,
                hypothesis_type TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 0,
                test_count INTEGER DEFAULT 0,
                success_rate REAL,
                status TEXT DEFAULT 'active',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_hypotheses_checkpoint
            ON hypotheses(checkpoint_id)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status)
        """)

        # =================================================================
        # TOOLS
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS tool_executions (
                id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                tool_version TEXT,
                input_hash TEXT NOT NULL,
                input_data TEXT NOT NULL,
                output_data TEXT,
                success INTEGER NOT NULL,
                error TEXT,
                execution_time_ms REAL,
                memory_used_bytes INTEGER,
                context_id TEXT,
                user_id TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_executions_name
            ON tool_executions(tool_name)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_executions_hash
            ON tool_executions(input_hash)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_executions_context
            ON tool_executions(context_id)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_executions_created
            ON tool_executions(created_at DESC)
        """)

        # Tool patterns for learning
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS tool_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                tool_sequence TEXT NOT NULL,
                context_pattern TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                avg_execution_time_ms REAL,
                confidence REAL DEFAULT 0.5,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_patterns_type
            ON tool_patterns(pattern_type)
        """)

        # =================================================================
        # CONFIGURATION
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS config_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                value_type TEXT DEFAULT 'string',
                encrypted INTEGER DEFAULT 0,
                description TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, key)
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_config_namespace ON config_entries(namespace)
        """)

        # =================================================================
        # SESSIONS
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_type TEXT DEFAULT 'interactive',
                state TEXT DEFAULT 'active',
                context TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                metadata TEXT
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_state ON sessions(state)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_activity
            ON sessions(last_activity_at DESC)
        """)

        # =================================================================
        # SYSTEM METADATA
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert schema version
        await connection.execute("""
            INSERT OR REPLACE INTO system_metadata (key, value, updated_at)
            VALUES ('schema_version', '001', CURRENT_TIMESTAMP)
        """)

        # =================================================================
        # CHANGE DATA CAPTURE (CDC)
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS cdc_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                operation TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                old_data TEXT,
                new_data TEXT,
                changed_fields TEXT,
                transaction_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed INTEGER DEFAULT 0,
                processed_at TIMESTAMP
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_cdc_events_table
            ON cdc_events(table_name, created_at DESC)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_cdc_events_entity
            ON cdc_events(entity_id)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_cdc_events_unprocessed
            ON cdc_events(processed) WHERE processed = 0
        """)

        # =================================================================
        # SNAPSHOTS
        # =================================================================

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                snapshot_name TEXT NOT NULL UNIQUE,
                snapshot_type TEXT DEFAULT 'full',
                tables_included TEXT NOT NULL,
                data_path TEXT NOT NULL,
                size_bytes INTEGER,
                checksum TEXT,
                compressed INTEGER DEFAULT 1,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_name ON snapshots(snapshot_name)
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_created ON snapshots(created_at DESC)
        """)

    async def down(self, connection) -> None:
        """Drop all tables in reverse order."""

        tables = [
            "snapshots",
            "cdc_events",
            "system_metadata",
            "sessions",
            "config_entries",
            "tool_patterns",
            "tool_executions",
            "hypotheses",
            "evolution_checkpoints",
            "events",
            "tasks",
            "processes",
            "plan_checkpoints",
            "plans",
            "faiss_indices",
            "memory_relations",
            "memory_embeddings",
            "memories",
        ]

        for table in tables:
            await connection.execute(f"DROP TABLE IF EXISTS {table}")
