"""Episodic memory: stores and retrieves past execution traces."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Episode(BaseModel):
    """A recorded execution episode with full trace."""

    episode_id: str
    task_description: str
    task_embedding: list[float] = []
    agent_trace: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    outcome: str = "unknown"  # success, failure, partial
    duration_seconds: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = {}


class EpisodicMemory:
    """
    Stores complete execution episodes and retrieves similar past experiences.

    Uses vector similarity search over task embeddings to find relevant
    prior executions, enabling the system to learn from past attempts
    at similar problems.

    Backend: PostgreSQL with pgvector for embedding storage and retrieval.
    """

    def __init__(
        self,
        db_pool: Any,
        embedding_client: Any,
        table_name: str = "episodic_memory",
        embedding_dim: int = 1536,
        ttl_days: int = 30,
    ) -> None:
        self.db_pool = db_pool
        self.embedding_client = embedding_client
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.ttl_days = ttl_days

    async def initialize(self) -> None:
        """Create the episodic memory table and index if they don't exist."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    episode_id TEXT PRIMARY KEY,
                    task_description TEXT NOT NULL,
                    task_embedding vector({self.embedding_dim}),
                    agent_trace JSONB DEFAULT '[]'::jsonb,
                    artifacts JSONB DEFAULT '{{}}'::jsonb,
                    outcome TEXT DEFAULT 'unknown',
                    duration_seconds REAL DEFAULT 0.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'::jsonb
                );
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                ON {self.table_name}
                USING ivfflat (task_embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

    async def store(self, episode: Episode) -> None:
        """Store a completed episode in memory."""
        if not episode.task_embedding:
            episode.task_embedding = await self._embed(episode.task_description)

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name}
                    (episode_id, task_description, task_embedding, agent_trace,
                     artifacts, outcome, duration_seconds, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (episode_id) DO UPDATE SET
                    agent_trace = EXCLUDED.agent_trace,
                    artifacts = EXCLUDED.artifacts,
                    outcome = EXCLUDED.outcome,
                    duration_seconds = EXCLUDED.duration_seconds
                """,
                episode.episode_id,
                episode.task_description,
                episode.task_embedding,
                episode.agent_trace,
                episode.artifacts,
                episode.outcome,
                episode.duration_seconds,
                episode.metadata,
            )

    async def retrieve(self, query: str, k: int = 5, min_similarity: float = 0.6) -> list[Episode]:
        """
        Retrieve the most similar episodes to the given query.

        Uses cosine similarity over task embeddings, filtered by minimum
        similarity threshold and TTL.
        """
        query_embedding = await self._embed(query)

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    episode_id, task_description, agent_trace, artifacts,
                    outcome, duration_seconds, metadata,
                    1 - (task_embedding <=> $1::vector) AS similarity
                FROM {self.table_name}
                WHERE created_at > NOW() - INTERVAL '{self.ttl_days} days'
                    AND 1 - (task_embedding <=> $1::vector) > $2
                ORDER BY task_embedding <=> $1::vector
                LIMIT $3
                """,
                query_embedding,
                min_similarity,
                k,
            )

        episodes = []
        for row in rows:
            episodes.append(Episode(
                episode_id=row["episode_id"],
                task_description=row["task_description"],
                agent_trace=row["agent_trace"],
                artifacts=row["artifacts"],
                outcome=row["outcome"],
                duration_seconds=row["duration_seconds"],
                metadata=row["metadata"],
            ))

        return episodes

    async def retrieve_successful(self, query: str, k: int = 3) -> list[Episode]:
        """Retrieve only successful episodes similar to the query."""
        query_embedding = await self._embed(query)

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    episode_id, task_description, agent_trace, artifacts,
                    outcome, duration_seconds, metadata
                FROM {self.table_name}
                WHERE outcome = 'success'
                    AND created_at > NOW() - INTERVAL '{self.ttl_days} days'
                ORDER BY task_embedding <=> $1::vector
                LIMIT $2
                """,
                query_embedding,
                k,
            )

        return [
            Episode(
                episode_id=row["episode_id"],
                task_description=row["task_description"],
                agent_trace=row["agent_trace"],
                artifacts=row["artifacts"],
                outcome=row["outcome"],
                duration_seconds=row["duration_seconds"],
                metadata=row["metadata"],
            )
            for row in rows
        ]

    async def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics for monitoring."""
        async with self.db_pool.acquire() as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name}")
            successful = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE outcome = 'success'"
            )
            avg_duration = await conn.fetchval(
                f"SELECT AVG(duration_seconds) FROM {self.table_name}"
            )

        return {
            "total_episodes": total or 0,
            "successful_episodes": successful or 0,
            "success_rate": (successful or 0) / max(total or 1, 1),
            "avg_duration_seconds": avg_duration or 0.0,
        }

    async def prune_expired(self) -> int:
        """Remove episodes older than TTL. Returns count of removed episodes."""
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE created_at < NOW() - INTERVAL '{self.ttl_days} days'
                """
            )
        # asyncpg returns "DELETE N"
        return int(result.split()[-1]) if result else 0

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding for the given text."""
        response = await self.embedding_client.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(dot / norm)
