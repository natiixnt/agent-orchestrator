"""Procedural memory: stores and retrieves learned task decomposition strategies."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class Strategy(BaseModel):
    """A learned decomposition strategy with performance history."""

    strategy_id: str
    task_pattern: str
    subtasks: list[dict[str, Any]]
    success_rate: float = 0.0
    total_uses: int = 0
    avg_duration: float = 0.0
    embedding: list[float] = []
    conditions: dict[str, Any] = {}  # When to apply this strategy
    created_at: float = Field(default_factory=time.time)
    last_used: float = Field(default_factory=time.time)


class ProceduralMemory:
    """
    Stores learned task decomposition strategies and agent configurations
    that have proven effective for specific task patterns.

    Strategies are ranked by success rate and recency, with decay applied
    to strategies that haven't been used recently. New strategies are
    formed by generalizing successful episodes.

    Backend: PostgreSQL with pgvector for strategy matching.
    """

    def __init__(
        self,
        db_pool: Any,
        embedding_client: Any,
        table_name: str = "procedural_memory",
        embedding_dim: int = 1536,
        decay_rate: float = 0.95,
        min_uses_for_confidence: int = 3,
    ) -> None:
        self.db_pool = db_pool
        self.embedding_client = embedding_client
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.min_uses_for_confidence = min_uses_for_confidence

    async def initialize(self) -> None:
        """Create procedural memory tables."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    strategy_id TEXT PRIMARY KEY,
                    task_pattern TEXT NOT NULL,
                    subtasks JSONB NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    total_uses INTEGER DEFAULT 0,
                    avg_duration REAL DEFAULT 0.0,
                    embedding vector({self.embedding_dim}),
                    conditions JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_used TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50);
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_success
                ON {self.table_name} (success_rate DESC);
            """)

    async def store_strategy(self, strategy: Strategy) -> None:
        """Store a new or updated strategy."""
        if not strategy.embedding:
            strategy.embedding = await self._embed(strategy.task_pattern)

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name}
                    (strategy_id, task_pattern, subtasks, success_rate,
                     total_uses, avg_duration, embedding, conditions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (strategy_id) DO UPDATE SET
                    success_rate = EXCLUDED.success_rate,
                    total_uses = EXCLUDED.total_uses,
                    avg_duration = EXCLUDED.avg_duration,
                    last_used = NOW()
                """,
                strategy.strategy_id,
                strategy.task_pattern,
                strategy.subtasks,
                strategy.success_rate,
                strategy.total_uses,
                strategy.avg_duration,
                strategy.embedding,
                strategy.conditions,
            )

    async def retrieve_similar_strategies(
        self, task_description: str, k: int = 5, min_success_rate: float = 0.5
    ) -> list[Strategy]:
        """
        Find strategies that worked for similar tasks in the past.

        Results are ranked by a composite score combining:
        - Vector similarity to the current task
        - Historical success rate
        - Recency (with decay)
        - Usage count (confidence)
        """
        query_embedding = await self._embed(task_description)

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    strategy_id, task_pattern, subtasks, success_rate,
                    total_uses, avg_duration, conditions, last_used,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM {self.table_name}
                WHERE success_rate >= $2
                    AND total_uses >= $4
                ORDER BY
                    (1 - (embedding <=> $1::vector)) * success_rate *
                    POWER($5, EXTRACT(EPOCH FROM (NOW() - last_used)) / 86400.0)
                DESC
                LIMIT $3
                """,
                query_embedding,
                min_success_rate,
                k,
                self.min_uses_for_confidence,
                self.decay_rate,
            )

        return [
            Strategy(
                strategy_id=row["strategy_id"],
                task_pattern=row["task_pattern"],
                subtasks=row["subtasks"],
                success_rate=row["success_rate"],
                total_uses=row["total_uses"],
                avg_duration=row["avg_duration"],
                conditions=row["conditions"],
                last_used=row["last_used"].timestamp(),
            )
            for row in rows
        ]

    async def update_outcome(
        self, strategy_id: str, success: bool, duration: float
    ) -> None:
        """
        Update a strategy's statistics after use.

        Uses exponential moving average for success rate to weight
        recent outcomes more heavily.
        """
        async with self.db_pool.acquire() as conn:
            current = await conn.fetchrow(
                f"SELECT success_rate, total_uses, avg_duration FROM {self.table_name} WHERE strategy_id = $1",
                strategy_id,
            )
            if not current:
                return

            alpha = 0.3  # EMA smoothing factor
            new_success_rate = (
                alpha * (1.0 if success else 0.0) + (1 - alpha) * current["success_rate"]
            )
            new_total = current["total_uses"] + 1
            new_avg_duration = (
                current["avg_duration"] * current["total_uses"] + duration
            ) / new_total

            await conn.execute(
                f"""
                UPDATE {self.table_name}
                SET success_rate = $2, total_uses = $3, avg_duration = $4, last_used = NOW()
                WHERE strategy_id = $1
                """,
                strategy_id,
                new_success_rate,
                new_total,
                new_avg_duration,
            )

    async def generalize_from_episodes(
        self, episodes: list[dict[str, Any]], task_pattern: str
    ) -> Strategy | None:
        """
        Create a new strategy by generalizing from successful episodes.

        Identifies common subtask patterns across multiple successful
        executions and creates a reusable strategy template.
        """
        if len(episodes) < 2:
            return None

        # Extract subtask sequences from episodes
        all_sequences = []
        for ep in episodes:
            if ep.get("outcome") == "success" and ep.get("agent_trace"):
                sequence = [
                    {"agent_type": step.get("agent"), "description": step.get("action", "")}
                    for step in ep["agent_trace"]
                    if step.get("agent")
                ]
                all_sequences.append(sequence)

        if not all_sequences:
            return None

        # Find common agent type patterns
        common_types = self._find_common_sequence(
            [[s["agent_type"] for s in seq] for seq in all_sequences]
        )

        # Build generalized subtask list
        subtasks = []
        for agent_type in common_types:
            descriptions = [
                s["description"]
                for seq in all_sequences
                for s in seq
                if s["agent_type"] == agent_type
            ]
            generalized_desc = descriptions[0] if descriptions else f"Execute {agent_type} phase"
            subtasks.append({
                "agent_type": agent_type,
                "description": generalized_desc,
                "required": True,
            })

        import hashlib
        strategy_id = hashlib.sha256(f"{task_pattern}:{common_types}".encode()).hexdigest()[:12]

        success_count = sum(1 for ep in episodes if ep.get("outcome") == "success")

        strategy = Strategy(
            strategy_id=strategy_id,
            task_pattern=task_pattern,
            subtasks=subtasks,
            success_rate=success_count / len(episodes),
            total_uses=len(episodes),
            avg_duration=sum(ep.get("duration", 0) for ep in episodes) / len(episodes),
        )

        await self.store_strategy(strategy)
        return strategy

    @staticmethod
    def _find_common_sequence(sequences: list[list[str]]) -> list[str]:
        """Find the longest common subsequence pattern across multiple sequences."""
        if not sequences:
            return []

        # Use frequency-based ordering for common elements
        from collections import Counter
        all_elements = [elem for seq in sequences for elem in seq]
        counts = Counter(all_elements)

        # Keep elements that appear in at least half the sequences
        threshold = len(sequences) / 2
        common = [elem for elem, count in counts.most_common() if count >= threshold]

        # Preserve typical ordering based on first occurrence across sequences
        order_scores: dict[str, float] = {}
        for elem in common:
            positions = []
            for seq in sequences:
                if elem in seq:
                    positions.append(seq.index(elem) / max(len(seq), 1))
            order_scores[elem] = sum(positions) / len(positions) if positions else 0

        return sorted(set(common), key=lambda x: order_scores.get(x, 0))

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = await self.embedding_client.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
