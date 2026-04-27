"""Semantic memory: knowledge base with vector retrieval over code and documentation."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel


class SemanticEntry(BaseModel):
    """A semantic memory entry representing a code pattern or knowledge fragment."""

    entry_id: str
    content: str
    file_path: str | None = None
    entry_type: str = "code"  # code, doc, pattern, api
    language: str | None = None
    embedding: list[float] = []
    metadata: dict[str, Any] = {}


class SemanticMemory:
    """
    Indexes and retrieves code patterns, API documentation, and domain knowledge.

    Supports incremental indexing of repositories with change detection,
    chunked embedding of large files, and hybrid search combining vector
    similarity with keyword matching.

    Backend: PostgreSQL with pgvector, with Redis caching for hot queries.
    """

    def __init__(
        self,
        db_pool: Any,
        embedding_client: Any,
        redis_client: Any | None = None,
        table_name: str = "semantic_memory",
        embedding_dim: int = 1536,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.db_pool = db_pool
        self.embedding_client = embedding_client
        self.redis = redis_client
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def initialize(self) -> None:
        """Create semantic memory tables and indexes."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    entry_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    file_path TEXT,
                    entry_type TEXT DEFAULT 'code',
                    language TEXT,
                    embedding vector({self.embedding_dim}),
                    content_hash TEXT,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 200);
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_file
                ON {self.table_name} (file_path);
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_type
                ON {self.table_name} (entry_type);
            """)

    async def index_file(self, file_path: str, content: str, language: str | None = None) -> int:
        """
        Index a file into semantic memory with chunking.

        Returns the number of chunks indexed.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Check if content has changed
        existing_hash = await self._get_content_hash(file_path)
        if existing_hash == content_hash:
            return 0

        # Remove old entries for this file
        await self._remove_file_entries(file_path)

        # Chunk the content
        chunks = self._chunk_content(content, file_path)

        # Batch embed all chunks
        embeddings = await self._batch_embed([c["content"] for c in chunks])

        # Store all chunks
        async with self.db_pool.acquire() as conn:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                entry_id = f"{file_path}::{i}::{content_hash[:8]}"
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name}
                        (entry_id, content, file_path, entry_type, language,
                         embedding, content_hash, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    entry_id,
                    chunk["content"],
                    file_path,
                    chunk.get("type", "code"),
                    language,
                    embedding,
                    content_hash,
                    chunk.get("metadata", {}),
                )

        # Invalidate cache for this file
        if self.redis:
            await self.redis.delete(f"semantic:file:{file_path}")

        return len(chunks)

    async def retrieve(self, query: str, k: int = 5, entry_type: str | None = None) -> list[SemanticEntry]:
        """
        Retrieve semantically similar entries using hybrid search.

        Combines vector similarity with optional type filtering.
        Results are cached in Redis for repeated queries.
        """
        # Check cache
        cache_key = f"semantic:query:{hashlib.md5(query.encode()).hexdigest()}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                import json
                return [SemanticEntry(**e) for e in json.loads(cached)]

        query_embedding = await self._embed(query)

        type_filter = ""
        params: list[Any] = [query_embedding, k]
        if entry_type:
            type_filter = "AND entry_type = $3"
            params.append(entry_type)

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    entry_id, content, file_path, entry_type, language, metadata,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM {self.table_name}
                WHERE TRUE {type_filter}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                *params,
            )

        entries = [
            SemanticEntry(
                entry_id=row["entry_id"],
                content=row["content"],
                file_path=row["file_path"],
                entry_type=row["entry_type"],
                language=row["language"],
                metadata=row["metadata"],
            )
            for row in rows
        ]

        # Cache results for 5 minutes
        if self.redis and entries:
            import json
            await self.redis.setex(
                cache_key, 300, json.dumps([e.model_dump() for e in entries])
            )

        return entries

    async def retrieve_for_file(self, file_path: str) -> list[SemanticEntry]:
        """Retrieve all indexed entries for a specific file."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.table_name} WHERE file_path = $1 ORDER BY entry_id",
                file_path,
            )
        return [
            SemanticEntry(
                entry_id=row["entry_id"],
                content=row["content"],
                file_path=row["file_path"],
                entry_type=row["entry_type"],
                language=row["language"],
                metadata=row["metadata"],
            )
            for row in rows
        ]

    async def keyword_search(self, keywords: list[str], k: int = 10) -> list[SemanticEntry]:
        """Full-text keyword search as complement to vector search."""
        query_str = " & ".join(keywords)
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT entry_id, content, file_path, entry_type, language, metadata
                FROM {self.table_name}
                WHERE to_tsvector('english', content) @@ to_tsquery('english', $1)
                LIMIT $2
                """,
                query_str,
                k,
            )
        return [
            SemanticEntry(
                entry_id=row["entry_id"],
                content=row["content"],
                file_path=row["file_path"],
                entry_type=row["entry_type"],
                language=row["language"],
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def _chunk_content(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """Split content into overlapping chunks preserving logical boundaries."""
        lines = content.split("\n")
        chunks: list[dict[str, Any]] = []
        current_chunk: list[str] = []
        current_size = 0

        for i, line in enumerate(lines):
            tokens_estimate = len(line.split()) + 1
            if current_size + tokens_estimate > self.chunk_size and current_chunk:
                chunks.append({
                    "content": "\n".join(current_chunk),
                    "type": self._classify_chunk("\n".join(current_chunk)),
                    "metadata": {"start_line": i - len(current_chunk), "end_line": i},
                })
                # Keep overlap
                overlap_lines = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_lines
                current_size = sum(len(line_.split()) + 1 for line_ in overlap_lines)

            current_chunk.append(line)
            current_size += tokens_estimate

        if current_chunk:
            chunks.append({
                "content": "\n".join(current_chunk),
                "type": self._classify_chunk("\n".join(current_chunk)),
                "metadata": {"start_line": len(lines) - len(current_chunk), "end_line": len(lines)},
            })

        return chunks

    @staticmethod
    def _classify_chunk(content: str) -> str:
        """Classify a chunk as code, docstring, comment, or mixed."""
        lines = content.strip().split("\n")
        if not lines:
            return "code"

        comment_lines = sum(1 for ln in lines if ln.strip().startswith(("#", "//", "/*", "*")))
        doc_indicators = ('"""', "'''", "/**")

        if any(content.startswith(d) or content.endswith(d) for d in doc_indicators):
            return "doc"
        if comment_lines > len(lines) * 0.6:
            return "comment"
        return "code"

    async def _get_content_hash(self, file_path: str) -> str | None:
        """Get stored content hash for a file."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT content_hash FROM {self.table_name} WHERE file_path = $1 LIMIT 1",
                file_path,
            )

    async def _remove_file_entries(self, file_path: str) -> None:
        """Remove all entries for a file before re-indexing."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.table_name} WHERE file_path = $1", file_path
            )

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = await self.embedding_client.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call."""
        if not texts:
            return []
        response = await self.embedding_client.create(
            input=texts, model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
