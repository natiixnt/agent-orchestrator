"""Researcher agent for codebase navigation and context gathering."""

from __future__ import annotations

import time
from typing import Any

from agent_orchestrator.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)


class ResearcherAgent(BaseAgent):
    """
    Agent specialized in codebase exploration and context retrieval.

    Capabilities:
    - File tree traversal and pattern matching
    - Dependency graph construction
    - Symbol definition and reference lookup
    - Relevant code snippet extraction
    - Documentation and comment analysis
    """

    def __init__(
        self,
        llm_client: Any,
        tools: list[Any] | None = None,
        memory_stores: dict[str, Any] | None = None,
        max_search_depth: int = 5,
        max_files_to_read: int = 20,
    ) -> None:
        super().__init__(
            name="researcher",
            llm_client=llm_client,
            tools=tools,
            memory_stores=memory_stores,
            max_iterations=8,
        )
        self.max_search_depth = max_search_depth
        self.max_files_to_read = max_files_to_read

    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Research the codebase to gather context for the task.

        Strategy:
        1. Query episodic memory for similar past research sessions
        2. Identify relevant files using semantic search and file patterns
        3. Build a dependency graph around the target area
        4. Extract key code snippets and documentation
        5. Synthesize findings into a structured context object
        """
        start_time = time.perf_counter()
        self.status = AgentStatus.RUNNING
        self._tool_calls = []
        self._tokens_used = 0

        try:
            # Retrieve prior research on similar tasks
            episodic_results = await self.query_memory(
                context.task_description, "episodic", k=3
            )

            # Build initial file candidates from multiple signals
            file_candidates = await self._identify_relevant_files(context, episodic_results)

            # Deep analysis of top candidates
            code_context = await self._analyze_files(file_candidates, context)

            # Build dependency graph for affected area
            dependency_graph = await self._build_dependency_graph(
                code_context["primary_files"], context
            )

            # Synthesize research findings
            synthesis = await self._synthesize_findings(
                code_context, dependency_graph, context
            )

            artifacts = {
                "relevant_files": code_context["primary_files"],
                "code_snippets": code_context["snippets"],
                "dependency_graph": dependency_graph,
                "synthesis": synthesis,
                "search_queries_used": code_context.get("queries", []),
            }

            duration = time.perf_counter() - start_time
            return self._build_result(
                status=AgentStatus.COMPLETED,
                output=synthesis,
                artifacts=artifacts,
                duration=duration,
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            return self._build_result(
                status=AgentStatus.FAILED,
                output="",
                error=str(e),
                duration=duration,
            )

    def get_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a code research agent. Your role is to explore a codebase and gather "
            "all relevant context needed to complete a software engineering task.\n\n"
            "You have access to tools for searching files, reading code, and building "
            "dependency graphs. Be thorough but focused: identify the minimal set of files "
            "and code regions that are relevant to the task.\n\n"
            f"Repository: {context.repo_path}\n"
            f"Task: {context.task_description}\n"
            f"Known relevant files: {context.relevant_files}\n"
        )

    async def _identify_relevant_files(
        self, context: AgentContext, episodic_results: list[Any]
    ) -> list[str]:
        """Identify files relevant to the task using multiple strategies."""
        candidates: set[str] = set()

        # Start with any explicitly provided files
        candidates.update(context.relevant_files)

        # Use semantic search over the codebase index
        semantic_results = await self.query_memory(
            context.task_description, "semantic", k=10
        )
        for result in semantic_results:
            if hasattr(result, "file_path"):
                candidates.add(result.file_path)

        # Extract file paths from episodic memory of similar tasks
        for episode in episodic_results:
            if hasattr(episode, "artifacts") and "relevant_files" in episode.artifacts:
                candidates.update(episode.artifacts["relevant_files"][:5])

        # Use grep/search tools to find additional references
        search_results = await self.run_tool(
            "code_search",
            {"query": context.task_description, "repo_path": context.repo_path, "limit": 15},
        )
        if search_results:
            for hit in search_results:
                candidates.add(hit["file_path"])

        return sorted(candidates)[: self.max_files_to_read]

    async def _analyze_files(
        self, file_paths: list[str], context: AgentContext
    ) -> dict[str, Any]:
        """Read and analyze the content of candidate files."""
        snippets: list[dict[str, Any]] = []
        primary_files: list[str] = []

        messages = [
            {"role": "system", "content": self.get_system_prompt(context)},
            {
                "role": "user",
                "content": (
                    f"Analyze these files to find code relevant to: {context.task_description}\n"
                    f"Files: {file_paths}"
                ),
            },
        ]

        for iteration in range(min(len(file_paths), self.max_iterations)):
            response = await self._call_llm(messages)

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = await self.run_tool(
                        tool_call.function.name, tool_call.function.arguments
                    )
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    })

                    if tool_call.function.name == "read_file":
                        file_path = tool_call.function.arguments.get("path", "")
                        primary_files.append(file_path)
                        snippets.append({
                            "file_path": file_path,
                            "content": result,
                            "relevance": "high",
                        })
            else:
                break

        return {
            "primary_files": primary_files,
            "snippets": snippets,
            "queries": [context.task_description],
        }

    async def _build_dependency_graph(
        self, files: list[str], context: AgentContext
    ) -> dict[str, list[str]]:
        """Build a dependency graph for the identified files."""
        graph: dict[str, list[str]] = {}

        for file_path in files[:10]:
            imports = await self.run_tool(
                "get_imports", {"file_path": file_path, "repo_path": context.repo_path}
            )
            if imports:
                graph[file_path] = imports

        return graph

    async def _synthesize_findings(
        self, code_context: dict[str, Any], dep_graph: dict[str, list[str]], context: AgentContext
    ) -> str:
        """Use LLM to synthesize research findings into actionable context."""
        synthesis_prompt = (
            f"Synthesize the following research findings for the task: {context.task_description}\n\n"
            f"Primary files identified: {code_context['primary_files']}\n"
            f"Dependency graph: {dep_graph}\n"
            f"Number of code snippets: {len(code_context['snippets'])}\n\n"
            "Provide a structured summary covering:\n"
            "1. Which files need modification and why\n"
            "2. Key dependencies and potential side effects\n"
            "3. Patterns observed in the existing code\n"
            "4. Risks and edge cases to watch for"
        )

        response = await self._call_llm([
            {"role": "system", "content": "You are a technical analyst synthesizing code research."},
            {"role": "user", "content": synthesis_prompt},
        ])

        return response.content
