"""Base agent class with shared capabilities."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentMessage(BaseModel):
    """Message passed between agents in the orchestration graph."""

    sender: str
    recipient: str
    content: str
    artifacts: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    timestamp: float = Field(default_factory=time.time)


class AgentContext(BaseModel):
    """Execution context available to all agents."""

    task_id: str
    task_description: str
    repo_path: str
    repo_metadata: dict[str, Any] = {}
    working_branch: str = "main"
    relevant_files: list[str] = []
    previous_messages: list[AgentMessage] = []
    memory_results: dict[str, Any] = {}
    autonomy_level: str = "high"


class ToolCall(BaseModel):
    """Represents a tool invocation by an agent."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    duration_ms: float = 0.0
    success: bool = True


class AgentResult(BaseModel):
    """Result produced by an agent after execution."""

    agent_name: str
    status: AgentStatus
    output: str
    artifacts: dict[str, Any] = {}
    tool_calls: list[ToolCall] = []
    tokens_used: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.

    Provides shared functionality for LLM interaction, tool usage,
    memory access, and human-in-the-loop checkpoints.
    """

    def __init__(
        self,
        name: str,
        llm_client: Any,
        tools: list[Any] | None = None,
        memory_stores: dict[str, Any] | None = None,
        max_iterations: int = 10,
        temperature: float = 0.0,
    ) -> None:
        self.name = name
        self.llm = llm_client
        self.tools = tools or []
        self.memory_stores = memory_stores or {}
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.status = AgentStatus.IDLE
        self._tool_calls: list[ToolCall] = []
        self._tokens_used: int = 0

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's primary task given the context."""
        ...

    @abstractmethod
    def get_system_prompt(self, context: AgentContext) -> str:
        """Return the system prompt for this agent's role."""
        ...

    async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool and record the invocation."""
        tool = self._find_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in agent '{self.name}'")

        start = time.perf_counter()
        try:
            result = await tool.invoke(arguments)
            duration = (time.perf_counter() - start) * 1000
            self._tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    duration_ms=duration,
                    success=True,
                )
            )
            return result
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self._tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=str(e),
                    duration_ms=duration,
                    success=False,
                )
            )
            raise

    async def query_memory(self, query: str, memory_type: str, k: int = 5) -> list[Any]:
        """Retrieve relevant memories from the specified store."""
        store = self.memory_stores.get(memory_type)
        if store is None:
            return []
        return await store.retrieve(query, k=k)

    async def request_human_review(
        self, checkpoint_name: str, data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """
        Request human review at a checkpoint.

        Returns immediately if autonomy level permits skipping,
        otherwise blocks until human provides input.
        """
        if self._can_skip_checkpoint(checkpoint_name, context.autonomy_level):
            return {"approved": True, "skipped": True}

        self.status = AgentStatus.WAITING_HUMAN
        # In production this publishes to WebSocket and awaits response
        response = await self._publish_checkpoint(checkpoint_name, data)
        self.status = AgentStatus.RUNNING
        return response

    def _can_skip_checkpoint(self, checkpoint_name: str, autonomy_level: str) -> bool:
        """Determine if a checkpoint can be skipped based on autonomy level."""
        skip_rules = {
            "full": True,
            "high": checkpoint_name not in ("destructive_operation", "deploy"),
            "medium": False,
            "low": False,
        }
        return skip_rules.get(autonomy_level, False)

    async def _publish_checkpoint(
        self, checkpoint_name: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Publish checkpoint to human review channel. Override for WebSocket impl."""
        return {"approved": True, "feedback": None}

    def _find_tool(self, tool_name: str) -> Any:
        """Find a tool by name from the agent's tool list."""
        for tool in self.tools:
            if getattr(tool, "name", None) == tool_name:
                return tool
        return None

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        tools: list[Any] | None = None,
    ) -> Any:
        """Make an LLM call with tool definitions."""
        response = await self.llm.chat(
            messages=messages,
            tools=[t.schema for t in (tools or self.tools)],
            temperature=self.temperature,
        )
        self._tokens_used += response.usage.total_tokens
        return response

    def _build_result(
        self,
        status: AgentStatus,
        output: str,
        artifacts: dict[str, Any] | None = None,
        error: str | None = None,
        duration: float = 0.0,
    ) -> AgentResult:
        """Construct an AgentResult with accumulated metrics."""
        return AgentResult(
            agent_name=self.name,
            status=status,
            output=output,
            artifacts=artifacts or {},
            tool_calls=self._tool_calls.copy(),
            tokens_used=self._tokens_used,
            duration_seconds=duration,
            error=error,
        )
