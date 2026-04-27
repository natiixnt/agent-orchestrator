"""Full audit trail - every decision, tool call, and dollar spent gets logged."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of auditable events in the system."""

    PLAN_CREATED = "plan_created"
    PLAN_STEP_STARTED = "plan_step_started"
    PLAN_STEP_COMPLETED = "plan_step_completed"
    AGENT_INVOKED = "agent_invoked"
    AGENT_COMPLETED = "agent_completed"
    TOOL_CALLED = "tool_called"
    TOOL_RETURNED = "tool_returned"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    SANDBOX_EXEC = "sandbox_exec"
    ERROR = "error"
    DECISION = "decision"
    COST_INCURRED = "cost_incurred"


class AuditEvent(BaseModel):
    """A single auditable event with full context."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = Field(default_factory=time.time)
    event_type: EventType
    task_id: str
    agent_id: str | None = None
    parent_event_id: str | None = None
    payload: dict[str, Any] = {}
    duration_ms: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = {}


class CostBreakdown(BaseModel):
    """Detailed cost accounting for a task or session."""

    total_usd: float = 0.0
    by_model: dict[str, float] = {}
    by_agent: dict[str, float] = {}
    by_operation: dict[str, float] = {}
    total_tokens_in: int = 0
    total_tokens_out: int = 0


@dataclass
class TaskAuditTrail:
    """Complete audit trail for a single task execution."""

    task_id: str
    events: list[AuditEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def cost(self) -> CostBreakdown:
        breakdown = CostBreakdown()
        for event in self.events:
            if event.cost_usd > 0:
                breakdown.total_usd += event.cost_usd
                breakdown.total_tokens_in += event.tokens_in
                breakdown.total_tokens_out += event.tokens_out

                # track by model
                model = event.metadata.get("model", "unknown")
                breakdown.by_model[model] = breakdown.by_model.get(model, 0) + event.cost_usd

                # track by agent
                agent = event.agent_id or "system"
                breakdown.by_agent[agent] = breakdown.by_agent.get(agent, 0) + event.cost_usd

                # track by operation type
                op = event.event_type.value
                breakdown.by_operation[op] = breakdown.by_operation.get(op, 0) + event.cost_usd

        return breakdown

    @property
    def decisions(self) -> list[AuditEvent]:
        """Get all decision events for this task."""
        return [e for e in self.events if e.event_type == EventType.DECISION]

    @property
    def errors(self) -> list[AuditEvent]:
        """Get all error events for this task."""
        return [e for e in self.events if e.event_type == EventType.ERROR]


class AuditLogger:
    """
    Central audit logger that captures everything the system does.

    Every LLM call, tool invocation, agent decision, and cost gets recorded.
    This is critical for debugging failures, understanding cost drivers,
    and building trust that the system isn't doing sketchy stuff.

    Events are stored in memory during execution and can be flushed to
    persistent storage (postgres, file, etc) after task completion.
    """

    # pricing per 1M tokens - updated whenever anthropic/openai change their rates
    # these are the rates as of march 2025
    MODEL_PRICING: dict[str, dict[str, float]] = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    }

    def __init__(self, persist_fn: Any | None = None) -> None:
        self._trails: dict[str, TaskAuditTrail] = {}
        self._persist_fn = persist_fn
        self._global_stats: dict[str, Any] = defaultdict(float)

    def start_task(self, task_id: str) -> TaskAuditTrail:
        """Initialize audit trail for a new task."""
        trail = TaskAuditTrail(task_id=task_id)
        self._trails[task_id] = trail
        return trail

    def end_task(self, task_id: str) -> TaskAuditTrail:
        """Finalize audit trail for a completed task."""
        trail = self._trails[task_id]
        trail.end_time = time.time()

        if self._persist_fn:
            self._persist_fn(trail)

        self._update_global_stats(trail)
        return trail

    def log_event(
        self,
        task_id: str,
        event_type: EventType,
        agent_id: str | None = None,
        parent_event_id: str | None = None,
        payload: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        model: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log a single event to the task's audit trail."""
        cost = self._calculate_cost(model, tokens_in, tokens_out) if model else 0.0

        event = AuditEvent(
            event_type=event_type,
            task_id=task_id,
            agent_id=agent_id,
            parent_event_id=parent_event_id,
            payload=payload or {},
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            error=error,
            metadata={**(metadata or {}), "model": model} if model else (metadata or {}),
        )

        if task_id in self._trails:
            self._trails[task_id].events.append(event)

        return event

    def log_llm_call(
        self,
        task_id: str,
        agent_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
        prompt_summary: str = "",
        response_summary: str = "",
    ) -> AuditEvent:
        """Convenience method for logging LLM API calls with cost tracking."""
        return self.log_event(
            task_id=task_id,
            event_type=EventType.LLM_REQUEST,
            agent_id=agent_id,
            tokens_in=prompt_tokens,
            tokens_out=completion_tokens,
            model=model,
            duration_ms=duration_ms,
            payload={
                "prompt_summary": prompt_summary,
                "response_summary": response_summary,
            },
        )

    def log_tool_call(
        self,
        task_id: str,
        agent_id: str,
        tool_name: str,
        args: dict[str, Any],
        result_summary: str,
        duration_ms: float,
        success: bool = True,
    ) -> AuditEvent:
        """Log a tool invocation with args and result."""
        return self.log_event(
            task_id=task_id,
            event_type=EventType.TOOL_CALLED,
            agent_id=agent_id,
            duration_ms=duration_ms,
            payload={
                "tool": tool_name,
                "args": args,
                "result_summary": result_summary,
                "success": success,
            },
        )

    def log_decision(
        self,
        task_id: str,
        agent_id: str,
        decision: str,
        reasoning: str,
        alternatives: list[str] | None = None,
        confidence: float | None = None,
    ) -> AuditEvent:
        """
        Log an explicit decision point with reasoning.

        These are the high-signal events - when the system chose path A over B,
        we want to know why so we can debug failures later.
        """
        return self.log_event(
            task_id=task_id,
            event_type=EventType.DECISION,
            agent_id=agent_id,
            payload={
                "decision": decision,
                "reasoning": reasoning,
                "alternatives": alternatives or [],
                "confidence": confidence,
            },
        )

    def get_trail(self, task_id: str) -> TaskAuditTrail | None:
        """Retrieve the audit trail for a task."""
        return self._trails.get(task_id)

    def get_cost_summary(self) -> dict[str, Any]:
        """Get aggregate cost summary across all tracked tasks."""
        return {
            "total_tasks": len(self._trails),
            "total_cost_usd": round(self._global_stats["total_cost"], 4),
            "total_tokens_in": int(self._global_stats["total_tokens_in"]),
            "total_tokens_out": int(self._global_stats["total_tokens_out"]),
            "avg_cost_per_task": round(
                self._global_stats["total_cost"] / max(len(self._trails), 1), 4
            ),
        }

    def export_trail(self, task_id: str) -> dict[str, Any]:
        """Export a complete audit trail as a serializable dict."""
        trail = self._trails.get(task_id)
        if not trail:
            return {}
        return {
            "task_id": trail.task_id,
            "duration_seconds": round(trail.duration_seconds, 2),
            "cost": trail.cost.model_dump(),
            "event_count": len(trail.events),
            "decisions": [e.model_dump() for e in trail.decisions],
            "errors": [e.model_dump() for e in trail.errors],
            "events": [e.model_dump() for e in trail.events],
        }

    def _calculate_cost(self, model: str | None, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost based on model pricing."""
        if not model or model not in self.MODEL_PRICING:
            return 0.0
        pricing = self.MODEL_PRICING[model]
        cost_in = (tokens_in / 1_000_000) * pricing["input"]
        cost_out = (tokens_out / 1_000_000) * pricing["output"]
        return round(cost_in + cost_out, 6)

    def _update_global_stats(self, trail: TaskAuditTrail) -> None:
        """Update running global statistics."""
        cost = trail.cost
        self._global_stats["total_cost"] += cost.total_usd
        self._global_stats["total_tokens_in"] += cost.total_tokens_in
        self._global_stats["total_tokens_out"] += cost.total_tokens_out
