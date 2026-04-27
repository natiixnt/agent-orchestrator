"""
Minimal quickstart for agent-orchestrator.

Loads the public API, plans a task, and prints the decomposition.
Uses a stub LLM client - replace with your own to wire up real models.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from agent_orchestrator import MCTSPlanner


@dataclass
class StubResponse:
    content: str = ""
    parsed: list[dict] | None = None
    tool_calls: list = field(default_factory=list)


class StubLLM:
    """
    Replace this with your real LLM wrapper. The interface the planner expects:
        async def generate(prompt, response_format=None) -> object with .parsed
    """

    async def generate(self, prompt: str, response_format: str | None = None):
        # this stub returns a hardcoded plausible decomposition
        return StubResponse(
            parsed=[
                {
                    "description": "Identify files implicated by the task description",
                    "agent_type": "researcher",
                    "complexity": 0.3,
                },
                {
                    "description": "Generate a minimal patch to fix the issue",
                    "agent_type": "coder",
                    "complexity": 0.6,
                },
                {
                    "description": "Validate the patch against existing tests",
                    "agent_type": "reviewer",
                    "complexity": 0.3,
                },
            ],
        )


async def main() -> None:
    task = "Fix the off-by-one bug in pagination.py that drops the first page of results."
    repo_context = {
        "languages": ["python"],
        "framework": "flask",
        "test_runner": "pytest",
    }

    planner = MCTSPlanner(
        llm_client=StubLLM(),
        simulations=20,  # default is 100; lower here for quick demo
        max_depth=4,
    )
    plan = await planner.plan(task, repo_context)

    print(f"Task ID:     {plan.task_id}")
    print(f"Subtasks:    {len(plan.subtasks)}")
    print(f"Expected pass rate: {plan.expected_success_rate:.2f}")
    print(f"Tree depth explored: {plan.tree_depth}")
    print()
    print("Plan:")
    for i, subtask in enumerate(plan.subtasks, start=1):
        print(f"  {i}. [{subtask.agent_type:10s}] {subtask.description}")

    print()
    print("Next: hand each subtask to the corresponding agent.")
    print("See benchmarks/run_demo.py for a full pipeline walkthrough.")


if __name__ == "__main__":
    asyncio.run(main())
