"""LangGraph workflow definition for agent orchestration."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from agent_orchestrator.agents.base import AgentContext, AgentMessage, AgentStatus
from agent_orchestrator.agents.coder import CoderAgent
from agent_orchestrator.agents.researcher import ResearcherAgent
from agent_orchestrator.agents.reviewer import ReviewerAgent
from agent_orchestrator.context_compression import ContextCompressor, compress_for_task
from agent_orchestrator.planner import DecompositionPlan, MCTSPlanner
from agent_orchestrator.sandbox import SandboxExecutor
from agent_orchestrator.test_driven import TestDrivenPatchGenerator
from agent_orchestrator.tree_of_thought import TreeOfThoughtVerifier


class OrchestratorState(BaseModel):
    """State flowing through the orchestration graph."""

    task_id: str = ""
    task_description: str = ""
    repo_path: str = ""
    repo_metadata: dict[str, Any] = {}
    autonomy_level: str = "high"

    # Planning state
    plan: DecompositionPlan | None = None
    current_subtask_idx: int = 0

    # Agent outputs
    research_output: dict[str, Any] = {}
    code_output: dict[str, Any] = {}
    review_output: dict[str, Any] = {}

    # Communication
    messages: list[AgentMessage] = []
    errors: list[str] = []

    # v2 enhancements
    compressed_context: dict[str, Any] = {}
    tdd_result: dict[str, Any] = {}
    tot_result: dict[str, Any] = {}

    # Control flow
    iteration: int = 0
    max_iterations: int = 5
    status: str = "pending"
    use_test_driven: bool = True
    use_tree_of_thought: bool = True
    use_context_compression: bool = True


def build_workflow(
    llm_client: Any,
    planner: MCTSPlanner | None = None,
    sandbox: SandboxExecutor | None = None,
    memory_stores: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> StateGraph:
    """
    Build the LangGraph workflow for multi-agent orchestration.

    The graph follows this flow:
    1. Plan: MCTS decomposes the task into subtasks
    2. Research: Researcher agent gathers codebase context
    3. Code: Coder agent generates a patch
    4. Review: Reviewer agent validates the patch
    5. Decision: Route based on review verdict (approve/revise/fail)
    """
    config = config or {}
    memory_stores = memory_stores or {}

    researcher = ResearcherAgent(
        llm_client=llm_client,
        memory_stores=memory_stores,
    )
    coder = CoderAgent(
        llm_client=llm_client,
        memory_stores=memory_stores,
        sandbox=sandbox,
    )
    reviewer = ReviewerAgent(
        llm_client=llm_client,
        memory_stores=memory_stores,
        strict_mode=config.get("strict_review", False),
    )

    # v2: test-driven patch generator and tree-of-thought verifier
    tdd_generator = TestDrivenPatchGenerator(
        llm_client=llm_client,
        sandbox=sandbox,
    )
    tot_verifier = TreeOfThoughtVerifier(
        llm_client=llm_client,
        sandbox=sandbox,
    )
    context_compressor = ContextCompressor(
        max_context_tokens=config.get("max_context_tokens", 16000),
    )

    # Define the graph
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("plan", _make_plan_node(planner or MCTSPlanner(llm_client)))
    workflow.add_node("research", _make_agent_node(researcher))
    workflow.add_node(
        "compress_context",
        _make_compression_node(context_compressor),
    )
    workflow.add_node(
        "test_driven",
        _make_tdd_node(tdd_generator),
    )
    workflow.add_node("code", _make_agent_node(coder))
    workflow.add_node(
        "tree_of_thought",
        _make_tot_node(tot_verifier),
    )
    workflow.add_node("review", _make_agent_node(reviewer))
    workflow.add_node("handle_revision", _handle_revision)
    workflow.add_node("finalize", _finalize)

    # Define edges - v2 flow adds optional compression, TDD, and ToT nodes
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "research")
    workflow.add_conditional_edges(
        "research",
        _compression_router,
        {
            "compress": "compress_context",
            "skip": "test_driven",
        },
    )
    workflow.add_edge("compress_context", "test_driven")
    workflow.add_conditional_edges(
        "test_driven",
        _tdd_router,
        {
            "use_tdd": "code",
            "use_tot": "tree_of_thought",
            "skip": "code",
        },
    )
    workflow.add_edge("tree_of_thought", "review")
    workflow.add_edge("code", "review")
    workflow.add_conditional_edges(
        "review",
        _review_router,
        {
            "approve": "finalize",
            "revise": "handle_revision",
            "fail": "finalize",
        },
    )
    workflow.add_edge("handle_revision", "code")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def _make_plan_node(planner: MCTSPlanner):
    """Create the planning node function."""

    async def plan_node(state: OrchestratorState) -> dict[str, Any]:
        plan = await planner.plan(
            task_description=state.task_description,
            repo_context=state.repo_metadata,
        )
        return {
            "plan": plan,
            "task_id": plan.task_id,
            "status": "planning_complete",
        }

    return plan_node


def _make_agent_node(agent: Any):
    """Create a generic agent execution node."""

    async def agent_node(state: OrchestratorState) -> dict[str, Any]:
        context = AgentContext(
            task_id=state.task_id,
            task_description=state.task_description,
            repo_path=state.repo_path,
            repo_metadata=state.repo_metadata,
            relevant_files=state.research_output.get("relevant_files", []),
            previous_messages=state.messages,
            autonomy_level=state.autonomy_level,
        )

        result = await agent.execute(context)

        # Store result in appropriate state field
        updates: dict[str, Any] = {}
        message = AgentMessage(
            sender=agent.name,
            recipient="orchestrator",
            content=result.output,
            artifacts=result.artifacts,
        )
        updates["messages"] = state.messages + [message]

        if agent.name == "researcher":
            updates["research_output"] = result.artifacts
        elif agent.name == "coder":
            updates["code_output"] = result.artifacts
        elif agent.name == "reviewer":
            updates["review_output"] = result.artifacts

        if result.status == AgentStatus.FAILED:
            updates["errors"] = state.errors + [result.error or "Unknown error"]

        return updates

    return agent_node


def _make_compression_node(compressor: ContextCompressor):
    """Create the context compression node."""

    async def compress_node(state: OrchestratorState) -> dict[str, Any]:
        file_contents = state.research_output.get("file_contents", {})
        if not file_contents:
            return {}

        compressed = compress_for_task(
            file_contents=file_contents,
            task_description=state.task_description,
            max_tokens=16000,
        )
        return {
            "compressed_context": {
                "prompt_context": compressed.to_prompt_context(),
                "compression_ratio": compressed.compression_ratio,
                "tokens_saved": compressed.total_tokens_saved,
            },
        }

    return compress_node


def _make_tdd_node(tdd_generator: TestDrivenPatchGenerator):
    """Create the test-driven generation node."""

    async def tdd_node(state: OrchestratorState) -> dict[str, Any]:
        if not state.use_test_driven:
            return {}

        relevant_code = state.research_output.get("file_contents", {})

        result = await tdd_generator.generate(
            task_description=state.task_description,
            relevant_code=relevant_code,
            repo_path=state.repo_path,
        )
        return {
            "tdd_result": {
                "test": result.test,
                "patch": result.patch,
                "success": result.success,
                "retries": result.total_retries,
            },
        }

    return tdd_node


def _make_tot_node(tot_verifier: TreeOfThoughtVerifier):
    """Create the tree-of-thought verification node."""

    async def tot_node(state: OrchestratorState) -> dict[str, Any]:
        relevant_code = state.research_output.get("file_contents", {})
        test_code = state.tdd_result.get("test")

        result = await tot_verifier.generate_and_verify(
            task_description=state.task_description,
            relevant_code=relevant_code,
            repo_path=state.repo_path,
            test_code=test_code,
        )
        return {
            "tot_result": {
                "best_patch": result.best_patch,
                "any_passed": result.any_passed,
                "branches_evaluated": len(result.all_branches),
                "total_cost_tokens": result.total_cost_tokens,
            },
            "code_output": {"patch": result.best_patch},
        }

    return tot_node


def _compression_router(
    state: OrchestratorState,
) -> Literal["compress", "skip"]:
    """Route based on whether context compression is enabled."""
    if state.use_context_compression and state.research_output.get("file_contents"):
        return "compress"
    return "skip"


def _tdd_router(
    state: OrchestratorState,
) -> Literal["use_tdd", "use_tot", "skip"]:
    """Route based on TDD/ToT configuration."""
    if state.use_tree_of_thought:
        return "use_tot"
    if state.use_test_driven:
        return "use_tdd"
    return "skip"


def _review_router(state: OrchestratorState) -> Literal["approve", "revise", "fail"]:
    """Route based on review verdict and iteration count."""
    review = state.review_output.get("review", {})
    verdict = review.get("verdict", "request_changes")

    if verdict == "approve":
        return "approve"

    if state.iteration >= state.max_iterations:
        return "fail"

    return "revise"


async def _handle_revision(state: OrchestratorState) -> dict[str, Any]:
    """Prepare state for a revision iteration."""
    review = state.review_output.get("review", {})
    comments = review.get("comments", [])

    revision_message = AgentMessage(
        sender="orchestrator",
        recipient="coder",
        content=f"Revision requested. Address these issues: {comments}",
        artifacts={"review_comments": comments},
    )

    return {
        "iteration": state.iteration + 1,
        "messages": state.messages + [revision_message],
        "status": "revising",
    }


async def _finalize(state: OrchestratorState) -> dict[str, Any]:
    """Finalize the orchestration run."""
    review = state.review_output.get("review", {})
    verdict = review.get("verdict", "unknown")

    if verdict == "approve":
        status = "completed"
    elif state.iteration >= state.max_iterations:
        status = "max_iterations_reached"
    else:
        status = "failed"

    return {"status": status}
