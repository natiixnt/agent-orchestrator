"""Tests for the MCTS planner - UCB1, backprop, best path extraction."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_orchestrator.planner import (
    DecompositionPlan,
    MCTSNode,
    MCTSPlanner,
    Subtask,
)


# ---- fixtures ----

@pytest.fixture
def mock_llm():
    """Fake LLM client that returns canned decomposition suggestions."""
    llm = AsyncMock()
    llm.generate.return_value = MagicMock(
        parsed=[
            {"description": "Research the codebase", "agent_type": "researcher", "complexity": 0.3},
            {"description": "Implement the fix", "agent_type": "coder", "complexity": 0.7},
            {"description": "Review the patch", "agent_type": "reviewer", "complexity": 0.4},
        ]
    )
    return llm


@pytest.fixture
def planner(mock_llm):
    return MCTSPlanner(
        llm_client=mock_llm,
        simulations=20,  # keep it fast for tests
        max_depth=3,
    )


@pytest.fixture
def sample_subtasks():
    return [
        Subtask(id="s0", description="research", agent_type="researcher"),
        Subtask(id="s1", description="code", agent_type="coder", dependencies=["s0"]),
        Subtask(id="s2", description="review", agent_type="reviewer", dependencies=["s1"]),
    ]


# ---- UCB1 tests ----

class TestUCB1:
    """UCB1 selection policy - the heart of MCTS exploration vs exploitation."""

    def test_unvisited_node_returns_infinity(self):
        """unvisited nodes should always be selected first - infinite UCB1"""
        parent = MCTSNode(state=[], visits=10, total_reward=5.0)
        child = MCTSNode(state=[], parent=parent, visits=0, total_reward=0.0)
        parent.children = [child]
        assert child.ucb1 == float("inf")

    def test_ucb1_formula_correct(self):
        """sanity check the actual UCB1 computation against manual calculation"""
        parent = MCTSNode(state=[], visits=100, total_reward=50.0)
        child = MCTSNode(state=[], parent=parent, visits=20, total_reward=14.0)
        parent.children = [child]

        # exploitation = 14/20 = 0.7
        # exploration = sqrt(2 * ln(100) / 20) = sqrt(2 * 4.605 / 20) = sqrt(0.4605) = 0.6786
        # UCB1 = 0.7 + 1.414 * 0.6786 = 0.7 + 0.9595 = 1.6595
        expected_exploitation = 14.0 / 20.0
        expected_exploration = math.sqrt(2.0 * math.log(100) / 20.0)
        expected = expected_exploitation + 1.414 * expected_exploration

        assert abs(child.ucb1 - expected) < 0.001

    def test_higher_reward_increases_ucb1(self):
        """nodes with better track records should score higher (exploitation)"""
        parent = MCTSNode(state=[], visits=50, total_reward=25.0)
        good = MCTSNode(state=[], parent=parent, visits=10, total_reward=8.0)
        bad = MCTSNode(state=[], parent=parent, visits=10, total_reward=2.0)
        parent.children = [good, bad]

        assert good.ucb1 > bad.ucb1

    def test_fewer_visits_increases_ucb1(self):
        """less-explored nodes should get a bonus (exploration term)"""
        parent = MCTSNode(state=[], visits=100, total_reward=50.0)
        explored = MCTSNode(state=[], parent=parent, visits=50, total_reward=25.0)
        fresh = MCTSNode(state=[], parent=parent, visits=5, total_reward=2.5)
        parent.children = [explored, fresh]

        # same reward ratio (0.5) but fresh has way more exploration bonus
        assert fresh.ucb1 > explored.ucb1

    def test_root_node_ucb1_is_zero(self):
        """root has no parent so UCB1 is meaningless - should be 0"""
        root = MCTSNode(state=[], visits=100, total_reward=50.0)
        assert root.ucb1 == 0.0


# ---- backpropagation tests ----

class TestBackpropagation:
    """Backprop reward from leaf to root - every ancestor gets the update."""

    def test_single_node_backprop(self, planner):
        """reward propagates to the node itself"""
        node = MCTSNode(state=[], visits=0, total_reward=0.0)
        planner._backpropagate(node, 0.8)
        assert node.visits == 1
        assert node.total_reward == 0.8

    def test_chain_backprop(self, planner):
        """reward walks all the way up from leaf to root"""
        root = MCTSNode(state=[], visits=5, total_reward=2.0)
        mid = MCTSNode(state=[], parent=root, visits=3, total_reward=1.5)
        leaf = MCTSNode(state=[], parent=mid, visits=1, total_reward=0.5)
        root.children = [mid]
        mid.children = [leaf]

        planner._backpropagate(leaf, 0.9)

        # leaf gets the update
        assert leaf.visits == 2
        assert abs(leaf.total_reward - 1.4) < 0.001

        # mid gets it too
        assert mid.visits == 4
        assert abs(mid.total_reward - 2.4) < 0.001

        # root gets it
        assert root.visits == 6
        assert abs(root.total_reward - 2.9) < 0.001

    def test_backprop_doesnt_affect_siblings(self, planner):
        """only ancestors get updated, not uncle nodes"""
        root = MCTSNode(state=[], visits=10, total_reward=5.0)
        left = MCTSNode(state=[], parent=root, visits=5, total_reward=2.5)
        right = MCTSNode(state=[], parent=root, visits=5, total_reward=2.5)
        root.children = [left, right]

        planner._backpropagate(left, 1.0)

        assert left.visits == 6
        assert right.visits == 5  # untouched


# ---- best path extraction tests ----

class TestBestPathExtraction:
    """Extract the best decomposition by following most-visited children."""

    def test_empty_tree_returns_empty(self, planner):
        """no children means no path"""
        root = MCTSNode(state=[])
        result = planner._extract_best_path(root)
        assert result == []

    def test_follows_most_visited(self, planner, sample_subtasks):
        """should greedily follow the child with most visits at each level"""
        root = MCTSNode(state=[], visits=100)
        good_child = MCTSNode(
            state=[sample_subtasks[0]],
            parent=root,
            visits=60,
            total_reward=42.0,
        )
        bad_child = MCTSNode(
            state=[sample_subtasks[1]],
            parent=root,
            visits=40,
            total_reward=20.0,
        )
        root.children = [bad_child, good_child]

        result = planner._extract_best_path(root)
        assert len(result) == 1
        assert result[0].agent_type == "researcher"

    def test_multi_level_extraction(self, planner, sample_subtasks):
        """follows the chain through multiple levels"""
        root = MCTSNode(state=[], visits=100)
        level1 = MCTSNode(
            state=[sample_subtasks[0]],
            parent=root,
            visits=80,
        )
        level2 = MCTSNode(
            state=[sample_subtasks[0], sample_subtasks[1]],
            parent=level1,
            visits=50,
        )
        root.children = [level1]
        level1.children = [level2]

        result = planner._extract_best_path(root)
        assert len(result) == 2
        assert result[0].agent_type == "researcher"
        assert result[1].agent_type == "coder"


# ---- selection tests ----

class TestSelection:
    """Selection phase - walk to the most promising expandable leaf."""

    def test_selects_node_with_untried_actions(self, planner):
        """if current node has untried actions, stop there"""
        root = MCTSNode(
            state=[],
            visits=10,
            untried_actions=[{"description": "x", "agent_type": "coder"}],
        )
        child = MCTSNode(state=[], parent=root, visits=5, total_reward=2.5)
        root.children = [child]

        selected = planner._select(root)
        assert selected is root  # has untried actions, expand here

    def test_selects_leaf_when_no_untried(self, planner):
        """if no untried actions, walk to leaf following UCB1"""
        root = MCTSNode(state=[], visits=50, total_reward=25.0)
        left = MCTSNode(state=[], parent=root, visits=30, total_reward=20.0)
        right = MCTSNode(state=[], parent=root, visits=20, total_reward=5.0)
        root.children = [left, right]

        selected = planner._select(root)
        # left has better UCB1 (higher reward ratio) so should be selected
        assert selected is left


# ---- integration test ----

class TestPlannerIntegration:
    """End-to-end MCTS planning with mocked LLM."""

    @pytest.mark.asyncio
    async def test_plan_produces_valid_decomposition(self, planner):
        """full plan() call should return a valid DecompositionPlan"""
        result = await planner.plan(
            task_description="Fix the pagination bug in the user list endpoint",
            repo_context={"languages": ["python"], "framework": "django"},
        )

        assert isinstance(result, DecompositionPlan)
        assert result.total_simulations == 20
        assert result.expected_success_rate >= 0.0
        assert result.expected_success_rate <= 1.0
        assert len(result.subtasks) >= 1
        # should have generated a task id from the description
        assert len(result.task_id) == 12

    @pytest.mark.asyncio
    async def test_plan_respects_max_depth(self, mock_llm):
        """planner shouldn't produce plans deeper than max_depth"""
        shallow = MCTSPlanner(llm_client=mock_llm, simulations=10, max_depth=2)
        result = await shallow.plan(
            task_description="Add caching layer",
            repo_context={"languages": ["python"]},
        )
        assert result.tree_depth <= 2

    @pytest.mark.asyncio
    async def test_procedural_memory_integration(self, mock_llm):
        """when procedural memory is provided, it should influence scoring"""
        memory = AsyncMock()
        memory.retrieve_similar_strategies.return_value = [
            MagicMock(subtasks=[], success_rate=0.9),
            MagicMock(subtasks=[], success_rate=0.85),
        ]

        planner = MCTSPlanner(
            llm_client=mock_llm,
            procedural_memory=memory,
            simulations=10,
        )

        result = await planner.plan(
            task_description="Refactor auth module",
            repo_context={"languages": ["python"]},
        )

        # memory should have been consulted
        assert memory.retrieve_similar_strategies.called
        assert isinstance(result, DecompositionPlan)


# ---- helper method tests ----

class TestHelperMethods:
    """Test utility methods on the planner."""

    def test_has_dependency_coder_depends_on_researcher(self, planner):
        researcher = Subtask(id="r0", description="research", agent_type="researcher")
        action = {"agent_type": "coder", "description": "code stuff"}
        assert planner._has_dependency(researcher, action) is True

    def test_has_dependency_reviewer_depends_on_coder(self, planner):
        coder = Subtask(id="c0", description="write code", agent_type="coder")
        action = {"agent_type": "reviewer", "description": "review"}
        assert planner._has_dependency(coder, action) is True

    def test_has_dependency_no_dep(self, planner):
        researcher = Subtask(id="r0", description="research", agent_type="researcher")
        action = {"agent_type": "researcher", "description": "more research"}
        assert planner._has_dependency(researcher, action) is False

    def test_generate_task_id_deterministic(self, planner):
        """same input should always produce the same task ID"""
        id1 = planner._generate_task_id("fix the bug")
        id2 = planner._generate_task_id("fix the bug")
        assert id1 == id2
        assert len(id1) == 12

    def test_generate_task_id_unique(self, planner):
        """different inputs should produce different IDs"""
        id1 = planner._generate_task_id("fix the bug")
        id2 = planner._generate_task_id("add the feature")
        assert id1 != id2

    def test_measure_depth_empty(self, planner):
        root = MCTSNode(state=[])
        assert planner._measure_depth(root) == 0

    def test_measure_depth_chain(self, planner):
        root = MCTSNode(state=[])
        child = MCTSNode(state=[], parent=root)
        grandchild = MCTSNode(state=[], parent=child)
        root.children = [child]
        child.children = [grandchild]
        assert planner._measure_depth(root) == 2

    def test_default_actions_always_returns_something(self, planner):
        """fallback actions should never be empty"""
        actions = planner._default_actions("whatever task")
        assert len(actions) == 3
        types = {a["agent_type"] for a in actions}
        assert "researcher" in types
        assert "coder" in types
        assert "reviewer" in types
