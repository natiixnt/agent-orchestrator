"""Tree-of-Thought verification: branch into parallel approaches, pick the best."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from agent_orchestrator.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)


@dataclass
class BranchResult:
    """Result from a single ToT branch."""

    branch_id: int
    temperature: float
    patch: str
    test_passed: bool = False
    existing_tests_passed: bool = False
    sandbox_output: str = ""
    score: float = 0.0
    cost_tokens: int = 0


@dataclass
class TreeOfThoughtResult:
    """Aggregated result from Tree-of-Thought evaluation."""

    best_patch: str
    best_branch: BranchResult | None = None
    all_branches: list[BranchResult] = field(default_factory=list)
    total_cost_tokens: int = 0
    any_passed: bool = False


class TreeOfThoughtVerifier:
    """
    Branch into multiple parallel coding attempts and pick the best one.

    # 3 parallel attempts sound expensive but it's cheaper than 3 sequential retries

    Instead of generating one patch and retrying on failure, we generate N patches
    in parallel with different temperatures, evaluate each in sandbox, and pick the
    winner. This exploits the fact that LLM inference is embarrassingly parallel
    while sandbox evaluation is fast.

    # diversity trick: vary temperature (0.2, 0.5, 0.8) across branches to get genuinely different approaches
    """

    DEFAULT_TEMPERATURES = [0.2, 0.5, 0.8]

    def __init__(
        self,
        llm_client: Any,
        sandbox: SandboxExecutor | None = None,
        num_branches: int = 3,
        temperatures: list[float] | None = None,
        selection_strategy: str = "best_score",
    ):
        self.llm_client = llm_client
        self.sandbox = sandbox
        self.num_branches = num_branches
        self.temperatures = temperatures or self.DEFAULT_TEMPERATURES[: num_branches]
        self.selection_strategy = selection_strategy

        # pad temperatures if fewer than branches
        while len(self.temperatures) < self.num_branches:
            self.temperatures.append(0.5)

    async def generate_and_verify(
        self,
        task_description: str,
        relevant_code: dict[str, str],
        repo_path: str,
        test_code: str | None = None,
    ) -> TreeOfThoughtResult:
        """
        Generate multiple patch candidates in parallel, evaluate each, pick best.

        Returns the best patch along with all branch results for debugging.
        """
        # generate patches in parallel across different temperatures
        branch_tasks = [
            self._generate_branch(
                branch_id=i,
                temperature=self.temperatures[i],
                task_description=task_description,
                relevant_code=relevant_code,
            )
            for i in range(self.num_branches)
        ]

        branches: list[BranchResult] = await asyncio.gather(*branch_tasks)

        # evaluate each branch in parallel (sandbox runs are independent)
        if self.sandbox:
            eval_tasks = [
                self._evaluate_branch(
                    branch=branch,
                    repo_path=repo_path,
                    test_code=test_code,
                )
                for branch in branches
            ]
            branches = await asyncio.gather(*eval_tasks)

        # score and select the best branch
        for branch in branches:
            branch.score = self._compute_score(branch)

        result = TreeOfThoughtResult(
            best_patch="",
            all_branches=branches,
            total_cost_tokens=sum(b.cost_tokens for b in branches),
            any_passed=any(b.test_passed for b in branches),
        )

        # select winner
        best = self._select_best(branches)
        if best:
            result.best_patch = best.patch
            result.best_branch = best

        return result

    async def _generate_branch(
        self,
        branch_id: int,
        temperature: float,
        task_description: str,
        relevant_code: dict[str, str],
    ) -> BranchResult:
        """Generate a single patch candidate at a given temperature."""
        context_snippets = "\n\n".join(
            f"# {path}\n{code}" for path, code in relevant_code.items()
        )

        prompt = f"""Fix this issue. Generate a minimal, correct patch.

Issue: {task_description}

Code:
{context_snippets}

Output a unified diff patch.
"""
        response = await self.llm_client.generate(
            prompt,
            temperature=temperature,
        )

        patch = self._extract_patch(response)

        return BranchResult(
            branch_id=branch_id,
            temperature=temperature,
            patch=patch,
            cost_tokens=len(prompt.split()) + len(response.split()),  # rough estimate
        )

    async def _evaluate_branch(
        self,
        branch: BranchResult,
        repo_path: str,
        test_code: str | None = None,
    ) -> BranchResult:
        """Evaluate a branch's patch in sandbox."""
        if not self.sandbox or not branch.patch:
            return branch

        # run the TDD test if provided
        if test_code:
            result = await self.sandbox.execute(
                commands=[
                    f"cd {repo_path} && git apply <<'PATCH'\n{branch.patch}\nPATCH",
                    f"cd {repo_path} && python -m pytest _tot_test.py -x -q",
                ],
                files={"_tot_test.py": test_code},
                cwd=repo_path,
                timeout=60,
            )
            branch.test_passed = result.exit_code == 0
            branch.sandbox_output = result.stdout

        # run existing tests to check for regressions
        regression_result = await self.sandbox.execute(
            commands=[
                f"cd {repo_path} && git stash",  # clean up from test above
                f"cd {repo_path} && git apply <<'PATCH'\n{branch.patch}\nPATCH",
                f"cd {repo_path} && python -m pytest --timeout=120 -x -q",
            ],
            cwd=repo_path,
            timeout=180,
        )
        branch.existing_tests_passed = regression_result.exit_code == 0

        return branch

    def _compute_score(self, branch: BranchResult) -> float:
        """
        Score a branch based on evaluation results.

        Scoring priorities:
        1. TDD test passes (the new test we wrote to verify the fix)
        2. Existing tests pass (no regressions)
        3. Patch exists and is non-empty
        """
        score = 0.0

        if not branch.patch:
            return 0.0

        # base score for having a patch
        score += 1.0

        # big bonus for passing the reproducer test
        if branch.test_passed:
            score += 5.0

        # bonus for not breaking existing tests
        if branch.existing_tests_passed:
            score += 3.0

        # slight preference for lower temperature (more focused) if all else equal
        score -= branch.temperature * 0.1

        return score

    def _select_best(self, branches: list[BranchResult]) -> BranchResult | None:
        """Select the best branch based on selection strategy."""
        if not branches:
            return None

        if self.selection_strategy == "best_score":
            return max(branches, key=lambda b: b.score)
        elif self.selection_strategy == "first_passing":
            passing = [b for b in branches if b.test_passed and b.existing_tests_passed]
            if passing:
                return passing[0]
            return max(branches, key=lambda b: b.score)
        else:
            return max(branches, key=lambda b: b.score)

    @staticmethod
    def _extract_patch(response: str) -> str:
        """Extract patch content from LLM response."""
        if "```" in response:
            blocks = response.split("```")
            if len(blocks) >= 3:
                code = blocks[1]
                if code.startswith(("diff", "patch", "python")):
                    code = code.split("\n", 1)[1] if "\n" in code else ""
                return code.strip()
        return response.strip()
