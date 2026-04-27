"""SWE-bench evaluation harness - runs tasks, validates patches, scores results."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_orchestrator.sandbox import ExecutionResult, SandboxExecutor


class EvalTask(BaseModel):
    """A single evaluation task from SWE-bench or custom suite."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    test_patch: str
    patch_gold: str = ""
    difficulty: str = "medium"
    issue_type: str = "bug_fix"


class EvalResult(BaseModel):
    """Result of running one evaluation task."""

    instance_id: str
    resolved: bool
    generated_patch: str
    test_output: str
    exit_code: int
    time_seconds: float
    cost_usd: float
    tokens_used: int
    agents_invoked: list[str]
    retries: int
    error: str | None = None


@dataclass
class EvalMetrics:
    """Aggregate metrics across an evaluation run."""

    total: int = 0
    resolved: int = 0
    failed: int = 0
    errored: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.resolved / self.total if self.total > 0 else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.total if self.total > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "resolved": self.resolved,
            "pass_rate": round(self.pass_rate, 4),
            "avg_cost_usd": round(self.avg_cost, 4),
            "avg_time_seconds": round(self.avg_time, 1),
            "total_tokens": self.total_tokens,
        }


class SWEBenchEvaluator:
    """
    Runs the full agent pipeline against SWE-bench tasks and validates results.

    The flow for each task:
    1. Clone repo at the specified base commit
    2. Feed problem statement to the orchestrator
    3. Capture the generated patch
    4. Apply patch and run test suite
    5. Compare test results against expected (test_patch from SWE-bench)
    """

    # 5 min timeout per task - if you can't solve it by then it's cooked
    TASK_TIMEOUT: int = 300
    # max concurrent evals - limited by API rate limits more than compute
    MAX_CONCURRENCY: int = 4

    def __init__(
        self,
        orchestrator: Any,
        sandbox: SandboxExecutor | None = None,
        workspace_dir: str | None = None,
        timeout: int = TASK_TIMEOUT,
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        self.orchestrator = orchestrator
        self.sandbox = sandbox or SandboxExecutor()
        self.workspace = Path(workspace_dir or tempfile.mkdtemp(prefix="swebench_eval_"))
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run_evaluation(
        self,
        tasks: list[EvalTask],
        output_path: str | None = None,
    ) -> EvalMetrics:
        """
        Run evaluation across all tasks with bounded concurrency.

        Writes results incrementally so partial progress isn't lost if we crash.
        """
        metrics = EvalMetrics()

        # run tasks with a semaphore so we don't hammer the API
        coros = [self._eval_single_task(task, metrics) for task in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # task blew up in a way we didn't expect - log it and keep going
                metrics.errored += 1
                metrics.total += 1
                metrics.results.append(EvalResult(
                    instance_id=tasks[i].instance_id,
                    resolved=False,
                    generated_patch="",
                    test_output="",
                    exit_code=-1,
                    time_seconds=0,
                    cost_usd=0,
                    tokens_used=0,
                    agents_invoked=[],
                    retries=0,
                    error=str(result),
                ))

        if output_path:
            self._write_results(metrics, output_path)

        return metrics

    async def _eval_single_task(self, task: EvalTask, metrics: EvalMetrics) -> EvalResult:
        """Evaluate a single task end-to-end."""
        async with self._semaphore:
            start = time.perf_counter()

            try:
                # step 1: prepare the repo checkout
                repo_path = await self._prepare_repo(task)

                # step 2: run the orchestrator to generate a patch
                orchestrator_result = await asyncio.wait_for(
                    self.orchestrator.solve(
                        problem_statement=task.problem_statement,
                        repo_path=str(repo_path),
                        hints=task.hints_text,
                    ),
                    timeout=self.timeout,
                )

                patch = orchestrator_result.patch
                agents = orchestrator_result.agents_used
                tokens = orchestrator_result.total_tokens
                cost = orchestrator_result.cost_usd
                retries = orchestrator_result.retries

                # step 3: apply patch and validate
                validation = await self._validate_patch(
                    patch=patch,
                    repo_path=repo_path,
                    test_patch=task.test_patch,
                )

                elapsed = time.perf_counter() - start
                resolved = validation.exit_code == 0

                result = EvalResult(
                    instance_id=task.instance_id,
                    resolved=resolved,
                    generated_patch=patch,
                    test_output=validation.stdout + validation.stderr,
                    exit_code=validation.exit_code,
                    time_seconds=elapsed,
                    cost_usd=cost,
                    tokens_used=tokens,
                    agents_invoked=agents,
                    retries=retries,
                )

            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - start
                result = EvalResult(
                    instance_id=task.instance_id,
                    resolved=False,
                    generated_patch="",
                    test_output="",
                    exit_code=-1,
                    time_seconds=elapsed,
                    cost_usd=0,
                    tokens_used=0,
                    agents_invoked=[],
                    retries=0,
                    error=f"Task timed out after {self.timeout}s",
                )

            except Exception as e:
                elapsed = time.perf_counter() - start
                result = EvalResult(
                    instance_id=task.instance_id,
                    resolved=False,
                    generated_patch="",
                    test_output="",
                    exit_code=-1,
                    time_seconds=elapsed,
                    cost_usd=0,
                    tokens_used=0,
                    agents_invoked=[],
                    retries=0,
                    error=str(e),
                )

            # update aggregate metrics
            metrics.total += 1
            if result.resolved:
                metrics.resolved += 1
            else:
                metrics.failed += 1
            metrics.total_cost += result.cost_usd
            metrics.total_tokens += result.tokens_used
            metrics.total_time += result.time_seconds
            metrics.results.append(result)

            return result

    async def _prepare_repo(self, task: EvalTask) -> Path:
        """Clone and checkout the repo at the specified base commit."""
        repo_dir = self.workspace / task.instance_id.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)

        # shallow clone - we only need the one commit
        # full clone would be insane for django (2GB+)
        await asyncio.to_thread(
            subprocess.run,
            [
                "git", "clone", "--depth", "1",
                f"https://github.com/{task.repo}.git",
                str(repo_dir),
            ],
            capture_output=True,
            check=True,
        )

        # fetch the specific commit we need
        await asyncio.to_thread(
            subprocess.run,
            ["git", "fetch", "origin", task.base_commit, "--depth", "1"],
            cwd=str(repo_dir),
            capture_output=True,
        )

        await asyncio.to_thread(
            subprocess.run,
            ["git", "checkout", task.base_commit],
            cwd=str(repo_dir),
            capture_output=True,
        )

        return repo_dir

    async def _validate_patch(
        self,
        patch: str,
        repo_path: Path,
        test_patch: str,
    ) -> ExecutionResult:
        """
        Apply the generated patch and run tests to validate correctness.

        Steps:
        1. Apply the agent's patch to the repo
        2. Apply the test patch (new tests from the issue resolution)
        3. Run the test suite in sandbox
        4. Return pass/fail based on exit code
        """
        commands = []

        # apply the generated patch
        if patch:
            patch_file = repo_path / "agent_patch.diff"
            patch_file.write_text(patch)
            commands.append(f"cd {repo_path} && git apply agent_patch.diff")

        # apply the test patch (these are the tests that should pass if fixed)
        if test_patch:
            test_file = repo_path / "test_patch.diff"
            test_file.write_text(test_patch)
            commands.append(f"cd {repo_path} && git apply test_patch.diff")

        # run the test suite
        commands.append(f"cd {repo_path} && python -m pytest --tb=short -q 2>&1 | tail -50")

        return await self.sandbox.execute(
            commands=commands,
            repo_path=str(repo_path),
            timeout=120,  # tests shouldn't take more than 2 min
        )

    def _write_results(self, metrics: EvalMetrics, output_path: str) -> None:
        """Write evaluation results to JSON."""
        output = {
            "summary": metrics.to_dict(),
            "results": [r.model_dump() for r in metrics.results],
        }
        Path(output_path).write_text(json.dumps(output, indent=2))

    @staticmethod
    def load_tasks(dataset_path: str) -> list[EvalTask]:
        """Load evaluation tasks from a JSON dataset file."""
        data = json.loads(Path(dataset_path).read_text())
        return [EvalTask(**item) for item in data]
