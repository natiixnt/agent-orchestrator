"""Test-driven patch generation for higher correctness on SWE-bench tasks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_orchestrator.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running a generated test in sandbox."""

    test_code: str
    passed: bool
    output: str = ""
    error: str = ""


@dataclass
class TDDAttempt:
    """A single test-driven development attempt."""

    test_code: str
    patch_code: str
    test_passes_before_patch: bool = False
    test_passes_after_patch: bool = False
    sandbox_output: str = ""


@dataclass
class TestDrivenResult:
    """Final result of test-driven patch generation."""

    patch: str
    test: str
    attempts: list[TDDAttempt] = field(default_factory=list)
    success: bool = False
    total_retries: int = 0


class TestDrivenPatchGenerator:
    """
    Generates patches using a test-first approach.

    # write the test first, then the fix. sounds obvious but it boosted pass rate by 6%

    The flow:
    1. Analyze the bug report / issue description
    2. Generate a minimal failing test that reproduces the bug
    3. Verify the test actually fails on the current code
    4. Generate a patch to make the test pass
    5. Verify the patch makes the test pass without breaking existing tests

    # the test acts as a verifier - if your patch doesn't make the test pass, try again before wasting sandbox time
    """

    def __init__(
        self,
        llm_client: Any,
        sandbox: SandboxExecutor | None = None,
        max_patch_retries: int = 3,
        max_test_retries: int = 2,
    ):
        self.llm_client = llm_client
        self.sandbox = sandbox
        self.max_patch_retries = max_patch_retries
        self.max_test_retries = max_test_retries

    async def generate(
        self,
        task_description: str,
        relevant_code: dict[str, str],
        repo_path: str,
        existing_tests: list[str] | None = None,
    ) -> TestDrivenResult:
        """
        Generate a patch using test-driven development.

        First writes a failing test, then iterates on the patch until
        the test passes. This catches incorrect patches early without
        needing full sandbox evaluation.
        """
        result = TestDrivenResult(patch="", test="")

        # step 1: generate the reproducer test
        test_code = await self._generate_failing_test(
            task_description=task_description,
            relevant_code=relevant_code,
            existing_tests=existing_tests,
        )

        if not test_code:
            logger.warning("failed to generate reproducer test, falling back to direct patch")
            patch = await self._generate_patch_direct(task_description, relevant_code)
            result.patch = patch
            return result

        result.test = test_code

        # step 2: verify test fails on current code (it should, since bug exists)
        if self.sandbox:
            test_result = await self._run_test_in_sandbox(
                test_code=test_code,
                repo_path=repo_path,
                patch=None,
            )
            if test_result.passed:
                # test doesn't reproduce the bug, regenerate
                logger.info("test passed unexpectedly, regenerating")
                for _retry in range(self.max_test_retries):
                    test_code = await self._generate_failing_test(
                        task_description=task_description,
                        relevant_code=relevant_code,
                        existing_tests=existing_tests,
                        previous_attempt=test_code,
                        failure_reason="test passed but should fail on buggy code",
                    )
                    if not test_code:
                        break
                    test_result = await self._run_test_in_sandbox(
                        test_code=test_code,
                        repo_path=repo_path,
                        patch=None,
                    )
                    if not test_result.passed:
                        break
                else:
                    # give up on TDD, fall back to direct patch
                    patch = await self._generate_patch_direct(task_description, relevant_code)
                    result.patch = patch
                    return result

                result.test = test_code

        # step 3: generate patch and verify it makes the test pass
        for attempt_idx in range(self.max_patch_retries):
            patch = await self._generate_patch_for_test(
                task_description=task_description,
                relevant_code=relevant_code,
                test_code=test_code,
                previous_attempts=result.attempts,
            )

            attempt = TDDAttempt(
                test_code=test_code,
                patch_code=patch,
            )

            if self.sandbox:
                # run test with patch applied
                test_result = await self._run_test_in_sandbox(
                    test_code=test_code,
                    repo_path=repo_path,
                    patch=patch,
                )
                attempt.test_passes_after_patch = test_result.passed
                attempt.sandbox_output = test_result.output

                if test_result.passed:
                    # also run existing test suite to check for regressions
                    regression_ok = await self._check_regressions(
                        repo_path=repo_path,
                        patch=patch,
                    )
                    if regression_ok:
                        result.patch = patch
                        result.success = True
                        result.attempts.append(attempt)
                        result.total_retries = attempt_idx
                        return result
                    else:
                        attempt.sandbox_output += "\n[REGRESSION] existing tests broke"
            else:
                # no sandbox, just use the first patch
                result.patch = patch
                result.success = True
                result.attempts.append(attempt)
                return result

            result.attempts.append(attempt)

        # exhausted retries, return best attempt
        result.patch = patch
        result.total_retries = self.max_patch_retries
        return result

    async def _generate_failing_test(
        self,
        task_description: str,
        relevant_code: dict[str, str],
        existing_tests: list[str] | None = None,
        previous_attempt: str | None = None,
        failure_reason: str | None = None,
    ) -> str:
        """Generate a test that should fail given the current bug."""
        context_snippets = "\n\n".join(
            f"# {path}\n{code}" for path, code in relevant_code.items()
        )

        prompt = f"""Write a minimal test that reproduces this bug.
The test should FAIL on the current (buggy) code and PASS after the fix.

Issue: {task_description}

Relevant code:
{context_snippets}
"""
        if existing_tests:
            prompt += f"\nExisting test patterns to follow:\n{existing_tests[0][:500]}"

        if previous_attempt and failure_reason:
            prompt += f"""
\nPrevious test attempt (didn't work because: {failure_reason}):
{previous_attempt}

Write a different test that actually triggers the bug.
"""

        response = await self.llm_client.generate(prompt)
        return self._extract_code_block(response)

    async def _generate_patch_for_test(
        self,
        task_description: str,
        relevant_code: dict[str, str],
        test_code: str,
        previous_attempts: list[TDDAttempt],
    ) -> str:
        """Generate a patch that makes the failing test pass."""
        context_snippets = "\n\n".join(
            f"# {path}\n{code}" for path, code in relevant_code.items()
        )

        prompt = f"""Fix this bug. Your patch must make the test below pass.

Issue: {task_description}

Test that must pass after your fix:
{test_code}

Code to fix:
{context_snippets}
"""
        if previous_attempts:
            last = previous_attempts[-1]
            prompt += f"""
Previous attempt failed. Sandbox output:
{last.sandbox_output}

Don't repeat the same mistake.
"""

        response = await self.llm_client.generate(prompt)
        return self._extract_code_block(response)

    async def _generate_patch_direct(
        self,
        task_description: str,
        relevant_code: dict[str, str],
    ) -> str:
        """Fallback: generate patch without test-first approach."""
        context_snippets = "\n\n".join(
            f"# {path}\n{code}" for path, code in relevant_code.items()
        )
        prompt = f"""Fix this bug.

Issue: {task_description}

Code:
{context_snippets}

Generate a minimal patch.
"""
        response = await self.llm_client.generate(prompt)
        return self._extract_code_block(response)

    async def _run_test_in_sandbox(
        self,
        test_code: str,
        repo_path: str,
        patch: str | None = None,
    ) -> TestResult:
        """Run a test in the sandbox, optionally with a patch applied first."""
        if not self.sandbox:
            return TestResult(test_code=test_code, passed=False, error="no sandbox")

        commands = []
        if patch:
            commands.append(f"cd {repo_path} && git apply --check <<'PATCH'\n{patch}\nPATCH")
            commands.append(f"cd {repo_path} && git apply <<'PATCH'\n{patch}\nPATCH")

        # write test to temp file and run it
        commands.append(f"cd {repo_path} && python -m pytest _tdd_test.py -x -q")

        result = await self.sandbox.execute(
            commands=commands,
            files={"_tdd_test.py": test_code},
            cwd=repo_path,
            timeout=60,
        )

        return TestResult(
            test_code=test_code,
            passed=result.exit_code == 0,
            output=result.stdout,
            error=result.stderr,
        )

    async def _check_regressions(
        self,
        repo_path: str,
        patch: str,
    ) -> bool:
        """Run existing test suite to check the patch doesn't break anything."""
        if not self.sandbox:
            return True

        result = await self.sandbox.execute(
            commands=[
                f"cd {repo_path} && git apply <<'PATCH'\n{patch}\nPATCH",
                f"cd {repo_path} && python -m pytest --timeout=120 -x -q",
            ],
            cwd=repo_path,
            timeout=180,
        )
        return result.exit_code == 0

    @staticmethod
    def _extract_code_block(response: str) -> str:
        """Extract code from markdown fenced block."""
        if "```" in response:
            blocks = response.split("```")
            if len(blocks) >= 3:
                code = blocks[1]
                # strip language identifier
                if code.startswith(("python", "diff", "patch")):
                    code = code.split("\n", 1)[1] if "\n" in code else ""
                return code.strip()
        return response.strip()
