"""Reviewer agent for code review and quality assurance."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel

from agent_orchestrator.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)


class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


class ReviewComment(BaseModel):
    """A single review comment on a specific code location."""

    file_path: str
    line_start: int
    line_end: int | None = None
    severity: str  # critical, warning, suggestion
    category: str  # correctness, style, performance, security
    message: str
    suggested_fix: str | None = None


class ReviewResult(BaseModel):
    """Structured result of a code review."""

    verdict: ReviewVerdict
    summary: str
    comments: list[ReviewComment]
    tests_pass: bool
    style_compliant: bool
    security_issues: list[str]
    confidence: float


class ReviewerAgent(BaseAgent):
    """
    Agent specialized in reviewing code patches for correctness, style,
    security, and best practices.

    Uses procedural memory to apply learned review heuristics and adapts
    its review criteria based on the repository's conventions.
    """

    def __init__(
        self,
        llm_client: Any,
        tools: list[Any] | None = None,
        memory_stores: dict[str, Any] | None = None,
        strict_mode: bool = False,
    ) -> None:
        super().__init__(
            name="reviewer",
            llm_client=llm_client,
            tools=tools,
            memory_stores=memory_stores,
            max_iterations=8,
            temperature=0.0,
        )
        self.strict_mode = strict_mode

    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Review a code patch produced by the coder agent.

        Review pipeline:
        1. Extract patch from coder agent's artifacts
        2. Retrieve review heuristics from procedural memory
        3. Run automated checks (tests, lint, type check)
        4. LLM-based review for correctness and logic issues
        5. Security analysis for common vulnerability patterns
        6. Produce structured review with verdict
        """
        start_time = time.perf_counter()
        self.status = AgentStatus.RUNNING
        self._tool_calls = []
        self._tokens_used = 0

        try:
            patch = self._extract_patch(context)
            if not patch:
                raise ValueError("No patch found in context to review")

            # Retrieve learned review strategies
            review_heuristics = await self.query_memory(
                f"review patterns for: {context.task_description}", "procedural", k=5
            )

            # Run automated validation
            auto_results = await self._run_automated_checks(patch, context)

            # LLM-based deep review
            review = await self._perform_llm_review(
                patch, context, review_heuristics, auto_results
            )

            # Security scan
            security_issues = await self._security_analysis(patch, context)
            review.security_issues = security_issues

            # Determine final verdict
            review.verdict = self._determine_verdict(review, auto_results)

            artifacts = {
                "review": review.model_dump(),
                "automated_results": auto_results,
                "patch_reviewed": patch[:500],
            }

            duration = time.perf_counter() - start_time
            return self._build_result(
                status=AgentStatus.COMPLETED,
                output=f"Review complete: {review.verdict.value} "
                       f"({len(review.comments)} comments, "
                       f"{len(security_issues)} security issues)",
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
        strictness = "strict" if self.strict_mode else "balanced"
        return (
            f"You are a code review agent operating in {strictness} mode. "
            "Your role is to review code patches for:\n"
            "1. Correctness: Does the code solve the stated problem?\n"
            "2. Edge cases: Are boundary conditions handled?\n"
            "3. Style: Does it match the repository's conventions?\n"
            "4. Performance: Are there unnecessary allocations or O(n^2) patterns?\n"
            "5. Security: Are there injection risks, unsafe operations, or data leaks?\n\n"
            "Provide specific, actionable feedback with line references.\n"
            f"Task being solved: {context.task_description}\n"
        )

    async def _run_automated_checks(
        self, patch: str, context: AgentContext
    ) -> dict[str, Any]:
        """Run linting, type checking, and tests on the patch."""
        results: dict[str, Any] = {
            "tests_pass": False,
            "lint_clean": False,
            "type_check_pass": False,
            "errors": [],
        }

        # Apply patch and run checks via tools
        try:
            test_result = await self.run_tool(
                "run_tests",
                {"repo_path": context.repo_path, "patch": patch},
            )
            results["tests_pass"] = test_result.get("passed", False)
            results["test_output"] = test_result.get("output", "")
        except Exception as e:
            results["errors"].append(f"Test execution failed: {e}")

        try:
            lint_result = await self.run_tool(
                "run_lint",
                {"repo_path": context.repo_path, "patch": patch},
            )
            results["lint_clean"] = lint_result.get("clean", False)
            results["lint_issues"] = lint_result.get("issues", [])
        except Exception as e:
            results["errors"].append(f"Lint failed: {e}")

        return results

    async def _perform_llm_review(
        self,
        patch: str,
        context: AgentContext,
        heuristics: list[Any],
        auto_results: dict[str, Any],
    ) -> ReviewResult:
        """Perform LLM-based code review with learned heuristics."""
        heuristic_text = ""
        if heuristics:
            heuristic_text = "Review heuristics from past experience:\n" + "\n".join(
                f"- {h.content}" for h in heuristics if hasattr(h, "content")
            )

        messages = [
            {"role": "system", "content": self.get_system_prompt(context)},
            {
                "role": "user",
                "content": (
                    f"Review this patch:\n```diff\n{patch}\n```\n\n"
                    f"{heuristic_text}\n\n"
                    f"Automated check results: tests_pass={auto_results['tests_pass']}, "
                    f"lint_clean={auto_results['lint_clean']}\n\n"
                    "Respond with JSON containing:\n"
                    "- summary: brief review summary\n"
                    "- comments: array of {file_path, line_start, severity, category, message}\n"
                    "- confidence: 0-1 confidence in your review"
                ),
            },
        ]

        response = await self._call_llm(messages)
        parsed = self._parse_review_response(response.content, auto_results)
        return parsed

    async def _security_analysis(
        self, patch: str, context: AgentContext
    ) -> list[str]:
        """Scan patch for common security vulnerability patterns."""
        security_patterns = [
            ("SQL injection", ["execute(", "raw(", "f\"SELECT", "f'SELECT"]),
            ("Command injection", ["subprocess.call(", "os.system(", "shell=True"]),
            ("Path traversal", ["../", "os.path.join(user_input"]),
            ("Hardcoded secrets", ["password=", "api_key=", "secret="]),
            ("Unsafe deserialization", ["pickle.loads(", "yaml.load(", "eval("]),
        ]

        issues = []
        added_lines = [
            line[1:] for line in patch.split("\n") if line.startswith("+") and not line.startswith("+++")
        ]
        added_content = "\n".join(added_lines)

        for vuln_name, patterns in security_patterns:
            for pattern in patterns:
                if pattern in added_content:
                    issues.append(
                        f"Potential {vuln_name}: found '{pattern}' in added lines"
                    )

        return issues

    def _determine_verdict(
        self, review: ReviewResult, auto_results: dict[str, Any]
    ) -> ReviewVerdict:
        """Determine final verdict based on review findings and automated checks."""
        critical_comments = [c for c in review.comments if c.severity == "critical"]

        if critical_comments or review.security_issues:
            return ReviewVerdict.REJECT

        if not auto_results.get("tests_pass", False) and self.strict_mode:
            return ReviewVerdict.REQUEST_CHANGES

        warning_comments = [c for c in review.comments if c.severity == "warning"]
        if len(warning_comments) > 3:
            return ReviewVerdict.REQUEST_CHANGES

        return ReviewVerdict.APPROVE

    def _parse_review_response(
        self, content: str, auto_results: dict[str, Any]
    ) -> ReviewResult:
        """Parse LLM review response into structured ReviewResult."""
        import json

        try:
            data = json.loads(content)
            comments = [
                ReviewComment(
                    file_path=c.get("file_path", "unknown"),
                    line_start=c.get("line_start", 0),
                    line_end=c.get("line_end"),
                    severity=c.get("severity", "suggestion"),
                    category=c.get("category", "correctness"),
                    message=c.get("message", ""),
                    suggested_fix=c.get("suggested_fix"),
                )
                for c in data.get("comments", [])
            ]
            return ReviewResult(
                verdict=ReviewVerdict.APPROVE,  # Will be overridden by _determine_verdict
                summary=data.get("summary", "Review completed"),
                comments=comments,
                tests_pass=auto_results.get("tests_pass", False),
                style_compliant=auto_results.get("lint_clean", False),
                security_issues=[],
                confidence=data.get("confidence", 0.7),
            )
        except (json.JSONDecodeError, KeyError):
            return ReviewResult(
                verdict=ReviewVerdict.REQUEST_CHANGES,
                summary="Could not parse structured review; manual review recommended",
                comments=[],
                tests_pass=auto_results.get("tests_pass", False),
                style_compliant=auto_results.get("lint_clean", False),
                security_issues=[],
                confidence=0.3,
            )

    def _extract_patch(self, context: AgentContext) -> str:
        """Extract patch content from previous agent messages."""
        for msg in reversed(context.previous_messages):
            if msg.sender == "coder" and "patch" in msg.artifacts:
                return msg.artifacts["patch"]
        return ""
