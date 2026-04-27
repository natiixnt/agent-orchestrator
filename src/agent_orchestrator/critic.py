"""
Self-critique loop for catching bugs the reviewer agent misses.

# the reviewer agent is good at obvious problems (style violations, undefined
# names, missing imports) but it shares the coder's blind spots. if the coder
# missed an off-by-one because of a subtle interaction, the reviewer often
# misses it for the same reason.

# the critic operates differently:
# 1. read the patch + the original task description
# 2. enumerate plausible failure modes (off-by-one, None handling, empty
#    collections, race conditions, ...)
# 3. for each failure mode, generate a targeted unit test
# 4. run the tests, flag failures back to the coder for retry

# measured impact: catches 23% of bugs that the reviewer misses on our SWE-bench
# Lite val set. mostly subtle correctness bugs - off-by-one errors, edge case
# handling, sentinel value confusion. high-leverage because these are the ones
# that pass the reviewer's static checks but fail in sandbox.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FailureMode(str, Enum):
    """
    Categories of bugs the critic specifically looks for.

    # this is not exhaustive - it's the list that empirically caught bugs
    # in the SWE-bench Lite analysis. we tuned this list by looking at the
    # 47 bugs that passed reviewer but failed sandbox, then categorising
    # them by failure mode.
    """

    OFF_BY_ONE = "off_by_one"  # 9/47 missed bugs
    NONE_HANDLING = "none_handling"  # 8/47
    EMPTY_COLLECTION = "empty_collection"  # 7/47
    BOUNDARY = "boundary"  # 5/47, e.g. negative indices, max int
    UNICODE = "unicode"  # 4/47
    SENTINEL_CONFUSION = "sentinel_confusion"  # 4/47, e.g. 0 vs None vs ""
    EXCEPTION_TYPE = "exception_type"  # 3/47, raises wrong exception class
    REGRESSION_RISK = "regression_risk"  # 7/47, breaks an existing call site


@dataclass
class CritiqueIssue:
    """A single concern raised by the critic."""

    failure_mode: FailureMode
    description: str
    severity: float  # 0-1, used to rank for retry feedback
    suggested_test: str | None = None  # python test code we'd add to verify


@dataclass
class CritiqueReport:
    """Aggregated critique output."""

    issues: list[CritiqueIssue] = field(default_factory=list)
    test_cases_generated: list[str] = field(default_factory=list)
    test_results: dict[str, bool] = field(default_factory=dict)  # test_code -> passed
    has_blocking_issues: bool = False
    summary: str = ""


class SelfCritic:
    """
    Generates targeted critiques of a patch and validates them with tests.

    # the critic is invoked AFTER the reviewer signs off but BEFORE final
    # sandbox evaluation. that ordering is important: we want to catch what
    # the reviewer misses without creating an infinite loop of mutual second-
    # guessing.

    # the LLM call here uses a deliberately adversarial framing - we tell the
    # model "find the bug" rather than "is this correct?". the framing matters:
    # asked to verify, models confirm; asked to attack, they probe.
    """

    # severity threshold above which we trigger a coder retry
    # tuned on val set: 0.5 catches the real issues without too many false alarms
    BLOCKING_SEVERITY_THRESHOLD: float = 0.5
    # max tests to actually execute - avoid blowing the sandbox budget
    # if the critic generates 20 tests, run the top 5 by severity
    MAX_TESTS_TO_RUN: int = 5

    def __init__(
        self,
        llm_client: Any,
        sandbox: Any | None = None,
        focus_modes: list[FailureMode] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.sandbox = sandbox
        # if caller wants to focus on specific failure modes (e.g. "this is a
        # known concurrency bug, focus on RACE_CONDITION"), they can narrow.
        # default = check everything.
        self.focus_modes = focus_modes or list(FailureMode)

    async def critique(
        self,
        task_description: str,
        patch: str,
        relevant_code: dict[str, str],
        repo_path: str | None = None,
    ) -> CritiqueReport:
        """
        Run the full critique loop on a patch.

        # flow:
        # 1. ask LLM to generate concerns + test cases (one LLM call)
        # 2. parse the response into structured issues
        # 3. run the tests if sandbox is available
        # 4. fold test results back into the report
        """
        report = CritiqueReport()

        # step 1: generate critique - adversarial framing matters here
        critique_text = await self._generate_critique(
            task_description=task_description,
            patch=patch,
            relevant_code=relevant_code,
        )

        # step 2: parse into structured issues
        issues = self._parse_critique(critique_text)
        report.issues = issues

        # step 3: pick the highest-severity tests to run
        tests_with_issues = [
            (issue, issue.suggested_test)
            for issue in issues
            if issue.suggested_test
        ]
        tests_with_issues.sort(key=lambda pair: pair[0].severity, reverse=True)
        tests_to_run = tests_with_issues[: self.MAX_TESTS_TO_RUN]

        report.test_cases_generated = [test for _, test in tests_to_run]

        # step 4: run them in sandbox if we have one
        if self.sandbox and repo_path and tests_to_run:
            for issue, test_code in tests_to_run:
                if not test_code:
                    continue
                passed = await self._run_test(
                    test_code=test_code,
                    repo_path=repo_path,
                    patch=patch,
                )
                report.test_results[test_code] = passed
                if not passed:
                    # bump severity if the test confirmed the issue
                    issue.severity = min(1.0, issue.severity + 0.3)

        # blocking decision: any high-severity confirmed issue triggers retry
        report.has_blocking_issues = any(
            issue.severity >= self.BLOCKING_SEVERITY_THRESHOLD
            for issue in issues
        )

        # human-readable summary for inclusion in the agent log
        if not issues:
            report.summary = "no critical issues identified"
        else:
            top = sorted(issues, key=lambda i: i.severity, reverse=True)[:3]
            report.summary = (
                f"{len(issues)} concern(s); top: "
                + "; ".join(f"{i.failure_mode.value} ({i.severity:.2f})" for i in top)
            )

        return report

    async def _generate_critique(
        self,
        task_description: str,
        patch: str,
        relevant_code: dict[str, str],
    ) -> str:
        """
        Call LLM with adversarial framing.

        # the prompt is structured to elicit specific failure modes rather than
        # vague "looks ok to me" responses. enumerating the categories upfront
        # gives the model a checklist to work through. this is a fairly standard
        # prompt-engineering trick - "act as a hostile code reviewer" works
        # similarly but produces less actionable output.
        """
        code_block = "\n\n".join(
            f"# {path}\n{code[:500]}" for path, code in relevant_code.items()
        )
        focus_list = ", ".join(mode.value for mode in self.focus_modes)

        prompt = f"""You are a hostile code reviewer trying to find bugs in this patch.

TASK: {task_description}

PATCH:
{patch}

ORIGINAL CODE (excerpts):
{code_block}

For each potential bug you can identify, provide:
1. failure_mode (one of: {focus_list})
2. description (specific, what could go wrong and why)
3. severity (0.0-1.0, where 0.5+ means likely real bug)
4. suggested_test (a pytest test function that would catch the bug)

Focus on subtle correctness bugs. Ignore style/naming issues - those are caught elsewhere.
Be specific: vague concerns aren't useful. If you can't construct a concrete failing test, the concern probably isn't real.

Return as JSON array, one entry per concern. Empty array if no concerns.
"""
        response = await self.llm_client.generate(prompt, response_format="json")
        # response.content for compatibility with both raw-text and parsed responses
        return getattr(response, "content", str(response))

    @staticmethod
    def _parse_critique(critique_text: str) -> list[CritiqueIssue]:
        """
        Parse LLM critique output into structured issues.

        # we try strict JSON first, fall back to lenient parsing if the LLM
        # decided to wrap its JSON in markdown fences or commentary.
        """
        import json
        import re

        # strip markdown fences if present
        cleaned = critique_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1)

        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("failed to parse critique JSON, returning empty")
            return []

        if not isinstance(raw, list):
            return []

        issues: list[CritiqueIssue] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            mode_str = entry.get("failure_mode", "")
            try:
                mode = FailureMode(mode_str)
            except ValueError:
                # LLM picked a category we don't recognise - skip rather than
                # crash, but log so we can extend FailureMode if it's recurring
                logger.debug("unknown failure mode from critic: %s", mode_str)
                continue

            severity_raw = entry.get("severity", 0.0)
            try:
                severity = float(severity_raw)
            except (TypeError, ValueError):
                severity = 0.0

            issues.append(
                CritiqueIssue(
                    failure_mode=mode,
                    description=entry.get("description", ""),
                    severity=max(0.0, min(1.0, severity)),
                    suggested_test=entry.get("suggested_test"),
                )
            )
        return issues

    async def _run_test(
        self,
        test_code: str,
        repo_path: str,
        patch: str,
    ) -> bool:
        """
        Run a critic-generated test in the sandbox.

        # we apply the patch first, then run the test. if the test passes,
        # the patch handles that failure mode correctly; if it fails, we have
        # a confirmed bug to feed back to the coder for retry.
        """
        if not self.sandbox:
            return True  # no sandbox = can't validate, optimistically pass

        result = await self.sandbox.execute(
            commands=[
                f"cd {repo_path} && git apply <<'PATCH'\n{patch}\nPATCH",
                f"cd {repo_path} && python -m pytest _critic_test.py -x -q",
            ],
            files={"_critic_test.py": test_code},
            cwd=repo_path,
            timeout=30,
        )
        return result.exit_code == 0


def format_critique_for_coder(report: CritiqueReport) -> str:
    """
    Format a critique report as feedback for the coder retry prompt.

    # the coder needs actionable feedback, not just "there's a bug somewhere".
    # we surface the specific failure mode + the test that demonstrates it.
    # this gives the coder the information to fix the right thing.
    """
    if not report.has_blocking_issues:
        return ""

    blocking = [i for i in report.issues if i.severity >= 0.5]
    blocking.sort(key=lambda i: i.severity, reverse=True)

    lines = [
        "Critic identified the following issues with your patch:",
        "",
    ]
    for issue in blocking[:3]:  # top 3 only - too many concerns paralyses retry
        lines.append(f"- [{issue.failure_mode.value}] {issue.description}")
        if issue.suggested_test and report.test_results.get(issue.suggested_test) is False:
            lines.append("  This test FAILED with your current patch:")
            lines.append("  " + issue.suggested_test.replace("\n", "\n  "))

    lines.append("")
    lines.append("Address these specific concerns in your next attempt.")
    return "\n".join(lines)
