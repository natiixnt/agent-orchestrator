"""Coder agent for code generation and patch creation."""

from __future__ import annotations

import time
from typing import Any

from agent_orchestrator.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)


class CoderAgent(BaseAgent):
    """
    Agent specialized in code generation, modification, and patch creation.

    Uses retrieved context from the researcher agent along with semantic memory
    to produce high-quality code patches that match existing patterns and style.
    """

    def __init__(
        self,
        llm_client: Any,
        tools: list[Any] | None = None,
        memory_stores: dict[str, Any] | None = None,
        sandbox: Any | None = None,
        max_patch_attempts: int = 3,
    ) -> None:
        super().__init__(
            name="coder",
            llm_client=llm_client,
            tools=tools,
            memory_stores=memory_stores,
            max_iterations=15,
            temperature=0.0,
        )
        self.sandbox = sandbox
        self.max_patch_attempts = max_patch_attempts

    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Generate a code patch for the given task.

        Strategy:
        1. Consume context from the researcher agent
        2. Query semantic memory for relevant code patterns
        3. Generate initial patch using LLM with full context
        4. Validate patch in sandbox (syntax, tests, lint)
        5. Iterate on failures up to max_patch_attempts
        6. Return final patch with confidence score
        """
        start_time = time.perf_counter()
        self.status = AgentStatus.RUNNING
        self._tool_calls = []
        self._tokens_used = 0

        try:
            # Extract researcher context from previous messages
            research_context = self._extract_research_context(context)

            # Query semantic memory for relevant patterns
            patterns = await self.query_memory(
                context.task_description, "semantic", k=5
            )

            # Generate patch iteratively
            patch = None
            validation_result = None

            for attempt in range(self.max_patch_attempts):
                patch = await self._generate_patch(
                    context, research_context, patterns, validation_result
                )

                if self.sandbox:
                    validation_result = await self._validate_in_sandbox(patch, context)
                    if validation_result["passed"]:
                        break
                else:
                    break

            if patch is None:
                raise RuntimeError("Failed to generate any patch")

            # Determine confidence based on validation
            confidence = self._compute_confidence(patch, validation_result)

            artifacts = {
                "patch": patch,
                "files_modified": self._extract_modified_files(patch),
                "validation": validation_result,
                "confidence": confidence,
                "attempts": attempt + 1 if "attempt" in dir() else 1,
            }

            duration = time.perf_counter() - start_time
            return self._build_result(
                status=AgentStatus.COMPLETED,
                output=f"Generated patch modifying {len(artifacts['files_modified'])} files "
                       f"(confidence: {confidence:.2f}, attempts: {artifacts['attempts']})",
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
        return (
            "You are a code generation agent. Your role is to produce precise, minimal "
            "code patches that solve the given task.\n\n"
            "Guidelines:\n"
            "- Match the existing code style exactly (indentation, naming, patterns)\n"
            "- Make minimal changes; do not refactor unrelated code\n"
            "- Include proper error handling\n"
            "- Add or update tests when appropriate\n"
            "- Produce output as a unified diff\n\n"
            f"Repository: {context.repo_path}\n"
            f"Task: {context.task_description}\n"
        )

    async def _generate_patch(
        self,
        context: AgentContext,
        research: dict[str, Any],
        patterns: list[Any],
        previous_validation: dict[str, Any] | None,
    ) -> str:
        """Generate a code patch using LLM with full context."""
        messages = [
            {"role": "system", "content": self.get_system_prompt(context)},
        ]

        # Add research context
        if research:
            messages.append({
                "role": "user",
                "content": (
                    "Research findings:\n"
                    f"Relevant files: {research.get('relevant_files', [])}\n"
                    f"Synthesis: {research.get('synthesis', 'No synthesis available')}\n"
                ),
            })

        # Add pattern context from semantic memory
        if patterns:
            pattern_text = "\n".join(
                f"- {p.content[:200]}" for p in patterns if hasattr(p, "content")
            )
            if pattern_text:
                messages.append({
                    "role": "user",
                    "content": f"Relevant code patterns from memory:\n{pattern_text}",
                })

        # Add validation feedback for retry attempts
        if previous_validation and not previous_validation["passed"]:
            messages.append({
                "role": "user",
                "content": (
                    "Previous attempt failed validation:\n"
                    f"Errors: {previous_validation.get('errors', [])}\n"
                    f"Test output: {previous_validation.get('test_output', '')}\n"
                    "Please fix these issues in the new patch."
                ),
            })

        messages.append({
            "role": "user",
            "content": (
                f"Generate a unified diff patch to solve: {context.task_description}\n"
                "Output ONLY the diff, no explanation."
            ),
        })

        # Run agentic loop with tools for file reading if needed
        for _ in range(self.max_iterations):
            response = await self._call_llm(messages)

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = await self.run_tool(
                        tool_call.function.name, tool_call.function.arguments
                    )
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    })
            else:
                return response.content

        return ""

    async def _validate_in_sandbox(
        self, patch: str, context: AgentContext
    ) -> dict[str, Any]:
        """Validate a patch by applying and testing it in the sandbox."""
        result = await self.sandbox.execute(
            commands=[
                f"cd {context.repo_path}",
                f"echo '{patch}' | git apply --check",
                f"echo '{patch}' | git apply",
                "python -m pytest --tb=short -q 2>&1 | tail -20",
                "ruff check . --fix --quiet 2>&1 | tail -10",
            ],
            timeout=120,
            memory_limit="1g",
        )

        passed = result.exit_code == 0
        return {
            "passed": passed,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "errors": self._parse_errors(result.stderr) if not passed else [],
            "test_output": result.stdout,
        }

    def _extract_research_context(self, context: AgentContext) -> dict[str, Any]:
        """Extract research findings from previous agent messages."""
        for msg in reversed(context.previous_messages):
            if msg.sender == "researcher" and msg.artifacts:
                return msg.artifacts
        return {}

    def _extract_modified_files(self, patch: str) -> list[str]:
        """Parse file paths from a unified diff."""
        files = []
        for line in patch.split("\n"):
            if line.startswith("+++ b/"):
                files.append(line[6:])
            elif line.startswith("--- a/"):
                files.append(line[6:])
        return list(set(files))

    def _compute_confidence(
        self, patch: str, validation: dict[str, Any] | None
    ) -> float:
        """Compute confidence score for the generated patch."""
        score = 0.5

        if validation and validation.get("passed"):
            score += 0.3
        elif validation:
            score -= 0.2

        # Smaller patches are generally more reliable
        line_count = len(patch.split("\n"))
        if line_count < 20:
            score += 0.1
        elif line_count > 100:
            score -= 0.1

        return max(0.0, min(1.0, score))

    @staticmethod
    def _parse_errors(stderr: str) -> list[str]:
        """Parse error messages from stderr output."""
        errors = []
        for line in stderr.split("\n"):
            line = line.strip()
            if line and ("error" in line.lower() or "Error" in line):
                errors.append(line)
        return errors[:10]
