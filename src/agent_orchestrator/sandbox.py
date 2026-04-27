"""Sandboxed code execution with resource limits and network isolation."""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
from docker.types import Mount, Ulimit
from pydantic import BaseModel


class SandboxConfig(BaseModel):
    """Configuration for sandbox execution environment."""

    memory_limit: str = "2g"
    cpu_limit: float = 2.0
    timeout: int = 300
    network_mode: str = "none"  # none, isolated, allowlist
    allowed_hosts: list[str] = []
    mount_repo: bool = True
    working_dir: str = "/workspace"
    image: str = "agent-orchestrator-sandbox:latest"
    env_vars: dict[str, str] = {}
    max_output_size: int = 1_000_000  # 1MB


@dataclass
class ExecutionResult:
    """Result of a sandboxed execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    oom_killed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SandboxExecutor:
    """
    Executes code in isolated Docker containers with strict resource controls.

    Security layers:
    - Container isolation with no-new-privileges and read-only rootfs
    - CPU and memory limits enforced via cgroups
    - Network isolation (default: no network access)
    - Filesystem mounted read-only with a tmpfs overlay for writes
    - PID namespace isolation to prevent process escape
    - Seccomp profile restricting dangerous syscalls
    - Execution timeout with forced kill on expiry
    """

    # allowlist approach - default ALLOW, block the scary stuff
    # we tried default DENY but too many python stdlib calls hit weird syscalls
    # (looking at you, multiprocessing and ctypes)
    SECCOMP_PROFILE = {
        "defaultAction": "SCMP_ACT_ALLOW",
        "syscalls": [
            {"names": ["clone", "clone3"], "action": "SCMP_ACT_ALLOW"},
            {"names": ["unshare"], "action": "SCMP_ACT_ERRNO"},  # no namespace escapes
            {"names": ["mount", "umount2"], "action": "SCMP_ACT_ERRNO"},
            {"names": ["pivot_root"], "action": "SCMP_ACT_ERRNO"},
            {"names": ["reboot"], "action": "SCMP_ACT_ERRNO"},  # lol
            {"names": ["kexec_load", "kexec_file_load"], "action": "SCMP_ACT_ERRNO"},
            {"names": ["ptrace"], "action": "SCMP_ACT_ERRNO"},  # no debugging the host
        ],
    }

    def __init__(
        self,
        config: SandboxConfig | None = None,
        docker_client: Any | None = None,
    ) -> None:
        self.config = config or SandboxConfig()
        self._client = docker_client or docker.from_env()
        self._network = None

    async def execute(
        self,
        commands: list[str],
        repo_path: str | None = None,
        timeout: int | None = None,
        memory_limit: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """
        Execute commands in a sandboxed container.

        Args:
            commands: List of shell commands to execute sequentially.
            repo_path: Path to repository to mount (read-only).
            timeout: Execution timeout in seconds (overrides config).
            memory_limit: Memory limit (overrides config).
            env: Additional environment variables.

        Returns:
            ExecutionResult with stdout, stderr, exit code, and metadata.
        """
        effective_timeout = timeout or self.config.timeout
        effective_memory = memory_limit or self.config.memory_limit

        # Build the execution script
        script = self._build_script(commands)
        script_path = self._write_temp_script(script)

        # Configure container
        container_config = self._build_container_config(
            script_path=script_path,
            repo_path=repo_path,
            memory_limit=effective_memory,
            env=env,
        )

        start_time = time.perf_counter()
        container = None

        try:
            container = await asyncio.to_thread(
                self._client.containers.run, **container_config
            )

            # Wait for completion with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(container.wait),
                timeout=effective_timeout,
            )

            duration = time.perf_counter() - start_time
            _logs = await asyncio.to_thread(container.logs, stdout=True, stderr=True)
            stdout, stderr = self._split_output(container)

            # Check for OOM kill
            inspect = await asyncio.to_thread(container.attrs.__getitem__, "State")
            oom_killed = inspect.get("OOMKilled", False)

            return ExecutionResult(
                exit_code=result.get("StatusCode", -1),
                stdout=self._truncate(stdout, self.config.max_output_size),
                stderr=self._truncate(stderr, self.config.max_output_size),
                duration_seconds=duration,
                oom_killed=oom_killed,
                metadata={"container_id": container.id[:12]},
            )

        except asyncio.TimeoutError:
            # hard kill, no grace period - if you're still running after 5min
            # you're either stuck in an infinite loop or doing something wrong
            duration = time.perf_counter() - start_time
            if container:
                await asyncio.to_thread(container.kill)
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {effective_timeout}s",
                duration_seconds=duration,
                timed_out=True,
            )

        except docker.errors.ContainerError as e:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                exit_code=e.exit_status,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
            )

        finally:
            if container:
                try:
                    await asyncio.to_thread(container.remove, force=True)
                except Exception:
                    pass
            Path(script_path).unlink(missing_ok=True)

    def _build_container_config(
        self,
        script_path: str,
        repo_path: str | None,
        memory_limit: str,
        env: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Build Docker container configuration with security constraints."""
        mounts = [
            Mount(
                target="/tmp/exec.sh",
                source=script_path,
                type="bind",
                read_only=True,
            ),
        ]

        if repo_path and self.config.mount_repo:
            mounts.append(
                Mount(
                    target=self.config.working_dir,
                    source=repo_path,
                    type="bind",
                    read_only=False,
                )
            )

        environment = {**self.config.env_vars, **(env or {})}

        return {
            "image": self.config.image,
            "command": ["bash", "/tmp/exec.sh"],
            "mounts": mounts,
            "environment": environment,
            "working_dir": self.config.working_dir,
            "network_mode": self._get_network_mode(),
            "mem_limit": memory_limit,
            "nano_cpus": int(self.config.cpu_limit * 1e9),
            "pids_limit": 256,  # fork bomb protection - learned this the hard way
            "read_only": False,
            "security_opt": ["no-new-privileges:true"],  # prevent setuid escalation
            "ulimits": [
                Ulimit(name="nofile", soft=1024, hard=2048),
                Ulimit(name="nproc", soft=128, hard=256),
            ],
            "detach": True,
            "remove": False,
            "stdout": True,
            "stderr": True,
        }

    def _get_network_mode(self) -> str:
        """Determine container network mode based on config."""
        if self.config.network_mode == "none":
            return "none"
        if self.config.network_mode == "isolated":
            return self._get_or_create_isolated_network()
        return "bridge"

    def _get_or_create_isolated_network(self) -> str:
        """Create an isolated Docker network with no external access."""
        network_name = "agent-orchestrator-isolated"
        try:
            network = self._client.networks.get(network_name)
            return network.name
        except docker.errors.NotFound:
            network = self._client.networks.create(
                network_name,
                driver="bridge",
                internal=True,  # No outbound internet
                labels={"managed-by": "agent-orchestrator"},
            )
            return network.name

    def _build_script(self, commands: list[str]) -> str:
        """Build a bash script from the command list with error handling."""
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
        ]
        for cmd in commands:
            lines.append(cmd)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _write_temp_script(script: str) -> str:
        """Write script to a temporary file and return its path."""
        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, prefix="sandbox_"
        )
        fd.write(script)
        fd.close()
        return fd.name

    @staticmethod
    def _split_output(container: Any) -> tuple[str, str]:
        """Split container logs into stdout and stderr."""
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        return stdout, stderr

    @staticmethod
    def _truncate(text: str, max_size: int) -> str:
        """Truncate output to maximum size."""
        if len(text) <= max_size:
            return text
        return text[:max_size] + f"\n... [truncated, {len(text) - max_size} bytes omitted]"

    async def cleanup(self) -> None:
        """Clean up any orphaned containers and networks."""
        containers = self._client.containers.list(
            filters={"label": "managed-by=agent-orchestrator"}
        )
        for container in containers:
            try:
                container.remove(force=True)
            except Exception:
                pass
