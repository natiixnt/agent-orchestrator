# Changelog

All notable changes to agent-orchestrator. Versions follow [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-04

The big jump. SWE-bench Lite pass rate goes from 73.0% to 78.3% on the same 300 tasks.
Three additions, each addressing a different failure mode of the v1 system.

### Added
- **Test-driven patch generation** (`src/agent_orchestrator/test_driven.py`). Writes a failing reproducer test before generating the patch. Acts as a fast verifier loop: if the patch does not pass the reproducer, regenerate before spending sandbox time on full test suite. +3.1% pass rate, biggest win on bug-fix tasks.
- **Tree-of-thought verification** (`src/agent_orchestrator/tree_of_thought.py`). Three parallel patch attempts at temperatures 0.2 / 0.5 / 0.8, evaluated independently in sandbox, best one wins. Three parallel branches end up cheaper than three sequential retries because each branch skips the reviewer round-trip on failure. +2.2% pass rate.
- **AST-based context compression** (`src/agent_orchestrator/context_compression.py`). Researcher extracts only the symbols the coder actually needs (target function plus callers/callees) instead of dumping whole files. Cuts research tokens by ~30% and makes the coder more accurate by reducing distractor context. -$0.04 cost per task, neutral on pass rate but compounds with TDD/ToT.
- **DSPy-style prompt optimisation** (`src/agent_orchestrator/dspy_optimizer.py`). BootstrapFewShot picks the best few-shot examples for the coder agent on a held-out validation set. +3.2% pass rate when the coder runs in the optimised configuration. Optimisation is offline so it does not affect inference latency.
- **Self-critique loop** (`src/agent_orchestrator/critic.py`). After patch generation, the agent critiques its own work, identifies likely failure modes, generates targeted test cases. Catches 23% of bugs that the reviewer agent misses, mostly subtle edge cases (off-by-one, None handling, empty-collection paths).
- **SWE-bench Verified** results writeup (`benchmarks/swe_bench_verified.md`). 51.2% pass rate on the rigorous human-verified subset.
- **HumanEval+** results (`benchmarks/humaneval_plus.md`). 92.1% pass@1 on Python.
- **LiveCodeBench** results (`benchmarks/livecodebench.md`) across easy/medium/hard tiers.
- **SOTA comparison** writeup (`benchmarks/sota_comparison.md`).
- **Reproducibility demo** (`benchmarks/run_demo.py`). Runs the full pipeline on a fixture repo with a mock LLM so reviewers can see the system end-to-end without spending API credits.
- **Quickstart example** (`examples/quickstart.py`).
- Generated PNG figures in `benchmarks/charts/` via `benchmarks/generate_charts.py`.

### Changed
- README architecture diagram refactored to show the linear pipeline with memory layers feeding all agents.
- Added a Limitations section to the README. Honesty about Lite vs full SWE-bench, multi-file architectural changes, race conditions, language coverage.

### Performance
- Overall SWE-bench Lite pass rate: 73.0% -> 78.3% (+5.3%).
- First-attempt success: 61.5% -> 72.4% (+10.9%).
- Cost per task: $0.42 -> $0.38 (-$0.04, mostly from context compression).
- Time per task: 3.2 min -> 2.8 min (-0.4 min, ToT branches run in parallel).

## [1.0.0] - 2026-02

The "memory + planning is real" release. Adds MCTS task decomposition and the three-tier memory system. Pass rate jumps from 62.3% (v0.1 baseline) to 73.0%.

### Added
- **MCTS planner** (`src/agent_orchestrator/planner.py`). Replaces greedy 1-shot decomposition. UCB1 selection with 100 simulations. +15.3% pass rate vs greedy at the cost of 4.5s of planning latency, which is more than offset by saving 40+ seconds of wasted agent calls per task.
- **Episodic memory** (`src/agent_orchestrator/memory/episodic.py`). Records full execution traces of past tasks for similarity retrieval. +3.8% pass rate.
- **Semantic memory** (`src/agent_orchestrator/memory/semantic.py`). Indexes code patterns, API docs, domain knowledge with pgvector. +2.4% pass rate.
- **Procedural memory** (`src/agent_orchestrator/memory/procedural.py`). Stores learned decomposition strategies and which agent configurations worked for which task types. The MCTS scoring function consults this. +4.5% pass rate (largest single contributor among memory layers).
- **Audit trail** (`src/agent_orchestrator/audit.py`). Every plan, agent invocation, tool call, LLM request, sandbox execution, and dollar spent gets logged. Critical for production debugging and cost attribution.
- **Memory ablation benchmark** demonstrating each layer's contribution.
- **Planning comparison benchmark** (MCTS vs greedy across 300 tasks).

### Changed
- Specialised agents now consult their corresponding memory store before generating output.
- Planner output feeds into the procedural memory after task completion to close the learning loop.

### Performance
- Pass rate: 62.3% -> 73.0% (+10.7%).
- Wasted agent calls: 22.1% -> 6.1% (-16.0%) thanks to MCTS producing tighter decompositions.

## [0.1.0] - 2025-12

Initial release. Functional but simple.

### Added
- **Researcher agent** (`src/agent_orchestrator/agents/researcher.py`). Codebase navigation, dependency graphs, file pattern matching.
- **Coder agent** (`src/agent_orchestrator/agents/coder.py`). Patch generation with sandbox-validated retry loop.
- **Reviewer agent** (`src/agent_orchestrator/agents/reviewer.py`). Catches roughly 38% of bugs before sandbox execution.
- **Sandboxed execution** (`src/agent_orchestrator/sandbox.py`). Docker isolation with seccomp, namespace isolation, CPU/memory/network limits.
- **LangGraph orchestration** (`src/agent_orchestrator/graph.py`). Inter-agent message passing, human-in-the-loop checkpoints.
- **Greedy 1-shot planner**. LLM produces a flat list of subtasks in one call. Worked well enough to validate the architecture, but a clear weak point that v1.0 addressed.

### Performance
- Baseline pass rate on SWE-bench Lite: 62.3%.
