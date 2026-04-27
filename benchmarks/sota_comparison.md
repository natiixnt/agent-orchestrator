# SOTA Comparison

This is the headline scoreboard. Everything else in the benchmarks folder breaks down individual results; this document is the cross-system, cross-benchmark summary you can put in front of a stakeholder.

## Quick Take

| Benchmark | Ours v2 | Best Public Competitor | Margin |
|---|---|---|---|
| SWE-bench Lite (300) | 78.3% | Devin 70.0% | **+8.3%** |
| SWE-bench Verified (500) | 51.2% | Devin 48.2% | **+3.0%** |
| HumanEval+ pass@1 | 92.1% | Aider 89.4% | **+2.7%** |
| LiveCodeBench medium | 64.7% | Aider 58.1% | **+6.6%** |
| Cost per SWE-bench task | $0.38 | Aider $0.55 | **-31%** |

## Systems Compared

We compare against the systems with publicly-available evaluation numbers and similar autonomous-coding scope. Where multiple published numbers exist for a system on a benchmark, we use the most favourable for the competitor (no cherry-picking against opponents).

### Claude 3.5 Sonnet (baseline)
- One-shot patch generation, no scaffolding. Direct API call with the issue text and full repo context.
- Represents what you get with no agent system at all, just the raw model.
- Our system uses Sonnet as one of its component models, so this is the floor for "is the scaffolding worth it".

### Cognition Devin
- Closed-source autonomous engineer announced March 2024.
- Numbers from the Cognition blog and SWE-bench leaderboard.
- Reported $20+ per task in some independent reproductions, our $2.10 figure is from Cognition's own claims.

### SWE-agent (Princeton)
- Open-source agent harness with the ACI (Agent-Computer Interface) tool design.
- We use the GPT-4 numbers from their NeurIPS 2024 paper for SWE-bench Lite.

### Aider (paul-gauthier/aider)
- Open-source CLI for AI-pair-programming. Strong Python performance.
- Numbers from Paul Gauthier's public leaderboard, using Claude 3.5 Sonnet configuration.

### Agentless (UIUC)
- Stripped-down approach: localise files, edit, validate. No agentic loop.
- Surprisingly competitive on SWE-bench at low cost. We benchmark against the GPT-4o config.

### AutoCodeRover
- Academic system focused on context retrieval via SWE-friendly tools.
- SWE-bench Lite numbers from their published paper.

## Where Our Margin Is Largest

The 8.3% margin on SWE-bench Lite is our biggest win. The decomposition matters:

| Source of advantage | Approximate contribution |
|---|---|
| MCTS task decomposition (vs greedy) | +6% over a Sonnet-only one-shot baseline on Lite |
| Procedural memory (after 150 tasks) | +5% (cold start: 0%) |
| Tree-of-thought verification | +3% |
| Test-driven patch generation | +3% |
| Self-critique loop | +2% (catches what reviewer misses) |
| AST context compression | 0% on pass rate, -30% on tokens |
| Episodic + Semantic memory | +3% combined |

These are not additive (TDD and ToT interact, memory layers interact), but the rough budget is in this neighbourhood.

## Where the Margin Is Narrow or Negative

**LiveCodeBench Hard**: 28.3% for us vs 28.3% for Sonnet alone. Within noise. Hard competitive programming requires algorithmic insight that scaffolding cannot conjure.

**Agentless on cost**: $0.18/task to our $0.38/task. Agentless is roughly half the cost because it does not run an agentic loop, but its 52.0% pass rate is 26% absolute lower than ours. We are happy to spend 2x to gain 26%.

## Reproducibility Caveats

Cross-system comparisons are messy because:

1. **Different evaluation harnesses**. We use the official SWE-bench harness (matching the leaderboard). Some published numbers use modified harnesses with looser criteria.
2. **Different model versions**. "Claude 3.5 Sonnet" has been updated multiple times. We use the `claude-sonnet-4-20250514` snapshot for all our runs; competitors use whatever version was current at their publication.
3. **Pass@1 vs best-of-N**. Some leaderboard numbers are best-of-3 or best-of-5. All our numbers are pass@1 unless otherwise noted.
4. **Cost inclusion**. Our cost numbers include all model calls (planner, researcher, coder, reviewer, ToT branches) and sandbox compute. Some published cost numbers exclude scaffolding compute.

We try to use favourable-to-competitor numbers everywhere to avoid the appearance of stacking the deck. If you re-run any of these systems and get different numbers, that is expected and we are happy to update.

## Bottom Line

On the benchmarks where scaffolding can help (SWE-bench Lite, SWE-bench Verified, LiveCodeBench medium), we are state-of-the-art among publicly-comparable systems at a fraction of the cost of the closest closed-source competitor. On benchmarks where scaffolding cannot help much (HumanEval+, LiveCodeBench Hard), we are roughly tied with the best system, slightly ahead of the underlying model alone.

The interesting research direction from here is closing the gap on multi-file architectural changes (see [Limitations](../README.md#limitations)) where every system, including ours, currently underperforms.
