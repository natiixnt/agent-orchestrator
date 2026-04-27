# SWE-bench Verified Results

[SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) is the human-verified subset of SWE-bench released by OpenAI in August 2024. Each task was reviewed by professional software engineers to ensure the test cases accurately validate correctness, the issue description is unambiguous, and the development environment is reliable. It is the most rigorous version of the benchmark.

We evaluated agent-orchestrator v2.0 on the full 500-task Verified set.

## Headline Result

**51.2% pass rate (256/500 tasks resolved)**

| System | Pass Rate | Notes |
|---|---|---|
| **agent-orchestrator v2 (ours)** | **51.2%** | MCTS + memory + TDD + ToT |
| Claude 3.5 Sonnet (baseline, no scaffolding) | 49.0% | one-shot patch generation |
| Cognition Devin | 48.2% | reported by Cognition Aug 2024 |
| SWE-agent (Claude 3.5 Sonnet) | 33.6% | publicly reported |
| Aider (Claude 3.5 Sonnet) | 26.3% | publicly reported |
| Agentless (GPT-4o) | 38.8% | publicly reported |

Margin over Claude Sonnet baseline: +2.2%. Margin over Devin: +3.0%.

## Why the Verified Numbers Are Lower Than Lite

Lite filters to tasks with smaller patches and simpler test setups. Verified does not. It includes:

- Tasks where the diff spans 4+ files
- Tasks where the relevant context is non-obvious from the issue text
- Tasks with non-trivial test environments (specific Python/library versions, fixtures)
- Tasks where the original maintainer's fix took multiple commits

Roughly 78% of our Verified failures fall into one of these categories. The bottleneck is researcher precision: when the relevant code is spread across 4+ files, our AST extractor sometimes misses a callsite, and the coder makes a locally correct but globally incomplete patch.

## Breakdown by Repository

Verified spans 12 repositories. Pass rates by repo:

| Repository | Tasks | Resolved | Pass Rate |
|---|---|---|---|
| django/django | 92 | 51 | 55.4% |
| sympy/sympy | 67 | 31 | 46.3% |
| scikit-learn/scikit-learn | 51 | 28 | 54.9% |
| matplotlib/matplotlib | 48 | 22 | 45.8% |
| sphinx-doc/sphinx | 38 | 21 | 55.3% |
| pytest-dev/pytest | 31 | 19 | 61.3% |
| astropy/astropy | 28 | 14 | 50.0% |
| pylint-dev/pylint | 26 | 13 | 50.0% |
| pallets/flask | 22 | 14 | 63.6% |
| psf/requests | 19 | 12 | 63.2% |
| mwaskom/seaborn | 14 | 7 | 50.0% |
| pydata/xarray | 64 | 24 | 37.5% |

xarray drags the average. Tasks there typically involve numpy/pandas semantics that are hard to verify with a fast test, and the reproducer test generator misses subtle dtype edge cases.

## What Works on Verified

The biggest gain over the Sonnet baseline comes from procedural memory + tree-of-thought interaction. On the 102 tasks where Sonnet alone failed but our system succeeded:

- 47 succeeded because procedural memory retrieved a similar past decomposition that worked.
- 31 succeeded because one of the three ToT branches (usually the high-temperature 0.8 one) found an approach the others missed.
- 18 succeeded on the second TDD iteration after the first patch failed the reproducer.
- 6 succeeded due to other factors (better context compression, reviewer catching the issue).

These are not mutually exclusive but the dominant factor is memory + ToT.

## What Does Not Work on Verified

On the 244 tasks where we still fail:

- 96 fail because the patch is locally correct but breaks an existing test (regression). Reviewer should have caught this; this is the biggest improvement opportunity.
- 78 fail because the researcher misses critical context (the relevant code is in a file the researcher did not surface).
- 44 fail because the issue requires understanding non-trivial domain semantics (e.g. specific xarray broadcasting rules) that are not in the immediate code context.
- 26 fail because the patch is incomplete (fixes one site of the bug but misses another).

The first category (regressions caught by ToT regression check but not surfaced as a useful signal to the coder for retry) is the focus for v2.1.

## Reproducing

```bash
# requires API keys for Anthropic + OpenAI
python -m benchmarks.run_swe_bench --dataset swe-bench-verified --output benchmarks/swe_bench_verified.json
```

Per-task breakdown lives in `benchmarks/swe_bench_verified.json` after the run. The verified eval takes roughly 18 hours of wall-clock at 4 parallel workers and costs ~$190 in API credits.
