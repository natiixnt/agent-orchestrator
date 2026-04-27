# HumanEval+ Results

[HumanEval+](https://github.com/evalplus/evalplus) is the augmented version of OpenAI's HumanEval. It adds 80x more test cases per problem to catch the kind of "passes the visible tests but is actually broken" patches that the original HumanEval missed.

We evaluated agent-orchestrator v2.0 on all 164 HumanEval+ Python problems.

## Headline Result

**92.1% pass@1 on Python**

| System | pass@1 | pass@10 |
|---|---|---|
| **agent-orchestrator v2 (ours)** | **92.1%** | **96.9%** |
| Claude 3.5 Sonnet (direct, no scaffolding) | 89.0% | 94.5% |
| GPT-4o | 86.0% | 93.3% |
| Aider (Claude 3.5 Sonnet) | 89.4% | 95.1% |
| SWE-agent | 84.8% | 92.7% |
| DeepSeek-Coder-V2 | 83.5% | 91.5% |

Margin over the closest competitor (Aider): +2.7% on pass@1, +1.8% on pass@10.

## Why the Improvement Is Smaller Than on SWE-bench

HumanEval+ problems are self-contained algorithmic tasks: write a function from a docstring. There is no codebase to navigate, no dependency graph, no architectural context. So the things that make agent-orchestrator strong on SWE-bench (researcher, MCTS planning, procedural memory of past projects) contribute almost nothing.

What helps:
- **Tree-of-thought** is the primary contributor (+1.8%). Three temperature branches catch problems where the lowest-temperature greedy decode picks a subtly wrong approach.
- **Test-driven generation** helps another +0.7%. Our generated reproducer tests sometimes overlap with the hidden test cases, which means we are filtering bad solutions before submission.
- **Self-critique loop** catches +0.6%, mostly off-by-one errors on edge cases the visible docstring examples did not cover.
- **MCTS planning, memory, context compression** contribute 0.0% on HumanEval+ and add latency. These are disabled in our HumanEval+ run.

Net: +3.1% over the Sonnet baseline.

## Where We Still Fail

13 of 164 problems fail. Categorising the failures:

- **5 numerical edge cases**: precision issues in floating-point comparisons (e.g. `is_close` with default tolerance behaving differently than the test expects).
- **4 ambiguous specs**: the docstring is consistent with multiple implementations but only one passes the hidden test suite.
- **3 sneaky off-by-one**: ToT did not generate a branch with the correct boundary handling; self-critic flagged the issue but the regenerated patch repeated the same mistake.
- **1 unicode bug**: docstring did not mention unicode handling but a test case used it.

The numerical and ambiguity cases are essentially unfixable at the prompt level; they would require richer specification than the docstring provides.

## Per-Problem Latency and Cost

| Metric | Value |
|---|---|
| Avg time per problem | 8.2s |
| Avg cost per problem | $0.014 |
| Total eval cost | $2.30 |
| Total eval time | 22 min (4 parallel workers) |

HumanEval+ is cheap to evaluate, which is why we run it on every PR as a smoke test (see `.github/workflows/ci.yml`).

## Reproducing

```bash
# requires evalplus
pip install evalplus
python -m benchmarks.run_humaneval_plus --output benchmarks/humaneval_plus.json
```
