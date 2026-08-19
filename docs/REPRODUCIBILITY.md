# Reproducibility

## What is fixed and what is free

| Fixed | Free |
|---|---|
| Data generation seed (`20260818`) | Machine, core count, memory |
| Workload shape (rows, engines, θ, μ, sensors) | Spark minor version |
| Spark configuration per cell | Absolute wall-clock times |
| Arm definitions and terminal aggregation | Fitted `γ` (platform-dependent by design) |
| Analysis code and figure generation | — |

`γ = a/b` is *expected* to differ per platform. It is a calibrated parameter, not a published constant. What should reproduce across platforms is the **structure**: convexity in `k`, the `√(H/P)` dependence, the ordering of the arms, and the deterministic shuffle-byte counts.

## Reproducing from scratch

```bash
git clone <repo> && cd spark-skew-bench
python -m venv .venv && source .venv/bin/activate
make install
make test          # 36 unit tests, no Spark required for most
make reproduce     # gen → run → analyze → paper
```

Outputs land in:

- `results/pilot.jsonl` — one row per cell, carrying its full specification
- `results/eventlogs/` — raw Spark event logs, re-analysable without re-running
- `results/analysis/` — CSV + LaTeX tables, `report.json`, figures
- `paper/numbers.tex` — every number the manuscript quotes, as LaTeX macros
- `paper/main.pdf`

## Why the event log rather than a listener

The Spark event log is a JSON-lines file written by Spark itself. It survives the driver exiting, needs no JVM callback plumbing from Python, and can be re-analysed months later without re-running anything. Every measurement is tagged with a unique Spark job group, so the mapping from a results row back to the tasks that produced it is **exact**, not inferred from timestamps.

This means a reviewer can audit a claim without trusting the analysis code: the raw events are in the repo.

## Known sources of variance

| Source | Mitigation |
|---|---|
| JIT compilation, class loading, cold page cache | One warmup execution per cell, discarded |
| Long right tail in run times | Median and IQR reported, never mean and SD |
| Co-tenancy on shared compute | Dedicated local compute only; never serverless |
| Optimizer silently changing the plan | `autoBroadcastJoinThreshold = -1`; executed physical plan captured per arm |
| Column pruning skipping the probe | Terminal op is a grouped aggregation touching both sides, not `count()` |

## What a single node does not license

Absolute wall-clock times do not transfer to a cluster, and the paper claims they do not. A single-node shuffle writes to local disk; a cluster shuffle crosses the network, where salting's replication cost is **higher**. So the estimate of `b` here is conservative and the true `k*` on a cluster is *smaller* than reported — an asymmetry stated explicitly in the paper because it cuts in favour of the conclusion and should therefore be held to a higher standard, not a lower one.
