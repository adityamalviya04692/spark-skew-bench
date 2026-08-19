# skewbench

**A reproducible benchmark for Spark skewed-join mitigation, and a cost model for salt cardinality.**

Apache Spark ships an automatic remedy for skewed joins — the `OptimizeSkewedJoin`
rule of Adaptive Query Execution. Practitioners keep salting by hand anyway. Nobody
has published which one you should use, or how much salt to use when you do.

This repository is the harness and the analysis behind
*"Adaptive Query Execution or Salting? A Cost Model and Decision Procedure for
Skewed Joins in Apache Spark."*

---

## The result in one paragraph

Salting a hot key across `k` sub-keys cuts the straggler's input in proportion to
`1/k` and replicates the matching probe-side rows in proportion to `k`. Total time
is therefore **convex in k** with an interior minimum:

```
T(k) = a·H/k + b·P·(k−1) + c        →        k* = √(γ · H/P),   γ = a/b
```

where `H` and `P` are the hot key's row counts on the two sides. Two ceilings bound
the useful range: **saturation** at `ρ = H/(N/p)`, beyond which splitting the hot key
relieves nothing, and **parallelism** at `C`, the number of concurrent task slots.

```
k_eff = max(1, round(min(k*, ρ, C)))
```

Because `k*` grows only as the **square root** of the skew, the intuition that severe
skew calls for aggressive salting is structurally wrong.

## What we actually measured

Three findings from 48 configurations, all reproducible with `make reproduce`:

**1. AQE's benefit collapses as the probe side thickens — and salting's grows.**

| Probe-side hot-key rows `P` | 10 | 100 | 320 |
|---|---|---|---|
| AQE speedup vs. baseline | **1.36×** | 1.06× | **1.01×** |
| Salting speedup vs. baseline | 1.27× | 1.40× | **1.68×** |

They cross over. Which mechanism wins depends on a statistic nobody tells you to look at.

**2. Spark's skew-join rule never fired — on a join with a 101:1 task-time skew ratio.**

Reading the *final* adaptive plans (not the pre-adaptive ones — see below) shows
`AQEShuffleRead coalesced` and never `AQEShuffleRead skewed`. The entire measured AQE
benefit came from partition **coalescing**, which does nothing for a straggler. That
explains the collapse: coalescing amortises a fixed cost, and fixed cost is a shrinking
fraction of a job that grows with `P`.

**3. The blocking condition was not either knob with "skew" in its name.**

Sweeping `skewedPartitionThresholdInBytes` from 64 MB → 256 KB and
`skewedPartitionFactor` from 5 → 1 changed nothing. The binding precondition was
**`advisoryPartitionSizeInBytes`** — that is the size a skewed partition is split
*into*, so a hot partition smaller than it cannot be divided at all:

| Partitions | Advisory | Threshold | Skew-join fired? |
|---|---|---|---|
| 16 | 16 MB | 8 MB | ❌ no |
| 16 | **1 MB** | 1 MB | ✅ **yes** |
| 200 | **256 KB** | 256 KB | ✅ **yes** |

One config change, no code change. This is the finding most likely to be
immediately useful to you.

**And one honest negative:** the wall-clock U-curve did **not** resolve at this scale
for *selective* salting. Beyond `k=2` runtime is flat to within the IQR, because
selective salting replicates only `P` rows — about 20k rows against a 2M-row fact table
even at `k=64`. The replication cost is real and visible in shuffle bytes
(0.065 MB/k selective vs 0.433 MB/k uniform, a **7×** gap), but too small to dominate
wall-clock here. So: the folklore warning against large `k` is well founded for
*uniform* salting and largely misplaced for *selective* salting.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test

# End-to-end: generate data, run the grid, build figures, compile the paper
make reproduce
```

The pilot grid runs on a 2-core laptop in roughly 40 minutes and needs about
400 MB of scratch space. `config/full.yaml` is the grid reported in the paper and
wants 8+ cores.

### Just want the recommendation for your own join?

```bash
PYTHONPATH=src python -m skewbench.cli advise \
  --H 1170000 --P 300 --n-fact 2000000 \
  --partitions 200 --cores 8 --row-bytes 80 --two-sided
```

```
decision: salt in addition to AQE
  - hot partition is approximately 93.6 MB against a trigger threshold of 22.1 MB …
  - both sides are concentrated on the same key. AQE splits the skewed partition and
    replicates the matching partition to every split, so the probe-side work
    multiplies with the split count and the straggler is only partly relieved.

k* (unconstrained) = 39.50; saturation bound rho = 9.36; parallelism bound C = 8;
recommended k = 8 (binding: parallelism)
```

---

## What it measures

Six strategies, on the same workload, with the same terminal aggregation:

| Arm | Description |
|---|---|
| `baseline` | sort-merge join, AQE disabled, no mitigation |
| `aqe` | AQE enabled with skew-join optimisation |
| `salt_uniform(k)` | whole probe side replicated `k` times |
| `salt_selective(k)` | **hot-key rows only** replicated — cost scales with `P`, not `M` |
| `aqe_salt(k)` | AQE enabled *and* selective salting |
| `broadcast` | broadcast hash join |

Metrics are separated by what they license:

- **Deterministic** — shuffle bytes, records shuffled, spill bytes. Properties of the
  physical plan and the data, not of the machine. These are identical on a laptop and
  on a hundred nodes, and the transferable claims rest on them.
- **Structural** — join-stage skew ratio (max ÷ median task duration) and task-time
  CV. Within-run ratios, so they cancel most machine-specific scaling.
- **Machine-dependent** — wall-clock. Reported with median and IQR, used to calibrate
  `γ`, never load-bearing for a transferable claim.

All of it is recovered by parsing the **Spark event log**, so a result can be
re-analysed months later without re-running anything. Each measurement is tagged with
a unique Spark job group, giving an exact rather than timestamp-inferred mapping from
a results row back to the tasks that produced it.

---

## The workload

Two Parquet tables with an aerospace engine-telemetry schema following the NASA
C-MAPSS convention (unit id, operational cycle, three operational settings, N sensor
channels):

- **`fact_telemetry`** — the skewed side. `engine_id` drawn from a truncated Zipf law,
  so a few problem engines dominate, as in a real fleet.
- **`dim_maintenance`** — the probe side. Uniform keys give **one-sided** skew; Zipf
  keys give **two-sided** skew, the case where AQE's split-and-replicate degenerates.

That two-sided case is why the harness generates its own data rather than using
skewed TPC-H: the standard skewed generators don't produce it, and it is the condition
the decision rule turns on.

Generation is deterministic given a seed. A `manifest.json` records `H` and `P`
exactly, so nothing downstream has to re-derive them.

---

## Layout

```
src/skewbench/
  config.py      experiment specifications; every result row carries its full spec
  datagen.py     Zipfian generator + engine-telemetry schema
  arms.py        the six join strategies
  metrics.py     Spark event-log parser
  runner.py      grid driver: warmup, repetitions, job-group tagging
  costmodel.py   T(k), closed-form k*, the two ceilings, the decision rule
  cli.py         gen / run / analyze / advise
analysis/
  analyze.py     tables, paper macros, report.json
  plots.py       figures (print- and grayscale-safe)
config/          pilot.yaml (laptop) and full.yaml (paper)
paper/           IEEEtran manuscript; numbers.tex is generated, never hand-edited
tests/           correctness, parameter recovery, and event-log parsing
```

---

## Reproducibility notes

- **Every number in the paper is generated.** `analysis/analyze.py` writes
  `paper/numbers.tex` as LaTeX macros. The manuscript contains no hand-typed
  measurement, so the paper and the data cannot drift apart.
- **Correctness before performance.** `tests/test_arms.py` asserts that all six
  strategies return results *identical* to the unmitigated baseline. A mitigation that
  changes the answer is not a mitigation.
- **Warmup is discarded, not averaged in.** The first execution of a plan pays JIT and
  page-cache costs, and including it inflates arms unevenly.
- **Median and IQR, not mean and SD.** Run times have a long right tail.
- **Automatic broadcast is disabled** except in the broadcast arm, so a strategy under
  test is never silently replaced by a different plan. Each arm's executed physical
  plan is captured into the results file.

## Honest limitations

Measurements in the paper were taken on a **single node**. Absolute wall-clock times
do not transfer to a cluster and no claim is made that they do. A single-node shuffle
writes to local disk; a cluster shuffle crosses the network, where salting's
replication cost is **higher** — so the true `k*` on a cluster is *smaller* than
reported here. That asymmetry strengthens rather than weakens the practical finding.

Zipf is a model of skew, not skew itself. The harness accepts skewed TPC-H and real
C-MAPSS data as drop-in replacements.

---

## Citing

See `CITATION.cff`. If you use the cost model or the decision procedure, please cite
the paper.

## Licence

MIT.
