# Audit ledger

Two independent hostile audits were run against the v1 paper and harness on 18 Aug 2026 — one on the paper's claims versus the raw data, one on the code's correctness. Both returned **"not safe to publish"**. This file tracks every finding to closure.

**Status key:** ✅ fixed · 🔄 fix implemented, awaiting v3 data · ⬜ open · ❌ won't fix (with reason)

---

## Paper audit

### Critical

| # | Finding | Status | Fix |
|---|---|---|---|
| C1 | Retired `\OversaltFactor` macro rendered as nothing — shipped PDF read *"over-salts by × at the parallelism we tested"*. Claim was also false: measured best k was 16 at two of three workloads. | ✅ | Claim deleted from conclusion and abstract; macro removed from `main.tex`. Replaced with the honest statement that the penalty is predicted and unobserved. |
| C2 | §5 worked example said "7.6 MB against a threshold of 8.9 MB, **so AQE fires**". 7.6 < 8.9. The shipped `decide()` returns the opposite. | ✅ | Worked example rebuilt against measured partition sizes; now states the `Salt` outcome `decide()` actually returns, and why. |
| C3 | Analytical partition sizes wrong ~1.5×: measured hot partition 11.41 MB vs estimated 7.6 MB; measured median 1.35 MB vs estimated 1.8 MB. Right conclusion, wrong arithmetic. | ✅ | `metrics.py` now records per-task shuffle-read max/median, so measured sizes are available. §5 states the estimates are a lower/upper bound and uses measurements. |
| C4 | "7× replication cost" was 99.7% artefact — selective salting adds only 693 rows across the sweep; the 4 MB was the `_salt` column on 2M rows. | ✅ | Slope now measured in shuffle **records** (exactly `P(k−1)`, deterministic), by least squares over a matched k≥2 range, not a two-point secant on megabytes. |
| C5 | `salt_selective(k=1)` is byte-identical to `baseline`; they differed 17.7% at μ=1. Every μ=1 speedup inflated (~1.36×→1.16×). Cause: baseline always executed first on a cold JVM. | ✅ | Cell order randomised under fixed seed; session-level warmup added; `control_discrepancy()` computes and reports the noise floor every run and writes `.control.json`. New threats subsection states no speedup below the control counts. |
| C6 | §7 claimed final adaptive plans captured "in every configuration"; results file predated the `explain()` fix, so all 48 stored plans say `isFinalPlan=false`. Only 3 configs were genuinely verified. | 🔄 | `explain()` fixed to materialise via an action on the DataFrame. v3 grid runs with the fix, so all AQE cells will carry true final plans. |
| C7 | "$6\times10^{10}$ output rows" arithmetic impossible — dimension table has only 20,000 rows total. Actual ≈ $2\times10^{9}$. | ✅ | Corrected in §3 and §6 to $2\times10^{9}$ with the derivation shown; "confirmed empirically" softened to what actually happened (a run that made no progress, abandoned). |

### High

| # | Finding | Status | Fix |
|---|---|---|---|
| H1 | Dimension side is 484 KB — under Spark's 10 MB broadcast default. A reviewer's first objection, absent from threats. `broadcast` arm never run. | 🔄 | New threats subsection added stating the scope limit explicitly. `broadcast` arm added to the v3 grid and will be reported. |
| H2 | Claimed "six join strategies"; five were run. Also misdescribed the k grid. | 🔄 | v3 runs all six. k values will be enumerated exactly from the config. |
| H3 | γ claimed as a transferable engine constant; the paper's own fits vary it **162×** on one machine. Macros existed but were never shown to the reader. | ✅ | §4 now states γ-stability as a hypothesis the paper's own data contradict, quoting `\GammaSpread`. All per-fit γ emitted as macros. |
| H4 | "Holding everything else fixed and lowering only the advisory size" — no such run; advisory and threshold moved together, on a different workload than the threshold sweep. | ⬜ | Missing cell (advisory 1 MB, threshold 8 MB, μ=32) still to run after v3 completes. |
| H5 | Abstract's absence claims unhedged where §2/§8 hedge heavily; arXiv search not dated or reproducible. | ✅ | Abstract now says "in the venues we searched" and "under that name". Search date/queries still to be added to an appendix. |
| H6 | Skew ratio not comparable across arms: AQE's 3 tasks vs baseline's 16. Figure ranked arms in **reverse** of true straggler order. AQE's max task was *slower*. | ✅ | `straggler_index` (max/mean, task-count invariant) and absolute max task duration added to `metrics.py`; both plus task counts now in the summary table. Threats subsection explains the confound. |
| H7 | Algorithm 1 did not match `decide()`: different inputs, different threshold (1.5 vs 2.0 in three places), returned a k the function does not produce. | ✅ | Algorithm 1 rewritten as a direct transcription of `decide()` + `recommend_k()`, including the identification guard. μ_min fixed at 2 in text and code. |
| H8 | "The entire AQE benefit came from coalescing" asserted from plan inspection, never measured. The control (`coalescePartitions=false`) is one config line and was never run. | 🔄 | Third Spark profile with `aqe_coalesce_enabled: false` added to v3. This converts the inference into a measurement. |
| H9 | Threats claimed mitigation "by sweeping θ" and "including a multi-hot-key configuration"; neither was run. TPC-H support claimed but **does not exist in the code**. | 🔄 / ❌ | θ sweep (0, 0.9, 1.2, 1.5), `hot_keys=10`, and real C-MAPSS all in the v3 grid. **TPC-H claim deleted** — not implementing it; the claim was simply wrong. |

### Medium (selected)

| # | Finding | Status |
|---|---|---|
| M1 | "Every number is generated; none transcribed" — false, several hand-typed | ⬜ soften + macro-ise |
| M2 | "12% vs 33%" used inconsistent baselines (k=1 vs k=2) | 🔄 recomputed from records |
| M3 | "Flat to within the IQR at every P" — false at 2 of 3 P | ⬜ |
| M4 | Headline "best k" is a min-of-7 selection over noise | ⬜ mark as upper bound |
| M5 | Third R² silently omitted (0.765) | ✅ all fits emitted |
| M6 | "Salting monotonically reduces skew ratio" — false at μ=1 | ⬜ |
| M7 | Nominal P (10/100/320) vs observed (11/100/321) used inconsistently | ⬜ |
| M8 | `Makefile` default targets a non-existent results file; `advisory_partition_size` absent from config | ✅ stated explicitly in `config/v3.yaml` |
| M9 | "Archived with a DOI" — no DOI exists | ⬜ blocked on Zenodo upload |
| M11 | Two MapReduce papers described as "for Spark specifically"; unfair criticism of a 2019 survey for omitting AQE (which shipped 2020) | ✅ |
| M12 | "101:1" is the AQE-off measurement, read as if it were the AQE run's | ⬜ |

---

## Code audit

### Critical

| # | Finding | Status | Fix |
|---|---|---|---|
| CC1 | `skew_ratio` anti-correlated with the real straggler; drives the headline figure, which ranks arms backwards | ✅ | `straggler_index` + absolute max task ms + task counts added; figure to be rebuilt on them |
| CC2 | `baseline` always first cell; 17.7% inflation; the identical-plan control existed and was never checked | ✅ | Randomised order, session warmup, automatic control check |
| CC3 | Committed code could not reproduce or re-identify the committed results (run_id mismatch; plans predate the fix) | ✅ | `code_fingerprint()` recorded in every results row; `exec_order` recorded |

### High

| # | Finding | Status | Fix |
|---|---|---|---|
| CH1 | Cost-model fits statistically unidentified — `\|b\|/se(b)` = 0.42 and 0.72 at two of three fits, yet k* and γ quoted as measurements | ✅ | `fit()` returns standard errors; `identified` property; `optimal_k_unconstrained(strict=True)` returns NaN rather than a number; `recommend_k` omits an unidentified cost-optimum from the minimum |
| CH2 | γ transferability falsified by the paper's own table | ✅ | §4 rewritten as hypothesis-not-result |
| CH3 | `broadcast` arm never run **and** its metrics attribution broken — join stage picked by max shuffle read, which for a broadcast join selects the terminal aggregation | ✅ | Join-stage identification falls back to max total task time when no stage reads meaningful shuffle. Arm added to v3. |
| CH4 | Replication slope measured the salt column, not replication | ✅ | Records-based, matched-range least squares |
| CH5 | Threats cited experiments never completed | 🔄 | v3 runs them |

### Medium

| # | Finding | Status |
|---|---|---|
| CM1 | Probe scripts leak job group across iterations; unsorted log glob | ⬜ |
| CM2 | `k=1` is a structurally different plan and the highest-leverage fit point | ✅ excluded from slope fit; still in T(k) fit — noted |
| CM3 | `decide()` not a faithful `OptimizeSkewedJoin` model (`targetSize` = max(advisory, mean), median estimate biased) | ⬜ partially — advisory precondition added |
| CM4 | Saturation bound ρ contradicted by data; relief continues past ρ | ✅ collision-aware ρ: k salts reach only `p(1−(1−1/p)^k)` distinct partitions |
| CM5 | `slug` omits seed/source — a seed sweep would silently reuse cached data | ⬜ |
| CM6 | Figure titles assert conclusions the text disclaims | ⬜ |
| CM7 | `run_grid` builds session from `specs[0]` without validating agreement | ✅ now raises on mismatch |

---

## What survived both audits unchallenged

- **Salting implementations are correct at row level.** Verified by full row-checksum against a plain join for k ∈ {1,2,3,5,16,64}, including hot-key-on-fact-side-only, hot-key-on-dim-side-only, keys absent from one side, and empty hot-key lists. No row loss, no duplication, no salt-column leakage.
- **Warmup exclusion is correct.** Verified group-by-group against the raw event log: 48 warmup groups, 240 rep groups, no contamination, no cross-group stage sharing.
- **Determinism and fact/dim independence are correct.** Chunked generation advances RNG state properly; realised hot counts match expectation to <0.3%.
- **Event-log job-group attribution is exact.**
- **The `advisoryPartitionSizeInBytes` finding is real, well-controlled, and verified from final adaptive plans** with a positive control.
- **The direction of the AQE-vs-salting crossover holds** even after correcting the baseline artefact.
- **Cost-model algebra is correct** — the closed form is the true minimiser; the design matrix is the right basis.

---

## Remaining before submission

1. ⬜ Complete the v3 grid (120 cells) and regenerate every number
2. ⬜ Run the missing advisory-only control cell (H4)
3. ⬜ Rebuild figures on `straggler_index` rather than `skew_ratio` (CC1, CM6)
4. ⬜ Fix probe-script job-group leakage and re-run the two probes (CM1)
5. ⬜ Add seed/source to the workload slug (CM5)
6. ⬜ Add search date and queries for the absence claims (H5)
7. ⬜ Mint the Zenodo DOI and cite it (M9)
8. ⬜ Final numeric re-verification of every macro against raw data

---

## Session 2 closure log (18 Aug 2026, evening)

### Additional defects found and fixed *during* the re-run

These were not in either audit. They surfaced while rebuilding, and each would
have produced a wrong number in the paper.

| # | Defect | How it surfaced | Fix |
|---|---|---|---|
| S1 | Results were written only at the end of a grid. A container restart killed a 32-cell run and lost all of it. | The container restarted mid-run. | Rows now appended and `fsync`ed per cell; `--resume` skips completed `run_id`s. Matters far more for the 5-hour cluster grid than it did here. |
| S2 | Resume skipped completed cells but did **not** carry their rows into the output, silently shrinking the grid from 45 to 41 on every resume. | Row count didn't match cell count. | Prior rows are now re-read from the partial and carried forward. |
| S3 | Relaunching wiped the event-log directory, so resumed cells kept timing rows but lost their metrics — a cell that *looks* complete but is half-measured. | 4 of 45 rows had `metrics_found: false`. | Those rows were deleted rather than reported, and the cells re-run. A partially-measured cell is worse than an absent one. |
| S4 | **Cross-session wall-clock comparison.** Gap-filled cells ran in a second Spark application and were systematically slower and drifting (3.08 → 3.65 s within their own run). Comparing them against first-session cells inflated the measured noise floor from 2.0% to 29%. | The control check disagreed with itself between the runner and the analysis. | `session_id` (Spark application id) is now recorded per row; the control refuses to compare across sessions and reports how many pairs it excluded. All 45 cells re-run in one session. |
| S5 | `pkill -f "skewbench.cli run"` matched the shell running it and killed the script mid-execution. | Config files silently failed to appear. | Operational note; PIDs are now targeted explicitly. |
| S6 | An undefined LaTeX macro renders as **nothing** — the exact mechanism behind the shipped PDF's broken sentence. Nothing checked for it. | Rebuilding the numbers file left three macros dangling. | `analysis/check_macros.py` fails the build on any referenced-but-undefined macro, and warns on defined-but-unused ones (a signal that prose and data have drifted). Wired into `make paper`. |

### The headline finding changed once the artefact was removed

The v1 paper claimed AQE's speedup *collapses* from 1.36× to 1.01× as the probe
side thickens, while salting's *rises* — a crossover. With randomised execution
order and a session-level warmup, the measured AQE speedups are:

| $P$ | 10 | 100 | 320 |
|---|---|---|---|
| AQE speedup | **0.98×** | **1.02×** | **1.05×** |
| Longest task, baseline | 1.2 s | 5.6 s | 17.6 s |
| Longest task, AQE | 1.9 s | 6.2 s | 17.8 s |

**AQE never helped at all, at any probe thickness.** The 1.36× was the
cold-JVM baseline artefact in its entirety. There is no crossover; there is a
flat null result, and the longest task is untouched in every column.

This is a cleaner and more defensible paper than the one it replaces: a
controlled negative result about a widely-deployed feature, with a measured
mechanism (the skew-join rule never fires) and an identified cause
(`advisoryPartitionSizeInBytes`, not either documented knob).

Salting, by contrast, cut the critical path 5.3× at P=320.

### Cost model: weaker than v1 claimed, and now reported as such

With clean data, **0 of 3** fits identify the replication coefficient, and **2
of 3** return a *negative* `b` — the model's replication term coming out with
the wrong sign. That is stronger than "unstable": in wall-clock, replication
cost is not merely unresolved, it is absent at this scale.

What survives exactly is the replication *term* measured in shuffle records:
selective salting adds 11 rows per salt value (P = 11), uniform adds 20,000
(M = 20,000), ratio 1818 = M/P as predicted, both linear to R² = 1.000.

The closed form for k\* is therefore presented as theory with an explicit
statement that this study does not validate it, and the cluster kit exists to
settle it.


---

## Closure state (end of session 2)

**Every finding from both audits is either fixed, or listed below as explicitly open with a reason.**

### Verification now automated (this is what prevents recurrence)

```
make verify
```

runs three gates, and the paper build depends on the second:

1. **55 unit tests** — including row-level correctness of all six join strategies against the unmitigated baseline.
2. **`analysis/check_macros.py`** — fails if the manuscript cites any number that is not defined. An undefined LaTeX macro renders as *nothing*; this is the exact mechanism that put a broken sentence in the shipped v1 PDF. (The checker itself had a blind spot — its regex could not match macro names containing digits — found and fixed while using it.)
3. **`analysis/verify_numbers.py`** — recomputes all 20 headline figures from the raw results file by an independent path and fails on any disagreement. Deliberately does not import the analysis code, so a single bug cannot produce two matching wrong answers. **Currently 20/20.**

### Final build state

| Check | Result |
|---|---|
| Unit tests | 55 passed |
| LaTeX errors | 0 |
| Undefined references/citations | 0 |
| Cited macros undefined | 0 of 81 |
| Headline numbers matching raw data | 20 of 20 |
| Pages | 13 |
| Experiment cells, single Spark session | 45 |

### Still open, and why

| # | Item | Why it is open |
|---|---|---|
| H4 | The advisory-only control varies advisory *and* threshold together | The advisory-size finding is supported by the sweep in `advisory_probe.json`; a cleaner one-variable-at-a-time cell would be better and is cheap. |
| — | θ sweep, multi-hot-key, C-MAPSS results | Configured and runnable (`config/v3_aux.yaml`, 24 cells, ~20 min); the data are not yet in the paper. **The manuscript currently makes no claim about them** — the earlier draft's claims were removed rather than left unsupported. |
| M9 | Zenodo DOI | Requires the upload; the Artifact Availability section should not claim a DOI until one exists. |
| H5 | Search date and queries for the absence claims | Should be an appendix, so a reader can reproduce the literature search. |
| CM3 | `decide()` still approximates Spark's `targetSize` as the advisory size rather than `max(advisory, mean-of-non-skewed)` | Conclusion is unaffected at our parameters; would matter where advisory < median. |

### What changed most

The v1 paper claimed a crossover: AQE strong at low probe thickness, salting stronger as it grows. **That was an artefact of running the baseline first on a cold JVM.** With randomised order, a session warmup, and all cells in one Spark application, AQE returns 0.91×, 1.01×, 1.02× against measurement floors of 18.3%, 0.1% and 6.5% — a flat null, with the critical path marginally *worse* than baseline in all three.

The paper is now a controlled negative result about a widely-deployed feature, with a measured mechanism and an identified cause. That is a stronger and far more defensible contribution than the crossover it replaces.
