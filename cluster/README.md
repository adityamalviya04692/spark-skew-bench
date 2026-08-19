# Cluster validation kit

> **New to Azure or Databricks? Read [`../docs/CLUSTER_GUIDE.md`](../docs/CLUSTER_GUIDE.md) instead.**
> It is a click-by-click walkthrough from creating an Azure account through to
> downloading results. This file is the condensed reference.

> ⚠️ **Two facts that break most tutorials, verified August 2026:**
> 1. An Azure **Free Trial subscription cannot run a Databricks cluster** — it is
>    capped near 4 vCPUs and is *not permitted to request more*. You must upgrade
>    to Pay-As-You-Go first; the unused $200 credit carries over.
> 2. **DBFS is deprecated and new workspaces are provisioned without it.** Any
>    instruction using `dbfs:/` will fail. Use Unity Catalog volumes
>    (`/Volumes/<catalog>/<schema>/<volume>/...`) throughout.

This kit exists to settle the one prediction the single-node study could not: that the optimal salt cardinality falls as **1/√P**, and that **k\* is smaller on a cluster than on a laptop**.

## Why a cluster can settle it and a laptop cannot

The cost model is

```
T(k) = a·H/k + b·P·(k−1) + c
```

`b` is the cost of moving one replicated probe-side row through the shuffle. On a single host that is a write to local disk and a read back. On a cluster it crosses the network. So `b` should be **materially larger**, the replication term should become visible against the straggler term, and the convex minimum should appear at a **smaller k**.

On the single node the replication term was real and measurable in shuffle records — exactly `P·(k−1)`, to the row — but far too small to move wall-clock. The paper reports that as an unconfirmed prediction rather than dressing it up. This kit is how it gets confirmed or refuted.

There is a second reason. On 2 cores the **parallelism ceiling C bound the recommendation in every single fit**, so the cost-optimum branch of the decision rule was never exercised. With 32 task slots it should be.

## What you need

- A Spark 3.5+ cluster. 4 workers × 8 cores is enough; more is better.
- ~200 GB of scratch on distributed storage.
- 3–5 hours of cluster time for the full grid.
- **Photon OFF** on Databricks — it changes the execution engine, and the results would not be comparable to open-source Spark or to the single-node run.

## Databricks

Push the repo to a **public** GitHub repo, then in the workspace sidebar:
**Workspace → Create → Git folder** → paste the HTTPS URL. Public repos clone
without any credential, which is by far the fewest steps.

Then, in a notebook attached to the cluster:

```python
%pip install PyYAML
dbutils.library.restartPython()

import sys
sys.path.insert(0, "/Workspace/Users/<you>/spark-skew-bench/src")

from skewbench.cli import load_config, specs_from_config
from skewbench.runner import run_grid

specs = specs_from_config(load_config(
    "/Workspace/Users/<you>/spark-skew-bench/config/cluster_salt.yaml"))

rows = run_grid(specs,
                data_root="/Volumes/skewbench_ws/default/skewbench/data",
                out_path="/Volumes/skewbench_ws/default/skewbench/cluster_salt.jsonl",
                progress=True)
```

The harness detects Databricks automatically (`DATABRICKS_RUNTIME_VERSION`),
attaches to the existing session, and leaves event logging to the platform
rather than fighting it. Metrics are attached afterwards from the delivered
logs — see below.

## EMR / self-managed YARN

```bash
aws s3 cp cluster/emr_bootstrap.sh s3://<bucket>/
# launch the cluster with that bootstrap action, then:
spark-submit --deploy-mode client \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir=hdfs:///var/log/spark/skewbench \
  cluster/run_cluster.py \
  --config config/cluster_salt.yaml \
  --data-root hdfs:///tmp/skewbench/data \
  --out /tmp/cluster_salt.jsonl
```

## Smoke test first

Do not start a 5-hour run blind:

```bash
python cluster/run_cluster.py --config config/cluster_salt.yaml \
  --data-root <path> --out /tmp/smoke.jsonl --limit 3
```

Three cells, a few minutes. Check `metrics_found: true` in the output before committing to the full grid.

## After the run

Timing rows are written as the grid proceeds, but **metrics are not attached on
a managed platform**: Databricks delivers event logs asynchronously, roughly
every five minutes, so they are not readable when the grid finishes.

1. Leave the cluster running for **~5 minutes** after completion so delivery
   flushes. Terminating immediately loses the final batch.
2. Then attach metrics from the delivered logs:

```bash
python analysis/reparse.py cluster_salt.jsonl \
  /Volumes/skewbench_ws/default/skewbench/logs/<cluster-id>/eventlog
```

## Send back

1. `cluster_salt.jsonl` — the results
2. `cluster_salt.jsonl.control.json` — **read this first** (see below)
3. The **whole log folder** — this is the important one. Because every metric is derived from the event log rather than from a live listener, the whole run can be re-analysed later, with new metrics, without re-running anything. That is how the measured partition sizes were recovered from the single-node run after the fact.

## Read the control file before anything else

`cluster.control.json` compares `baseline` against `salt_selective(k=1)`. Those two compile to **the same physical plan under the same configuration**, so any difference between them is pure measurement noise.

**This is your noise floor. Any speedup smaller than this number is not a result.**

On the single node, an early run showed a 17.7% discrepancy here, caused by executing `baseline` first on a cold JVM. It invalidated every speedup on that workload. Cell order is now randomised and a session-level warmup runs first, but check the control anyway — that is what it is for.

## What the results will look like

Three outcomes, all publishable:

| Outcome | What it means | What the paper says |
|---|---|---|
| `k*` identified, falls as 1/√P | **Prediction confirmed** | The limitation becomes a validated result; the closed form is usable |
| `k*` identified but does not follow 1/√P | Model is wrong in a specific, informative way | Report the actual scaling; that is a finding |
| `b` still not identified (`\|b\|/se(b) < 2`) | Replication cost is negligible even at cluster scale | Strong practical result: **selective salting is essentially free to over-provision**, and the folklore warning applies only to uniform salting |

The harness will tell you which: `analysis/analyze.py` reports `identified` and `|b|/se(b)` per fit and **refuses to print a k\* it cannot support**.

## Analysing what comes back

```bash
python analysis/analyze.py results/cluster_salt.jsonl results/analysis_cluster
python analysis/compare_scales.py results/v3.jsonl results/cluster_salt.jsonl
```

The second produces the single-node-vs-cluster comparison table the paper needs.
