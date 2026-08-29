"""Run one grid stage from a checkout of this repo, without a Git folder.

Why this file exists
--------------------
The notebooks in ``cluster/notebooks`` are the readable way to drive a run, but
they only work if the Databricks Git folder has actually been pulled to the
commit you think it has. Twice now a run has failed in a way that looked like a
code bug and was in fact a stale checkout: the notebook on the cluster was days
behind the repository. There is no cheap way to see that from inside a notebook,
and every occurrence costs a full cluster cycle.

This driver removes the question. Clone the repository onto the driver, run one
stage from that clone, and print the commit it ran. Then "which code ran?" has
an answer that is a git sha rather than an assumption::

    %sh rm -rf /tmp/sk && git clone -q --depth 1 \
        https://github.com/adityamalviya04692/spark-skew-bench /tmp/sk

    STAGE = "smoke"
    exec(open("/tmp/sk/cluster/run_stage.py").read())

It shares the notebook's ``spark`` session because it is exec'd, not spawned.

Stages: ``smoke`` -> config/cluster_smoke.yaml, ``salt`` -> cluster_salt.yaml,
``aqe`` -> cluster_aqe.yaml, ``collect`` -> reparse + verify + bundle only.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------- parameters

try:
    STAGE
except NameError:
    STAGE = "smoke"

REPO = globals().get("REPO_DIR", "/tmp/sk")
VOLUME = globals().get("VOLUME", "/Volumes/skewbench_ws/default/skewbench")
DATA_ROOT = f"{VOLUME}/data"
RESULTS = f"{VOLUME}/results"
os.makedirs(RESULTS, exist_ok=True)

STAGES = {
    "smoke": ("config/cluster_smoke.yaml", "smoke.jsonl"),
    "salt": ("config/cluster_salt.yaml", "cluster_salt.jsonl"),
    "aqe": ("config/cluster_aqe.yaml", "cluster_aqe.jsonl"),
    "collect": (None, None),
}
if STAGE not in STAGES:
    raise SystemExit(f"unknown STAGE {STAGE!r}; expected one of {sorted(STAGES)}")

if f"{REPO}/src" not in sys.path:
    sys.path.insert(0, f"{REPO}/src")

try:
    import yaml  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "PyYAML"],
                   check=True)

commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
subject = subprocess.run(["git", "-C", REPO, "log", "-1", "--pretty=%s"],
                         capture_output=True, text=True).stdout.strip()
print(f"stage       {STAGE}")
print(f"code        {REPO} @ {commit or '(no git metadata)'}  {subject}")
print(f"volume      {VOLUME}")

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", "")
LOG_ROOT = f"{VOLUME}/logs/{cluster_id}"
print(f"cluster id  {cluster_id}")
print(f"log root    {LOG_ROOT}")


# ------------------------------------------------------------------- helpers

def event_logs(newer_than=0.0):
    """Delivered event-log files, optionally only those written after a cutoff.

    The cutoff is the whole point. Earlier runs leave their logs in this
    destination forever, so a bare existence check succeeds instantly against
    files that predate the run -- and the reparse then attributes nothing,
    which looks exactly like broken attribution and is not.
    """
    out = []
    for path in glob.glob(f"{LOG_ROOT}/**/*", recursive=True):
        if not (os.path.isfile(path) and "eventlog" in path):
            continue
        if os.path.basename(path).startswith("."):
            continue
        if os.path.getmtime(path) < newer_than:
            continue
        out.append(path)
    return out


def wait_for_logs(cutoff, budget_s=600):
    print(f"logs already present from earlier runs: {len(event_logs())}")
    deadline = time.time() + budget_s
    found = []
    while time.time() < deadline:
        found = event_logs(cutoff)
        if found:
            break
        print(f"  waiting for THIS run's event log "
              f"({int(deadline - time.time())}s left)...", flush=True)
        time.sleep(30)
    print(f"event log files newer than this run's start: {len(found)}")
    if not found:
        raise SystemExit(
            "No event logs delivered within the budget. Check Compute -> Edit "
            "-> Advanced -> Logging; delivery cannot be added to an already "
            "running cluster."
        )
    return found


def reparse(target):
    out = subprocess.run(
        [sys.executable, f"{REPO}/analysis/reparse.py", target, LOG_ROOT],
        capture_output=True, text=True)
    print(out.stdout or "")
    if out.stderr:
        print(out.stderr)


def report(target):
    rows = [json.loads(line) for line in open(target) if line.strip()]
    for row in rows:
        print(f"  {row['arm_label']:26s} median {row['wall_median_s']:7.2f}s  "
              f"straggler {row.get('join_straggler_index')}  "
              f"tasks {row.get('n_tasks')}  "
              f"attribution {row.get('attribution')}  "
              f"metrics {row.get('metrics_found')}")
    missing = [r["arm_label"] for r in rows if not r.get("metrics_found")]
    print("cells without metrics:", missing or "none")
    return rows, missing


# --------------------------------------------------------------------- stages

if STAGE == "collect":
    newest_ms = 0
    for name in ("cluster_salt", "cluster_aqe", "smoke"):
        target = f"{RESULTS}/{name}.jsonl"
        if not os.path.exists(target):
            continue
        for line in open(target):
            if line.strip():
                for span in json.loads(line).get("rep_windows", []):
                    newest_ms = max(newest_ms, int(span[1]))
    wait_for_logs(newest_ms / 1000 if newest_ms else 0)

    summary = {}
    for name in ("cluster_salt", "cluster_aqe"):
        target = f"{RESULTS}/{name}.jsonl"
        if not os.path.exists(target):
            summary[name] = "MISSING"
            continue
        print(f"\n--- reparse {name} ---")
        reparse(target)
        rows, missing = report(target)
        summary[name] = {
            "rows": len(rows),
            "with_metrics": sum(1 for r in rows if r.get("metrics_found")),
            "sessions": sorted({r.get("session_id") for r in rows}),
            "fingerprints": sorted({r.get("code_fingerprint") for r in rows}),
        }
    print("\n" + json.dumps(summary, indent=2))

    bundle = f"{RESULTS}/cluster_bundle"
    shutil.rmtree(bundle, ignore_errors=True)
    os.makedirs(bundle, exist_ok=True)
    for name in os.listdir(RESULTS):
        if name.endswith((".jsonl", ".json")):
            shutil.copy2(f"{RESULTS}/{name}", f"{bundle}/{name}")
    shutil.make_archive(bundle, "zip", bundle)
    size_mb = os.path.getsize(bundle + ".zip") / 1e6
    print(f"\n{bundle}.zip   {size_mb:.1f} MB")

else:
    from skewbench.cli import load_config, specs_from_config
    from skewbench.runner import run_grid

    config_rel, out_name = STAGES[STAGE]
    out_path = f"{RESULTS}/{out_name}"

    # Start clean. run_grid resumes from <out>.partial by design, which is what
    # you want after a crash mid-grid and exactly what you do not want when
    # re-running a stage whose earlier attempt is the thing being replaced:
    # the resumed rows carry the OLD repetition windows, so attribution is
    # matched against logs from a different day.
    for stale in (out_path, out_path + ".partial"):
        if os.path.exists(stale):
            os.remove(stale)
            print(f"removed stale {os.path.basename(stale)}")

    specs = specs_from_config(load_config(f"{REPO}/{config_rel}"))
    print(f"\n{len(specs)} cells in {config_rel}\n")

    RUN_STARTED = time.time()
    rows = run_grid(specs, data_root=DATA_ROOT, out_path=out_path, progress=True)
    print(f"\n{len(rows)} rows in {(time.time() - RUN_STARTED) / 60:.1f} min")

    # Metrics are absent right now on purpose: Databricks delivers event logs
    # asynchronously, so nothing is readable at the moment a grid finishes.
    wait_for_logs(RUN_STARTED)
    reparse(out_path)
    rows, missing = report(out_path)
    if missing:
        raise SystemExit(
            "Event-log metrics are still missing after reparse. Timings exist "
            "but no task-level evidence does. Do not start the next stage."
        )
    print("\nMetrics attach correctly.")

    if STAGE == "smoke":
        print(subprocess.run(
            [sys.executable, f"{REPO}/cluster/project_runtime.py",
             "--smoke", out_path, "--repo", REPO, "--gen-minutes", "2"],
            capture_output=True, text=True).stdout)
