"""Re-derive metrics from saved event logs, without re-running any experiment.

This is the payoff of storing raw Spark event logs rather than only derived
numbers. When the metrics layer gains a field -- as it did when the analytical
partition-size estimates turned out to be wrong and the measured ones were
needed -- the existing runs can be re-analysed rather than repeated. Nothing
about the experiment changes; only what we compute from it.

Usage:
    python analysis/reparse.py results/v3.jsonl results/eventlogs/v3
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skewbench.metrics import RunMetrics, parse_event_log  # noqa: E402
from skewbench.runner import METRIC_FIELDS as NUMERIC_FIELDS  # noqa: E402


def main(results_path: str, log_dir: str, out_path: str | None = None) -> None:
    rows: List[Dict[str, Any]] = [
        json.loads(line) for line in open(results_path, encoding="utf-8")
        if line.strip()
    ]

    by_group: Dict[str, RunMetrics] = {}
    # Recursive, because Databricks cluster log delivery nests the event log
    # several levels below the destination you configure:
    #   <dest>/<cluster-id>/eventlog/<cluster-id>_<driver-ip>/<app-id>/eventlog
    # A non-recursive glob at the destination finds nothing and reports zero
    # job groups, which looks identical to "the tags never reached the log".
    # Databricks cluster log delivery drops stdout, stderr, log4j output,
    # init-script logs and metrics files into the same tree as the event log.
    # Feeding those to the event parser is how "0 job groups" gets produced by
    # a run whose tagging was fine, so select rather than take everything --
    # and say out loud what was skipped, because a silent filter that is too
    # aggressive looks exactly like tagging that never worked.
    candidates = sorted((q for q in Path(log_dir).rglob("*")
                         if q.is_file() and not q.name.startswith(".")),
                        key=lambda q: q.stat().st_mtime)
    logs = [q for q in candidates if "eventlog" in str(q).lower()]
    skipped = len(candidates) - len(logs)
    if not logs:
        raise SystemExit(
            f"No event-log files under {log_dir} ({len(candidates)} other "
            "files present). Cluster log delivery writes them to "
            "<destination>/<cluster-id>/eventlog/... -- check the path."
        )
    # Attribution strategy, decided by evidence rather than assumed.
    #
    # Preferred: the marker the runner wrote. Exact -- a job either carries the
    # run's id or it does not. Fallback: submission time inside a repetition's
    # recorded window. Inferred, and used only when no marker survives, which is
    # the case on Databricks: it propagates neither spark.jobGroup.id (it
    # overwrites that with its own internal id) nor our job tag into the
    # delivered event log.
    windows: List[tuple] = []
    for row in rows:
        for group, span in zip(row.get("group_ids", []),
                               row.get("rep_windows", [])):
            if isinstance(span, (list, tuple)) and len(span) == 2:
                windows.append((int(span[0]), int(span[1]), group))
    windows.sort()

    # The upper bound of each window is padded, because the driver stops its
    # timer once the last job is submitted rather than when the log records it.
    # The padding is clamped to just before the NEXT window opens: without that
    # clamp a padded window swallows the first job of the following repetition,
    # which silently merges two repetitions into one and drops the other. That
    # is not hypothetical -- it cost one repetition of four in testing.
    padded = []
    for i, (start, end, group) in enumerate(windows):
        limit = end + 1000
        if i + 1 < len(windows):
            limit = min(limit, windows[i + 1][0] - 1)
        padded.append((start, max(end, limit), group))

    def by_window(props: Dict[str, Any], submitted: int):
        # Last window that opened at or before this job, so a job can never be
        # claimed by an earlier repetition than the one it actually ran in.
        chosen = None
        for start, end, group in padded:
            if start <= submitted <= end:
                chosen = group
        return chosen  # None: platform overhead or generation, not a measurement

    for log in logs:
        by_group.update(parse_event_log(log))
    print(f"parsed {len(logs)} event log file(s) "
          f"({skipped} non-event files skipped), {len(by_group)} job groups")

    # Distinguish the two ways this can go wrong, because they look identical
    # in the results file and have completely different fixes:
    #   (a) our tag is in the log but the rows do not match it  -> naming
    #   (b) our tag is not in the log at all                    -> attribution
    from skewbench.metrics import iter_events  # noqa: E402
    tagged = untagged = 0
    seen_props: set = set()
    for log in logs:
        for event in iter_events(log):
            if event.get("Event") != "SparkListenerJobStart":
                continue
            props = event.get("Properties") or {}
            raw = props.get("spark.job.tags") or ""
            if any(t.strip().startswith("skewbench:") for t in raw.split(",")):
                tagged += 1
            else:
                untagged += 1
                for key in props:
                    if "tag" in key.lower() or "jobGroup" in key:
                        seen_props.add(key)
    print(f"  job starts carrying a skewbench: tag -> {tagged}")
    print(f"  job starts without one             -> {untagged}")
    if not tagged:
        print("  NO skewbench tag reached the event log. Tag-carrying "
              "properties actually present on job starts:")
        for key in sorted(seen_props):
            print(f"    {key}")
    if by_group:
        sample = sorted(by_group)[:3]
        print(f"  sample group ids: {sample}")

    wanted = {g for row in rows for g in row.get("group_ids", [])}
    strategy = "marker"
    if not (wanted & set(by_group)):
        if not windows:
            print("  no marker matched and no repetition windows recorded -- "
                  "re-run the grid with a build that records rep_windows")
        else:
            print(f"  no marker matched; attributing by submission time across "
                  f"{len(windows)} repetition windows instead (INFERRED)")
            by_group = {}
            for log in logs:
                by_group.update(parse_event_log(log, resolver=by_window))
            strategy = "time-window"
            print(f"  attributed {len(by_group)} run group(s) by time window")
            if not by_group:
                # Zero attributed means the jobs in these logs and the windows in
                # these results do not overlap in time at all. Almost always that
                # is stale logs -- a previous session's event log is still in the
                # destination and the current run's has not been delivered yet.
                # Print both spans so the mismatch is visible rather than
                # something to be reverse-engineered from job-group ids.
                import datetime as _dt

                def _stamp(ms):
                    return _dt.datetime.utcfromtimestamp(ms / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S UTC")

                seen = []
                for log in logs:
                    for event in iter_events(log):
                        if event.get("Event") == "SparkListenerJobStart":
                            t = event.get("Submission Time")
                            if t:
                                seen.append(int(t))
                if seen:
                    print(f"  jobs in the logs span   {_stamp(min(seen))} .. "
                          f"{_stamp(max(seen))}")
                print(f"  result windows span     "
                      f"{_stamp(min(w[0] for w in windows))} .. "
                      f"{_stamp(max(w[1] for w in windows))}")
                print("  The two do not overlap. The event logs are from a "
                      "different run than these results -- wait for delivery of "
                      "THIS run's log and reparse again.")
    for row in rows:
        row["attribution"] = strategy

    missing = 0
    for row in rows:
        reps = [by_group[g] for g in row.get("group_ids", []) if g in by_group]
        if not reps:
            missing += 1
            row["metrics_found"] = False
            continue
        row["metrics_found"] = True
        row["n_reps_measured"] = len(reps)
        for field in NUMERIC_FIELDS:
            row[field] = statistics.median(getattr(m, field) for m in reps)
        row["shuffle_read_mb"] = round(row["shuffle_read_bytes"] / 1e6, 3)
        row["shuffle_write_mb"] = round(row["shuffle_write_bytes"] / 1e6, 3)
        row["spill_mb"] = round(
            (row["memory_spilled"] + row["disk_spilled"]) / 1e6, 3)
        row["join_hot_partition_mb"] = round(
            row["join_shuffle_read_max_bytes"] / 1e6, 3)
        row["join_median_partition_mb"] = round(
            row["join_shuffle_read_median_bytes"] / 1e6, 3)

    target = Path(out_path or results_path)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    print(f"rewrote {len(rows)} rows -> {target}"
          + (f" ({missing} without metrics)" if missing else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
