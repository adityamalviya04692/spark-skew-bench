"""Spark event-log parsing into hardware-independent metrics.

Why the event log rather than a ``SparkListener``: the log is a plain JSON-lines
file written by Spark itself, so parsing it needs no JVM callback plumbing from
Python, survives the driver exiting, and can be re-analysed months later without
re-running anything. That last property is what makes a result auditable.

The metrics deliberately separate two families:

*Deterministic* -- shuffle bytes read and written, records shuffled, bytes
spilled. These are properties of the physical plan and the data, not of the
machine. They are identical on a laptop and on a hundred-node cluster given the
same plan and partitioning, which is what allows a single-node study to say
something honest about cluster behaviour.

*Machine-dependent* -- wall-clock and executor run time. Reported, but never
load-bearing for a claim, and always with dispersion attached.

Between them sits the metric that actually measures skew: the ratio of maximum
to median task duration within the join stage. It is a within-run ratio, so it
cancels most machine-specific scaling while still capturing the straggler that
skew produces.
"""

from __future__ import annotations

import gzip
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TASK_END = "SparkListenerTaskEnd"
JOB_START = "SparkListenerJobStart"
JOB_END = "SparkListenerJobEnd"
STAGE_COMPLETED = "SparkListenerStageCompleted"

# Below this, a run is treated as having no meaningful shuffle (broadcast join),
# and the join stage is identified by task time instead of by shuffle bytes.
SHUFFLE_READ_FLOOR_BYTES = 1_000_000


@dataclass
class TaskRecord:
    """One completed task, reduced to the fields the study needs."""

    task_id: int
    stage_id: int
    attempt: int
    duration_ms: int
    executor_run_ms: int
    executor_cpu_ms: int
    gc_ms: int
    shuffle_read_bytes: int
    shuffle_read_records: int
    shuffle_write_bytes: int
    shuffle_write_records: int
    memory_spilled: int
    disk_spilled: int
    peak_execution_memory: int
    input_bytes: int
    failed: bool = False


@dataclass
class StageMetrics:
    """Aggregate of all tasks in one stage, plus the skew statistics."""

    stage_id: int
    n_tasks: int
    task_time_max_ms: int
    task_time_median_ms: float
    task_time_mean_ms: float
    task_time_p95_ms: float
    task_time_stdev_ms: float
    skew_ratio: float          # max / median -- NOT comparable across task counts
    straggler_index: float     # max / mean -- task-count invariant
    task_time_cv: float        # stdev / mean
    shuffle_read_bytes: int
    shuffle_read_records: int
    shuffle_write_bytes: int
    shuffle_write_records: int
    memory_spilled: int
    disk_spilled: int
    peak_execution_memory: int
    total_task_time_ms: int
    # Per-task shuffle-read distribution. These are the *measured* partition
    # sizes, and they matter: AQE's skew-join trigger compares the largest
    # shuffle partition against an absolute threshold, a multiple of the median,
    # and the advisory split size. Estimating those sizes analytically from row
    # counts understates the maximum (the hot partition also absorbs cold keys)
    # and overstates the median (it ignores variance among the cold partitions).
    # Reading them from the event log removes the estimate entirely.
    shuffle_read_max_bytes: int = 0
    shuffle_read_median_bytes: float = 0.0


@dataclass
class RunMetrics:
    """Everything measured for one execution of one arm."""

    job_group: str
    wall_clock_ms: int = 0
    n_jobs: int = 0
    n_stages: int = 0
    n_tasks: int = 0
    failed_tasks: int = 0
    shuffle_read_bytes: int = 0
    shuffle_read_records: int = 0
    shuffle_write_bytes: int = 0
    shuffle_write_records: int = 0
    memory_spilled: int = 0
    disk_spilled: int = 0
    total_task_time_ms: int = 0
    peak_execution_memory: int = 0
    join_stage_id: Optional[int] = None
    join_skew_ratio: float = 0.0
    join_task_time_cv: float = 0.0
    join_task_time_max_ms: int = 0
    join_task_time_median_ms: float = 0.0
    join_task_time_mean_ms: float = 0.0
    join_total_task_time_ms: int = 0
    join_straggler_index: float = 0.0
    join_n_tasks: int = 0
    join_shuffle_read_max_bytes: int = 0
    join_shuffle_read_median_bytes: float = 0.0
    stages: List[StageMetrics] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        """Flat dict for a results file; per-stage detail is dropped."""
        row = {k: v for k, v in asdict(self).items() if k != "stages"}
        row["shuffle_read_mb"] = round(self.shuffle_read_bytes / 1e6, 3)
        row["shuffle_write_mb"] = round(self.shuffle_write_bytes / 1e6, 3)
        row["spill_mb"] = round((self.memory_spilled + self.disk_spilled) / 1e6, 3)
        return row


def _num(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parse_task_end(event: Dict[str, Any]) -> TaskRecord:
    info = event.get("Task Info", {})
    metrics = event.get("Task Metrics") or {}
    read = metrics.get("Shuffle Read Metrics", {}) or {}
    write = metrics.get("Shuffle Write Metrics", {}) or {}
    inp = metrics.get("Input Metrics", {}) or {}
    launch, finish = _num(info.get("Launch Time")), _num(info.get("Finish Time"))
    return TaskRecord(
        task_id=_num(info.get("Task ID")),
        stage_id=_num(event.get("Stage ID")),
        attempt=_num(info.get("Attempt")),
        duration_ms=max(0, finish - launch),
        executor_run_ms=_num(metrics.get("Executor Run Time")),
        executor_cpu_ms=_num(metrics.get("Executor CPU Time")) // 1_000_000,
        gc_ms=_num(metrics.get("JVM GC Time")),
        shuffle_read_bytes=_num(read.get("Remote Bytes Read")) + _num(read.get("Local Bytes Read")),
        shuffle_read_records=_num(read.get("Total Records Read")),
        shuffle_write_bytes=_num(write.get("Shuffle Bytes Written")),
        shuffle_write_records=_num(write.get("Shuffle Records Written")),
        memory_spilled=_num(metrics.get("Memory Bytes Spilled")),
        disk_spilled=_num(metrics.get("Disk Bytes Spilled")),
        peak_execution_memory=_num(metrics.get("Peak Execution Memory")),
        input_bytes=_num(inp.get("Bytes Read")),
        failed=bool(info.get("Failed", False)),
    )


def _stage_metrics(stage_id: int, tasks: List[TaskRecord]) -> StageMetrics:
    durations = sorted(t.duration_ms for t in tasks) or [0]
    median = statistics.median(durations)
    mean = statistics.fmean(durations)
    stdev = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    p95_index = max(0, min(len(durations) - 1, int(round(0.95 * (len(durations) - 1)))))
    return StageMetrics(
        stage_id=stage_id,
        n_tasks=len(tasks),
        task_time_max_ms=max(durations),
        task_time_median_ms=float(median),
        task_time_mean_ms=float(mean),
        task_time_p95_ms=float(durations[p95_index]),
        task_time_stdev_ms=float(stdev),
        # Guard the degenerate case where the median task is sub-millisecond:
        # a ratio against zero is meaningless, so fall back to the mean.
        skew_ratio=float(max(durations) / median) if median > 0
        else (float(max(durations) / mean) if mean > 0 else 1.0),
        # Straggler index: max task duration over the MEAN task duration, i.e.
        # over (total task time / n_tasks). Unlike max/median this does not move
        # when an optimisation changes the task count: coalescing 16 partitions
        # into 3 leaves total task time roughly unchanged, so the denominator is
        # stable and the numerator still reports the critical path. max/median
        # is retained for continuity but must never be compared across arms with
        # different task counts.
        straggler_index=float(max(durations) / mean) if mean > 0 else 1.0,
        task_time_cv=float(stdev / mean) if mean > 0 else 0.0,
        shuffle_read_bytes=sum(t.shuffle_read_bytes for t in tasks),
        shuffle_read_records=sum(t.shuffle_read_records for t in tasks),
        shuffle_write_bytes=sum(t.shuffle_write_bytes for t in tasks),
        shuffle_write_records=sum(t.shuffle_write_records for t in tasks),
        memory_spilled=sum(t.memory_spilled for t in tasks),
        disk_spilled=sum(t.disk_spilled for t in tasks),
        peak_execution_memory=max((t.peak_execution_memory for t in tasks), default=0),
        total_task_time_ms=sum(t.duration_ms for t in tasks),
        shuffle_read_max_bytes=max((t.shuffle_read_bytes for t in tasks), default=0),
        shuffle_read_median_bytes=float(
            statistics.median([t.shuffle_read_bytes for t in tasks] or [0])),
    )


def iter_events(log_path: str | Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed Spark events, tolerating a truncated tail and foreign files.

    Two things this must survive. A rolled event log is gzipped, so the reader
    sniffs the magic bytes rather than trusting the extension. And Databricks
    cluster log delivery drops stdout, stderr, log4j output and metrics files
    into the same tree as the event log -- a line of one of those can be valid
    JSON that is not an object (a bare integer, say), which would otherwise
    surface as ``'int' object has no attribute 'get'`` several frames away from
    the file that caused it. Only JSON objects are events; everything else is
    silently not one.
    """
    path = Path(log_path)
    with open(path, "rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if gzipped else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially flushed tail
            if isinstance(event, dict):
                yield event


SKEWBENCH_TAG_PREFIX = "skewbench:"


def _group_of(props: Dict[str, Any]) -> str:
    """Recover the run group a Spark job belongs to, from either mechanism.

    A classic session writes ``spark.jobGroup.id``. A Spark Connect session --
    which is what Databricks Standard access mode and Serverless give you,
    because they withhold ``sparkContext`` -- writes ``spark.job.tags`` instead,
    a comma-separated list that also carries tags the platform sets for its own
    purposes. Only our own prefixed tag identifies a measurement, so the others
    are discarded rather than guessed at.

    Precedence matters here; see the comment below.
    """
    # OUR tag is checked FIRST, and this ordering is the whole point.
    #
    # Databricks sets ``spark.jobGroup.id`` itself, to an internal value of the
    # form <epoch>_<n>_<hex>. An earlier revision read that property first, so
    # on Databricks every job resolved to a platform id and our own tag was
    # never consulted -- the parse "succeeded", reported job groups, and
    # matched none of them to a results row. Reading our own marker first is
    # correct on both platforms: on a classic session we own
    # ``spark.jobGroup.id`` and set no tags, so the fallback below is what runs.
    raw = props.get("spark.job.tags") or ""
    for tag in raw.split(","):
        tag = tag.strip()
        if tag.startswith(SKEWBENCH_TAG_PREFIX):
            return tag[len(SKEWBENCH_TAG_PREFIX):]
    group = props.get("spark.jobGroup.id")
    if group:
        return group
    return "ungrouped"



def parse_event_log(log_path: str | Path,
                    resolver: Optional[Any] = None) -> Dict[str, RunMetrics]:
    """Group an event log by run and reduce each group to metrics.

    By default a job is attributed to a run by the marker the runner wrote --
    ``spark.jobGroup.id`` on a classic session, our own ``skewbench:`` job tag
    on a Spark Connect one. That mapping is exact.

    ``resolver`` overrides it with ``f(props, submission_time_ms) -> group``.
    That exists because Databricks propagates neither marker into the delivered
    event log: measured on DBR 16.4, zero of 56 job starts carried a
    ``skewbench:`` tag, and the only tag-shaped properties present were
    ``spark.databricks.clusterUsageTags.*`` cluster metadata. Where no marker
    survives, the caller supplies attribution by submission time instead --
    inference rather than identity, and the analysis must say so.
    """
    job_group_of_job: Dict[int, str] = {}
    stages_of_job: Dict[int, List[int]] = {}
    job_bounds: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    tasks_by_stage: Dict[int, List[TaskRecord]] = defaultdict(list)

    for event in iter_events(log_path):
        kind = event.get("Event")
        if kind == JOB_START:
            job_id = _num(event.get("Job ID"))
            props = event.get("Properties") or {}
            submitted = _num(event.get("Submission Time"))
            group = resolver(props, submitted) if resolver else _group_of(props)
            if group is None:
                continue  # resolver disowned this job: platform overhead, not ours
            job_group_of_job[job_id] = group
            stages_of_job[job_id] = [_num(s) for s in event.get("Stage IDs", [])]
            job_bounds[job_id][0] = submitted
        elif kind == JOB_END:
            job_id = _num(event.get("Job ID"))
            job_bounds[job_id][1] = _num(event.get("Completion Time"))
        elif kind == TASK_END:
            record = _parse_task_end(event)
            tasks_by_stage[record.stage_id].append(record)

    grouped: Dict[str, RunMetrics] = {}
    for job_id, group in job_group_of_job.items():
        run = grouped.setdefault(group, RunMetrics(job_group=group))
        run.n_jobs += 1
        start, end = job_bounds[job_id]
        if end > start > 0:
            run.wall_clock_ms += end - start
        for stage_id in stages_of_job.get(job_id, []):
            tasks = tasks_by_stage.get(stage_id)
            if not tasks:
                continue
            run.stages.append(_stage_metrics(stage_id, tasks))

    for run in grouped.values():
        _finalise(run)
    return grouped


def _finalise(run: RunMetrics) -> None:
    """Aggregate stage metrics and identify the join stage."""
    # Deduplicate: a stage can appear under more than one job when reused.
    seen: Dict[int, StageMetrics] = {s.stage_id: s for s in run.stages}
    stages = sorted(seen.values(), key=lambda s: s.stage_id)
    run.stages = stages
    run.n_stages = len(stages)
    run.n_tasks = sum(s.n_tasks for s in stages)
    run.shuffle_read_bytes = sum(s.shuffle_read_bytes for s in stages)
    run.shuffle_read_records = sum(s.shuffle_read_records for s in stages)
    run.shuffle_write_bytes = sum(s.shuffle_write_bytes for s in stages)
    run.shuffle_write_records = sum(s.shuffle_write_records for s in stages)
    run.memory_spilled = sum(s.memory_spilled for s in stages)
    run.disk_spilled = sum(s.disk_spilled for s in stages)
    run.total_task_time_ms = sum(s.total_task_time_ms for s in stages)
    run.peak_execution_memory = max((s.peak_execution_memory for s in stages), default=0)

    # Identifying the join stage.
    #
    # The obvious heuristic -- the stage that reads the most shuffle bytes -- is
    # correct for every shuffle-based join, but silently wrong for a broadcast
    # hash join, which reads no shuffle at all. There the heuristic selects the
    # tiny terminal aggregation and reports its task statistics as the join's.
    # We therefore fall back to the most expensive stage by total task time
    # whenever no stage reads a meaningful amount of shuffle.
    if stages:
        max_read = max(s.shuffle_read_bytes for s in stages)
        if max_read < SHUFFLE_READ_FLOOR_BYTES:
            join = max(stages, key=lambda s: (s.total_task_time_ms, s.stage_id))
        else:
            join = max(stages, key=lambda s: (s.shuffle_read_bytes, s.total_task_time_ms))
        run.join_stage_id = join.stage_id
        run.join_skew_ratio = join.skew_ratio
        run.join_task_time_cv = join.task_time_cv
        run.join_task_time_max_ms = join.task_time_max_ms
        run.join_task_time_median_ms = join.task_time_median_ms
        run.join_task_time_mean_ms = join.task_time_mean_ms
        run.join_total_task_time_ms = join.total_task_time_ms
        run.join_straggler_index = join.straggler_index
        run.join_n_tasks = join.n_tasks
        run.join_shuffle_read_max_bytes = join.shuffle_read_max_bytes
        run.join_shuffle_read_median_bytes = join.shuffle_read_median_bytes
