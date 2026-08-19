"""Tests for event-log parsing.

Built from a synthetic event log so the expected aggregates are known exactly.
"""

import json

import pytest

from skewbench.metrics import RunMetrics, parse_event_log


def _task_end(stage, task_id, launch, finish, shuffle_read=0, shuffle_write=0,
              spill_mem=0, spill_disk=0):
    return {
        "Event": "SparkListenerTaskEnd",
        "Stage ID": stage,
        "Task Info": {"Task ID": task_id, "Attempt": 0, "Launch Time": launch,
                      "Finish Time": finish, "Failed": False},
        "Task Metrics": {
            "Executor Run Time": finish - launch,
            "Executor CPU Time": (finish - launch) * 1_000_000,
            "JVM GC Time": 0,
            "Memory Bytes Spilled": spill_mem,
            "Disk Bytes Spilled": spill_disk,
            "Peak Execution Memory": 1024,
            "Shuffle Read Metrics": {"Remote Bytes Read": 0,
                                     "Local Bytes Read": shuffle_read,
                                     "Total Records Read": shuffle_read // 10},
            "Shuffle Write Metrics": {"Shuffle Bytes Written": shuffle_write,
                                      "Shuffle Records Written": shuffle_write // 10},
            "Input Metrics": {"Bytes Read": 0},
        },
    }


@pytest.fixture
def log(tmp_path):
    events = [
        {"Event": "SparkListenerJobStart", "Job ID": 0, "Stage IDs": [0, 1],
         "Submission Time": 1000,
         "Properties": {"spark.jobGroup.id": "run-a::rep0"}},
        _task_end(0, 1, 1000, 1100, shuffle_write=5_000),
        _task_end(0, 2, 1000, 1120, shuffle_write=5_000),
        # Stage 1 is the join stage: heaviest shuffle read, one clear straggler.
        _task_end(1, 3, 1200, 1300, shuffle_read=10_000),
        _task_end(1, 4, 1200, 1300, shuffle_read=10_000),
        _task_end(1, 5, 1200, 1800, shuffle_read=80_000, spill_disk=2_048),
        {"Event": "SparkListenerJobEnd", "Job ID": 0, "Completion Time": 1900},
        {"Event": "SparkListenerJobStart", "Job ID": 1, "Stage IDs": [2],
         "Submission Time": 2000,
         "Properties": {"spark.jobGroup.id": "run-b::rep0"}},
        _task_end(2, 6, 2000, 2100, shuffle_read=1_000),
        {"Event": "SparkListenerJobEnd", "Job ID": 1, "Completion Time": 2150},
    ]
    path = tmp_path / "eventlog"
    path.write_text("\n".join(json.dumps(e) for e in events))
    return path


def test_groups_are_separated(log):
    runs = parse_event_log(log)
    assert set(runs) == {"run-a::rep0", "run-b::rep0"}


def test_wall_clock_from_job_bounds(log):
    assert parse_event_log(log)["run-a::rep0"].wall_clock_ms == 900


def test_join_stage_is_the_heaviest_shuffle_reader(log):
    run = parse_event_log(log)["run-a::rep0"]
    assert run.join_stage_id == 1
    assert run.join_n_tasks == 3


def test_skew_ratio_is_max_over_median(log):
    run = parse_event_log(log)["run-a::rep0"]
    # Join-stage durations are 100, 100 and 600 ms; median 100.
    assert run.join_skew_ratio == pytest.approx(6.0)


def test_deterministic_counters_aggregate(log):
    run = parse_event_log(log)["run-a::rep0"]
    assert run.shuffle_read_bytes == 100_000
    assert run.shuffle_write_bytes == 10_000
    assert run.disk_spilled == 2_048


def test_truncated_final_line_is_tolerated(log, tmp_path):
    broken = tmp_path / "broken"
    broken.write_text(log.read_text() + '\n{"Event": "SparkListenerTa')
    assert parse_event_log(broken)["run-a::rep0"].join_skew_ratio == pytest.approx(6.0)


def test_to_row_is_flat_and_serialisable(log):
    row = parse_event_log(log)["run-a::rep0"].to_row()
    assert "stages" not in row
    assert row["shuffle_read_mb"] == pytest.approx(0.1)
    json.dumps(row)
