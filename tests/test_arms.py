"""Tests for the join strategies.

Correctness first: a mitigation that changes the answer is not a mitigation.
Every arm must produce results identical to the unmitigated baseline.
"""

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from skewbench.arms import JOIN_KEY, build  # noqa: E402
from skewbench.config import ArmSpec  # noqa: E402

JVM_OPENS = ("--add-opens=java.base/java.lang=ALL-UNNAMED "
             "--add-opens=java.base/java.nio=ALL-UNNAMED "
             "--add-opens=java.base/java.util=ALL-UNNAMED "
             "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED")


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[2]").appName("skewbench-tests")
               .config("spark.driver.extraJavaOptions", JVM_OPENS)
               .config("spark.sql.shuffle.partitions", "4")
               .config("spark.ui.enabled", "false")
               .config("spark.ui.showConsoleProgress", "false")
               .getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def tables(spark):
    fact = spark.createDataFrame(
        [("ENG-000000", i, float(i), "cruise") for i in range(200)]
        + [("ENG-000001", i, float(i), "climb") for i in range(20)],
        [JOIN_KEY, "cycle", "sensor_01", "flight_phase"],
    )
    dim = spark.createDataFrame(
        [("ENG-000000", 5.0, "FC-BLADE"), ("ENG-000000", 2.0, "FC-SEAL"),
         ("ENG-000001", 1.0, "FC-BLADE")],
        [JOIN_KEY, "labour_hours", "finding_code"],
    )
    return fact, dim


HOT = ["ENG-000000"]


def _rows(df):
    return sorted(tuple(r) for r in df.collect())


@pytest.mark.parametrize("arm,k", [
    ("aqe", 1), ("broadcast", 1),
    ("salt_selective", 2), ("salt_selective", 8), ("salt_selective", 32),
    ("salt_uniform", 2), ("salt_uniform", 8),
    ("aqe_salt", 4),
])
def test_every_arm_matches_the_baseline_result(spark, tables, arm, k):
    fact, dim = tables
    expected = _rows(build(fact, dim, ArmSpec("baseline"), HOT))
    actual = _rows(build(fact, dim, ArmSpec(arm, k), HOT))
    assert actual == expected, f"arm {arm}(k={k}) changed the query result"


def test_selective_salting_with_k_one_degenerates_to_plain_join(spark, tables):
    fact, dim = tables
    a = _rows(build(fact, dim, ArmSpec("salt_selective", 1), HOT))
    b = _rows(build(fact, dim, ArmSpec("baseline"), HOT))
    assert a == b


def test_salt_column_does_not_leak_into_output(spark, tables):
    fact, dim = tables
    columns = build(fact, dim, ArmSpec("salt_selective", 4), HOT).columns
    assert not any(c.startswith("_salt") for c in columns)


def test_arm_spec_rejects_salt_on_non_salting_arms():
    with pytest.raises(ValueError):
        ArmSpec("baseline", 4)
    with pytest.raises(ValueError):
        ArmSpec("not_an_arm")
    with pytest.raises(ValueError):
        ArmSpec("salt_selective", 0)
