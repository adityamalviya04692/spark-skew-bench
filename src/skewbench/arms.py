"""The six join strategies under evaluation.

Each arm is a pure function of ``(fact, dim, spec)`` returning a DataFrame whose
materialisation forces the join. The action is a grouped aggregation rather than
``count()``: counting invites the optimizer to prune the payload columns and, on
some plans, to skip the probe entirely, which would measure the optimizer's
column pruning rather than the join strategy under test.

Salting, as practitioners write it
----------------------------------
Salting spreads one hot key across ``k`` synthetic sub-keys on the skewed side,
and replicates the matching rows ``k`` times on the other side so that every
sub-key still finds its partners. The replication is the cost, and it is the
half of the trade-off that the folklore advice ("just pick a reasonable k")
leaves unpriced.

Two variants are implemented because they have materially different cost curves:

``salt_uniform``
    Every row on the probe side is replicated ``k`` times. Simple, and the
    version most commonly shown in tutorials. Replication cost scales with the
    whole probe table.

``salt_selective``
    Only rows whose key is hot are replicated; everything else carries salt 0
    and is untouched. Replication cost scales with the hot-key rows alone, which
    is typically two to three orders of magnitude smaller.
"""

from __future__ import annotations

from typing import List, Sequence

from pyspark.sql import DataFrame, functions as F
from pyspark.sql import SparkSession

from skewbench.config import ArmSpec, SparkSpec

JOIN_KEY = "engine_id"
SALT_COL = "_salt"


def apply_spark_conf(spark: SparkSession, spec: SparkSpec) -> None:
    """Push a SparkSpec into the live session.

    All of these are runtime-settable, which is what allows one session to serve
    the whole grid. Anything that is not runtime-settable (event log location,
    master, driver memory) belongs to session construction instead.
    """
    conf = spark.conf
    conf.set("spark.sql.shuffle.partitions", str(spec.shuffle_partitions))
    conf.set("spark.sql.adaptive.enabled", str(spec.aqe_enabled).lower())
    conf.set("spark.sql.adaptive.skewJoin.enabled", str(spec.aqe_skew_join_enabled).lower())
    conf.set("spark.sql.adaptive.coalescePartitions.enabled",
             str(spec.aqe_coalesce_enabled).lower())
    conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor",
             str(spec.skewed_partition_factor))
    conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
             spec.skewed_partition_threshold)
    conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes",
             spec.advisory_partition_size)
    conf.set("spark.sql.autoBroadcastJoinThreshold", str(spec.autobroadcast_threshold))


def _aggregate(joined: DataFrame) -> DataFrame:
    """Terminal aggregation.

    Touches columns from both sides so neither can be pruned away, and returns a
    small result so that the measurement is not dominated by driver collection.
    """
    return (
        joined.groupBy("finding_code", "flight_phase")
        .agg(
            F.count(F.lit(1)).alias("n_matches"),
            F.sum("labour_hours").alias("total_labour_hours"),
            F.avg("sensor_01").alias("avg_sensor_01"),
            F.max("cycle").alias("max_cycle"),
        )
    )


def _plain_join(fact: DataFrame, dim: DataFrame) -> DataFrame:
    return fact.join(dim, on=JOIN_KEY, how="inner")


def _broadcast_join(fact: DataFrame, dim: DataFrame) -> DataFrame:
    return fact.join(F.broadcast(dim), on=JOIN_KEY, how="inner")


def _salt_uniform(fact: DataFrame, dim: DataFrame, k: int, seed: int = 7) -> DataFrame:
    """Replicate the entire probe side ``k`` times."""
    if k <= 1:
        return _plain_join(fact, dim)
    fact_salted = fact.withColumn(
        SALT_COL, (F.rand(seed) * F.lit(k)).cast("int")
    )
    dim_replicated = dim.withColumn(
        SALT_COL, F.explode(F.array(*[F.lit(i) for i in range(k)]))
    )
    return fact_salted.join(dim_replicated, on=[JOIN_KEY, SALT_COL], how="inner")


def _salt_selective(fact: DataFrame, dim: DataFrame, k: int,
                    hot_keys: Sequence[str], seed: int = 7) -> DataFrame:
    """Replicate only the hot-key rows on the probe side.

    Cold keys are assigned salt 0 on both sides, so they join exactly as before
    and pay no replication. This is the variant whose replication cost is
    proportional to ``P_hot * (k - 1)`` rather than to the whole probe table --
    the quantity that appears in the cost model of Section IV.
    """
    if k <= 1 or not hot_keys:
        return _plain_join(fact, dim)

    hot = F.array(*[F.lit(key) for key in hot_keys])
    is_hot_fact = F.array_contains(hot, F.col(JOIN_KEY))
    fact_salted = fact.withColumn(
        SALT_COL,
        F.when(is_hot_fact, (F.rand(seed) * F.lit(k)).cast("int")).otherwise(F.lit(0)),
    )

    is_hot_dim = F.array_contains(hot, F.col(JOIN_KEY))
    dim_replicated = dim.withColumn(
        SALT_COL,
        F.explode(
            F.when(is_hot_dim, F.array(*[F.lit(i) for i in range(k)]))
            .otherwise(F.array(F.lit(0)))
        ),
    )
    return fact_salted.join(dim_replicated, on=[JOIN_KEY, SALT_COL], how="inner")


def build(fact: DataFrame, dim: DataFrame, arm: ArmSpec,
          hot_keys: Sequence[str]) -> DataFrame:
    """Return the aggregated query plan for one arm."""
    if arm.name in ("baseline", "aqe"):
        joined = _plain_join(fact, dim)
    elif arm.name == "broadcast":
        joined = _broadcast_join(fact, dim)
    elif arm.name == "salt_uniform":
        joined = _salt_uniform(fact, dim, arm.k)
    elif arm.name in ("salt_selective", "aqe_salt"):
        joined = _salt_selective(fact, dim, arm.k, hot_keys)
    else:  # pragma: no cover - guarded by ArmSpec validation
        raise ValueError(f"unhandled arm {arm.name!r}")
    return _aggregate(joined)


def explain(fact: DataFrame, dim: DataFrame, arm: ArmSpec,
            hot_keys: Sequence[str], materialise: bool = True) -> str:
    """Physical plan for the arm, captured into results for auditability.

    A reviewer asking "was this really a sort-merge join?", or more pointedly
    "did AQE's skew-join rule actually fire?", should not have to take the
    author's word for it.

    Materialisation is not optional in practice. Under AQE the plan is rewritten
    between stages, so a plan captured before execution reports
    ``isFinalPlan=false`` and shows the *pre*-adaptive shape -- precisely hiding
    the rewrite one wants to audit. We therefore execute into the ``noop`` sink
    first and read the plan afterwards.
    """
    plan = build(fact, dim, arm, hot_keys)
    if materialise:
        # The action must run on *this* DataFrame. Writing to the noop sink
        # builds a separate QueryExecution, leaving this one unexecuted and its
        # plan still reporting isFinalPlan=false. collect() is safe here: the
        # terminal aggregation returns one row per (finding_code, flight_phase)
        # pair, a few dozen rows.
        plan.collect()
    return plan._jdf.queryExecution().executedPlan().toString()


def aqe_evidence(plan_text: str) -> dict:
    """Extract, from a final plan, what AQE actually did.

    Distinguishes the two AQE behaviours that are easily conflated. Partition
    *coalescing* merges small post-shuffle partitions and lowers fixed overhead;
    *skew-join* splitting divides an oversized partition and is the only one
    that addresses a straggler. A speedup from the former says nothing about
    skew handling, and reporting them separately is what keeps the claim honest.
    """
    return {
        "is_final_plan": "isFinalPlan=true" in plan_text,
        "aqe_shuffle_read": "AQEShuffleRead" in plan_text,
        "coalesced": "coalesced" in plan_text,
        "skew_join_applied": "skewed" in plan_text.lower(),
        "broadcast_join": "BroadcastHashJoin" in plan_text,
        "sort_merge_join": "SortMergeJoin" in plan_text,
    }
