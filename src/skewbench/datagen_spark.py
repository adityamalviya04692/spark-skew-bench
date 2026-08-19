"""Distributed workload generation, for when a cluster is available.

The pandas generator in :mod:`skewbench.datagen` runs on one core. That is fine
at the two-million-row scale used for the single-node study, where it costs
about ninety seconds. It is not fine at cluster scale: sixty million rows of
twenty-one sensor columns is roughly fifty times the work, so a single-threaded
generator spends over an hour per workload before any measurement begins, and
writes it all through one driver to network-backed storage.

This module produces the identical schema and the identical statistical
properties using Spark itself, so generation is spread across the cluster and
finishes in minutes.

Equivalence with the pandas generator is what makes the two studies comparable,
so it is enforced rather than assumed:

* The Zipf key law is the same truncated law over the same ranks. We compute the
  CDF once on the driver with the same ``zipf_weights`` function the pandas path
  uses, invert it with the same ``searchsorted`` call, and ship the result as a
  small broadcast lookup table joined on a bucket key. No Python UDF is
  involved: inversion happens once on the driver over a fixed grid, not per row.
  That keeps generation in the JVM, avoids an Arrow dependency, and is
  substantially faster than a per-row UDF.
* The probe side uses the same amplified-uniform construction from
  :func:`skewbench.datagen.probe_weights`.
* Column names, types and value ranges match ``datagen`` exactly.
* The manifest is written in the same shape, so downstream code cannot tell
  which generator produced a workload.

What is deliberately *not* preserved is row-for-row identity with the pandas
generator: Spark's RNG is per-partition and seeded differently. The key
distribution, payload distribution and hot-key counts agree; the individual rows
do not. Nothing in the study depends on row identity -- only on the
distributions and on the realised hot-key counts, which the manifest records
from the data rather than from theory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame, SparkSession, functions as F
from skewbench.config import WorkloadSpec
from skewbench.datagen import (FINDING_CODES, FLIGHT_PHASES, probe_weights,
                               theoretical_skew_ratio, zipf_weights)


# Buckets used to discretise the unit interval for key sampling. The rarest key
# under the distributions we generate carries probability of order 1e-5, so
# 2**18 buckets give it several buckets and the discretisation error is far
# below sampling noise. The hot key, which is the only one any measurement
# depends on, spans tens of thousands of buckets.
KEY_BUCKETS = 1 << 18


def _rank_lookup(spark: SparkSession, cdf: np.ndarray) -> DataFrame:
    """Broadcastable ``(bucket, rank)`` table inverting the CDF.

    The inversion is done once, on the driver, with the same ``searchsorted``
    the single-node generator uses -- so the key law is identical by
    construction rather than by reimplementation. Sampling a row is then an
    equi-join against a small table, which the planner broadcasts.
    """
    edges = (np.arange(KEY_BUCKETS, dtype=np.float64) + 0.5) / KEY_BUCKETS
    ranks = np.searchsorted(cdf, edges, side="right").astype(np.int32)
    ranks = np.minimum(ranks, len(cdf) - 1)
    pdf = pd.DataFrame({"bucket": np.arange(KEY_BUCKETS, dtype=np.int32),
                        "rank": ranks})
    return F.broadcast(spark.createDataFrame(pdf))


def _with_sampled_key(frame: DataFrame, lookup: DataFrame, seed: int) -> DataFrame:
    """Attach ``engine_id`` sampled from the law encoded in ``lookup``."""
    bucketed = frame.withColumn(
        "bucket", (F.rand(seed) * F.lit(KEY_BUCKETS)).cast("int"))
    return (bucketed.join(lookup, on="bucket", how="inner")
            .withColumn("engine_id", _engine_id_col(F.col("rank")))
            .drop("bucket", "rank"))


def _engine_id_col(rank_col):
    """``ENG-000123`` from a rank, matching the pandas generator's format."""
    return F.concat(F.lit("ENG-"), F.lpad(rank_col.cast("string"), 6, "0"))


def _fact(spark: SparkSession, spec: WorkloadSpec, cdf: np.ndarray) -> DataFrame:
    lookup = _rank_lookup(spark, cdf)
    # Partition count is chosen so each task writes a few hundred thousand rows
    # -- large enough that Parquet blocks are efficient, small enough to spread
    # across the cluster.
    partitions = max(8, int(spec.n_fact // 400_000))
    base = spark.range(0, spec.n_fact, numPartitions=partitions) \
                .withColumnRenamed("id", "event_id")
    frame = (
        _with_sampled_key(base, lookup, spec.seed)
        .withColumn("cycle", (F.rand(spec.seed + 1) * 399 + 1).cast("int"))
        .withColumn("ts", (F.lit(1_750_000_000)
                           + (F.rand(spec.seed + 2) * 31_536_000)).cast("long"))
        .withColumn("flight_phase",
                    F.element_at(F.array(*[F.lit(p) for p in FLIGHT_PHASES]),
                                 (F.rand(spec.seed + 3) * len(FLIGHT_PHASES))
                                 .cast("int") + 1))
        .withColumn("op_setting_1", F.randn(spec.seed + 4).cast("float"))
        .withColumn("op_setting_2", F.randn(spec.seed + 5).cast("float"))
        .withColumn("op_setting_3", F.lit(100.0).cast("float"))
    )
    for s in range(1, spec.n_sensors + 1):
        centre = 100.0 * (1 + (s % 7))
        frame = frame.withColumn(
            f"sensor_{s:02d}",
            (F.lit(centre) + F.randn(spec.seed + 100 + s) * F.lit(centre * 0.02))
            .cast("float"))
    return frame


def _dim(spark: SparkSession, spec: WorkloadSpec, cdf: np.ndarray) -> DataFrame:
    lookup = _rank_lookup(spark, cdf)
    partitions = max(4, int(spec.n_dim // 400_000))
    base = spark.range(0, spec.n_dim, numPartitions=partitions) \
                .withColumnRenamed("id", "wo_id")
    return (
        _with_sampled_key(base, lookup, spec.seed + 11)
        .withColumn("wo_ts", (F.lit(1_750_000_000)
                              + (F.rand(spec.seed + 12) * 31_536_000)).cast("long"))
        .withColumn("shop_visit", (F.rand(spec.seed + 13) * 11 + 1).cast("int"))
        .withColumn("part_no", F.concat(
            F.lit("P-"), F.lpad((F.rand(spec.seed + 14) * 50_000)
                                .cast("int").cast("string"), 5, "0")))
        .withColumn("finding_code",
                    F.element_at(F.array(*[F.lit(c) for c in FINDING_CODES]),
                                 (F.rand(spec.seed + 15) * len(FINDING_CODES))
                                 .cast("int") + 1))
        .withColumn("labour_hours",
                    F.round(F.rand(spec.seed + 16) * 20 + 0.5, 2).cast("float"))
    )


def generate(spec: WorkloadSpec, root: str, spark: SparkSession | None = None,
             overwrite: bool = False) -> Dict:
    """Materialise a workload with Spark and return a manifest.

    ``root`` is a distributed path (a Unity Catalog volume, DBFS, S3, HDFS).
    """
    spark = spark or SparkSession.builder.getOrCreate()
    base = f"{root.rstrip('/')}/{spec.slug}"
    fact_path, dim_path = f"{base}/fact_telemetry", f"{base}/dim_maintenance"
    manifest_path = f"{base}/manifest.json"

    fact_cdf = np.cumsum(zipf_weights(spec.n_engines, spec.theta))
    dim_cdf = np.cumsum(probe_weights(spec))

    # Write, unless the data is already there and we were not asked to redo it.
    # The earlier version wrapped this in a bare except that re-raised only when
    # NOT overwriting -- which silently swallowed every genuine failure on an
    # overwrite, leaving an empty directory and a confusing schema error later.
    mode = "overwrite" if overwrite else "errorifexists"
    already_present = False
    if not overwrite:
        try:
            spark.read.parquet(fact_path).limit(1).count()
            spark.read.parquet(dim_path).limit(1).count()
            already_present = True
        except Exception:
            already_present = False

    if not already_present:
        _fact(spark, spec, fact_cdf).write.mode(mode).parquet(fact_path)
        _dim(spark, spec, dim_cdf).write.mode(mode).parquet(dim_path)

    fact = spark.read.parquet(fact_path)
    dim = spark.read.parquet(dim_path)

    # Hot keys are read from the realised data, not assumed from the law. The
    # cost model consumes these as H and P, so an assumption here would
    # propagate silently into every downstream number.
    hot = [r["engine_id"] for r in
           fact.groupBy("engine_id").count()
               .orderBy(F.desc("count")).limit(spec.hot_keys).collect()]
    hot_fact = fact.filter(F.col("engine_id").isin(hot)).count()
    hot_dim = dim.filter(F.col("engine_id").isin(hot)).count()

    manifest = {
        "spec": spec.__dict__,
        "slug": spec.slug,
        "root": base,
        "fact_path": fact_path,
        "dim_path": dim_path,
        "hot_keys": hot,
        "generator": "spark",
        "expected_hot_fact_rows": float(hot_fact),
        "expected_hot_dim_rows": float(hot_dim),
        "expected_hot_fact_share": hot_fact / max(1, spec.n_fact),
        "expected_hot_dim_share": hot_dim / max(1, spec.n_dim),
        "hot_dim_multiplier": spec.hot_dim_multiplier,
        "theoretical_skew_ratio": theoretical_skew_ratio(spec),
        "expected_hot_key_output_rows": float(hot_fact) * float(hot_dim)
        / max(1, spec.hot_keys),
    }
    spark.createDataFrame([(json.dumps(manifest, default=str),)], ["json"]) \
        .coalesce(1).write.mode("overwrite").text(manifest_path)
    return manifest
