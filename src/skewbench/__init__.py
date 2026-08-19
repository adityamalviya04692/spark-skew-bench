"""skewbench: a reproducible benchmark for Spark skewed-join mitigation.

Modules
-------
config      Experiment specifications (workload, Spark, arm, run).
datagen     Zipfian-skewed synthetic data generation with an engine-telemetry schema.
arms        The six join strategies under evaluation.
metrics     Spark event-log parsing into hardware-independent metrics.
runner      Experiment driver: grid execution, warmup, repetitions.
costmodel   Analytical salt-cardinality cost model and the AQE-sufficiency rule.
cli         Command-line interface.
"""

__version__ = "1.0.0"

from skewbench.config import ArmSpec, RunSpec, SparkSpec, WorkloadSpec  # noqa: F401

__all__ = ["ArmSpec", "RunSpec", "SparkSpec", "WorkloadSpec", "__version__"]
