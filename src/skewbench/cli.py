"""Command-line interface.

    skewbench gen     --config config/pilot.yaml
    skewbench run     --config config/pilot.yaml --out results/pilot.jsonl
    skewbench analyze --results results/pilot.jsonl --outdir results/analysis
    skewbench advise  --H 1170000 --P 300 --n-fact 3000000 --partitions 16 --cores 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from skewbench.config import ArmSpec, RunSpec, SparkSpec, WorkloadSpec, expand_grid


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def specs_from_config(cfg: Dict[str, Any]) -> List[RunSpec]:
    workloads = [WorkloadSpec(**w) for w in cfg["workloads"]]
    sparks = [SparkSpec(**s) for s in cfg["spark"]]
    arms: List[ArmSpec] = []
    for entry in cfg["arms"]:
        name = entry["name"]
        for k in entry.get("k", [1]):
            arms.append(ArmSpec(name=name, k=int(k)))
    return expand_grid(workloads, sparks, arms,
                       repetitions=int(cfg.get("repetitions", 5)),
                       warmup=int(cfg.get("warmup", 1)))


def cmd_gen(args: argparse.Namespace) -> int:
    from skewbench import datagen

    cfg = load_config(args.config)
    for entry in cfg["workloads"]:
        spec = WorkloadSpec(**entry)
        manifest = datagen.generate(spec, args.data_root, overwrite=args.overwrite)
        share, gini = datagen.summarise(spec)
        print(f"{spec.slug}: hot share {share*100:.2f}%, gini {gini:.3f}, "
              f"{manifest['bytes_on_disk']/1e6:.1f} MB on disk")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from skewbench.runner import run_grid

    cfg = load_config(args.config)
    specs = specs_from_config(cfg)
    if args.limit:
        specs = specs[: args.limit]
    print(f"{len(specs)} cells to execute")
    run_grid(specs, args.data_root, args.out, overwrite_data=args.overwrite)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
    from analyze import main as analyze_main  # type: ignore

    analyze_main(args.results, args.outdir)
    return 0


def cmd_advise(args: argparse.Namespace) -> int:
    from skewbench.costmodel import CostParams, decide, recommend_k

    params = CostParams(a=args.a, b=args.b, c=args.c)
    rec = recommend_k(args.H, args.P, args.n_fact, args.partitions, args.cores, params)
    dec = decide(args.H, args.P, args.n_fact, args.row_bytes, args.partitions,
                 args.two_sided, args.skew_factor, args.skew_threshold)
    print(dec.explain())
    print()
    print(rec.explain())
    if args.json:
        print(json.dumps({"decision": dec.__dict__, "k": rec.__dict__}, default=str))
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skewbench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen", help="materialise the synthetic workloads")
    p_gen.add_argument("--config", required=True)
    p_gen.add_argument("--data-root", default="data")
    p_gen.add_argument("--overwrite", action="store_true")
    p_gen.set_defaults(func=cmd_gen)

    p_run = sub.add_parser("run", help="execute the experiment grid")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default="results/run.jsonl")
    p_run.add_argument("--data-root", default="data")
    p_run.add_argument("--overwrite", action="store_true")
    p_run.add_argument("--limit", type=int, default=0)
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="tables and figures from a results file")
    p_an.add_argument("--results", required=True)
    p_an.add_argument("--outdir", default="results/analysis")
    p_an.set_defaults(func=cmd_analyze)

    p_ad = sub.add_parser("advise", help="apply the decision rule to a workload")
    p_ad.add_argument("--H", type=float, required=True, help="hot-key rows, skewed side")
    p_ad.add_argument("--P", type=float, required=True, help="hot-key rows, probe side")
    p_ad.add_argument("--n-fact", dest="n_fact", type=int, required=True)
    p_ad.add_argument("--partitions", type=int, default=200)
    p_ad.add_argument("--cores", type=int, default=8)
    p_ad.add_argument("--row-bytes", dest="row_bytes", type=float, default=100.0)
    p_ad.add_argument("--two-sided", dest="two_sided", action="store_true")
    p_ad.add_argument("--skew-factor", dest="skew_factor", type=int, default=5)
    p_ad.add_argument("--skew-threshold", dest="skew_threshold", default="256MB")
    p_ad.add_argument("--a", type=float, default=2.0e-6)
    p_ad.add_argument("--b", type=float, default=5.0e-6)
    p_ad.add_argument("--c", type=float, default=0.0)
    p_ad.add_argument("--json", action="store_true")
    p_ad.set_defaults(func=cmd_advise)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
