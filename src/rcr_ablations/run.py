"""Run B0–B3 with the original RCR training, calibration and evaluation pipeline."""

import argparse
from pathlib import Path

import yaml

from rcr.common.config import load_config, override_config, parse_overrides
from rcr_ablations.config import (
    RCR_ROOT,
    ROOT,
    load_effective_config,
    materialize_native_config,
)
from rcr_ablations.factory import MODEL_CLASSES


def _calibrate(args):
    from rcr.proposed.calibration import calibrate

    if args.config:
        cfg = load_config(args.config)
    else:
        base = load_config(RCR_ROOT / "configs/proposed.yaml")
        cfg = {
            "rankings": args.rankings,
            "output_dir": args.output_dir,
            "selection_metric": "checkpoint_score",
            **base["evaluation"]["calibration"],
        }
    cfg = override_config(cfg, parse_overrides(args.set))
    if not cfg.get("rankings") or not cfg.get("output_dir"):
        raise ValueError("calibrate requires --config or --rankings and --output-dir")
    cfg["rankings"] = [
        str((ROOT / Path(p).expanduser()).resolve()) for p in cfg["rankings"]
    ]
    cfg["output_dir"] = str((ROOT / Path(cfg["output_dir"]).expanduser()).resolve())
    return calibrate(cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "validate", "retrieve", "evaluate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", help="Variant overlay or saved effective YAML")
        command.add_argument("--variant", choices=MODEL_CLASSES)
        command.add_argument("--seed", type=int)
        command.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
        command.add_argument("--print-config", action="store_true")
        if name == "train":
            command.add_argument("--resume", nargs="?", const="auto", metavar="LAST_PT")
        else:
            command.add_argument("--checkpoint")
        if name in ("retrieve", "evaluate", "run"):
            command.add_argument("--split", choices=("train", "val", "test"))
    command = commands.add_parser("calibrate")
    command.add_argument("--config")
    command.add_argument("--rankings", nargs="+")
    command.add_argument("--output-dir")
    command.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    try:
        if args.command == "calibrate":
            return _calibrate(args)
        cfg = load_effective_config(
            args.config,
            variant=args.variant,
            seed=args.seed,
            checkpoint=getattr(args, "checkpoint", None),
            overrides=parse_overrides(args.set),
        )
        if args.command == "validate":
            cfg["split"] = "val"
        elif getattr(args, "split", None):
            cfg["split"] = args.split
        if args.command == "run" and cfg["split"] == "train":
            raise ValueError("run supports validation or test, as in the original CLI")
        if args.print_config:
            print(yaml.safe_dump(cfg, sort_keys=False), end="")
            return cfg
        cfg = materialize_native_config(cfg)
        if args.command == "train":
            from rcr.proposed.train import _training_config, train
            from rcr_ablations.provenance import experiment_metadata, write_run_manifest

            cfg = _training_config(cfg)
            cfg["experiment"] = experiment_metadata(cfg)
            resume = args.resume
            if resume is not None and resume != "auto":
                resume = str(Path(resume).expanduser().resolve())
            result = train(cfg, resume=resume)
            write_run_manifest(Path(cfg["output"]["dir"]))
            return result
        if args.command in ("validate", "run"):
            from rcr.proposed.runner import run_experiment

            return run_experiment(cfg, RCR_ROOT / "tools/run.py")
        if args.command == "retrieve":
            from rcr.proposed.retrieve import retrieve

            return retrieve(cfg)
        from rcr.evaluation.runner import evaluate_run

        return evaluate_run(cfg)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
