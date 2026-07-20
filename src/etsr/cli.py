from __future__ import annotations

import argparse
from pathlib import Path

from etsr.config import load_config
from etsr.data.matched_dvsgc import prepare_matched_dvsgc
from etsr.evaluation.mechanistic import (
    parse_seed_checkpoints,
    run_mechanistic_audit,
)
from etsr.runner import run_temporal_audit, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal Event Spiking Research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a configured model")
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int)

    audit = subparsers.add_parser(
        "temporal-audit", help="Run perturbation and prefix diagnostics"
    )
    audit.add_argument("--config", required=True)
    audit.add_argument("--checkpoint", required=True)

    prepare = subparsers.add_parser(
        "prepare-matched-dvsgc", help="Prepare grouped, boundary-aware DVS-GC samples"
    )
    prepare.add_argument("--config", required=True)

    mechanistic = subparsers.add_parser(
        "mechanistic-audit", help="Run the multi-seed mechanistic temporal audit"
    )
    mechanistic.add_argument("--config", required=True)
    mechanistic.add_argument(
        "--checkpoint", action="append", required=True, metavar="SEED=PATH"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "train":
        print(train_experiment(config, seed=args.seed))
    elif args.command == "temporal-audit":
        print(run_temporal_audit(config, args.checkpoint))
    elif args.command == "prepare-matched-dvsgc":
        manifest = prepare_matched_dvsgc(config)
        root = Path(config["dataset"]["root"])
        print(
            {
                "dataset_root": str(root.resolve()),
                "dataset_manifest": str((root / "dataset_manifest.json").resolve()),
                "samples": len(manifest["samples"]),
                "source_groups": len(manifest["source_filenames"]),
                "official_test_used": False,
            }
        )
    elif args.command == "mechanistic-audit":
        result = run_mechanistic_audit(config, parse_seed_checkpoints(args.checkpoint))
        print(
            {
                "audit_id": result["audit_id"],
                "artifact_dir": result["artifact_dir"],
                "official_test_used": result["official_test_used"],
            }
        )
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
