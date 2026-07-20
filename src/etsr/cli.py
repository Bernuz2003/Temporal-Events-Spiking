from __future__ import annotations

import argparse

from etsr.config import load_config
from etsr.phase2.runner import (
    parse_checkpoint_arguments,
    phase2_audit,
    phase2_prepare,
    phase2_train,
)
from etsr.runner import audit_experiment, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal Event Spiking Research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train, validate, test and profile a model")
    train.add_argument("--config", required=True)

    audit = subparsers.add_parser("audit", help="Run temporal perturbation and prefix audits")
    audit.add_argument("--config", required=True)
    audit.add_argument("--checkpoint", required=True)

    phase2_prepare_parser = subparsers.add_parser(
        "phase2-prepare", help="Prepare the grouped metadata-rich Phase 2 dataset"
    )
    phase2_prepare_parser.add_argument("--config", required=True)

    phase2_train_parser = subparsers.add_parser(
        "phase2-train", help="Train one frozen-architecture Phase 2 seed"
    )
    phase2_train_parser.add_argument("--config", required=True)
    phase2_train_parser.add_argument("--seed", required=True, type=int)

    phase2_audit_parser = subparsers.add_parser(
        "phase2-audit", help="Run the multi-seed mechanistic temporal audit"
    )
    phase2_audit_parser.add_argument("--config", required=True)
    phase2_audit_parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="SEED=PATH"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "train":
        train_experiment(config)
    elif args.command == "audit":
        audit_experiment(config, args.checkpoint)
    elif args.command == "phase2-prepare":
        print(phase2_prepare(config))
    elif args.command == "phase2-train":
        print(phase2_train(config, args.seed))
    elif args.command == "phase2-audit":
        checkpoints = parse_checkpoint_arguments(args.checkpoint)
        result = phase2_audit(config, checkpoints)
        print(
            {
                "phase2_id": result["phase2_id"],
                "artifact_dir": result["artifact_dir"],
                "official_test_used": result["official_test_used"],
            }
        )
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
