from __future__ import annotations

import argparse

from etsr.config import load_config
from etsr.runner import audit_experiment, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal Event Spiking Research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train, validate, test and profile a model")
    train.add_argument("--config", required=True)

    audit = subparsers.add_parser("audit", help="Run temporal perturbation and prefix audits")
    audit.add_argument("--config", required=True)
    audit.add_argument("--checkpoint", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "train":
        train_experiment(config)
    elif args.command == "audit":
        audit_experiment(config, args.checkpoint)
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
