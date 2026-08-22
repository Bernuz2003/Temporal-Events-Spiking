from __future__ import annotations

import argparse
from pathlib import Path

from etsr.config import load_config
from etsr.data.dvslip_preflight import run_dvslip_preflight
from etsr.data.matched_dvsgc import prepare_matched_dvsgc
from etsr.evaluation.mechanistic import (
    parse_seed_checkpoints,
    run_mechanistic_audit,
)
from etsr.runner import run_temporal_audit, train_experiment
from etsr.smoke import run_smoke_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal Event Spiking Research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a configured model")
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int)

    smoke = subparsers.add_parser(
        "smoke", help="Run the bounded synthetic end-to-end integration check"
    )
    smoke.add_argument("--config", default="configs/smoke.yaml")

    preflight = subparsers.add_parser(
        "preflight-dvslip",
        help="Validate the prospective official-train DVS-Lip archive without opening test",
    )
    preflight.add_argument("--train-root", required=True)
    preflight.add_argument("--speaker-manifest")
    preflight.add_argument("--split-manifest")
    preflight.add_argument(
        "--class-groups",
        default="configs/dvslip_class_groups.json",
        help="Versioned paper-semantic Acc1/Acc2 class manifest",
    )
    preflight.add_argument("--terms")
    preflight.add_argument("--output", default="artifacts/dvslip_preflight.json")
    preflight.add_argument("--samples-per-class", type=int, default=1)
    preflight.add_argument(
        "--hash-samples",
        action="store_true",
        help="Hash every official-train sample; potentially slow on the full archive",
    )

    audit = subparsers.add_parser(
        "temporal-audit", help="Run the frozen frame/DVS-GC perturbation regression"
    )
    audit.add_argument("--config", required=True)
    audit.add_argument("--checkpoint", required=True)

    prepare = subparsers.add_parser(
        "prepare-matched-dvsgc", help="Prepare frozen grouped DVS-GC regression data"
    )
    prepare.add_argument("--config", required=True)

    mechanistic = subparsers.add_parser(
        "mechanistic-audit", help="Run the frozen multi-seed DVS-GC mechanistic audit"
    )
    mechanistic.add_argument("--config", required=True)
    mechanistic.add_argument(
        "--checkpoint", action="append", required=True, metavar="SEED=PATH"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        config = load_config(args.config)
        print(train_experiment(config, seed=args.seed))
    elif args.command == "smoke":
        config = load_config(args.config)
        print(run_smoke_test(config))
    elif args.command == "preflight-dvslip":
        report = run_dvslip_preflight(
            args.train_root,
            speaker_manifest=args.speaker_manifest,
            split_manifest=args.split_manifest,
            class_groups_manifest=args.class_groups,
            terms_path=args.terms,
            output_path=args.output,
            hash_samples=args.hash_samples,
            samples_per_class=args.samples_per_class,
        )
        print(
            {
                "validation_status": report["validation_status"],
                "preflight_gate_status": report["preflight_gate_status"],
                "protocol_gate_status": report["protocol_gate_status"],
                "protocol_blockers": report["protocol_blockers"],
                "official_test_used": report["official_test_used"],
                "output": str(Path(args.output).resolve()),
            }
        )
    elif args.command == "temporal-audit":
        config = load_config(args.config)
        print(run_temporal_audit(config, args.checkpoint))
    elif args.command == "prepare-matched-dvsgc":
        config = load_config(args.config)
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
        config = load_config(args.config)
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
