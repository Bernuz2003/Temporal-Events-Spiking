from __future__ import annotations

import argparse
from pathlib import Path


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
    preflight.add_argument("--split-manifest")
    preflight.add_argument(
        "--class-groups",
        default="configs/dvslip_class_groups.json",
        help="Versioned paper-semantic Acc1/Acc2 class manifest",
    )
    preflight.add_argument("--output", default="artifacts/dvslip_preflight.json")
    preflight.add_argument("--samples-per-class", type=int, default=1)
    preflight.add_argument(
        "--hash-samples",
        action="store_true",
        help="Hash every official-train sample; potentially slow on the full archive",
    )

    prepare_dvslip = subparsers.add_parser(
        "prepare-dvslip-split",
        help="Generate the deterministic sample-stratified DVS-Lip development split",
    )
    prepare_dvslip.add_argument("--train-root", required=True)
    prepare_dvslip.add_argument("--output", default="data/dvslip_development_split.json")

    profile_dvslip = subparsers.add_parser(
        "profile-dvslip",
        help="Profile every official-train DVS-Lip sample without opening test",
    )
    profile_dvslip.add_argument("--train-root", required=True)
    profile_dvslip.add_argument("--split-manifest", required=True)
    profile_dvslip.add_argument("--output", default="artifacts/dvslip_dataset_profile.json")

    shortcut_dvslip = subparsers.add_parser(
        "shortcut-dvslip",
        help="Measure global duration/count/polarity shortcut floors without official-test access",
    )
    shortcut_dvslip.add_argument("--config", default="configs/dvslip_e0_recipe_r0.yaml")
    shortcut_dvslip.add_argument(
        "--output", default="artifacts/dvslip_shortcut_control.json"
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
    mechanistic.add_argument("--checkpoint", action="append", required=True, metavar="SEED=PATH")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        from etsr.config import load_config
        from etsr.runner import train_experiment

        config = load_config(args.config)
        print(train_experiment(config, seed=args.seed))
    elif args.command == "smoke":
        from etsr.config import load_config
        from etsr.smoke import run_smoke_test

        config = load_config(args.config)
        print(run_smoke_test(config))
    elif args.command == "preflight-dvslip":
        from etsr.dvslip.preflight import run_dvslip_preflight

        report = run_dvslip_preflight(
            args.train_root,
            split_manifest=args.split_manifest,
            class_groups_manifest=args.class_groups,
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
    elif args.command == "prepare-dvslip-split":
        from etsr.dvslip.split import prepare_dvslip_development_split

        manifest = prepare_dvslip_development_split(
            args.train_root,
            args.output,
        )
        print(
            {
                "output": str(Path(args.output).resolve()),
                "split_seed": manifest["split_seed"],
                "sample_counts": manifest["sample_counts"],
                "speaker_disjoint": manifest["speaker_disjoint"],
                "official_test_used": manifest["official_test_used"],
            }
        )
    elif args.command == "profile-dvslip":
        from etsr.dvslip.profile import run_dvslip_profile

        report = run_dvslip_profile(
            args.train_root,
            args.split_manifest,
            args.output,
        )
        print(
            {
                "output": str(Path(args.output).resolve()),
                "samples": report["dataset"]["sample_count"],
                "classes": report["dataset"]["class_count"],
                "all_samples_valid": report["validation"]["all_samples_valid"],
                "official_test_used": report["official_test_used"],
            }
        )
    elif args.command == "shortcut-dvslip":
        from etsr.config import load_config
        from etsr.dvslip.shortcut import run_dvslip_shortcut_control

        config = load_config(args.config)
        if config["dataset"]["name"] != "dvslip":
            raise ValueError("shortcut-dvslip requires a DVS-Lip configuration.")
        report = run_dvslip_shortcut_control(
            config["dataset"]["root"],
            config["dataset"]["split_manifest"],
            args.output,
            bin_width_us=int(config["representation"]["bin_width_us"]),
        )
        print(
            {
                "output": str(Path(args.output).resolve()),
                "controls": {
                    name: values["validation"]
                    for name, values in report["controls"].items()
                },
                "official_test_used": report["official_test_used"],
            }
        )
    elif args.command == "temporal-audit":
        from etsr.config import load_config
        from etsr.runner import run_temporal_audit

        config = load_config(args.config)
        print(run_temporal_audit(config, args.checkpoint))
    elif args.command == "prepare-matched-dvsgc":
        from etsr.config import load_config
        from etsr.data.matched_dvsgc import prepare_matched_dvsgc

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
        from etsr.config import load_config
        from etsr.evaluation.mechanistic import (
            parse_seed_checkpoints,
            run_mechanistic_audit,
        )

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
