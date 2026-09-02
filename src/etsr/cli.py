from __future__ import annotations

import argparse
import os
from pathlib import Path

# Required before the command-local PyTorch import when deterministic CUDA matmul is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal Event Spiking Research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a configured model")
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--epochs", type=int, help="Override the configured run length")
    train.add_argument(
        "--resume",
        help="Resume the same run from its epoch-boundary last.pt checkpoint",
    )
    train.add_argument(
        "--overfit",
        nargs=2,
        type=int,
        metavar=("CLASSES", "SAMPLES_PER_CLASS"),
        help="Train and evaluate on one small balanced train-only subset",
    )

    evaluate = subparsers.add_parser(
        "evaluate-checkpoint",
        help="Apply current validation metrics to a compatible selected best.pt checkpoint",
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)

    profile = subparsers.add_parser(
        "profile-checkpoint",
        help="Profile operations, activity and LIF state for a compatible best.pt checkpoint",
    )
    profile.add_argument("--config", required=True)
    profile.add_argument("--checkpoint", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--samples", type=int, default=64)

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
    prepare_dvslip.add_argument(
        "--output",
        default="data/DVS-Lip/dvslip_development_split.json",
    )

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
    shortcut_dvslip.add_argument("--config", default="configs/dvslip_e0.yaml")
    shortcut_dvslip.add_argument("--output", default="artifacts/dvslip_shortcut_control.json")

    prepare_dvsgesture = subparsers.add_parser(
        "prepare-dvsgesture",
        help="Segment official-train DVS-Gesture AEDAT recordings into raw-event samples",
    )
    prepare_dvsgesture.add_argument("--source-root", required=True)
    prepare_dvsgesture.add_argument("--output-root", required=True)
    prepare_dvsgesture.add_argument(
        "--report",
        default="artifacts/dvsgesture_preparation.json",
    )

    profile_dvsgesture = subparsers.add_parser(
        "profile-dvsgesture",
        help="Validate and profile every prepared official-train DVS-Gesture sample",
    )
    profile_dvsgesture.add_argument("--train-root", required=True)
    profile_dvsgesture.add_argument(
        "--output",
        default="artifacts/dvsgesture_dataset_profile.json",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        from etsr.config import load_config
        from etsr.runner import train_experiment

        config = load_config(args.config)
        if args.epochs is not None:
            if args.epochs <= 0:
                raise ValueError("--epochs must be positive")
            config["training"]["epochs"] = args.epochs
            config["training"]["warmup_epochs"] = min(
                int(config["training"].get("warmup_epochs", 0)), args.epochs - 1
            )
        if args.overfit is not None:
            class_count, samples_per_class = args.overfit
            if class_count <= 0 or samples_per_class <= 0:
                raise ValueError("--overfit values must be positive")
            config["training"]["overfit"] = {
                "class_count": class_count,
                "samples_per_class": samples_per_class,
            }
            config["training"]["amp"] = False
            config["training"]["select_metric"] = "accuracy"
            config["experiment"]["name"] += "_overfit"
            if "recipe_id" in config["training"]:
                config["training"]["recipe_id"] += "_overfit"
        print(train_experiment(config, seed=args.seed, resume_from=args.resume))
    elif args.command == "evaluate-checkpoint":
        from etsr.config import load_config
        from etsr.runner import evaluate_checkpoint

        config = load_config(args.config)
        summary = evaluate_checkpoint(config, args.checkpoint, args.output)
        print(
            {
                "output": str(Path(args.output).resolve()),
                "checkpoint_epoch": summary["checkpoint_epoch"],
                "accuracy": summary["validation"]["accuracy"],
                "macro_f1": summary["validation"]["macro_f1"],
                "official_test_used": summary["official_test_used"],
            }
        )
    elif args.command == "profile-checkpoint":
        from etsr.config import load_config
        from etsr.runner import profile_checkpoint

        config = load_config(args.config)
        profile = profile_checkpoint(
            config,
            args.checkpoint,
            args.output,
            max_samples=args.samples,
        )
        print(
            {
                "output": str(Path(args.output).resolve()),
                "samples_profiled": profile["samples_profiled"],
                "trainable_parameters": profile["parameters"]["trainable_elements"],
                "official_test_used": profile["official_test_used"],
            }
        )
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
                    name: values["validation"] for name, values in report["controls"].items()
                },
                "official_test_used": report["official_test_used"],
            }
        )
    elif args.command == "prepare-dvsgesture":
        from etsr.dvsgesture.prepare import prepare_dvsgesture_train

        report = prepare_dvsgesture_train(
            args.source_root,
            args.output_root,
            args.report,
        )
        print(
            {
                "output": str(Path(args.output_root).resolve()),
                "report": str(Path(args.report).resolve()),
                "recordings": report["recording_count"],
                "samples": report["sample_count"],
                "subjects": len(report["subject_counts"]),
                "official_test_used": report["official_test_used"],
            }
        )
    elif args.command == "profile-dvsgesture":
        from etsr.dvsgesture.profile import run_dvsgesture_profile

        report = run_dvsgesture_profile(args.train_root, args.output)
        print(
            {
                "output": str(Path(args.output).resolve()),
                "samples": report["dataset"]["sample_count"],
                "classes": report["dataset"]["class_count"],
                "subjects": report["dataset"]["subject_count"],
                "all_samples_valid": report["validation"]["all_samples_valid"],
                "official_test_used": report["official_test_used"],
            }
        )
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
