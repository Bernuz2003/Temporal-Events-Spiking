from __future__ import annotations

import argparse
import os
from pathlib import Path

# Required before the command-local PyTorch import when deterministic CUDA matmul is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal Event Spiking Research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate = subparsers.add_parser("candidate", help="Bounded overfit, gated full run, then profile")
    candidate.add_argument("--config", required=True)
    replicate = subparsers.add_parser(
        "replicate", help="Repeat a validated full configuration at one new seed, then profile"
    )
    replicate.add_argument("--config", required=True)
    replicate.add_argument("--seed", required=True, type=int)
    backfill = subparsers.add_parser("profile-runs", help="Reprofile completed full runs, no training")
    backfill.add_argument("--artifact-root", default="artifacts")
    backfill.add_argument("--checkpoint-root", default="checkpoints")
    backfill.add_argument("--samples", type=int, default=64)

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
    train.add_argument("--readout", choices=("mean", "last", "diagonal_gated"))
    train.add_argument(
        "--readout-time",
        choices=("fixed_window", "last_event"),
        help="Select the fixed horizon or the latest occupied event-bin snapshot for readout",
    )
    train.add_argument(
        "--bin-width-us",
        type=int,
        help="Override only E0 physical bin width for the controlled coarse/fine comparison",
    )
    train.add_argument(
        "--no-cross-time",
        action="store_true",
        help="Reset every LIF between timesteps for the P2 dependency control",
    )
    train.add_argument(
        "--temporal-mask",
        nargs=2,
        type=int,
        metavar=("COUNT", "MAX_STEPS"),
        help="Apply COUNT training-only temporal masks of up to MAX_STEPS",
    )
    train.add_argument(
        "--spatial-erasing",
        nargs=2,
        type=int,
        metavar=("COUNT", "MAX_PIXELS"),
        help="Apply COUNT training-only square cutouts of up to MAX_PIXELS",
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

    temporal_diagnostic = subparsers.add_parser(
        "temporal-diagnostic",
        help="Diagnose every causal prefix and the post-event tail of a selected checkpoint",
    )
    temporal_diagnostic.add_argument("--config", required=True)
    temporal_diagnostic.add_argument("--checkpoint", required=True)
    temporal_diagnostic.add_argument("--output", required=True)

    temporal_pair = subparsers.add_parser(
        "temporal-diagnostic-pair",
        help="Run the preregistered baseline and PLIF checkpoint diagnostics sequentially",
    )
    temporal_pair.add_argument("--baseline-config", required=True)
    temporal_pair.add_argument("--baseline-checkpoint", required=True)
    temporal_pair.add_argument("--plif-config", required=True)
    temporal_pair.add_argument("--plif-checkpoint", required=True)
    temporal_pair.add_argument("--output", required=True)

    tcap_diagnostic = subparsers.add_parser(
        "tcap-tap-diagnostic",
        help="Ablate each learned TCAP delay matrix in one selected checkpoint",
    )
    tcap_diagnostic.add_argument("--config", required=True)
    tcap_diagnostic.add_argument("--checkpoint", required=True)
    tcap_diagnostic.add_argument("--output", required=True)

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
    shortcut_dvslip.add_argument(
        "--temporal",
        action="store_true",
        help="Also compare time-aligned and order-invariant per-bin polarity counts",
    )

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
    if args.command == "candidate":
        from etsr.config import load_config
        from etsr.workflows import run_candidate

        print(run_candidate(load_config(args.config)))
    elif args.command == "replicate":
        from etsr.config import load_config
        from etsr.workflows import run_replication

        print(run_replication(load_config(args.config), args.seed))
    elif args.command == "profile-runs":
        from etsr.workflows import profile_completed_runs

        report = profile_completed_runs(args.artifact_root, args.checkpoint_root, args.samples)
        print(report)
        if not report["complete"]:
            raise SystemExit(f"Profiling incomplete; see {args.artifact_root}/profile_backfill.json.")
    elif args.command == "train":
        from etsr.config import load_config, validate_config
        from etsr.runner import train_experiment

        config = load_config(args.config)
        experiment_suffixes = []
        recipe_suffixes = []
        if args.bin_width_us is not None:
            if args.bin_width_us <= 0:
                raise ValueError("--bin-width-us must be positive")
            config["representation"]["bin_width_us"] = args.bin_width_us
            experiment_suffixes.append(f"bin_{args.bin_width_us}us")
        if args.readout is not None:
            config["model"]["readout"] = args.readout
            experiment_suffixes.append(f"readout_{args.readout}")
        if args.readout_time is not None:
            config["model"]["readout_time"] = args.readout_time
            experiment_suffixes.append(f"readout_time_{args.readout_time}")
        if args.no_cross_time:
            config["model"]["lif_cross_time"] = False
            experiment_suffixes.append("no_cross_time")
        for argument, field_prefix, suffix_prefix in (
            (args.temporal_mask, "temporal_mask", "tm"),
            (args.spatial_erasing, "spatial_erasing", "se"),
        ):
            if argument is None:
                continue
            count, maximum = argument
            if count <= 0 or maximum <= 0:
                raise ValueError(f"--{field_prefix.replace('_', '-')} values must be positive")
            extent_field = "max_steps" if field_prefix == "temporal_mask" else "max_pixels"
            config["augmentation"][f"{field_prefix}_count"] = count
            config["augmentation"][f"{field_prefix}_{extent_field}"] = maximum
            experiment_suffixes.append(f"{suffix_prefix}{count}x{maximum}")
            recipe_suffixes.append(f"{suffix_prefix}{count}x{maximum}")
        if experiment_suffixes:
            config["experiment"]["name"] += f"_{'_'.join(experiment_suffixes)}"
        if recipe_suffixes:
            config["training"]["recipe_id"] += f"_{'_'.join(recipe_suffixes)}"
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
        validate_config(config)
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
    elif args.command == "temporal-diagnostic":
        from etsr.config import load_config
        from etsr.evaluation.temporal_diagnostic import diagnose_checkpoint_temporal_dynamics

        summary = diagnose_checkpoint_temporal_dynamics(
            load_config(args.config), args.checkpoint, args.output
        )
        print(
            {
                "output": str(Path(args.output).resolve()),
                "checkpoint_epoch": summary["checkpoint_epoch"],
                "samples": summary["samples"],
                "prefix_auc": summary["prefix_auc"],
                "official_test_used": summary["official_test_used"],
            }
        )
    elif args.command == "temporal-diagnostic-pair":
        from etsr.config import load_config
        from etsr.evaluation.temporal_diagnostic import diagnose_baseline_plif_pair

        summary = diagnose_baseline_plif_pair(
            load_config(args.baseline_config),
            args.baseline_checkpoint,
            load_config(args.plif_config),
            args.plif_checkpoint,
            args.output,
        )
        print(
            {
                "output": str(Path(args.output).resolve()),
                "samples": summary["samples"],
                "plif_minus_baseline_prefix_auc": summary[
                    "plif_minus_baseline_prefix_auc"
                ],
                "official_test_used": summary["official_test_used"],
            }
        )
    elif args.command == "tcap-tap-diagnostic":
        from etsr.config import load_config
        from etsr.evaluation.temporal_diagnostic import diagnose_tcap_taps

        summary = diagnose_tcap_taps(
            load_config(args.config), args.checkpoint, args.output
        )
        print(
            {
                "output": str(Path(args.output).resolve()),
                "checkpoint_epoch": summary["checkpoint_epoch"],
                "samples": summary["samples"],
                "ablation_effects": summary["ablation_effects"],
                "official_test_used": summary["official_test_used"],
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
            time_steps=(
                int(config["representation"]["window_us"])
                // int(config["representation"]["bin_width_us"])
                if args.temporal
                else None
            ),
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
