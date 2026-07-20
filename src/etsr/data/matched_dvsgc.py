from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from etsr.data.common import DatasetBundle, IndexedDataset
from etsr.utils.io import write_json

MATCHED_SPLITS = ("train_core", "checkpoint_validation", "development_audit")


def ordered_sequences(
    primitive_ids: list[str], sequence_length: int, allow_repetition: bool
) -> list[str]:
    if sequence_length != 2:
        raise ValueError("The matched generator currently supports order-2 sequences only.")
    if len(set(primitive_ids)) < 2:
        raise ValueError("At least two distinct primitive identifiers are required.")
    if any(len(identifier) != 1 for identifier in primitive_ids):
        raise ValueError(
            "The matched DVS-GC generator currently requires single-character primitive IDs."
        )
    if allow_repetition:
        raise ValueError("The temporal-utilization protocol disables consecutive repetitions.")
    sequences = []
    for sequence in itertools.product(primitive_ids, repeat=sequence_length):
        if any(
            a == b for a, b in zip(sequence, sequence[1:], strict=False)
        ):
            continue
        sequences.append("".join(sequence))
    return sequences


def grouped_split(
    source_filenames: list[str],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    if set(fractions) != set(MATCHED_SPLITS):
        raise ValueError(f"Split fractions must define exactly: {', '.join(MATCHED_SPLITS)}")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to one.")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("Every split fraction must be positive.")

    groups = sorted(set(source_filenames))
    if len(groups) < len(MATCHED_SPLITS):
        raise ValueError("At least three source groups are required.")
    random.Random(seed).shuffle(groups)

    train_end = int(len(groups) * fractions["train_core"])
    validation_end = train_end + int(len(groups) * fractions["checkpoint_validation"])
    train_end = min(max(train_end, 1), len(groups) - 2)
    validation_end = min(max(validation_end, train_end + 1), len(groups) - 1)

    assignments: dict[str, str] = {}
    for position, source in enumerate(groups):
        if position < train_end:
            split = "train_core"
        elif position < validation_end:
            split = "checkpoint_validation"
        else:
            split = "development_audit"
        assignments[source] = split
    assert_grouped_split(assignments)
    return assignments


def assert_grouped_split(assignments: dict[str, str]) -> None:
    invalid = sorted(set(assignments.values()) - set(MATCHED_SPLITS))
    if invalid:
        raise ValueError(f"Unknown split names: {invalid}")
    missing = [name for name in MATCHED_SPLITS if name not in assignments.values()]
    if missing:
        raise ValueError(f"Empty splits: {missing}")


def _stable_seed(*values: object) -> int:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _paired_lengths(
    total_frames: int,
    primitive_frames: int,
    alpha_min: float,
    alpha_max: float,
    seed: int,
) -> tuple[int, int]:
    minimum = math.ceil(primitive_frames * alpha_min)
    maximum = math.floor(primitive_frames * alpha_max)
    candidates = [
        first for first in range(minimum, maximum + 1) if minimum <= total_frames - first <= maximum
    ]
    if not candidates:
        raise ValueError("No valid segment lengths for the requested frames_number/alpha interval.")
    first = random.Random(seed).choice(candidates)
    return first, total_frames - first


def _load_and_integrate_primitive(
    path: Path,
    split_by: str,
    primitive_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    try:
        from spikingjelly import datasets as sjds
    except ImportError as exc:
        raise RuntimeError(
            "Matched DVS-GC preparation requires SpikingJelly. "
            "Use the project container or make install."
        ) from exc

    with np.load(path) as archive:
        events = {key: archive[key] for key in archive.files}
    frames = sjds.integrate_events_by_fixed_frames_number(
        events, split_by, primitive_frames, height, width
    )
    return np.asarray(frames)


def prepare_matched_dvsgc(config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = config["dataset"]
    split_config = config["split"]
    if dataset_config.get("official_split") != "train":
        raise ValueError("Matched-data preparation may only use the official training partition.")
    if not bool(split_config.get("forbid_official_test", True)):
        raise ValueError("The protocol requires split.forbid_official_test=true.")
    if str(split_config.get("group_by")) != "source_filename":
        raise ValueError("The protocol requires grouping by source_filename.")

    output_root = Path(dataset_config["root"])
    events_root = Path(dataset_config["events_root"])
    if events_root.name != "train":
        raise ValueError(
            "dataset.events_root must point to the official events_np/train directory."
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output is not empty at {output_root}; use a new versioned root."
        )

    primitive_ids = [str(value) for value in dataset_config["primitive_ids"]]
    primitive_dirs = [events_root / primitive for primitive in primitive_ids]
    missing_dirs = [str(path) for path in primitive_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing primitive event directories: {missing_dirs}")

    source_sets = [
        {path.name for path in directory.iterdir() if path.suffix == ".npz"}
        for directory in primitive_dirs
    ]
    source_filenames = sorted(set.intersection(*source_sets))
    if not source_filenames:
        raise RuntimeError("No source filenames are shared by all configured primitives.")

    split_fractions = {name: float(split_config[name]) for name in MATCHED_SPLITS}
    assignments = grouped_split(source_filenames, split_fractions, int(split_config["split_seed"]))
    audit_group_count = sum(
        split == "development_audit" for split in assignments.values()
    )
    if audit_group_count < 2:
        raise RuntimeError(
            "At least two development-audit source groups are required for bootstrap."
        )
    sequences = ordered_sequences(
        primitive_ids,
        int(dataset_config["sequence_length"]),
        bool(dataset_config.get("allow_consecutive_repetition", False)),
    )
    classes = sorted(sequences)
    class_to_idx = {name: index for index, name in enumerate(classes)}
    if any(sequence[::-1] not in class_to_idx for sequence in classes):
        raise ValueError("Every class must have a reversed counterpart.")

    output_root.mkdir(parents=True, exist_ok=True)
    samples_root = output_root / "samples"
    for class_name in classes:
        (samples_root / class_name).mkdir(parents=True, exist_ok=True)

    frames_number = int(dataset_config["frames_number"])
    primitive_frames = int(dataset_config["primitive_frames"])
    generation_seed = int(dataset_config["generation_seed"])
    alpha_min = float(dataset_config["alpha_min"])
    alpha_max = float(dataset_config["alpha_max"])
    split_by = str(dataset_config["split_by"])
    height = int(dataset_config.get("height", 128))
    width = int(dataset_config.get("width", 128))
    sample_entries: list[dict[str, Any]] = []

    unordered_pairs = list(itertools.combinations(primitive_ids, 2))
    for source_filename in source_filenames:
        primitive_blocks = {
            primitive: _load_and_integrate_primitive(
                events_root / primitive / source_filename,
                split_by,
                primitive_frames,
                height,
                width,
            )
            for primitive in primitive_ids
        }
        for primitive, block in primitive_blocks.items():
            expected_shape = (primitive_frames, 2, height, width)
            if block.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected integrated shape for {primitive}/{source_filename}: "
                    f"{block.shape}, expected {expected_shape}."
                )
        for first_primitive, second_primitive in unordered_pairs:
            # Both orders reuse the very same primitive chunks. Reversing AB into BA therefore
            # swaps complete actions at their true boundary instead of cutting arbitrary frames.
            length_seed = _stable_seed(
                generation_seed, source_filename, first_primitive, second_primitive
            )
            first_length, second_length = _paired_lengths(
                frames_number, primitive_frames, alpha_min, alpha_max, length_seed
            )
            length_by_primitive = {
                first_primitive: first_length,
                second_primitive: second_length,
            }
            for sequence in (
                first_primitive + second_primitive,
                second_primitive + first_primitive,
            ):
                lengths = [length_by_primitive[primitive] for primitive in sequence]
                chunks = [
                    primitive_blocks[primitive][:length]
                    for primitive, length in zip(sequence, lengths, strict=True)
                ]
                frames = np.concatenate(chunks, axis=0)
                if frames.shape[0] != frames_number:
                    raise RuntimeError("Generated sample has an invalid time dimension.")

                relative_path = Path("samples") / sequence / source_filename
                output_path = output_root / relative_path
                transition = lengths[0]
                np.savez_compressed(
                    output_path,
                    frames=frames,
                    source_filename=np.asarray(source_filename),
                    primitive_sequence=np.asarray(list(sequence)),
                    segment_lengths=np.asarray(lengths, dtype=np.int64),
                    transition_indices=np.asarray([transition], dtype=np.int64),
                    generation_seed=np.asarray(generation_seed, dtype=np.int64),
                    frames_number=np.asarray(frames_number, dtype=np.int64),
                    split_by=np.asarray(split_by),
                    alpha_min=np.asarray(alpha_min, dtype=np.float64),
                    alpha_max=np.asarray(alpha_max, dtype=np.float64),
                    generator_version=np.asarray("matched_dvsgc_order2_v1"),
                )
                sample_id = f"{sequence}/{source_filename}"
                sample_entries.append(
                    {
                        "sample_id": sample_id,
                        "path": relative_path.as_posix(),
                        "source_filename": source_filename,
                        "primitive_sequence": list(sequence),
                        "class_name": sequence,
                        "target": class_to_idx[sequence],
                        "split": assignments[source_filename],
                        "segment_lengths": lengths,
                        "transition_indices": [transition],
                        "reverse_sample_id": f"{sequence[::-1]}/{source_filename}",
                    }
                )

    sample_entries.sort(key=lambda item: (item["class_name"], item["source_filename"]))
    manifest = {
        "schema_version": 1,
        "generator_version": "matched_dvsgc_order2_v1",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_source_split": "train",
        "official_test_used": False,
        "generation_seed": generation_seed,
        "split_seed": int(split_config["split_seed"]),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "primitive_ids": primitive_ids,
        "frames_number": frames_number,
        "primitive_frames": primitive_frames,
        "split_by": split_by,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "source_filenames": source_filenames,
        "samples": sample_entries,
    }
    split_manifest = {
        "schema_version": 1,
        "group_by": "source_filename",
        "split_seed": int(split_config["split_seed"]),
        "fractions": split_fractions,
        "assignments": assignments,
        "official_test_used": False,
    }
    write_json(manifest, output_root / "dataset_manifest.json")
    write_json(split_manifest, output_root / "split_manifest.json")
    return manifest


class MatchedDVSGestureChain(Dataset):
    def __init__(self, root: str | Path, split: str):
        if split not in MATCHED_SPLITS:
            raise ValueError(f"Unknown matched-data split: {split}")
        self.root = Path(root)
        manifest_path = self.root / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Matched-data manifest not found: {manifest_path}. "
                "Run prepare-matched-dvsgc first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("official_test_used") is not False:
            raise RuntimeError("Dataset manifest does not certify the official-test embargo.")
        if self.manifest.get("official_source_split") != "train":
            raise RuntimeError("Dataset manifest does not originate from official training data.")
        if self.manifest.get("generator_version") != "matched_dvsgc_order2_v1":
            raise RuntimeError("Unsupported matched-data generator version.")
        split_manifest_path = self.root / "split_manifest.json"
        if not split_manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if split_manifest.get("official_test_used") is not False:
            raise RuntimeError("Split manifest does not certify the official-test embargo.")
        if split_manifest.get("group_by") != "source_filename":
            raise RuntimeError("Split manifest is not grouped by source filename.")
        self.split = split
        self.classes = list(self.manifest["classes"])
        self.class_to_idx = {
            str(name): int(index) for name, index in self.manifest["class_to_idx"].items()
        }
        self.metadata = [item for item in self.manifest["samples"] if item["split"] == split]
        if not self.metadata:
            raise RuntimeError(f"Matched-data split is empty: {split}")
        assignments = split_manifest.get("assignments", {})
        if any(assignments.get(item["source_filename"]) != split for item in self.metadata):
            raise RuntimeError("Dataset and split manifests disagree on source assignments.")
        self.samples = [
            (str(self.root / item["path"]), int(item["target"])) for item in self.metadata
        ]
        local_by_id = {item["sample_id"]: index for index, item in enumerate(self.metadata)}
        try:
            self.reverse_indices = {
                index: local_by_id[item["reverse_sample_id"]]
                for index, item in enumerate(self.metadata)
            }
        except KeyError as exc:
            raise RuntimeError("A reverse-action pair crosses or is missing from a split.") from exc

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        with np.load(path) as archive:
            frames = np.asarray(archive["frames"])
        return torch.as_tensor(frames, dtype=torch.float32), target

    def metadata_for_index(self, index: int) -> dict[str, Any]:
        return self.metadata[index]

    def matched_reverse_pairs(self) -> list[tuple[int, int]]:
        """Return each exact reverse-action pair once, in canonical dataset order."""
        return [
            (index, reverse_index)
            for index, reverse_index in self.reverse_indices.items()
            if index < reverse_index
        ]


def build_matched_dvsgc_bundle(dataset_config: dict[str, Any]) -> DatasetBundle:
    if dataset_config.get("name") != "matched_dvsgc":
        raise ValueError("Matched DVS-GC requires dataset.name=matched_dvsgc")
    wrapper_args = {
        "input_clip": dataset_config.get("input_clip"),
        "input_scale": str(dataset_config.get("input_scale", "none")),
    }
    raw = {
        split: MatchedDVSGestureChain(dataset_config["root"], split)
        for split in MATCHED_SPLITS
    }
    classes = raw["train_core"].classes
    if any(dataset.classes != classes for dataset in raw.values()):
        raise RuntimeError("Class mappings differ across matched-data splits.")
    return DatasetBundle(
        train=IndexedDataset(raw["train_core"], **wrapper_args),
        validation=IndexedDataset(raw["checkpoint_validation"], **wrapper_args),
        holdout=IndexedDataset(raw["development_audit"], **wrapper_args),
        classes=classes,
    )
