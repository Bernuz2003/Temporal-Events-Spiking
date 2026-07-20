import json

import numpy as np
import pytest

from etsr.data.matched_dvsgc import (
    MatchedDVSGestureChain,
    assert_grouped_split,
    grouped_split,
    prepare_matched_dvsgc,
)


def test_grouped_split_is_deterministic_disjoint_and_complete():
    sources = [f"sample_{index}.npz" for index in range(20)]
    fractions = {
        "train_core": 0.70,
        "checkpoint_validation": 0.15,
        "development_audit": 0.15,
    }
    first = grouped_split(sources, fractions, seed=17)
    second = grouped_split(list(reversed(sources)), fractions, seed=17)

    assert first == second
    assert set(first) == set(sources)
    assert_grouped_split(first)
    assert set(first.values()) == set(fractions)


def test_matched_dataset_keeps_reverse_pairs_inside_the_split(tmp_path):
    sample_entries = []
    for class_name, target, reverse_name in (("13", 0, "31"), ("31", 1, "13")):
        relative = f"samples/{class_name}/source.npz"
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, frames=np.zeros((4, 2, 8, 8), dtype=np.float32))
        sample_entries.append(
            {
                "sample_id": f"{class_name}/source.npz",
                "path": relative,
                "source_filename": "source.npz",
                "primitive_sequence": list(class_name),
                "class_name": class_name,
                "target": target,
                "split": "train_core",
                "segment_lengths": [2, 2],
                "transition_indices": [2],
                "reverse_sample_id": f"{reverse_name}/source.npz",
            }
        )
    manifest = {
        "official_test_used": False,
        "official_source_split": "train",
        "generator_version": "matched_dvsgc_order2_v1",
        "classes": ["13", "31"],
        "class_to_idx": {"13": 0, "31": 1},
        "samples": sample_entries,
    }
    split_manifest = {
        "official_test_used": False,
        "group_by": "source_filename",
        "assignments": {"source.npz": "train_core"},
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")

    dataset = MatchedDVSGestureChain(tmp_path, "train_core")

    assert len(dataset) == 2
    assert dataset.reverse_indices == {0: 1, 1: 0}
    assert dataset.matched_reverse_pairs() == [(0, 1)]
    assert dataset[0][0].shape == (4, 2, 8, 8)


def test_matched_dataset_rejects_a_manifest_without_test_embargo(tmp_path):
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"official_test_used": True}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="official-test embargo"):
        MatchedDVSGestureChain(tmp_path, "train_core")


def test_preparation_builds_frame_exact_reverse_action_pairs(tmp_path, monkeypatch):
    events_root = tmp_path / "events_np" / "train"
    for primitive in ("1", "3", "8"):
        directory = events_root / primitive
        directory.mkdir(parents=True)
        for index in range(20):
            (directory / f"source_{index}.npz").touch()

    def fake_integrate(path, _split_by, primitive_frames, height, width):
        primitive = int(path.parent.name)
        time = np.arange(primitive_frames, dtype=np.float32).reshape(-1, 1, 1, 1)
        return np.broadcast_to(
            primitive * 100 + time, (primitive_frames, 2, height, width)
        ).copy()

    monkeypatch.setattr("etsr.data.matched_dvsgc._load_and_integrate_primitive", fake_integrate)
    output_root = tmp_path / "matched_v1"
    config = {
        "dataset": {
            "root": str(output_root),
            "events_root": str(events_root),
            "official_split": "train",
            "primitive_ids": ["1", "3", "8"],
            "sequence_length": 2,
            "allow_consecutive_repetition": False,
            "frames_number": 16,
            "primitive_frames": 13,
            "generation_seed": 123,
            "alpha_min": 0.5,
            "alpha_max": 0.7,
            "split_by": "number",
            "height": 4,
            "width": 4,
        },
        "split": {
            "group_by": "source_filename",
            "split_seed": 19,
            "train_core": 0.70,
            "checkpoint_validation": 0.15,
            "development_audit": 0.15,
            "forbid_official_test": True,
        },
    }

    manifest = prepare_matched_dvsgc(config)
    by_id = {item["sample_id"]: item for item in manifest["samples"]}
    ab_meta = by_id["13/source_0.npz"]
    ba_meta = by_id["31/source_0.npz"]
    with np.load(output_root / ab_meta["path"]) as archive:
        ab = archive["frames"]
    with np.load(output_root / ba_meta["path"]) as archive:
        ba = archive["frames"]

    length_a, length_b = ab_meta["segment_lengths"]
    assert ba_meta["segment_lengths"] == [length_b, length_a]
    assert np.array_equal(ab[:length_a], ba[length_b:])
    assert np.array_equal(ab[length_a:], ba[:length_b])
    assert ab_meta["split"] == ba_meta["split"]
    assert manifest["official_test_used"] is False
