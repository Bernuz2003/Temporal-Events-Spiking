from __future__ import annotations

from typing import Any


def create_dvsgc_split(config: dict[str, Any], split: str):
    try:
        import dvsgc
    except ImportError as exc:
        raise RuntimeError(
            "DVS-Gesture-Chain is not installed. Run `make install` "
            "or use the provided Singularity container."
        ) from exc

    return dvsgc.DVSGestureChain(
        root=config["root"],
        frames_number=int(config["frames_number"]),
        split=split,
        validation=float(config.get("validation_fraction", 0.2)),
        split_by=str(config.get("split_by", "number")),
        alpha_min=float(config.get("alpha_min", 0.5)),
        alpha_max=float(config.get("alpha_max", 0.7)),
        seq_len=int(config.get("sequence_length", 2)),
        class_num=int(config.get("primitive_classes", 3)),
        repeat=bool(config.get("allow_consecutive_repetition", False)),
        dvsg_path=config.get("events_path"),
    )
