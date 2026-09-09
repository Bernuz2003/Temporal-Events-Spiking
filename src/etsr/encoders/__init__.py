"""Explicit raw-event representations used by the active research phase."""

from etsr.encoders.count import (
    CountFrameEncoder,
    EncodedRepresentation,
    MultiGranularCountFrameEncoder,
    SpikeTemporalBinaryFrameEncoder,
    TemporalBinaryFrameEncoder,
)

__all__ = [
    "CountFrameEncoder",
    "EncodedRepresentation",
    "MultiGranularCountFrameEncoder",
    "SpikeTemporalBinaryFrameEncoder",
    "TemporalBinaryFrameEncoder",
]
