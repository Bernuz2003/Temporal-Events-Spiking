"""Explicit raw-event representations used by the active research phase."""

from etsr.encoders.count import (
    CountFrameEncoder,
    EncodedRepresentation,
    SpikeTemporalBinaryFrameEncoder,
    TemporalBinaryFrameEncoder,
)

__all__ = [
    "CountFrameEncoder",
    "EncodedRepresentation",
    "SpikeTemporalBinaryFrameEncoder",
    "TemporalBinaryFrameEncoder",
]
