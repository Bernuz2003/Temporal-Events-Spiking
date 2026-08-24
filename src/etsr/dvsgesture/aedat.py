"""Minimal AEDAT 3 polarity-event reader required by the official DVS-Gesture archive."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def load_aedat_v3(path: str | Path) -> dict[str, np.ndarray]:
    """Read polarity packets from one AEDAT 3 recording without third-party conversion."""

    recording = Path(path)
    timestamps: list[int] = []
    x_coords: list[int] = []
    y_coords: list[int] = []
    polarities: list[int] = []

    with recording.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"AEDAT header terminator missing: {recording}")
            if line.rstrip(b"\r\n") == b"#!END-HEADER":
                break
            if not line.startswith(b"#"):
                raise ValueError(f"Invalid AEDAT header line in {recording}")

        while True:
            header = handle.read(28)
            if not header:
                break
            if len(header) != 28:
                raise ValueError(f"Truncated AEDAT packet header: {recording}")
            (
                event_type,
                _event_source,
                event_size,
                _timestamp_offset,
                timestamp_overflow,
                event_capacity,
                event_number,
                event_valid,
            ) = struct.unpack("<HHIIIIII", header)
            if event_size <= 0 or event_valid > event_number or event_number > event_capacity:
                raise ValueError(f"Invalid AEDAT packet dimensions in {recording}")

            payload = handle.read(event_capacity * event_size)
            if len(payload) != event_capacity * event_size:
                raise ValueError(f"Truncated AEDAT packet payload: {recording}")
            if event_type != 1:
                continue
            if event_size < 8:
                raise ValueError(f"Invalid polarity-event size in {recording}: {event_size}")

            valid_events = 0
            for offset in range(0, event_number * event_size, event_size):
                address, timestamp = struct.unpack_from("<II", payload, offset)
                if not address & 1:
                    continue
                valid_events += 1
                x_coords.append((address >> 17) & 0x1FFF)
                y_coords.append((address >> 2) & 0x1FFF)
                polarities.append((address >> 1) & 1)
                timestamps.append(timestamp | (timestamp_overflow << 31))
            if valid_events != event_valid:
                raise ValueError(f"AEDAT valid-event count mismatch in {recording}")

    if not timestamps:
        raise ValueError(f"No polarity events found in AEDAT recording: {recording}")
    timestamp_array = np.asarray(timestamps, dtype=np.uint64)
    return {
        "t": timestamp_array,
        "x": np.asarray(x_coords, dtype=np.uint16),
        "y": np.asarray(y_coords, dtype=np.uint16),
        "p": np.asarray(polarities, dtype=np.uint8),
    }
