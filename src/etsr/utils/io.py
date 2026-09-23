from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)


def append_csv(row: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not row:
        raise ValueError("Cannot append an empty CSV row.")
    if not output.exists() or output.stat().st_size == 0:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Existing CSV has no header: {output}")
        new_fields = [name for name in row if name not in fieldnames]
        existing_rows = list(reader) if new_fields else []

    if not new_fields:
        with output.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        return

    if any(None in existing for existing in existing_rows):
        raise ValueError(f"Existing CSV contains rows wider than its header: {output}")
    expanded_fields = [*fieldnames, *new_fields]
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=expanded_fields)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
        os.chmod(temporary_path, output.stat().st_mode)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
