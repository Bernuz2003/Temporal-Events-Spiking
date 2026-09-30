"""Small, read-only helpers shared by the research result notebooks."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "artifacts").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Repository root not found above the notebook working directory")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_run(root: Path, run_id: str, label: str) -> dict:
    """Load one exact artifact ID; never fall back to a similarly named run."""
    path = root / "artifacts" / run_id
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing complete run: {summary_path}")
    summary = _json(summary_path)
    if summary["run_id"] != run_id:
        raise ValueError(f"Run ID mismatch in {summary_path}")
    if summary.get("official_test_used") is not False:
        raise ValueError(f"Not a development-validation run: {run_id}")
    profile_path = path / "hardware_profile_v4.json"
    profile = _json(profile_path) if profile_path.exists() else None
    if profile is not None:
        if profile.get("official_test_used") is not False:
            raise ValueError(f"Profile includes official test: {run_id}")
        if profile["checkpoint_epoch"] != summary["best_epoch"]:
            raise ValueError(f"Profile is not for the best checkpoint: {run_id}")
    return {
        "id": run_id,
        "label": label,
        "path": path,
        "summary": summary,
        "profile": profile,
        "history": _csv(path / "history.csv"),
        "prefix": _csv(path / "prefix_curve_absolute_time.csv"),
    }


def number(value: str | float | int | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def macro_f1(run: dict) -> float:
    return 100 * run["summary"]["validation"]["macro_f1"]


def accuracy(run: dict) -> float:
    return 100 * run["summary"]["validation"]["accuracy"]


def group_accuracy(run: dict, group: str) -> float:
    metrics = run["summary"]["validation"]["class_group_accuracies"]["metrics"]
    return 100 * metrics[group]["accuracy"]


def best_train_accuracy(run: dict) -> float:
    epoch = run["summary"]["best_epoch"]
    row = next((row for row in run["history"] if int(row["epoch"]) == epoch), None)
    return 100 * number(row["train_accuracy"]) if row is not None else float("nan")


def late_f1(run: dict) -> float:
    late = run["summary"].get("late_window", {})
    return 100 * number(late.get("validation_macro_f1_mean"))


def deployment_parameters(run: dict) -> int:
    summary = run["summary"]
    return summary.get("deployment_parameters") or summary["trainable_parameters"]


def profile_value(run: dict, section: str, key: str) -> float:
    profile = run["profile"]
    return number(profile[section][key]) if profile is not None else float("nan")


def physical_f1_auc(run: dict) -> float:
    """Trapezoidal F1 AUC over the measured physical interval, not from t=0."""
    points = sorted(
        (int(row["requested_us"]), number(row["macro_f1"])) for row in run["prefix"]
    )
    if len(points) < 2:
        return float("nan")
    area = sum(
        (right_t - left_t) * (left_f1 + right_f1) / 2
        for (left_t, left_f1), (right_t, right_f1) in zip(points, points[1:], strict=False)
    )
    return 100 * area / (points[-1][0] - points[0][0])


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)
