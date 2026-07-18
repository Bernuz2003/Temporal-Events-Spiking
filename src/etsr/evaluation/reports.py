from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(matrix: np.ndarray, classes: list[str], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    if len(classes) <= 20:
        positions = np.arange(len(classes))
        axis.set_xticks(positions, classes, rotation=90)
        axis.set_yticks(positions, classes)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_prefix_curve(rows: list[dict], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["fraction"] for row in rows], [row["accuracy"] for row in rows], marker="o")
    axis.set_xlabel("Observed sequence fraction")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
