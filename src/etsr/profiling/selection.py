"""Versioned validation sampling independent of checkpoint, training seed and batch size."""

import random
from collections import defaultdict


def profile_indices(targets, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("Profile sample count must be positive.")
    groups = defaultdict(list)
    for index, target in enumerate(targets):
        groups[int(target)].append(index)
    rng = random.Random(0)
    classes = sorted(groups)
    rng.shuffle(classes)
    for indices in groups.values():
        rng.shuffle(indices)
    selected = []
    while len(selected) < min(count, len(targets)):
        for target in classes:
            if groups[target]:
                selected.append(groups[target].pop())
            if len(selected) == min(count, len(targets)):
                break
    return selected
