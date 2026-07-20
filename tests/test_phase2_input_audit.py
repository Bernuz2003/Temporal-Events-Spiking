import numpy as np
import torch

from etsr.phase2.input_audit import input_statistics


def test_order_invariant_shortcut_features_do_not_encode_frame_order():
    frames = torch.rand(1, 6, 2, 4, 4)
    metadata = [
        {
            "source_filename": "source.npz",
            "class_name": "13",
            "transition_indices": [3],
        }
    ]

    _rows, temporal, invariant = input_statistics(frames, metadata)
    _reversed_rows, reversed_temporal, reversed_invariant = input_statistics(
        frames.flip(1), metadata
    )

    assert not np.array_equal(temporal, reversed_temporal)
    assert np.allclose(invariant, reversed_invariant)
