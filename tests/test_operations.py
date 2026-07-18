import torch
from torch import nn

from etsr.profiling.operations import OperationProfiler


def test_conv_mac_count_is_reported_per_sample():
    model = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
    model.op_kind = "mac"
    profiler = OperationProfiler(model)
    batch = torch.zeros(5, 2, 8, 8)
    profiler.set_batch_size(batch.shape[0])
    model(batch)
    summary = profiler.summary(samples=5)
    profiler.close()

    expected = 4 * 8 * 8 * (3 * 3 * 2)
    assert summary["mac_ops_per_sample"] == expected
