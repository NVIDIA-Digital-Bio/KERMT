# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import pytest
import torch
from torch import nn

from kermt.util.scheduler import NoamLR


def _make_scheduler(num_groups=2, fine_tune_coff=0.5):
    """Helper to create a NoamLR scheduler with the given number of param groups."""
    model = nn.Linear(4, 2)
    param_groups = [{'params': model.parameters(), 'lr': 1e-4}]
    for _ in range(num_groups - 1):
        param_groups.append({'params': [], 'lr': 1e-4})
    optimizer = torch.optim.Adam(param_groups)
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=1, total_epochs=5, steps_per_epoch=10,
        init_lr=1e-4, max_lr=1e-3, final_lr=1e-5,
        fine_tune_coff=fine_tune_coff, fine_tune_param_idx=0,
    )


def test_lr_coff_dtype_is_float():
    """lr_coff must be float so multiplication with float lr values works correctly."""
    scheduler = _make_scheduler()
    assert scheduler.lr_coff.dtype == np.float64, \
        f"lr_coff dtype should be float64, got {scheduler.lr_coff.dtype}"


def test_lr_coff_values():
    scheduler = _make_scheduler(fine_tune_coff=0.5)
    assert scheduler.lr_coff[0] == 0.5
    assert scheduler.lr_coff[1] == 1.0


def test_lr_scaled_by_fine_tune_coff():
    """After stepping, the fine-tuned param group lr should be scaled down."""
    scheduler = _make_scheduler(fine_tune_coff=0.5)
    scheduler.step()
    lrs = scheduler.get_lr()
    assert lrs[0] < lrs[1], "Fine-tuned group should have lower lr"
    assert abs(lrs[0] / lrs[1] - 0.5) < 1e-6, "lr ratio should be 0.5"


def test_lr_coff_single_param_group():
    """Scheduler with a single param group and fine_tune_coff=1.0."""
    scheduler = _make_scheduler(num_groups=1, fine_tune_coff=1.0)
    scheduler.step()
    lrs = scheduler.get_lr()
    assert len(lrs) == 1
    assert lrs[0] > 0
