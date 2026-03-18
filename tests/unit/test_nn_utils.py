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
import pytest
import torch
from torch import nn

from kermt.util.nn_utils import param_count_trainable, param_count_total, initialize_weights


# ---------------------------------------------------------------------------
# param_count_trainable / param_count_total
# ---------------------------------------------------------------------------

def test_param_count_all_trainable():
    model = nn.Linear(10, 5)  # 10*5 + 5 = 55 params
    assert param_count_trainable(model) == 55
    assert param_count_total(model) == 55


def test_param_count_frozen_weight():
    model = nn.Linear(10, 5)
    model.weight.requires_grad = False
    assert param_count_trainable(model) == 5  # only bias
    assert param_count_total(model) == 55


def test_param_count_trainable_leq_total():
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    model[0].weight.requires_grad = False
    assert param_count_trainable(model) <= param_count_total(model)


# ---------------------------------------------------------------------------
# initialize_weights with init_param_names
# ---------------------------------------------------------------------------

def test_initialize_weights_no_param_names_is_noop():
    """When init_param_names is None (default), no parameters should be modified."""
    model = nn.Linear(4, 2)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    initialize_weights(model, init_param_names=None)
    for k, v in model.state_dict().items():
        assert torch.equal(v, original[k]), f"Parameter {k} should not change"


def test_initialize_weights_selective():
    """Only parameters whose names are in init_param_names should be modified.
    initialize_weights zeros biases (dim=1) and applies xavier_normal_ to weights (dim>=2).
    Unselected parameters should remain at their PyTorch default init (kaiming_uniform_)."""
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    original = {k: v.clone() for k, v in model.state_dict().items()}

    # Only init the second layer
    init_names = [n for n in model.state_dict().keys() if n.startswith("1.")]
    initialize_weights(model, init_param_names=init_names)

    for k, v in model.state_dict().items():
        if k in init_names:
            if v.dim() == 1:
                assert torch.all(v == 0), f"Bias {k} should be zeroed by initialize_weights"
            else:
                # Weight was re-initialized with xavier_normal_, should differ from default kaiming_uniform_
                assert not torch.equal(v, original[k]), f"Weight {k} should be re-initialized"
        else:
            assert torch.equal(v, original[k]), f"Parameter {k} should not change"


def test_initialize_weights_all_params():
    """When all param names are given, biases should be zeroed and weights re-initialized."""
    model = nn.Linear(4, 3, bias=True)
    original_weight = model.state_dict()["weight"].clone()
    all_names = list(model.state_dict().keys())
    initialize_weights(model, init_param_names=all_names)
    assert torch.all(model.state_dict()["bias"] == 0), "Bias should be zeroed"
    assert not torch.equal(model.state_dict()["weight"], original_weight), \
        "Weight should be re-initialized with xavier_normal_"


def test_initialize_weights_distinct_init():
    """Different model_idx values should produce different initializations."""
    torch.manual_seed(0)
    m1 = nn.Linear(8, 4, bias=False)
    names = list(m1.state_dict().keys())
    initialize_weights(m1, distinct_init=True, model_idx=0, init_param_names=names)
    w1 = m1.weight.clone()

    torch.manual_seed(0)
    m2 = nn.Linear(8, 4, bias=False)
    initialize_weights(m2, distinct_init=True, model_idx=1, init_param_names=names)
    w2 = m2.weight.clone()

    assert not torch.equal(w1, w2), "Different model_idx should produce different init"
