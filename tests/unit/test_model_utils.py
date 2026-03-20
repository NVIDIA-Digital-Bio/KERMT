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
"""
Tests for model building, optimizer construction, and checkpoint utilities.
Requires cuik_molmaker (skip if not installed).
"""
import os
import tempfile
from argparse import Namespace

import pytest
import torch
from torch import nn

cuik_molmaker = pytest.importorskip("cuik_molmaker")

from kermt.util.utils import (
    build_model, build_optimizer, get_ffn_layer_names, save_model_for_restart,
)
from kermt.util.nn_utils import param_count_trainable, param_count_total


def _minimal_finetune_args(**overrides):
    """Builds a minimal Namespace for constructing a KermtFinetuneTask."""
    defaults = dict(
        parser_name='finetune', hidden_size=64, depth=2,
        num_attn_head=2, num_mt_block=1, bias=False, dense=False,
        undirected=False, output_size=1, cuda=False, dropout=0.0,
        activation='ReLU', ffn_hidden_size=None, ffn_num_layers=2,
        self_attention=False, attn_hidden=4, attn_out=4,
        features_only=False, features_size=0, features_dim=0,
        use_input_features=False, embedding_output_type='atom',
        dataset_type='regression', dist_coff=0.1, bond_drop_rate=0.0,
        num_tasks=1, fine_tune_coff=1.0,
        init_lr=1e-4, weight_decay=0.0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# build_model / ffn_hidden_size default
# ---------------------------------------------------------------------------

def test_ffn_hidden_size_defaults_to_hidden_size():
    """When ffn_hidden_size is None, FFN layers should use hidden_size."""
    args = _minimal_finetune_args(ffn_hidden_size=None, ffn_num_layers=3, num_tasks=3)
    model = build_model(args)
    ffn = model.mol_atom_from_atom_ffn
    linear_layers = [m for m in ffn if isinstance(m, nn.Linear)]
    assert linear_layers[0].out_features == 64, \
        f"Expected hidden_size=64, got {linear_layers[0].out_features}"


def test_ffn_hidden_size_explicit():
    """When ffn_hidden_size is set, FFN layers should use that value."""
    args = _minimal_finetune_args(ffn_hidden_size=128, ffn_num_layers=3, num_tasks=3)
    model = build_model(args)
    ffn = model.mol_atom_from_atom_ffn
    linear_layers = [m for m in ffn if isinstance(m, nn.Linear)]
    assert linear_layers[0].out_features == 128, \
        f"Expected 128, got {linear_layers[0].out_features}"


def test_build_model_initializes_all_params():
    """build_model should call initialize_weights on all parameters."""
    args = _minimal_finetune_args()
    model = build_model(args)
    # Biases should be zeroed by initialize_weights
    for name, param in model.state_dict().items():
        if 'bias' in name and param.dim() == 1:
            assert torch.all(param == 0), f"Bias {name} should be zero-initialized"


# ---------------------------------------------------------------------------
# get_ffn_layer_names
# ---------------------------------------------------------------------------

def test_ffn_layer_names_excludes_encoder():
    args = _minimal_finetune_args()
    model = build_model(args)
    ffn_names = get_ffn_layer_names(model)
    for name in ffn_names:
        assert "kermt" not in name, f"FFN name {name} should not contain 'kermt'"
    assert len(ffn_names) > 0


def test_ffn_and_encoder_cover_all_params():
    args = _minimal_finetune_args()
    model = build_model(args)
    ffn_names = set(get_ffn_layer_names(model))
    all_names = set(n for n, _ in model.named_parameters())
    encoder_names = all_names - ffn_names
    assert len(encoder_names) > 0, "Should have encoder params"
    assert ffn_names | encoder_names == all_names


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------

def test_optimizer_has_two_param_groups():
    args = _minimal_finetune_args(fine_tune_coff=0.5)
    model = build_model(args)
    optimizer = build_optimizer(model, args)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]['lr'] == args.init_lr * 0.5
    assert optimizer.param_groups[1]['lr'] == args.init_lr


def test_encoder_frozen_when_fine_tune_coff_zero():
    args = _minimal_finetune_args(fine_tune_coff=0.0)
    model = build_model(args)
    build_optimizer(model, args)
    ffn_names = set(get_ffn_layer_names(model))
    for name, param in model.named_parameters():
        if name not in ffn_names:
            assert not param.requires_grad, f"Encoder param {name} should be frozen"
    # FFN linear layers (mol_atom_from_atom_ffn, mol_atom_from_bond_ffn) should be trainable.
    # readout.cached_zero_vector is a constant placeholder (requires_grad=False by design).
    for name, param in model.named_parameters():
        if name in ffn_names:
            if "mol_atom_from_atom_ffn" in name or "mol_atom_from_bond_ffn" in name:
                assert param.requires_grad, f"FFN param {name} should be trainable"
            elif name == "readout.cached_zero_vector":
                assert not param.requires_grad, f"{name} is a constant, should not be trainable"


def test_encoder_frozen_near_zero_coff():
    """fine_tune_coff close to zero (within tolerance) should also freeze encoder."""
    args = _minimal_finetune_args(fine_tune_coff=1e-7)
    model = build_model(args)
    build_optimizer(model, args)
    ffn_names = set(get_ffn_layer_names(model))
    for name, param in model.named_parameters():
        if name not in ffn_names:
            assert not param.requires_grad, f"Encoder param {name} should be frozen"


def test_frozen_encoder_param_counts():
    """With frozen encoder, trainable count should be less than total."""
    args = _minimal_finetune_args(fine_tune_coff=0.0)
    model = build_model(args)
    build_optimizer(model, args)
    assert param_count_trainable(model) < param_count_total(model)


# ---------------------------------------------------------------------------
# save_model_for_restart
# ---------------------------------------------------------------------------

def test_save_model_for_restart_contents():
    """Checkpoint should contain model, optimizer, scheduler state and epoch."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    args = Namespace(checkpoint_path="/some/pretrain.pt", other_arg="value")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_model_for_restart(path, model, optimizer, scheduler,
                               scaler=None, features_scaler=None,
                               args=args, epoch=5)
        state = torch.load(path, weights_only=False)
        assert 'state_dict' in state
        assert 'optimizer' in state
        assert 'scheduler' in state
        assert state['epoch'] == 5
        assert state['data_scaler'] is None
        assert state['features_scaler'] is None
        assert not hasattr(state['args'], 'checkpoint_path'), \
            "checkpoint_path should be stripped from saved args"
        assert state['args'].other_arg == "value"
    finally:
        os.unlink(path)


def test_save_model_for_restart_loadable():
    """Saved checkpoint should be loadable and contain correct model weights."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    args = Namespace(other_arg="value")

    expected_weight = model.weight.clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_model_for_restart(path, model, optimizer, scheduler,
                               scaler=None, features_scaler=None,
                               args=args, epoch=3)
        state = torch.load(path, weights_only=False)
        loaded_model = nn.Linear(4, 2)
        loaded_model.load_state_dict(state['state_dict'])
        assert torch.equal(loaded_model.weight, expected_weight)
    finally:
        os.unlink(path)
