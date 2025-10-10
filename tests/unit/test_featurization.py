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
import pandas as pd
from argparse import Namespace
from kermt.data.molgraph import mol2graph
from kermt.util.features import get_feature_range

import cuik_molmaker

@pytest.mark.parametrize("bond_drop_rate", [0.0, 0.1, 0.2, 0.3])
def test_cuik_molmaker_featurization(bond_drop_rate: float):
    smis = pd.read_csv('tests/data/smis.csv')['smiles'].tolist()

    # Form feature tensors for cuik-molmaker
    cmm_feature_tensors = {}
    atom_onehot_props = ["atomic-number", "total-degree", "formal-charge", "chirality",
                        "num-hydrogens", "hybridization",
                        "implicit-valence", 
                        "ring-size",
                        ]

    cmm_feature_tensors["atom_onehot"] = cuik_molmaker.atom_onehot_feature_names_to_tensor(atom_onehot_props)
    atom_float_props = ["aromatic", "mass", 
                        "hydrogen-bond-acceptor",
                            "hydrogen-bond-donor", 
                            "acidic", "basic"
                            ]
    cmm_feature_tensors["atom_float"] = cuik_molmaker.atom_float_feature_names_to_tensor(atom_float_props)
    bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]
    cmm_feature_tensors["bond"] = cuik_molmaker.bond_feature_names_to_tensor(bond_props)

    # Get feature ranges for cuik-molmaker
    cmm_feature_range = get_feature_range(atom_onehot_props, atom_float_props)
    
    shared_dict = {}
    mol2graph_args = Namespace(bond_drop_rate=bond_drop_rate, no_cache=True, use_cuikmolmaker_featurization=True, 
                               seed=42, # data was also generated with seed 42
                               )

    batch_mol_graph = mol2graph(smis, shared_dict, mol2graph_args, set_seed=True,
                                cmm_feature_range=cmm_feature_range,
                                cmm_tensors=cmm_feature_tensors,
    )

    batch_mol_graph_ref = torch.load(f'tests/data/batch_mol_graph_bond_drop_rate_{bond_drop_rate}.pt')

    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch_mol_graph.get_components()
    f_atoms_diff = torch.abs(f_atoms - batch_mol_graph_ref['f_atoms']).sum()
    assert torch.allclose(f_atoms, batch_mol_graph_ref['f_atoms']), f"f_atoms are not equal "
    assert torch.allclose(f_bonds, batch_mol_graph_ref['f_bonds']), f"f_bonds are not equal"
    assert torch.allclose(a2b, batch_mol_graph_ref['a2b']), f"a2b are not equal"
    assert torch.allclose(b2a, batch_mol_graph_ref['b2a']), f"b2a are not equal"
    assert torch.allclose(b2revb, batch_mol_graph_ref['b2revb']), f"b2revb are not equal"
    assert torch.allclose(a_scope, batch_mol_graph_ref['a_scope']), f"a_scope are not equal"
    assert torch.allclose(b_scope, batch_mol_graph_ref['b_scope']), f"b_scope are not equal"
    assert torch.allclose(a2a, batch_mol_graph_ref['a2a']), f"a2a are not equal"
