# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from dataclasses import dataclass
import cuik_molmaker
import torch


@dataclass
class FeatureRange:
    """Class to store the start and end indices of a feature range"""
    start: int
    end: int

def get_feature_range(atom_props_onehot, atom_props_float):
    """Get the feature ranges for atom properties"""
    
    smi = "CC(=O)O"  # Example molecule to compute feature ranges
    feature_ranges = {}
    feature_start_idx = 0
    
    # Get ranges for one-hot encoded features
    for atom_prop in atom_props_onehot:
        atom_prop_tensor = cuik_molmaker.atom_onehot_feature_names_to_tensor([atom_prop])
        atom_feats_cmm, _, _, _, _ = cuik_molmaker.mol_featurizer(smi, atom_prop_tensor, 
            torch.tensor([]), torch.tensor([]), False, False, True, False)
        feature_ranges[atom_prop] = FeatureRange(feature_start_idx, feature_start_idx + atom_feats_cmm.shape[1])
        feature_start_idx += atom_feats_cmm.shape[1]
    
    # Get ranges for float features
    for atom_prop in atom_props_float:
        atom_prop_tensor = cuik_molmaker.atom_float_feature_names_to_tensor([atom_prop])
        atom_feats_cmm, _, _, _, _ = cuik_molmaker.mol_featurizer(smi, torch.tensor([]), 
            atom_prop_tensor, torch.tensor([]), False, False, True, False)
        feature_ranges[atom_prop] = FeatureRange(feature_start_idx, feature_start_idx + atom_feats_cmm.shape[1])
        feature_start_idx += atom_feats_cmm.shape[1]
    
    return feature_ranges 
