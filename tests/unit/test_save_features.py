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
import pandas as pd
import numpy as np
import os
from argparse import Namespace
from kermt.data.task_labels import RDKIT_PROPS
from scripts.save_features import generate_and_save_features

@pytest.mark.parametrize("features_generator", ["fgtasklabel", "rdkit_2d_normalized", "rdkit_2d_normalized_onthefly", "rdkit_2d_normalized_cuik_molmaker"])
def test_generate_and_save_features(smis_only_csv_path, data_dir, features_generator):
    smi_df = pd.read_csv(smis_only_csv_path)

    save_path = f"{features_generator}_features.npz"

    args = Namespace(
        data_path=str(smis_only_csv_path),
        features_generator=features_generator,
        save_path=str(save_path),
        save_frequency=20000,
        restart=True,
        max_data_size=None,
        sequential=True,
    )

    generate_and_save_features(args)

    # Check that the features file was created
    assert os.path.exists(save_path)

    features = np.load(save_path)["features"]
    
    features_ref = np.load(data_dir / f"{features_generator}_features.npz")["features"]

    assert np.allclose(features, features_ref), "Generated features do not match reference features"

    # Clean up the generated features.npz file after the test
    if os.path.exists(save_path):
        os.remove(save_path)

