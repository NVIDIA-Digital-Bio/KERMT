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
"""
The vocabulary building scripts.
"""
import os
from kermt.data.torchvocab import MolVocab
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Path to the data file")
    parser.add_argument('--vocab_save_folder', type=str, help="Path to the folder where the vocab files will be saved")
    parser.add_argument('--dataset_name', type=str, default=None,
                        help="Will be the first part of the vocab file name. If it is None,"
                             "the vocab files will be: atom_vocab.pkl and bond_vocab.pkl")
    parser.add_argument('--vocab_max_size', type=int, default=None)
    parser.add_argument('--vocab_min_freq', type=int, default=1)
    args = parser.parse_args()
    return args

def build(args):
    for vocab_type in ['atom', 'bond']:
        vocab_file = f"{vocab_type}_vocab.pkl"
        if args.dataset_name is not None:
            vocab_file = args.dataset_name + '_' + vocab_file
        vocab_save_path = os.path.join(args.vocab_save_folder, vocab_file)

        os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
        vocab = MolVocab(file_path=args.data_path,
                         max_size=args.vocab_max_size,
                         min_freq=args.vocab_min_freq,
                         num_workers=100,
                         vocab_type=vocab_type)
        print(f"{vocab_type} vocab size", len(vocab))
        vocab.save_vocab(vocab_save_path)


if __name__ == '__main__':
    args = parse_args()
    build(args)
