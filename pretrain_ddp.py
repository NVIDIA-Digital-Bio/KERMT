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

import torch
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os


from kermt.data.kermtdataset import get_data, split_data, KermtCollator
from kermt.util.utils import create_logger
from kermt.model.models import KERMTEmbedding
from task.kermttrainer import KERMTTrainer
from kermt.data.torchvocab import MolVocab
from kermt.data.kermtdataset import BatchMolDataset
from kermt.util.parsing import parse_args_ddp
from kermt.util.nn_utils import param_count

def pre_load_data_ddp(dataset: BatchMolDataset, dataset_size: int, samples_per_file: int):
    for i in range(1, dataset_size, samples_per_file):
        dataset.load_data(i)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)

    # parse args
    args = parse_args_ddp()

    if rank == 0:
        print(f"{args=}")
    logger = create_logger(name='pretrain', save_dir=args.save_dir)

    # Build train, val, and test datasets
    train_data, train_sample_per_file = get_data(data_path=args.train_data_path)
    train_data_size = len(train_data)
    print(f"Training data size: {train_data_size}")
    pre_load_data_ddp(train_data, train_data_size, train_sample_per_file)

    if args.val_data_path is not None:
        val_data, val_sample_per_file = get_data(data_path=args.val_data_path)
        val_data_size = len(val_data)
        print(f"Validation data size: {val_data_size}")
        pre_load_data_ddp(val_data, val_data_size, val_sample_per_file)
    else:
        val_data = None
        val_data_size = 0

    if args.test_data_path is not None:
        raise NotImplementedError("Test data is not implemented")
        test_data, test_sample_per_file = get_data(data_path=args.test_data_path)
        test_data_size = len(test_data)
        print(f"Test data size: {test_data_size}")
        pre_load_data_ddp(test_data, test_data_size, test_sample_per_file)
    else:
        test_data = None
        test_data_size = 0
    # load atom and bond vocabulary and the semantic motif labels.
    atom_vocab = MolVocab.load_vocab(args.atom_vocab_path)
    bond_vocab = MolVocab.load_vocab(args.bond_vocab_path)
    atom_vocab_size, bond_vocab_size = len(atom_vocab), len(bond_vocab)

    train_sampler = DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True)
    
    if args.val_data_path is not None:
        val_sampler = DistributedSampler(
            val_data, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = None

    if args.test_data_path is not None:
        test_sampler = DistributedSampler(
            test_data, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        test_sampler = None

    # build collator
    # Hard coding here, since we haven't load any data yet!
    fg_size = 85
    shared_dict = {}
    mol_collator = KermtCollator(shared_dict=shared_dict, atom_vocab=atom_vocab, bond_vocab=bond_vocab, args=args)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, # batch size per GPU (aka micro batch size)
                                  shuffle=False, # because train_sampler does the shuffling
                                  num_workers=args.num_dataloader_workers,
                                  sampler=train_sampler,
                                  collate_fn=mol_collator,
                                  drop_last=True)
    
    if args.val_data_path is not None:
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, 
                                  shuffle=False, # because no shuffling needed
                                  num_workers=args.num_dataloader_workers,
                                  sampler=val_sampler,
                                  collate_fn=mol_collator,
                                  drop_last=True)
    else:
        val_dataloader = None
        
    # Build model
    kermt_model = KERMTEmbedding(args)

    print(f'Number of parameters = {param_count(kermt_model):,}')

    # Build trainer
    trainer = KERMTTrainer(args=args,
                            embedding_model=kermt_model,
                            atom_vocab_size=atom_vocab_size,
                            bond_vocab_size=bond_vocab_size,
                            fg_szie=fg_size,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            world_size=world_size,
                            gpu_id=rank,
                            n_steps=0,
                            logger=logger
                            )

    if args.save_dir is not None:
        last_ckpt_path = os.path.join(args.save_dir, "last_checkpoint.pt")
        if os.path.exists(last_ckpt_path):
            print(f"Loading checkpoint from {last_ckpt_path}")
            epoch, scheduler_step, prev_batch_idx = trainer.load(last_ckpt_path)
            print(f"Loaded checkpoint from epoch={epoch}, scheduler_step={scheduler_step}, prev_batch_idx={prev_batch_idx}")
        else:
            epoch = 0
            scheduler_step = 0
            prev_batch_idx = 0

    steps_per_epoch = train_data_size // (args.batch_size*world_size)
    print(f"Steps per epoch: {steps_per_epoch}")
    curr_epoch_batch_idx = scheduler_step % steps_per_epoch
    print(f"Current epoch batch index: {curr_epoch_batch_idx}")

    trainer.set_batch_idx(prev_batch_idx)
    # Train model
    trainer.train(start_epoch=epoch, max_epochs=args.epochs)
    destroy_process_group()


if __name__ == "__main__":

    world_size = os.environ.get("WORLD_SIZE", 1)
    world_size = int(world_size)
    print(f"World size: {world_size}")
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
