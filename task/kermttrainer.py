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
The KERMT trainer.
"""
import os
import time
from logging import Logger
from typing import List, Tuple
from collections.abc import Callable
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.cuda import nvtx

from kermt.model.models import KermtTask
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
class KERMTTrainer:
    def __init__(self,
                 args,
                 embedding_model: Module,
                 atom_vocab_size: int,  # atom vocab size
                 bond_vocab_size: int,
                 fg_szie: int,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer,
                 scheduler,
                 gpu_id,
                 n_steps: int,
                 logger: Logger = None):
        """
        The init function of KERMTTrainer
        :param args: the input arguments.
        :param embedding_model: the model to generate atom/bond embeddings.
        :param atom_vocab_size: the vocabulary size of atoms.
        :param bond_vocab_size: the vocabulary size of bonds.
        :param fg_szie: the size of semantic motifs (functional groups)
        :param train_dataloader: the training dataloader.
        :param val_dataloader: the validation dataloader.
        :param optimizer: the optimizer.
        :param scheduler: the scheduler.
        :param gpu_id: the gpu id.
        :param logger: the logger
        """

        self.args = args
        self.kermt = embedding_model
        self.model = KermtTask(args, embedding_model, atom_vocab_size, bond_vocab_size, fg_szie)
        self.loss_func = self.model.get_loss_func(args)
        self.gpu_id = gpu_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.atom_vocab_size = atom_vocab_size
        self.bond_vocab_size = bond_vocab_size
        self.debug = logger.debug if logger is not None else print

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.args = args
        self.n_iter = 0

        self.model.to(self.gpu_id)

        self.model = DDP(self.model, device_ids=[gpu_id])

        if self.args.tensorboard:
            self.writer = SummaryWriter(self.args.save_dir)

        # Var by SV (not sure what n_iter is doing)
        self.n_steps = n_steps
        self.first_epoch_post_resume = True
        self.curr_epoch_batch_idx = 0 # Number of batches to skip in first epoch of current training

    def train(self, start_epoch: int, max_epochs: int) -> List:
        """
        The training iteration
        :param max_epochs: the max epochs.
        :return: the loss terms of current epoch.
        """
        for epoch in range(start_epoch, max_epochs):
            s_time = time.time()
            _, train_loss, _ = self.iter(epoch, train=True)
            t_time = time.time() - s_time
            if self.gpu_id == 0:
                print(f"epoch={epoch:04d}, cur_lr={self.scheduler.get_lr()[0]:.5f}, train_loss={train_loss:.6f}, train_time={t_time:.2f}", flush=True)

    def validation(self, max_val_batches: int) -> List:
        """
        The validation iteration
        :param max_val_batches: the maximum number of batches to validate.
        :return: the loss terms as a list
        """
        self.model.eval()
        loss_sum, iter_count = 0, 0
        n_batches = 0
        av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum, bv_dist_loss_sum, fg_dist_loss_sum = 0, 0, 0, 0, 0, 0
        # loss_func = self.model.get_loss_func(self.args)

        for ibatch, item in enumerate(self.val_dataloader):
            batch_graph = item["graph_input"]
            targets = item["targets"]
            targets["av_task"] = targets["av_task"].to(self.gpu_id)
            targets["bv_task"] = targets["bv_task"].to(self.gpu_id)
            targets["fg_task"] = targets["fg_task"].to(self.gpu_id)

            preds = self.model(batch_graph)

            loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss = self.loss_func(preds, targets)

            loss_sum += loss.item()
            iter_count += self.args.batch_size

            av_loss_sum += av_loss.item()
            bv_loss_sum += bv_loss.item()
            fg_loss_sum += fg_loss.item()
            av_dist_loss_sum += av_dist_loss.item() if type(av_dist_loss) != float else av_dist_loss
            bv_dist_loss_sum += bv_dist_loss.item() if type(bv_dist_loss) != float else bv_dist_loss
            fg_dist_loss_sum += fg_dist_loss.item() if type(fg_dist_loss) != float else fg_dist_loss

            n_batches += 1
            if n_batches >= max_val_batches:
                break

        # Compute per batch losses
        loss_sum /= n_batches
        av_loss_sum /= n_batches
        bv_loss_sum /= n_batches
        fg_loss_sum /= n_batches
        av_dist_loss_sum /= n_batches
        bv_dist_loss_sum /= n_batches
        fg_dist_loss_sum /= n_batches


        if self.gpu_id == 0:
            print(f"Validation loss: {loss_sum:.4f}, av_loss: {av_loss_sum:.4f}, bv_loss: {bv_loss_sum:.4f}, fg_loss: {fg_loss_sum:.4f}", flush=True)
            if self.args.tensorboard:
                self.writer.add_scalar('val/loss', loss_sum, self.n_steps)
                self.writer.add_scalar('val/av_loss', av_loss_sum, self.n_steps)
                self.writer.add_scalar('val/bv_loss', bv_loss_sum, self.n_steps)
                self.writer.add_scalar('val/fg_loss', fg_loss_sum, self.n_steps)

        self.model.train()
        return loss_sum

    def test(self, epoch: int) -> List:
        """
        The test/validaiion iteration
        :param epoch: the current epoch number.
        :return:  the loss terms as a list
        """
        # return self.mock_iter(epoch, self.test_data, train=False)
        return self.iter(epoch, self.test_data, train=False)

    def mock_iter(self, epoch: int, data_loader: DataLoader, train: bool = True) -> List:
        """
        Perform a mock iteration. For test only.
        :param epoch: the current epoch number.
        :param data_loader: the data loader.
        :param train: True: train model, False: validation model.
        :return: the loss terms as a list
        """

        for _, _ in enumerate(data_loader):
            self.scheduler.step()
        cum_loss_sum = 0.0
        self.n_iter += self.args.batch_size
        return self.n_iter, cum_loss_sum, (0, 0, 0, 0, 0, 0)

    def set_batch_idx(self, batch_idx: int):
        self.curr_epoch_batch_idx = batch_idx

    def iter(self, epoch, train=True) -> List:
        """
        Perform a training / validation iteration.
        :param epoch: the current epoch number.
        :param data_loader: the data loader.
        :param train: True: train model, False: validation model.
        :return: the loss terms as a list
        """
        if train:
            self.model.train()
            self.train_dataloader.sampler.set_epoch(epoch)
        else:
            self.model.eval()

        loss_sum, iter_count = 0, 0
        cum_loss_sum, cum_iter_count = 0, 0
        av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum, bv_dist_loss_sum, fg_dist_loss_sum = 0, 0, 0, 0, 0, 0
        # loss_func = self.model.get_loss_func(self.args)

        for ibatch, item in enumerate(self.train_dataloader):
            if self.first_epoch_post_resume:
                if ibatch < self.curr_epoch_batch_idx:
                    print(f"Skipping batch {ibatch} because of curr_epoch_batch_idx={self.curr_epoch_batch_idx}", flush=True)
                    continue
                elif ibatch >= self.curr_epoch_batch_idx:
                    print(f"Stop skipping batches because of curr_epoch_batch_idx={self.curr_epoch_batch_idx}", flush=True)
                    self.first_epoch_post_resume = False

            if self.gpu_id == 0:
                print(f"{self.n_steps=}", flush=True)
            batch_graph = item["graph_input"]
            targets = item["targets"]
            # if next(self.model.parameters()).is_cuda:
            targets["av_task"] = targets["av_task"].to(self.gpu_id)
            targets["bv_task"] = targets["bv_task"].to(self.gpu_id)
            targets["fg_task"] = targets["fg_task"].to(self.gpu_id)

            preds = self.model(batch_graph)

            loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss = self.loss_func(preds, targets)

            loss_sum += loss.item()
            iter_count += self.args.batch_size

            if train:
                cum_loss_sum += loss.item()
                # Run model
                self.model.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            else:
                # For eval model, only consider the loss of three task.
                cum_loss_sum += av_loss.item()
                cum_loss_sum += bv_loss.item()
                cum_loss_sum += fg_loss.item()

            av_loss_sum += av_loss.item()
            bv_loss_sum += bv_loss.item()
            fg_loss_sum += fg_loss.item()
            av_dist_loss_sum += av_dist_loss.item() if type(av_dist_loss) != float else av_dist_loss
            bv_dist_loss_sum += bv_dist_loss.item() if type(bv_dist_loss) != float else bv_dist_loss
            fg_dist_loss_sum += fg_dist_loss.item() if type(fg_dist_loss) != float else fg_dist_loss

            # Save model
            if (self.gpu_id == 0)and (self.n_steps % self.args.save_interval) == 0:
                self.save(batch_idx=ibatch, n_steps=self.n_steps, epoch=epoch, file_path=self.args.save_dir, name=f"model_step_{self.n_steps}.pt", save_last=True)

            cum_iter_count += 1
            self.n_iter += self.args.batch_size
            self.n_steps += 1

            if self.gpu_id == 0 and self.args.tensorboard and self.n_steps % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.n_steps)
                self.writer.add_scalar('train/av_loss', av_loss.item(), self.n_steps)
                self.writer.add_scalar('train/bv_loss', bv_loss.item(), self.n_steps)
                self.writer.add_scalar('train/fg_loss', fg_loss.item(), self.n_steps)
                self.writer.add_scalar('train/av_dist_loss', av_dist_loss.item(), self.n_steps)
                self.writer.add_scalar('train/bv_dist_loss', bv_dist_loss.item(), self.n_steps)
                self.writer.add_scalar('train/fg_dist_loss', fg_dist_loss.item(), self.n_steps)
                self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], self.n_steps)
                self.writer.add_scalar('train/epoch', epoch, self.n_steps)
                self.writer.add_scalar('train/batch_idx', ibatch, self.n_steps)
            # Debug only.
            # if i % 50 == 0:
            #     print(f"epoch: {epoch}, batch_id: {i}, av_loss: {av_loss}, bv_loss: {bv_loss}, "
            #           f"fg_loss: {fg_loss}, av_dist_loss: {av_dist_loss}, bv_dist_loss: {bv_dist_loss}, "
            #           f"fg_dist_loss: {fg_dist_loss}")

        cum_loss_sum /= cum_iter_count
        av_loss_sum /= cum_iter_count
        bv_loss_sum /= cum_iter_count
        fg_loss_sum /= cum_iter_count
        av_dist_loss_sum /= cum_iter_count
        bv_dist_loss_sum /= cum_iter_count
        fg_dist_loss_sum /= cum_iter_count

        val_loss = self.validation(max_val_batches=self.args.max_val_batches)

        return self.n_iter, cum_loss_sum, (av_loss_sum, bv_loss_sum, fg_loss_sum, av_dist_loss_sum,
                                           bv_dist_loss_sum, fg_dist_loss_sum)

    def save(self, batch_idx, n_steps, epoch, file_path, name=None, save_last=False) -> str:
        """
        Save the intermediate models during training.
        :param n_steps: the step number.
        :param epoch: the epoch number.
        :param file_path: the file_path to save the model.
        :param save_last: whether to save the last model.
        :return: the output path.
        """
        # add specific time in model fine name, in order to distinguish different saved models
        now = time.localtime()
        if name is None:
            name = "_%04d_%02d_%02d_%02d_%02d_%02d" % (
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        output_path = os.path.join(file_path, name)
        scaler = None
        features_scaler = None
        state = {
            'args': self.args,
            'state_dict': self.model.module.state_dict(), # changed to self.model.module.state_dict() from self.model.state_dict()
            'optimizer': self.optimizer.state_dict(),
            'scheduler_step': n_steps,
            'batch_idx': batch_idx,
            "epoch": epoch,
            'data_scaler': {
                'means': scaler.means,
                'stds': scaler.stds
            } if scaler is not None else None,
            'features_scaler': {
                'means': features_scaler.means,
                'stds': features_scaler.stds
            } if features_scaler is not None else None
        }
        torch.save(state, output_path)
        if save_last:
            last_path = os.path.join(file_path, "last_checkpoint.pt")
            torch.save(state, last_path)

        # Is this necessary?
        # if self.with_cuda:
        #    self.model = self.model.cuda()
        print(f"Model at step={n_steps} saved at {output_path}", flush=True)
        return output_path

    def save_tmp(self, epoch, file_path, rank=0):
        """
        Save the models for auto-restore during training.
        The model are stored in file_path/tmp folder and will replaced on each epoch.
        :param epoch: the epoch number.
        :param file_path: the file_path to store the model.
        :param rank: the current rank (decrypted).
        :return:
        """
        store_path = os.path.join(file_path, "tmp")
        if not os.path.exists(store_path):
            os.makedirs(store_path, exist_ok=True)
        store_path = os.path.join(store_path, "model.%d" % rank)
        state = {
            'args': self.args,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            "epoch": epoch
        }
        torch.save(state, store_path)

    def load(self, checkpoint_path) -> Tuple[int, int]:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found")
            return
        # TODO(sveccham): Change this to weights_only=True
        ckpt = torch.load(checkpoint_path, weights_only=False)
        self.model.module.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.current_step = ckpt["scheduler_step"]
        epoch = ckpt["epoch"]
        scheduler_step = ckpt["scheduler_step"]
        batch_idx = ckpt["batch_idx"]
        print(f"Batch index from loaded checkpoint: {batch_idx}")
        self.n_steps = scheduler_step
        return epoch, scheduler_step, batch_idx
    