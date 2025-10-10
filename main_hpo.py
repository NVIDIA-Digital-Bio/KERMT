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

import random
import shutil
from functools import partial
import os
import numpy as np
import torch
from rdkit import RDLogger
import json

from kermt.util.parsing import parse_args, get_newest_train_args
from kermt.util.utils import create_logger
from task.cross_validate import cross_validate
from task.fingerprint import generate_fingerprints
from task.predict import make_predictions, write_prediction
from task.pretrain import pretrain_model
from kermt.data.torchvocab import MolVocab
from task.train import run_training

import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

INIT_LR_FACTOR = 10

def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)

def objective_all(trial, args, logger):

    ## Setup optuna stuff
    # Change save_dir to temp location
    print(f"Current trial.number: {trial.number}")
    failed_trial_number = RetryFailedTrialCallback.retried_trial_number(trial)
    print(f"failed_trial_number: {failed_trial_number}")

    trial_number = trial.number
    parent_save_dir = args.save_dir
    args.save_dir = os.path.join(args.save_dir, f"tmp_trial_{trial_number}")
    print(f"Saving temporarily to {args.save_dir}")

    ## HPO to tune
    # LR hyperparameters
    max_lr = trial.suggest_float("max_lr", 1E-4, 1E-3, step=2E-4)
    final_lr_factor = trial.suggest_int("final_lr_factor", 2, 10, step=2)
    final_lr = max_lr / final_lr_factor
    init_lr = max_lr / INIT_LR_FACTOR

    args.max_lr = max_lr 
    args.init_lr = init_lr
    args.final_lr = final_lr

    # Could not come up with a function that takes 0, 0.05, 0.1, 0.2
    dropout = trial.suggest_categorical("dropout", choices=[0, 0.05, 0.1, 0.2])
    attn_out = trial.suggest_int("attn_out", 4, 8, step=4)
    dist_coff = trial.suggest_float("dist_coff", 0.05, 0.15, step=0.05)
    bond_drop_rate = trial.suggest_float("bond_drop_rate", 0.0, 0.2, step=0.2)
    ffn_num_layers = trial.suggest_int("ffn_num_layers", 2, 3, step=1)
    ffn_hidden_size = trial.suggest_int("ffn_hidden_size", 700, 1300, step=600)

    args.dropout = dropout
    args.attn_out = attn_out
    args.dist_coff = dist_coff
    args.bond_drop_rate = bond_drop_rate
    args.ffn_num_layers = ffn_num_layers
    args.ffn_hidden_size = ffn_hidden_size

    print("Current set of hyperparameters used:")
    print(args)
    ensemble_scores, min_val_loss = run_training(args, logger, return_val=True)
    print(f"*************** min_val_loss for trial {trial_number}: {min_val_loss} ***************")
    trial_dict = vars(args)
    trial_dict["min_val_loss"] = min_val_loss
    trial_dict["test_metric"] = np.nanmean(ensemble_scores)
    with open(f"{args.save_dir}/params.json", "w") as outfile: 
        json.dump(trial_dict, outfile)

    # Move ckpt to actual path
    final_save_dir = os.path.join(parent_save_dir, f"trial_{trial_number}")
    print(f"Moving {args.save_dir} to {final_save_dir}")
    shutil.move(args.save_dir, final_save_dir)

    args.save_dir = parent_save_dir

    return min_val_loss


if __name__ == "__main__":

    args = parse_args()
    print(f"args: {args}")
    # setup random seed
    setup(seed=args.seed)

    # Set up Optuna storage
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{args.save_dir}/optuna.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage, study_name="pytorch_checkpoint", direction="minimize", load_if_exists=True,
    )

    # Avoid the pylint warning.
    a = MolVocab
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Initialize MolVocab
    mol_vocab = MolVocab

    if args.parser_name != 'finetune':
        raise ValueError(f"Not HPO NYI for {args.parser_name} mode")
    
    if args.n_trials is None:
        raise ValueError(f"--n_trials cannot be None during HPO")

    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
    
    print(f"Number of trials for HPO: {args.n_trials}")
    objective = partial(objective_all, args=args, logger=logger)
    study.optimize(objective, n_trials=args.n_trials, timeout=None,
                    callbacks=[MaxTrialsCallback(args.n_trials, states=(TrialState.COMPLETE,))])

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))