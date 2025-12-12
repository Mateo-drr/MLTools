# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:09:10 2024

@author: Mateo-drr
"""

import time
from model import SampleNet
import torch
import wandb

from config import config
from cstm_ds import make_train_dl
import utils as utils
import loops

#PARAMS
if config.threads is not None:
    torch.set_num_threads(config.threads)
    torch.set_num_interop_threads(config.threads)
torch.backends.cudnn.benchmark = config.cudnn_bench

train_dl = make_train_dl(config, "train")
valid_dl = make_train_dl(config, "valid")
test_dl = make_train_dl(config, "test")

# Instantiate the model
model = SampleNet()
model.to(config.device)

# Define a loss function and optimizer
criterion = config.criterion()
optimizer = config.optimizer(model.parameters(), lr=config.lr)
if config.scheduler is not None:
    scheduler = config.scheduler(optimizer, T_max=config.num_epochs)
scaler = torch.amp.GradScaler(device=config.device)

#init weights & biases
if config.wb:
    wandb.init(project=config.project_name,
               config=config.__dict__.copy())

utils.count_params(model)

timings = {"start": time.time()}
wb_metrics = {"train": {}, "valid": {}}
current_best = {}

for epoch in range(config.num_epochs):

    timings["epoch_start"] = time.time()

    loops.train_loop(
        model,
        train_dl,
        criterion,
        optimizer,
        scaler,
        wb_metrics,
        config,
        epoch,
        scheduler=None
    )

    loops.eval_loop(
        model,
        valid_dl,
        criterion,
        eval_name="valid",
        wb_metrics=wb_metrics,
        config=config,
    )

    current_best = utils.finish_epoch(
        epoch, wb_metrics, timings, current_best, optimizer.param_groups[0]["lr"], model, config
    )

if config.wb:
    wandb.finish()    











