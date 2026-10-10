# -*- coding: utf-8 -*-
"""
Train and Eval loops
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

from collections import defaultdict
from typing import Any
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_omni_tool import utils
from torch_omni_tool.config import Config


def run_model(
    model: nn.Module,
    samples: dict[str, torch.Tensor],
    criterion: nn.Module,
    config: Config,
) -> dict[str, Any]:
    """
    Run the model on the given input samples
    Args
        model: Model to run
        samples: Batch of samples, the input is expected under the data key
        criterion: Loss function, the input is used as the target
        config: Config holding the device
    Returns
        dict[str, Any]: Loss, input data and model output
    """
    # Move data to device
    data = samples["data"].to(config.device)
    # Run the model
    output = model(data)
    # Calc loss
    loss = criterion(data, output)

    return {
        "loss": loss,
        "data": data,
        "output": output,
    }


def train_loop(
    model: nn.Module,
    train_dl: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    wb_metrics: dict[str, dict[str, Any]],
    config: Config,
    epoch: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """
    Train the model for one epoch
    Args
        model: Model to train
        train_dl: Dataloader over the training split
        criterion: Loss function
        optim: Optimizer
        scaler: GradScaler used for the mixed precision backward pass
        wb_metrics: Metrics dictionary that holds the epoch results
        config: Config holding the device, mixed precision and grad clipping options
        epoch: Current epoch, 0 based
        scheduler: Scheduler stepped at the end of the epoch
    """
    model.train()
    results: defaultdict[str, list[Any]] = defaultdict(list)

    for sample in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
        optim.zero_grad()
        if config.half_p:
            with torch.amp.autocast(device_type=config.device):
                outputs = run_model(model, sample, criterion, config)

                scaler.scale(outputs["loss"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optim)
                scaler.update()
        else:
            outputs = run_model(model, sample, criterion, config)
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optim.step()

        # update metrics and loss
        outputs.pop("data")
        outputs.pop("output")
        results = utils.format_metrics(outputs, results)

    scheduler.step() if scheduler is not None else None

    utils.epoch_results(results, wb_metrics, split="train")


def eval_loop(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    eval_name: str,
    wb_metrics: dict[str, dict[str, Any]],
    config: Config,
) -> None:
    """
    Evaluate the model without computing gradients
    Args
        model: Model to evaluate
        dataloader: Dataloader over the split being evaluated
        criterion: Loss function
        eval_name: Evaluation name being run, e.g. valid
        wb_metrics: Metrics dictionary that holds the epoch results
        config: Config holding the device and the mixed precision option
    """
    model.eval()
    results: defaultdict[str, list[Any]] = defaultdict(list)
    with torch.no_grad():
        for sample in tqdm(dataloader, desc=f"{eval_name}"):
            if config.half_p:
                with torch.amp.autocast(device_type=config.device):
                    outputs = run_model(model, sample, criterion, config)
            else:
                outputs = run_model(model, sample, criterion, config)

            # update metrics and loss
            outputs.pop("data")
            outputs.pop("output")
            results = utils.format_metrics(outputs, results)

    utils.epoch_results(results, wb_metrics, split="valid")
