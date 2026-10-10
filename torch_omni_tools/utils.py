# -*- coding: utf-8 -*-
"""
Utility functions
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

from typing import Any
from collections import defaultdict
import numpy as np
import torch
import time
from datetime import datetime
from pprint import pprint
import copy

from torch_omni_tools.config import Config


def print_list(items: list[Any]) -> None:
    """
    Print every item of a list with its index
    Args
        items: Items to print
    """
    for i, item in enumerate(items):
        print(i, item)


def eta(
    start_time: float, epoch_start: float, epoch_end: float, epoch: int, num_epochs: int
) -> None:
    """
    Calculate and print estimated time of arrival (ETA) for training epochs
    Args
        start_time: Start time of the training process as a timestamp
        epoch_start: Start time of the current epoch as a timestamp
        epoch_end: End time of the current epoch as a timestamp
        epoch: Current epoch number, 0 based
        num_epochs: Total number of epochs
    """
    elapsed_total = epoch_end - start_time
    epoch_time = epoch_end - epoch_start
    epochs_completed = epoch + 1

    # Calculate remaining time
    avg_epoch_time = elapsed_total / epochs_completed
    remaining_epochs = num_epochs - epochs_completed
    eta_seconds = remaining_epochs * avg_epoch_time

    print(
        f"Epoch {epochs_completed}/{num_epochs} completed "
        f"in {epoch_time:.1f}s | "
        f"Elapsed: {format_time(elapsed_total)} | "
        f"ETA: {format_time(eta_seconds)} | "
        f"Total est: {format_time(elapsed_total + eta_seconds)}"
    )


def format_time(seconds: float) -> str:
    """
    Format seconds to h:m or only m
    Args
        seconds: Time in seconds to format
    Returns
        str: Formated time
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}h" if hours > 0 else f"{minutes:02d}m"


def count_params(model: torch.nn.Module) -> None:
    """
    Print trainable and total number of parameters in model
    Args
        model: Model to inspect
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


def format_metrics(
    outputs: dict[str, Any], results: defaultdict[str, list[Any]]
) -> defaultdict[str, list[Any]]:
    """
    Append every metric of a batch to the results of the epoch
    Args
        outputs: Metrics of a single batch, tensors are converted to floats
        results: Metrics of the epoch, each metric holds a list of values per batch
    Returns
        defaultdict[str, list[Any]]: Updated metrics of the epoch
    """
    for key, value in outputs.items():
        results[key].append(value.item() if isinstance(value, torch.Tensor) else value)
    return results


def epoch_results(
    results: dict[str, list[Any]], formated: dict[str, dict[str, Any]], split: str
) -> dict[str, dict[str, Any]]:
    """
    Calculate the average of all the metrics of an epoch
    Args
        results: Dictionary of metrics, each metric has to be a list of values per batch
        formated: Dictionary to place the formated epoch results
        split: Either train or valid
    Returns
        dict[str, dict[str, Any]]: Updated dictionary of epoch results
    """
    for key, values in results.items():
        mean_value = np.mean(values)
        formated[split][key] = mean_value.item()
    return formated


def finish_epoch(
    epoch: int,
    wb_metrics: dict[str, dict[str, Any]],
    timings: dict[str, float],
    current_best: dict[str, Any],
    current_lr: float,
    model: torch.nn.Module,
    config: Config,
) -> dict[str, Any]:
    """
    Log the epoch results, print the ETA and keep the best model so far
    Args
        epoch: Current epoch, 0 based
        wb_metrics: Metrics of the epoch, keyed by split and metric name
        timings: Timestamps of the start of the run and of the current epoch
        current_best: Metrics and model of the best epoch so far
        current_lr: Learning rate of the current epoch
        model: Model of the current epoch
        config: Config holding the total number of epochs and the wandb option
    Returns
        dict[str, Any]: Metrics and model of the best epoch so far
    """
    print(
        f"Epoch {epoch}: "
        f"Train Loss: {wb_metrics["train"]["loss"]},"
        f" Valid Loss: {wb_metrics["valid"]["loss"]}"
    )

    if config.wb:
        import wandb

        # TODO format your metrics if necessary
        formatted: dict[str, Any] = wb_metrics.copy()
        formatted["learning_rate"] = current_lr
        wandb.log(formatted)

    eta(timings["start"], timings["epoch_start"], time.time(), epoch, config.num_epochs)

    if epoch == 0 or current_best["loss"] > wb_metrics["valid"]["loss"]:
        print("=" * 10)
        print(f"New best model, epoch {epoch}")
        current_best = wb_metrics["valid"].copy()
        pprint(current_best)
        current_best["model"] = copy.deepcopy(model)
        print("=" * 10)

    if epoch == config.num_epochs - 1:
        print("=" * 10)
        print("Last epoch results")
        pprint(wb_metrics)
        print("=" * 10)

    return current_best
