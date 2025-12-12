"""
Utility functions
"""
import numpy as np
import torch
import wandb
import time
from datetime import datetime
from pprint import pprint
import copy

def print_list(items):
    for i,item in enumerate(items):
        print(i,item)

def eta(start_time: float,
        epoch_start: float,
        epoch_end: float,
        epoch: int,
        num_epochs: int):
    """
    Calculate and print estimated time of arrival (ETA) for training epochs.

    Args:
        start_time (datetime): Start time of the training process.
        epoch_start (datetime): Start time of the current epoch.
        epoch_end (datetime): End time of the current epoch.
        epoch (int): Current epoch number (0-based).
        num_epochs (int): Total number of epochs.
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
    Args:
        seconds (float): time in seconds to format
    Returns:
        str: formated time
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}h" if hours > 0 else f"{minutes:02d}m"


def count_params(model):
    """ Print trainable and total number of parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def format_metrics(outputs, results):
    for key, value in outputs.items():
        results[key].append(
            value.item() if isinstance(value, torch.Tensor) else value
        )
    return results

def epoch_results(results: dict, formated: dict, split: str) -> dict:
    """
    Calculate the average of all the metrics of an epoch.
    Args:
        results (dict): Dictionary of metrics, each metric has to be a list of
        values per batch
        formated (dict): Dictionary to place the formated epoch results
        split (str): either train or valid
    Returns:
        dict: Updated dictionary of epoch results
    """
    for key, values in results.items():
        mean_value = np.mean(values)
        formated[split][key] = mean_value.item()
    return formated

def finish_epoch(epoch, wb_metrics, timings, current_best, current_lr, model, config):
    print(
        f"Epoch {epoch}: "
        f"Train Loss: {wb_metrics["train"]["loss"]},"
        f" Valid Loss: {wb_metrics["valid"]["loss"]}"
    )

    if config.wb:
        # TODO format your metrics if necessary
        formatted = wb_metrics.copy()
        formatted["learning_rate"] = current_lr
        wandb.log(formatted)

    eta(
        timings["start"], timings["epoch_start"], time.time(), epoch,
        config.num_epochs
    )

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
