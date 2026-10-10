# -*- coding: utf-8 -*-
"""
Configuration used for the model
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import torch.optim as optim
import torch.nn as nn

cwd = Path(__file__).resolve().parent


@dataclass
class Config:
    """
    Hyperparameters and options of a training run
    """

    # training
    threads: int | None = 4  # None to disable
    cudnn_bench: bool = True
    lr: float = 1e-2
    optimizer: Callable[..., optim.Optimizer] = optim.AdamW
    criterion: Callable[..., nn.Module] = nn.MSELoss
    scheduler: Callable[..., optim.lr_scheduler.LRScheduler] | None = (
        optim.lr_scheduler.CosineAnnealingLR
    )  # None to disable it
    grad_clip: float = 1.0
    num_epochs: int = 12
    batch: int = 64
    num_workers: int = 4  # if 0 will disable prefetch factor
    prefetch_factor: int = 4
    device: str = "cpu"
    half_p: bool = True

    # wandb
    wb: bool = False
    project_name: str = "Sample"

    # others
    basePath: Path = cwd
    modelDir: Path = cwd / "weights"


config = Config()

if __name__ == "__main__":
    from pprint import pprint

    pprint(config.__dict__)
