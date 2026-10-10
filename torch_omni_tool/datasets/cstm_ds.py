# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:04:55 2024

@author: Mateo-drr
"""

from typing import Literal, Any

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

from torch_omni_tool.config import Config


class CustomDataset(Dataset[dict[str, Any]]):
    """
    Dataset that returns samples under the data key
    """

    def __init__(self, config: Config) -> None:
        """
        Build the dataset
        Args
            self: Dataset instance
            config: Config holding the batch and dataloader options
        """
        super().__init__()
        self.config = config
        self.data = [torch.rand(8, 16)]

    def __len__(self) -> int:
        """
        Count the samples of the dataset
        Args
            self: Dataset instance
        Returns
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single sample
        Args
            self: Dataset instance
            idx: Index of the sample
        Returns
            dict[str, Any]: Sample with shape [8, 16] under the data key
        """
        data = self.data[idx]
        data = torch.tensor(data)
        return {
            "data": data,
        }


def make_dl(
    config: Config, split: Literal["train", "valid", "test"]
) -> DataLoader[dict[str, Any]]:
    """
    Build the dataloader of a split
    Args
        config: Config holding the batch, worker and prefetch options
        split: Split to build the dataloader for, either train, valid or test
    Returns
        DataLoader[dict[str, Any]]: Dataloader over the dataset of the split
    """
    dataset = CustomDataset(config)
    dataloader: DataLoader[dict[str, Any]]
    match split:
        case "train":
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch,
                pin_memory=True,
                shuffle=True,
                num_workers=config.num_workers,
                prefetch_factor=(
                    config.prefetch_factor if config.num_workers > 0 else None
                ),
            )
        case "valid":
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch,
                pin_memory=True,
                shuffle=False,
                num_workers=config.num_workers,
            )
        case "test":
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch,
                pin_memory=True,
                shuffle=False,
                num_workers=config.num_workers,
            )
        case _:
            raise NotImplementedError

    return dataloader
