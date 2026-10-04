# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class MoDE(nn.Module):
    """
    MoDEConv2d wrapper to simplify usage
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_tasks: int,
        global_task_train_prob: float = 0.2,
        global_task_id: int = 0,
    ) -> None:
        """
        Build a MoDEConv2d wrapper
        Args
            self: MoDE wrapper instance
            in_chans: Number of input channels
            out_chans: Number of output channels
            num_tasks: Number of tasks. An additional global task is added internally, with id 0
            global_task_train_prob: Probability of a task being swapped for the global task id
            global_task_id: Id of the global task
        """
        super().__init__()

        self.num_tasks = num_tasks + 1  # additional global task
        self.global_task_id = global_task_id
        self.global_task_train_prob = global_task_train_prob
        self.mode = MoDEConv2D(
            num_experts=5,
            num_tasks=self.num_tasks,
            in_chan=in_chans,
            out_chan=out_chans,
            kernel_size=5,
            stride=1,
            padding="same",
            conv_type="final",
        )

    def get_task_weights(self, task_id: int | None = None) -> dict[int, np.ndarray]:
        """
        Extract learned gating weights for a specific task or for every task
        Args
            self: MoDE wrapper instance
            task_id: Id of the task, None returns the weights of every task
        Returns
            dict[int, np.ndarray]: Gating weights of shape [num_experts, out_chan] keyed by task id
        """
        weights: dict[int, np.ndarray] = {}

        task_ids: Sequence[int]
        if task_id is None:
            # Get weights for all tasks
            task_ids = range(self.num_tasks)
        else:
            task_ids = [task_id]

        for tid in task_ids:
            # Create one-hot encoding for this task
            # pylint: disable=not-callable
            t = (
                F.one_hot(torch.tensor([tid]), num_classes=self.num_tasks)
                .float()
                .to(next(self.parameters()).device)
            )

            # Get gating weights
            g = self.mode.gate(t)  # [1, num_experts * out_chan]
            g = g.view(
                self.mode.num_experts, self.mode.out_chan
            )  # [num_experts, out_chan]
            g = self.mode.softmax(g.unsqueeze(0)).squeeze(0)  # Apply softmax

            weights[tid] = g.detach().cpu().numpy()

        return weights

    def forward(
        self, x: torch.Tensor, task_id: torch.Tensor, grouped: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input through the experts of the given task ids
        Args
            self: MoDE wrapper instance
            x: Input tensor with shape [B, in_chans, H, W]
            task_id: Task id of every sample with shape [B]
            grouped: Whether the grouped convolution is used instead of one convolution per sample
        Returns
            tuple[torch.Tensor, torch.Tensor]: Output tensor [B, out_chans, H, W] and the used task ids
        """
        # During training, randomly use global task
        if self.training:
            assert task_id.min() >= 0
            assert task_id.max() < self.num_tasks
            b = x.shape[0]
            # Random mask: True for samples that should use global task
            mask = torch.rand(b, device=task_id.device) < self.global_task_train_prob
            # Replace masked task IDs with global task ID
            task_id = task_id.clone()  # Don't modify original
            task_id[mask] = self.global_task_id

        # pylint: disable=not-callable
        task = F.one_hot(task_id, num_classes=self.num_tasks).float().to(x.device)

        x = self.mode(x, task, grouped=grouped)
        return x, task_id


class MoDEConv2D(torch.nn.Module):
    """
    Mixture of Diverse Experts for 2d
    """

    def __init__(
        self,
        num_experts: int,
        num_tasks: int,
        in_chan: int,
        out_chan: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: str = "same",
        conv_type: str = "normal",
    ) -> None:
        """
        Build a mixture of diverse experts convolution
        Args
            self: MoDEConv2D instance
            num_experts: Number of experts, routing expects 5 of them
            num_tasks: Number of tasks the gate can select from
            in_chan: Number of input channels
            out_chan: Number of output channels
            kernel_size: Size of the expert kernels
            stride: Stride of the convolution
            padding: Padding mode passed to F.conv2d
            conv_type: normal applies InstanceNorm and Mish, any other value skips them
        """
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_type = conv_type

        # Expert convolutional kernels
        self.expert_conv5x5_conv = self.gen_conv_kernel(out_chan, in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(out_chan, in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(out_chan, in_chan, 1)

        # Expert pooled convolution kernels
        self.register_buffer("expert_avg3x3_pool", self.gen_avg_pool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(out_chan, in_chan, 1)

        self.register_buffer("expert_avg5x5_pool", self.gen_avg_pool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(out_chan, in_chan, 1)

        # Optional normalization and activation
        self.subsequent_layer: nn.Module
        if self.conv_type == "normal":
            self.subsequent_layer = nn.Sequential(
                nn.InstanceNorm2d(out_chan, affine=True),
                nn.Mish(inplace=True),
            )
        else:
            self.subsequent_layer = nn.Identity()

        # Gating mechanism
        self.gate = nn.Linear(num_tasks, num_experts * out_chan, bias=True)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def gen_conv_kernel(chans_out: int, chans_in: int, k_size: int) -> nn.Parameter:
        """
        Create an initialized expert kernel
        Args
            chans_out: Number of output channels of the kernel
            chans_in: Number of input channels of the kernel
            k_size: Spatial size of the kernel
        Returns
            nn.Parameter: Kernel with shape [chans_out, chans_in, k_size, k_size]
        """
        weight = nn.Parameter(torch.empty(chans_out, chans_in, k_size, k_size))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode="fan_out")
        return weight

    @staticmethod
    def gen_avg_pool_kernel(kernel_size: int) -> torch.Tensor:
        """
        Create an average pooling kernel
        Args
            kernel_size: Spatial size of the kernel
        Returns
            torch.Tensor: Kernel with shape [kernel_size, kernel_size]
        """
        return torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size**2)

    @staticmethod
    def trans_kernel(kernel: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Pad an expert kernel symmetrically to the target spatial size
        Args
            kernel: Kernel to pad
            target_size: Target spatial size of the kernel
        Returns
            torch.Tensor: Padded kernel with spatial size target_size
        """
        pad = (target_size - kernel.shape[2]) // 2
        return F.pad(kernel, [pad, pad, pad, pad])

    def routing(self, g: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Combine the expert kernels weighted by the gate output
        Args
            self: MoDEConv2D instance
            g: Gate output with shape [batch_size, num_experts, out_chan]
            batch_size: Number of samples in the batch
        Returns
            torch.Tensor: Mixed kernels with shape [batch_size, out_chan, in_chan, K, K]
        """
        # Resize and combine expert kernels with gate weights
        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)

        expert_avg3x3 = self.trans_kernel(
            torch.einsum(
                "oihw,hw->oihw", self.expert_avg3x3_conv, self.expert_avg3x3_pool
            ),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum(
            "oihw,hw->oihw", self.expert_avg5x5_conv, self.expert_avg5x5_pool
        )

        weights = []
        for n in range(batch_size):
            w = (
                torch.einsum("oihw,o->oihw", expert_conv5x5, g[n, 0, :])
                + torch.einsum("oihw,o->oihw", expert_conv3x3, g[n, 1, :])
                + torch.einsum("oihw,o->oihw", expert_conv1x1, g[n, 2, :])
                + torch.einsum("oihw,o->oihw", expert_avg3x3, g[n, 3, :])
                + torch.einsum("oihw,o->oihw", expert_avg5x5, g[n, 4, :])
            )
            weights.append(w)
        return torch.stack(weights)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, grouped: bool = True
    ) -> torch.Tensor:
        """
        Convolve the input with the expert kernels mixed by the task gate
        Args
            self: MoDEConv2D instance
            x: Input tensor with shape [B, in_chan, H, W]
            t: One hot task encoding with shape [B, num_tasks]
            grouped: Whether a single grouped convolution is used instead of one per sample
        Returns
            torch.Tensor: Output tensor with shape [B, out_chan, H, W]
        """
        batch_size = x.shape[0]  # batch size
        y: torch.Tensor

        g = self.gate(t)  # [batch_size, num_experts * out_chan]
        g = g.view(batch_size, self.num_experts, self.out_chan)
        g = self.softmax(g)

        w = self.routing(g, batch_size)  # [batch_size, out_chan, in_chan, K, K]

        if grouped:
            x_grouped = x.view(1, batch_size * self.in_chan, x.shape[2], x.shape[3])
            w_grouped = w.view(
                batch_size * self.out_chan,
                self.in_chan,
                self.kernel_size,
                self.kernel_size,
            )
            # pylint: disable=not-callable
            y_grouped = F.conv2d(
                x_grouped,
                w_grouped,
                padding=self.padding,
                stride=self.stride,
                groups=batch_size,
            )
            y = y_grouped.view(batch_size, self.out_chan, x.shape[2], x.shape[3])

        else:
            y = torch.cat(
                [
                    # pylint: disable=not-callable
                    F.conv2d(
                        x[i].unsqueeze(0),
                        w[i],
                        stride=self.stride,
                        padding=self.padding,
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )

        y = self.subsequent_layer(y)

        return y
