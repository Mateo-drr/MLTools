# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 22:54:53 2025

@author: Mateo-drr
"""

from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
Sample usage

# encoder
self.encoder_block1 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels, self.in_channels * self.mult_chan)
self.encoder_block2 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan, self.in_channels * self.mult_chan * 2)
self.encoder_block3 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 2, self.in_channels * self.mult_chan * 4)
self.encoder_block4 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 4, self.in_channels * self.mult_chan * 8)

# bottle
self.bottle_block = MoDESubNet2Conv(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 8, self.in_channels * self.mult_chan * 16)

# decoder
self.decoder_block4 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 16, self.in_channels * self.mult_chan * 8)
self.decoder_block3 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 8, self.in_channels * self.mult_chan * 4)
self.decoder_block2 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 4, self.in_channels * self.mult_chan * 2)
self.decoder_block1 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 2, self.in_channels * self.mult_chan)

# conv out
self.conv_out = MoDEConv(self.num_experts, self.num_tasks, self.mult_chan, self.out_channels, kernel_size=5, padding='same', conv_type='final')

def forward(self, x, t):
    # task embedding
    task_emb = self.one_hot_task_embedding(t)

    # encoding
    print(x.shape)
    x, x_skip1 = self.encoder_block1(x, task_emb)
    x, x_skip2 = self.encoder_block2(x, task_emb)
    x, x_skip3 = self.encoder_block3(x, task_emb)
    x, x_skip4 = self.encoder_block4(x, task_emb)

    # bottle
    x = self.bottle_block(x, task_emb)

    # decoding
    x = self.dropout_latent(x)
    x = self.decoder_block4(x, x_skip4, task_emb)
    x = self.decoder_block3(x, x_skip3, task_emb)
    x = self.decoder_block2(x, x_skip2, task_emb)
    x = self.decoder_block1(x, x_skip1, task_emb)
    outputs = self.conv_out(x, task_emb)
"""


def one_hot_task_embedding(self: Any, task_id: torch.Tensor) -> torch.Tensor:
    """
    Turn task ids into one hot encodings
    Args
        self: Module that owns the num_tasks and device attributes
        task_id: Task id of every sample with shape [B]
    Returns
        torch.Tensor: One hot task encoding with shape [B, num_tasks]
    """
    n_samples = task_id.shape[0]
    task_embedding = torch.zeros((n_samples, self.num_tasks))
    for i in range(n_samples):
        task_embedding[i, task_id[i]] = 1
    return task_embedding.to(self.device)


class MoDEEncoderBlock(torch.nn.Module):
    """
    Encoder block that increases the channels and halves the sequence length
    """

    def __init__(self, num_experts: int, num_tasks: int, in_chan: int, out_chan: int) -> None:
        """
        Build a MoDE encoder block
        Args
            self: Encoder block instance
            num_experts: Number of experts, routing expects 5 of them
            num_tasks: Number of tasks the gate can select from
            in_chan: Number of input channels
            out_chan: Number of output channels
        """
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv_more = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)
        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv1d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm1d(out_chan, affine=True),
            torch.nn.Mish(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convolve the input and downsample it by two
        Args
            self: Encoder block instance
            x: Input tensor with shape [B, in_chan, L]
            t: One hot task encoding with shape [B, num_tasks]
        Returns
            tuple[torch.Tensor, torch.Tensor]: Downsampled tensor and skip tensor with out_chan channels
        """
        x_skip = self.conv_more(x, t)
        x = self.conv_down(x_skip)
        return x, x_skip


class MoDEDecoderBlock(torch.nn.Module):
    """
    Decoder block that halves the channels and doubles the sequence length
    """

    def __init__(self, num_experts: int, num_tasks: int, in_chan: int, out_chan: int) -> None:
        """
        Build a MoDE decoder block
        Args
            self: Decoder block instance
            num_experts: Number of experts, routing expects 5 of them
            num_tasks: Number of tasks the gate can select from
            in_chan: Number of input channels, expected to be 2 * out_chan
            out_chan: Number of output channels
        """
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.convt = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_chan, out_chan, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm1d(out_chan, affine=True),
            torch.nn.Mish(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)

    def forward(
        self, x: torch.Tensor, x_skip: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Upsample the input, concatenate the skip connection and convolve the result
        Args
            self: Decoder block instance
            x: Input tensor with shape [B, in_chan, L]
            x_skip: Skip tensor from the matching encoder block
            t: One hot task encoding with shape [B, num_tasks]
        Returns
            torch.Tensor: Output tensor with shape [B, out_chan, 2L]
        """
        x = self.convt(x)
        x_cat: torch.Tensor = torch.cat((x_skip, x), 1)  # concatenate
        x_cat = self.conv_less(x_cat, t)
        return x_cat


class MoDESubNet2Conv(torch.nn.Module):
    """
    Two stacked MoDE convolutions that keep the sequence length
    """

    def __init__(self, num_experts: int, num_tasks: int, n_in: int, n_out: int) -> None:
        """
        Build a subnet of two MoDE convolutions
        Args
            self: Subnet instance
            num_experts: Number of experts, routing expects 5 of them
            num_tasks: Number of tasks the gate can select from
            n_in: Number of input channels
            n_out: Number of output channels
        """
        super().__init__()
        self.conv1 = MoDEConv(
            num_experts, num_tasks, n_in, n_out, kernel_size=5, padding="same"
        )
        self.conv2 = MoDEConv(
            num_experts, num_tasks, n_out, n_out, kernel_size=5, padding="same"
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply both MoDE convolutions in sequence
        Args
            self: Subnet instance
            x: Input tensor with shape [B, n_in, L]
            t: One hot task encoding with shape [B, num_tasks]
        Returns
            torch.Tensor: Output tensor with shape [B, n_out, L]
        """
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        return x


class MoDEConv(torch.nn.Module):
    """
    Mixture of Diverse Experts convolution for 1d data, optionally causal
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
        causal: bool = False,
    ) -> None:
        """
        Build a mixture of diverse experts convolution
        Args
            self: MoDEConv instance
            num_experts: Number of experts, routing expects 5 of them
            num_tasks: Number of tasks the gate can select from
            in_chan: Number of input channels
            out_chan: Number of output channels
            kernel_size: Size of the expert kernels
            stride: Stride of the convolution
            padding: Padding mode passed to F.conv1d
            conv_type: normal applies InstanceNorm and Mish, final skips them
            causal: Whether the convolution only looks at past samples
        """
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding
        self.causal = causal

        self.expert_conv5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer("expert_avg3x3_pool", self.gen_avgpool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer("expert_avg5x5_pool", self.gen_avgpool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)

        assert self.conv_type in ["normal", "final"]
        self.subsequent_layer: nn.Module
        if self.conv_type == "normal":
            self.subsequent_layer = torch.nn.Sequential(
                nn.InstanceNorm1d(out_chan, affine=True),
                torch.nn.Mish(inplace=True),
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def gen_conv_kernel(self, chans_out: int, chans_in: int, k_size: int) -> nn.Parameter:
        """
        Create an initialized expert kernel
        Args
            self: MoDEConv instance
            chans_out: Number of output channels of the kernel
            chans_in: Number of input channels of the kernel
            k_size: Size of the kernel
        Returns
            nn.Parameter: Kernel with shape [chans_out, chans_in, k_size]
        """
        # For 1D convolution, kernel shape is (Co, Ci, K)
        weight = torch.nn.Parameter(torch.empty(chans_out, chans_in, k_size))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode="fan_out")
        return weight

    def gen_avgpool_kernel(self, k_size: int) -> torch.Tensor:
        """
        Create an average pooling kernel
        Args
            self: MoDEConv instance
            k_size: Size of the kernel
        Returns
            torch.Tensor: Kernel with shape [k_size]
        """
        # For 1D convolution, kernel shape is (K)
        weight = torch.ones(k_size).mul(1.0 / k_size)
        return weight

    def trans_kernel(self, kernel: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Pad an expert kernel to the target size, only on the left when causal
        Args
            self: MoDEConv instance
            kernel: Kernel to pad
            target_size: Target size of the kernel
        Returns
            torch.Tensor: Padded kernel with size target_size
        """
        # For 1D convolution, padding is only applied to the last dimension
        if self.causal:
            pad = target_size - kernel.shape[2]
            return F.pad(kernel, [pad, 0])
        else:
            pad = (target_size - kernel.shape[2]) // 2
            return F.pad(kernel, [pad, pad])

    def routing(self, g: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Combine the expert kernels weighted by the gate output
        Args
            self: MoDEConv instance
            g: Gate output with shape [batch_size, num_experts, out_chan]
            batch_size: Number of samples in the batch
        Returns
            torch.Tensor: Mixed kernels with shape [batch_size, out_chan, in_chan, K]
        """
        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)

        # For 1D convolution, we use einsum with appropriate dimensions
        expert_avg3x3 = self.trans_kernel(
            torch.einsum(
                "oiw,w->oiw", self.expert_avg3x3_conv, self.expert_avg3x3_pool
            ),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum(
            "oiw,w->oiw", self.expert_avg5x5_conv, self.expert_avg5x5_pool
        )

        weights: list[torch.Tensor] = []
        for n in range(batch_size):
            weight_nth_sample = (
                torch.einsum("oiw,o->oiw", expert_conv5x5, g[n, 0, :])
                + torch.einsum("oiw,o->oiw", expert_conv3x3, g[n, 1, :])
                + torch.einsum("oiw,o->oiw", expert_conv1x1, g[n, 2, :])
                + torch.einsum("oiw,o->oiw", expert_avg3x3, g[n, 3, :])
                + torch.einsum("oiw,o->oiw", expert_avg5x5, g[n, 4, :])
            )
            weights.append(weight_nth_sample)
        weights_tensor = torch.stack(weights)

        return weights_tensor

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convolve the input with the expert kernels mixed by the task gate
        Args
            self: MoDEConv instance
            x: Input tensor with shape [B, in_chan, L]
            t: One hot task encoding with shape [B, num_tasks]
        Returns
            torch.Tensor: Output tensor with shape [B, out_chan, L]
        """
        batch_size = x.shape[0]  # batch size

        g = self.gate(t)  # [b, x out channels * experts]
        g = g.view((batch_size, self.num_experts, self.out_chan))  # [b,experts,x out channels]
        g = self.softmax(g)

        w = self.routing(g, batch_size)  # [b,x out chann, 1, 5] mix expert kernels

        y: torch.Tensor
        if self.causal:  # handle padding to not look into i+1 items
            pad_left = self.kernel_size - 1
            x = x.transpose(1, 2)  # makes [b,seqlen,dmodel] to [b,dmodel,seqlen]
            """
            processing everything as one batch
            """

            # Pad the entire batch at once
            x_padded = F.pad(x, (pad_left, 0), "constant", 0)  # Shape: [48, 768, 387]

            # Create a single large batch for all examples
            x_batched = x_padded.view(
                1, -1, x_padded.shape[2]
            )  # Shape: [1, 48*768, 387]

            # Reshape and concatenate all weights
            w_batched = w.view(
                batch_size * self.out_chan, self.in_chan, self.kernel_size
            )  # Shape: [48*768, 768, 5]

            # Use grouped convolution
            y_batched = F.conv1d(
                x_batched,  # [1, 48*768, 387]
                w_batched,  # [48*768, 768, 5]
                bias=None,
                stride=1,
                padding=0,
                groups=batch_size,  # 48 groups
            )

            # Reshape result back to original batch format
            y = y_batched.view(batch_size, self.out_chan, -1)  # Shape: [48, 768, 383]

        else:
            # non causal case (this is not optimized as the causal code but is equal)
            if self.training:
                outputs: list[torch.Tensor] = []
                for i in range(batch_size):
                    outputs.append(
                        F.conv1d(
                            x[i].unsqueeze(0), w[i], bias=None, stride=1, padding="same"
                        )
                    )
                y = torch.cat(outputs, dim=0)
            else:
                y = F.conv1d(x, w[0], bias=None, stride=1, padding="same")

        y = self.subsequent_layer(y)

        if self.causal:
            y = y.transpose(1, 2)

        return y
