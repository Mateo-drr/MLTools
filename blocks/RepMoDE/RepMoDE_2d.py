
import torch
from torch import nn
import torch.nn.functional as F
import math

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
    ):
        """
        Args:
            in_chans: number of input channels
            out_chans: number of output channels
            num_tasks: number of tasks. An additional global task is added
             internally, with id 0
            global_task_train_prob: probability of a task being swapped for the
             global task id
            global_task_id: id of the global task
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

    def get_task_weights(self, task_id=None):
        """
        Extract learned gating weights for specific task(s)
        None gives all task weights
        """
        weights = {}

        if task_id is None:
            # Get weights for all tasks
            task_ids = range(self.num_tasks)
        else:
            task_ids = [task_id]

        for tid in task_ids:
            # Create one-hot encoding for this task
            # pylint: disable=not-callable
            t = F.one_hot(
                torch.tensor([tid]), num_classes=self.num_tasks
            ).float().to(
                next(self.parameters()).device
            )

            # Get gating weights
            g = self.mode.gate(t)  # [1, num_experts * out_chan]
            g = g.view(self.mode.num_experts,
                       self.mode.out_chan)  # [num_experts, out_chan]
            g = self.mode.softmax(g.unsqueeze(0)).squeeze(0)  # Apply softmax

            weights[tid] = g.detach().cpu().numpy()

        return weights

    def forward(self, x, task_id, grouped=True):

        # During training, randomly use global task
        if self.training:
            assert task_id.min() >= 0
            assert task_id.max() < self.num_tasks
            b = x.shape[0]
            # Random mask: True for samples that should use global task
            mask = torch.rand(
                b, device=task_id.device
            ) < self.global_task_train_prob
            # Replace masked task IDs with global task ID
            task_id = task_id.clone()  # Don't modify original
            task_id[mask] = self.global_task_id

        # pylint: disable=not-callable
        task = F.one_hot(
            task_id, num_classes=self.num_tasks
        ).float().to(x.device)

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
    ):
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
        self.register_buffer(
            "expert_avg3x3_pool", self.gen_avg_pool_kernel(3)
        )
        self.expert_avg3x3_conv = self.gen_conv_kernel(out_chan, in_chan, 1)

        self.register_buffer(
            "expert_avg5x5_pool", self.gen_avg_pool_kernel(5)
        )
        self.expert_avg5x5_conv = self.gen_conv_kernel(out_chan, in_chan, 1)

        # Optional normalization and activation
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
    def gen_conv_kernel(chans_out, chans_in, k_size):
        weight = nn.Parameter(torch.empty(chans_out, chans_in, k_size, k_size))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode="fan_out")
        return weight

    @staticmethod
    def gen_avg_pool_kernel(kernel_size):
        return torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size ** 2)

    @staticmethod
    def trans_kernel(kernel, target_size):
        pad = (target_size - kernel.shape[2]) // 2
        return F.pad(kernel, [pad, pad, pad, pad])

    def routing(self, g, batch_size):
        # Resize and combine expert kernels with gate weights
        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(
            self.expert_conv3x3_conv, self.kernel_size
        )
        expert_conv1x1 = self.trans_kernel(
            self.expert_conv1x1_conv, self.kernel_size
        )

        expert_avg3x3 = self.trans_kernel(
            torch.einsum(
                "oihw,hw->oihw", self.expert_avg3x3_conv,
                self.expert_avg3x3_pool
            ),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum(
            "oihw,hw->oihw", self.expert_avg5x5_conv,
            self.expert_avg5x5_pool
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

    def forward(self, x, t, grouped=True):
        batch_size = x.shape[0]  # batch size

        g = self.gate(t)  # [batch_size, num_experts * out_chan]
        g = g.view(
            batch_size, self.num_experts, self.out_chan
        )
        g = self.softmax(g)

        w = self.routing(g, batch_size)  # [batch_size, out_chan, in_chan, K, K]

        if grouped:
            x_grouped = x.view(
                1, batch_size * self.in_chan, x.shape[2], x.shape[3]
            )
            w_grouped = w.view(batch_size * self.out_chan, self.in_chan,
                               self.kernel_size, self.kernel_size)
            # pylint: disable=not-callable
            y_grouped = F.conv2d(
                x_grouped,
                w_grouped,
                padding=self.padding,
                stride=self.stride,
                groups=batch_size
            )
            y = y_grouped.view(
                batch_size, self.out_chan, x.shape[2], x.shape[3]
            )

        else:
            y = torch.cat(
                [
                    # pylint: disable=not-callable
                    F.conv2d(
                        x[i].unsqueeze(0),
                        w[i],
                        stride=self.stride,
                        padding=self.padding
                    )
                    for i in range(batch_size)
                ], dim=0)

        return self.subsequent_layer(y)
