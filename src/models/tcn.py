"""
tcn.py
------
Temporal Convolutional Network (TCN) for axle pulse detection.

Based on:  Bai, S., Kolter, J. Z., & Koltun, V. (2018).
           "An Empirical Evaluation of Generic Convolutional and Recurrent
            Networks for Sequence Modeling."

Modifications for this task
---------------------------
* Non-causal (symmetric) padding — the full bridge-crossing signal is known
  at inference time, so we can use both past and future context.
* Dilation series: 1, 2, 4, 8, 16, 32  →  effective receptive field covers
  the entire 1300-sample sequence.
* Output head: Conv1d(num_channels, 1, 1) → raw logits shape [B, 1300].

Input : [B, 1, 1300]
Output: [B, 1300]   (logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dilated residual block (non-causal, symmetric padding)
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """
    Two-layer dilated Conv1d residual block with weight norm and dropout.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int   (should be odd for symmetric padding)
    dilation : int
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        pad = dilation * (kernel_size - 1) // 2  # keeps sequence length constant

        self.conv1 = nn.utils.parametrize.register_parametrization if False else \
            nn.utils.weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          dilation=dilation, padding=pad)
            )
        self.bn1    = nn.BatchNorm1d(out_channels)
        self.relu1  = nn.ReLU(inplace=True)
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=pad)
        )
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        # 1×1 projection for skip connection if channels differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))

        return F.relu(out + residual, inplace=True)


# ---------------------------------------------------------------------------
# Full TCN
# ---------------------------------------------------------------------------

class AxleTCN(nn.Module):
    """
    Stacked dilated-residual TCN for sequence-to-sequence axle detection.

    Parameters
    ----------
    in_channels : int       Input channels (1 for a single strain signal).
    num_channels : int      Hidden channels used in every TCN block.
    kernel_size : int       Convolution kernel size (odd integer).
    num_blocks : int        Number of residual blocks. Dilations: 1,2,4,...,2^(n-1).
    dropout : float         Dropout in each block.

    With kernel_size=3 and num_blocks=10:
      Receptive field = 1 + 2*(3-1)*(2^10 - 1) = 4095 samples → covers 1300.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_channels: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            in_ch  = in_channels if i == 0 else num_channels
            out_ch = num_channels
            blocks.append(
                TCNBlock(in_ch, out_ch, kernel_size=kernel_size,
                         dilation=dilation, dropout=dropout)
            )
        self.network = nn.Sequential(*blocks)

        # Output head: 1×1 conv → logits
        self.head = nn.Conv1d(num_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 1, L]

        Returns
        -------
        [B, L]  raw logits
        """
        out = self.network(x)          # [B, num_channels, L]
        out = self.head(out)           # [B, 1, L]
        return out.squeeze(1)          # [B, L]

    def receptive_field(self) -> int:
        """Compute the theoretical receptive field of the network."""
        num_blocks  = len(self.network)
        kernel_size = self.network[0].conv1.weight.shape[-1]
        rf = 1 + sum(2 * (kernel_size - 1) * (2 ** i) for i in range(num_blocks))
        return rf


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = AxleTCN()
    x = torch.randn(4, 1, 1300)
    y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")   # expect [4, 1300]
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(f"Receptive field     : {model.receptive_field()} samples")
