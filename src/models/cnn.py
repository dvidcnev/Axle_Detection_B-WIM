"""
cnn.py
------
1-D U-Net fully-convolutional network for axle pulse detection.

Architecture
------------
Encoder: 4 × (Conv1d → BatchNorm → ReLU → MaxPool2)
         Channels: 1 → 32 → 64 → 128 → 256

Bottleneck: Conv1d 256 → 256

Decoder: 4 × (Upsample × 2 → concat skip → Conv1d → BN → ReLU)
         Channels: 256 → 128 → 64 → 32 → 32

Head: Conv1d(32, 1, kernel_size=1)  — raw logits, shape [B, 1, 1300]

Input : [B, 1, 1300]  (1 channel, 1300 time-steps)
Output: [B, 1300]     (logits — apply sigmoid for probabilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class AxleUNet(nn.Module):
    """
    1-D U-Net for sequence-to-sequence axle pulse detection.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for a single strain signal).
    base_filters : int
        Number of filters in the first encoder block. Doubles each block.
    depth : int
        Number of encoder/decoder levels (default 4).
    dropout : float
        Dropout probability applied in the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.depth = depth

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(ConvBnRelu(ch, out_ch))
            self.pools.append(nn.MaxPool1d(2))
            ch = out_ch

        # --- Bottleneck ---
        bottleneck_ch = base_filters * (2 ** depth)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(ch, bottleneck_ch),
            nn.Dropout(dropout),
        )
        ch = bottleneck_ch

        # --- Decoder ---
        self.upsamples = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = base_filters * (2 ** i)
            out_ch  = skip_ch
            self.upsamples.append(nn.Upsample(scale_factor=2, mode="linear", align_corners=False))
            self.decoders.append(ConvBnRelu(ch + skip_ch, out_ch))
            ch = out_ch

        # --- Output head ---
        self.head = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 1, L]

        Returns
        -------
        [B, L]  raw logits
        """
        original_len = x.shape[-1]

        # Encoder pass — collect skip connections
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            # Handle size mismatch (if input length is not a power of 2)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        logits = self.head(x)  # [B, 1, L]

        # Final size guarantee
        if logits.shape[-1] != original_len:
            logits = F.interpolate(logits, size=original_len, mode="linear", align_corners=False)

        return logits.squeeze(1)  # [B, L]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = AxleUNet()
    x = torch.randn(4, 1, 1300)
    y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")   # expect [4, 1300]
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
