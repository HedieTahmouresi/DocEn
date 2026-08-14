"""Corner Detection Network Architectures (Approach A & Approach B).

Fulfills [REQ-30] (both coordinate regression and heatmap regression implementations)
and ADR-007 (fair comparison, no GAP for Approach A, shared U-Net backbone).
Enforces [CON-01] (no pre-built architectures), [CON-02] (no pretrained weights),
and [CON-04] (zero dropout, weight_decay=0.0).
"""

import torch
import torch.nn as nn
from src.models.unet import Encoder, Decoder, init_relu_trunk, init_sigmoid_head


class CornerRegNet(nn.Module):
    """Corner Approach A — Direct Coordinate Regression [REQ-30], ADR-007.

    Shared U-Net encoder followed by spatial reduction to 8x8 (NO Global Average Pooling
    per ADR-007) -> FC layers -> 8 normalized corner coordinates in [0, 1].
    Output: [N, 8] float32 in [0, 1], order: [x0, y0, x1, y1, x2, y2, x3, y3] (TL, TR, BR, BL).
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_channels: int = 64,
        levels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dropout == 0.0, f"[CON-04] Dropout must be 0.0 in Phase 04/06, got {dropout}"

        self.encoder = Encoder(in_ch=in_ch, base=base_channels, levels=levels)

        # Reduce spatial dimension from 32x32 to 8x8 (preserves 2D spatial layout per ADR-007)
        feat_ch = base_channels * (2 ** (levels - 1))  # 512 for base=64, levels=4
        self.extra_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten_dim = feat_ch * 8 * 8  # 512 * 8 * 8 = 32768

        self.fc_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming for the ReLU trunk and hidden FC layers, Xavier for the sigmoid head.

        ADR-007 §2 commits to a fair comparison between Approach A and Approach B.
        A saturated coordinate head would hand Approach B the win on an artefact of
        initialisation rather than on the property the spec is asking about.
        """
        init_relu_trunk(self)
        init_sigmoid_head(self.fc_head[-2])  # Linear(256, 8) before Sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_feat, _ = self.encoder(x)
        pooled = self.extra_pool(bottleneck_feat)
        flat = pooled.view(pooled.size(0), -1)
        coords = self.fc_head(flat)
        return coords


class CornerHeatmapNet(nn.Module):
    """Corner Approach B — Heatmap Regression [REQ-30], ADR-008.

    Shared U-Net encoder-decoder predicting 4 Gaussian heatmaps for document corners.
    Output: [N, 4, 512, 512] float32 in [0, 1], channels: TL, TR, BR, BL.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_channels: int = 64,
        levels: int = 4,
        upsample: str = "transpose",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dropout == 0.0, f"[CON-04] Dropout must be 0.0 in Phase 04/06, got {dropout}"

        self.encoder = Encoder(in_ch=in_ch, base=base_channels, levels=levels)
        self.decoder = Decoder(
            base=base_channels,
            levels=levels,
            out_ch=4,
            out_act="sigmoid",
            upsample=upsample,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming for the ReLU trunk, Xavier for the sigmoid heatmap head."""
        init_relu_trunk(self)
        init_sigmoid_head(self.decoder.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_feat, skips = self.encoder(x)
        heatmaps = self.decoder(bottleneck_feat, skips)
        return heatmaps
