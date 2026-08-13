"""U-Net architecture modules and shared encoder-decoder backbone.

Fulfills [REQ-19] (encoder-decoder with skip connections from scratch)
and ADR-005 (4-level U-Net, base=64, BatchNorm, concat skips, sigmoid head).
Enforces [CON-01] (no pre-built architectures) and [CON-04] (dropout == 0.0 in Phase 04/06).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm2d -> ReLU) * 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Encoder(nn.Module):
    """Configurable 4-level U-Net Encoder.

    Level 1: in_ch   -> base      (e.g., 3 -> 64)
    Level 2: base    -> base*2    (64 -> 128)
    Level 3: base*2  -> base*4    (128 -> 256)
    Level 4: base*4  -> base*8    (256 -> 512)
    Bottleneck: base*8 -> base*8  (512 -> 512)
    """

    def __init__(self, in_ch: int = 3, base: int = 64, levels: int = 4):
        super().__init__()
        self.levels = levels
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        curr_ch = in_ch
        for l in range(levels):
            next_ch = base * (2 ** l)
            self.enc_blocks.append(DoubleConv(curr_ch, next_ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_ch = next_ch

        self.bottleneck = DoubleConv(curr_ch, curr_ch)

    def forward(self, x: torch.Tensor):
        """Returns bottleneck_feature, list_of_skips [skip1, skip2, skip3, skip4]."""
        skips = []
        out = x
        for enc, pool in zip(self.enc_blocks, self.pools):
            feat = enc(out)
            skips.append(feat)
            out = pool(feat)

        bottleneck_feat = self.bottleneck(out)
        return bottleneck_feat, skips


class Decoder(nn.Module):
    """Configurable 4-level U-Net Decoder with concat skip connections."""

    def __init__(
        self,
        base: int = 64,
        levels: int = 4,
        out_ch: int = 3,
        out_act: str = "sigmoid",
        upsample: str = "transpose",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dropout == 0.0, f"[CON-04] Dropout must be 0.0 in Phase 04, got {dropout}"

        self.levels = levels
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        curr_ch = base * (2 ** (levels - 1))  # 512 for base=64, levels=4

        for l in reversed(range(levels)):
            skip_ch = base * (2 ** l)  # 512, 256, 128, 64
            out_dec_ch = base * (2 ** (l - 1)) if l > 0 else base  # 256, 128, 64, 64

            # Up-sampling keeps curr_ch or halves it to match skip_ch
            up_out_ch = skip_ch
            if upsample == "transpose":
                up = nn.ConvTranspose2d(curr_ch, up_out_ch, kernel_size=2, stride=2)
            elif upsample == "bilinear":
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(curr_ch, up_out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(up_out_ch),
                    nn.ReLU(inplace=True),
                )
            else:
                raise ValueError(f"Unknown upsample mode: {upsample}")

            # Concatenation: up_out_ch (skip_ch) + skip_ch = 2 * skip_ch
            dec = DoubleConv(up_out_ch + skip_ch, out_dec_ch)

            self.up_convs.append(up)
            self.dec_blocks.append(dec)
            curr_ch = out_dec_ch

        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

        if out_act == "sigmoid":
            self.act = nn.Sigmoid()
        elif out_act == "none" or out_act is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_act: {out_act}")

    def forward(self, bottleneck_feat: torch.Tensor, skips: list) -> torch.Tensor:
        """Forward pass through decoder with concatenated skip connections."""
        out = bottleneck_feat
        # skips are ordered [enc1 (64), enc2 (128), enc3 (256), enc4 (512)]
        # decoder processes them reversed [enc4, enc3, enc2, enc1]
        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            out = up(out)
            out = torch.cat([out, skip], dim=1)
            out = dec(out)

        out = self.head(out)
        out = self.act(out)
        return out


class EnhancementNet(nn.Module):
    """Document Enhancement U-Net Network [REQ-19], ADR-005.

    Transforms a degraded, perspective-rectified document image into a clean scan.
    Input: [N, 3, 512, 512] float32 (standardized)
    Output: [N, 3, 512, 512] float32 in [0, 1]
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_channels: int = 64,
        levels: int = 4,
        out_ch: int = 3,
        upsample: str = "transpose",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dropout == 0.0, f"[CON-04] Dropout must be 0.0 in Phase 04, got {dropout}"

        self.encoder = Encoder(in_ch=in_ch, base=base_channels, levels=levels)
        self.decoder = Decoder(
            base=base_channels,
            levels=levels,
            out_ch=out_ch,
            out_act="sigmoid",
            upsample=upsample,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming Normal for Conv2d and default for BatchNorm2d."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_feat, skips = self.encoder(x)
        out = self.decoder(bottleneck_feat, skips)
        return out


class CornerRegNet(nn.Module):
    """Corner Approach A — Direct Coordinate Regression [REQ-30], ADR-007.

    Shared U-Net encoder followed by FC head predicting 8 normalized corner coordinates.
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

        # Reduce spatial dimension from 32x32 to 8x8 (preserves spatial layout per ADR-007)
        feat_ch = base_channels * (2 ** (levels - 1))  # 512
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

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
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_feat, skips = self.encoder(x)
        heatmaps = self.decoder(bottleneck_feat, skips)
        return heatmaps
