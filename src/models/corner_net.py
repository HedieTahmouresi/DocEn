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
        allow_dropout: bool = False,
        spatial_pool: str = "max",
        head_norm: bool = True,
    ):
        """
        spatial_pool: "max" (default) or "avg". exp-009 used "avg" and collapsed to a
            near-constant prediction — its errors were identical on the synthetic test set
            and on the real photos, which only happens when the output barely depends on
            the input. Average-pooling 4x4 blocks of a post-BatchNorm ReLU map leaves a
            vector dominated by its per-channel DC term, so most of the positional signal
            is gone before the first Linear ever sees it. That satisfies the letter of
            ADR-007's "no GAP" while doing much of GAP's damage. Max-pooling keeps the
            peak response in each block, which is what a corner-like feature looks like.
            Neither has parameters, so this does not change the state_dict.
        head_norm: BatchNorm1d between the fully-connected layers. The trunk is normalised
            end to end, so the head is the one place a bad scale survives to the
            prediction — the same lesson the output-head initialisation defect taught in
            Phase 04. Permitted under [CON-04] on ADR-005's reading: BatchNorm is a
            normalisation/optimisation layer, not an explicit regulariser.
            This DOES add parameters, so a checkpoint trained with one setting cannot be
            loaded with the other. Loaders default to the legacy values for checkpoints
            whose config predates these keys.
        """
        super().__init__()
        # [CON-04] holds in Phases 04/06; [REQ-38] lifts it in Phase 07, opt-in only.
        if not allow_dropout:
            assert dropout == 0.0, f"[CON-04] Dropout must be 0.0 in Phase 04/06, got {dropout}"

        # Dropout for Approach A belongs between the FC layers, not in the encoder —
        # spec §6: "the fully connected layers are the classic place for Dropout".
        self.encoder = Encoder(in_ch=in_ch, base=base_channels, levels=levels, dropout=0.0)

        # Reduce spatial dimension from 32x32 to 8x8 (preserves 2D spatial layout per ADR-007)
        feat_ch = base_channels * (2 ** (levels - 1))  # 512 for base=64, levels=4
        if spatial_pool == "max":
            self.extra_pool = nn.AdaptiveMaxPool2d((8, 8))
        elif spatial_pool == "avg":
            self.extra_pool = nn.AdaptiveAvgPool2d((8, 8))
        else:
            raise ValueError(f"spatial_pool must be 'max' or 'avg', got {spatial_pool!r}")
        self.flatten_dim = feat_ch * 8 * 8  # 512 * 8 * 8 = 32768

        layers: list = [nn.Linear(self.flatten_dim, 512)]
        if head_norm:
            layers.append(nn.BatchNorm1d(512))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(512, 256))
        if head_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Keep Linear(256, 8) second-to-last so _init_weights can address it as [-2].
        layers.extend([nn.Linear(256, 8), nn.Sigmoid()])
        self.fc_head = nn.Sequential(*layers)

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
        allow_dropout: bool = False,
    ):
        super().__init__()
        # [CON-04] holds in Phases 04/06; [REQ-38] lifts it in Phase 07, opt-in only.
        if not allow_dropout:
            assert dropout == 0.0, f"[CON-04] Dropout must be 0.0 in Phase 04/06, got {dropout}"

        # Bottleneck only (model-specs §4). For Approach B specifically, dropping
        # bottleneck activations forces the network to infer a corner from global page
        # geometry rather than from one memorised local texture — the mechanism worth
        # stating in the report when answering [REQ-39].
        self.encoder = Encoder(in_ch=in_ch, base=base_channels, levels=levels, dropout=dropout)
        self.decoder = Decoder(
            base=base_channels,
            levels=levels,
            out_ch=4,
            out_act="sigmoid",
            upsample=upsample,
            dropout=0.0,
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
