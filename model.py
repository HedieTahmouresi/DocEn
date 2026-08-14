"""Model entry point [REQ-20].

Exposes all three network architectures and shared backbone machinery:
- EnhancementNet: Document enhancement network (4-level U-Net)
- CornerRegNet: Corner Approach A (direct coordinate regression)
- CornerHeatmapNet: Corner Approach B (heatmap regression)
- Encoder, Decoder, DoubleConv: Shared backbone modules

Constraints enforced:
- [CON-01] No pre-built architectures
- [CON-02] No pretrained weights
- [CON-04] Dropout == 0.0 in Phase 04 and Phase 06
"""

from src.models.unet import (
    DoubleConv,
    Encoder,
    Decoder,
    EnhancementNet,
    CornerRegNet,
    CornerHeatmapNet,
)

__all__ = [
    "DoubleConv",
    "Encoder",
    "Decoder",
    "EnhancementNet",
    "CornerRegNet",
    "CornerHeatmapNet",
]
