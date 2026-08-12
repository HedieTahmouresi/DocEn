"""
Loss functions for both tasks. EnhancementLoss: weighted combination of L1 + (1-SSIM) + Edge L1. CornerRegressionLoss: L1 or L2 on coordinate vectors. HeatmapLoss: MSE on heatmap predictions vs Gaussian targets.
"""
