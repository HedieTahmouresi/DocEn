"""
CPU/GPU benchmark script for 512x512 ConvNet steps.
Measures seconds per step for forward+backward pass at 512x512.
"""

import time
import torch
import torch.nn as nn


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def run_benchmark(batch_size: int = 2, steps: int = 50, device: str = "cpu"):
    print(f"Running benchmark on {device.upper()} with batch size {batch_size} for {steps} steps at 512x512...")
    model = SimpleConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    inputs = torch.randn(batch_size, 3, 512, 512, device=device)
    targets = torch.randn(batch_size, 3, 512, 512, device=device)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    start_time = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    total_time = end_time - start_time
    sec_per_step = total_time / steps
    print(f"Benchmark complete. Total time: {total_time:.2f}s | {sec_per_step*1000:.2f} ms/step ({sec_per_step:.4f} s/step)")
    return sec_per_step


if __name__ == "__main__":
    run_benchmark(batch_size=2, steps=50, device="cpu")

