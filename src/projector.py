"""Parallel beam Radon projector wrapper."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ParallelBeam:
    def __init__(self, size: int, views: int, device: torch.device):
        self.size = size
        self.views = views
        self.device = device
        self.angles = torch.linspace(0, torch.pi, views + 1, device=device)[:-1]
        # Differentiable rotate-and-sum parallel-beam projector, implemented in
        # native PyTorch so it works on CPU, CUDA, and Apple MPS.
        self.grid = self._make_grid(size, device)

    @staticmethod
    def _make_grid(size: int, device: torch.device) -> torch.Tensor:
        coords = torch.linspace(-1, 1, size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        return torch.stack((xx, yy), dim=-1)

    def _rotate(self, image: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        # grid_sample maps output coordinates to input coordinates. Rotation
        # sign has no effect on the parallel-beam angle set [0, pi).
        c, s = torch.cos(angle), torch.sin(angle)
        theta = torch.stack((c, -s, torch.zeros_like(c), s, c, torch.zeros_like(c))).reshape(1, 2, 3)
        theta = theta.expand(image.shape[0], -1, -1)
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        return F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(1)
        # Sum rotated images along rows, yielding (batch, views, detectors).
        projections = [self._rotate(image, angle).sum(dim=-2).squeeze(1) for angle in self.angles]
        return torch.stack(projections, dim=1)

    def fbp(self, sinogram: torch.Tensor) -> torch.Tensor:
        # Ramp-filter detector profiles before smearing them back over the
        # image. This is a discrete parallel-beam filtered backprojection.
        freq = torch.fft.rfftfreq(self.size, d=1.0, device=sinogram.device)
        ramp = (2.0 * freq).view(1, 1, -1)
        filtered = torch.fft.irfft(torch.fft.rfft(sinogram, dim=-1) * ramp, n=self.size, dim=-1)
        image = torch.zeros((sinogram.shape[0], 1, self.size, self.size), device=sinogram.device, dtype=sinogram.dtype)
        for j, angle in enumerate(self.angles[:filtered.shape[1]]):
            strip = filtered[:, j, :].unsqueeze(1).unsqueeze(2).expand(-1, 1, self.size, -1)
            image = image + self._rotate(strip, -angle)
        return image * (torch.pi / max(1, sinogram.shape[1]))


def sparse_indices(total_views: int, sparse_views: int, device: torch.device) -> torch.Tensor:
    if sparse_views < 1 or sparse_views > total_views:
        raise ValueError("sparse_views must be between 1 and total_views")
    return torch.linspace(0, total_views - 1, sparse_views, device=device).round().long().unique()
