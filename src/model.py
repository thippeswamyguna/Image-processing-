"""Compact residual U-Net for FBP refinement."""
import torch
from torch import nn


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.InstanceNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.InstanceNorm2d(cout), nn.ReLU(inplace=True))


class ReconNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.e1, self.e2, self.e3 = block(1, width), block(width, width*2), block(width*2, width*4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(width*4, width*2, 2, stride=2)
        self.d2 = block(width*4, width*2)
        self.up1 = nn.ConvTranspose2d(width*2, width, 2, stride=2)
        self.d1 = block(width*2, width)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, fbp: torch.Tensor) -> torch.Tensor:
        a = self.e1(fbp)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.d2(torch.cat((self.up2(c), b), dim=1))
        e = self.d1(torch.cat((self.up1(d), a), dim=1))
        return (fbp + self.out(e)).clamp(0, 1)
