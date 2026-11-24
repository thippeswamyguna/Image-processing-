import math
import torch
from skimage.metrics import structural_similarity


def batch_metrics(pred: torch.Tensor, target: torch.Tensor):
    p = pred.detach().clamp(0, 1).cpu().numpy()[:, 0]
    t = target.detach().clamp(0, 1).cpu().numpy()[:, 0]
    mse = ((p - t) ** 2).mean(axis=(1, 2))
    psnr = [float('inf') if m == 0 else 10 * math.log10(1.0 / float(m)) for m in mse]
    ssim = [structural_similarity(a, b, data_range=1.0) for a, b in zip(p, t)]
    return psnr, ssim
