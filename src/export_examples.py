"""Save held-out reference, sinogram, FBP, and model comparison figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import XrayDataset
from .model import ReconNet
from .projector import ParallelBeam, sparse_indices


def main():
    root, size = Path('data/chest_xray'), 64
    dataset = XrayDataset(root, 'test', size)
    indices = [next(i for i, p in enumerate(dataset.paths) if p.parent.name == label)
               for label in ('NORMAL', 'PNEUMONIA')]
    checkpoint = torch.load('outputs/best.pt', map_location='cpu', weights_only=False)
    model = ReconNet().cpu().eval()
    model.load_state_dict(checkpoint['model'])
    projector = ParallelBeam(size, checkpoint['config']['views'], torch.device('cpu'))
    view_ids = sparse_indices(checkpoint['config']['views'], 10, torch.device('cpu'))
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.2), constrained_layout=True)
    titles = ('Reference X-ray', '10-view sinogram', 'Filtered backprojection', 'Residual U-Net')
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11, weight='bold')
    with torch.no_grad():
        for row, i in enumerate(indices):
            image, path = dataset[i]
            image = image.unsqueeze(0)
            sinogram = projector.forward(image)
            sparse = torch.zeros_like(sinogram)
            sparse[:, view_ids, :] = sinogram[:, view_ids, :]
            fbp = projector.fbp(sparse)
            fbp = fbp / fbp.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            prediction = model(fbp)
            panels = (image[0, 0].numpy(), sparse[0].numpy().T,
                      fbp[0, 0].numpy(), prediction[0, 0].numpy())
            for col, panel in enumerate(panels):
                axes[row, col].imshow(panel, cmap='gray', aspect='auto' if col == 1 else 'equal',
                                      vmin=0 if col != 1 else None, vmax=1 if col != 1 else None)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            axes[row, 0].set_ylabel(Path(path).parent.name, fontsize=10, weight='bold')
    fig.suptitle('Held-out examples · 64×64 synthetic parallel-beam reconstruction', fontsize=14, weight='bold')
    output = Path('outputs/reconstruction_examples.png')
    fig.savefig(output, dpi=180, bbox_inches='tight')
    print(output.resolve())


if __name__ == '__main__':
    main()
