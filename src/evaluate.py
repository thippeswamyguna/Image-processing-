import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import XrayDataset
from .metrics import batch_metrics
from .model import ReconNet
from .projector import ParallelBeam, sparse_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/chest_xray')
    ap.add_argument('--checkpoint', default='outputs/best.pt')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output', default='outputs')
    ap.add_argument('--view-counts', type=int, nargs='+', help='Sparse-view counts to evaluate (defaults to the checkpoint ablation set)')
    args = ap.parse_args()
    device = torch.device(args.device); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']; size, views = cfg['image_size'], cfg['views']
    ds = XrayDataset(args.data_root, 'test', size); dl = DataLoader(ds, batch_size=1, shuffle=False)
    model = ReconNet().to(device); model.load_state_dict(ckpt['model']); model.eval()
    projector = ParallelBeam(size, views, device)
    all_rows = []
    with torch.no_grad():
        for x, paths in dl:
            x = x.to(device); sino = projector.forward(x)
            row = {'path': paths[0]}
            counts = args.view_counts or [max(4, views // 12), max(4, views // 6), cfg['sparse_views']]
            for nviews in sorted(set(counts)):
                idx = sparse_indices(views, nviews, device)
                sparse = torch.zeros_like(sino); sparse[:, idx, :] = sino[:, idx, :]
                fbp = projector.fbp(sparse); fbp = fbp / fbp.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                # A standard sparse-angle ablation. The network is trained at cfg['sparse_views'];
                # learned results at other view counts test view-count generalization.
                pred = model(fbp)
                for name, image in [('fbp', fbp), ('learned', pred)]:
                    p, s = batch_metrics(image, x)
                    row[f'{name}_views{len(idx)}_psnr'] = p[0]; row[f'{name}_views{len(idx)}_ssim'] = s[0]
            all_rows.append(row)
    keys = sorted({k for r in all_rows for k in r})
    with (out / 'test_per_image.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    metric_keys = [k for k in keys if k.endswith('_psnr') or k.endswith('_ssim')]
    summary = {k: sum(float(r[k]) for r in all_rows) / len(all_rows) for k in metric_keys}
    summary['n_test'] = len(all_rows); summary['checkpoint_epoch'] = ckpt['epoch']
    (out / 'test_metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
