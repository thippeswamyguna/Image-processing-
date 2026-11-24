import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .data import XrayDataset
from .model import ReconNet
from .projector import ParallelBeam, sparse_indices
from .prepare_data import find_dataset_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/chest_xray')
    ap.add_argument('--kagglehub', action='store_true', help='Fetch the dataset through KaggleHub if data-root is unavailable')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--image-size', type=int, default=128)
    ap.add_argument('--views', type=int, default=180)
    ap.add_argument('--sparse-views', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-train', type=int)
    ap.add_argument('--max-val', type=int)
    ap.add_argument('--seed', type=int, default=48)
    ap.add_argument('--output', default='outputs')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    if not data_root.exists():
        if not args.kagglehub:
            raise FileNotFoundError(f"Dataset is not at {data_root}. Pass --kagglehub to fetch it with KaggleHub.")
        try:
            import kagglehub
        except ImportError as exc:
            raise SystemExit('Install KaggleHub first: python -m pip install kagglehub') from exc
        cached = Path(kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia'))
        data_root = find_dataset_root(cached)
        print(f'Using KaggleHub cached dataset at: {data_root}')
    train_ds = XrayDataset(data_root, 'train', args.image_size)
    val_ds = XrayDataset(data_root, 'val', args.image_size)
    if args.max_train:
        train_ids = random.Random(args.seed).sample(range(len(train_ds)), min(args.max_train, len(train_ds)))
        train_ds = Subset(train_ds, train_ids)
    if args.max_val:
        val_ids = random.Random(args.seed + 1).sample(range(len(val_ds)), min(args.max_val, len(val_ds)))
        val_ds = Subset(val_ds, val_ids)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    projector = ParallelBeam(args.image_size, args.views, device)
    idx = sparse_indices(args.views, args.sparse_views, device)
    model = ReconNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pixel_loss = nn.L1Loss()
    history = []
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for x, _ in tqdm(train_dl, desc=f'Epoch {epoch}/{args.epochs} train'):
            x = x.to(device)
            with torch.no_grad():
                full_sino = projector.forward(x)
                y = full_sino[:, idx, :]
                sparse = torch.zeros_like(full_sino); sparse[:, idx, :] = y
                fbp = projector.fbp(sparse)
                fbp = fbp / fbp.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            pred = model(fbp)
            # Data consistency on measured angles, reference fidelity, and TV regularization.
            pred_sino = projector.forward(pred[:, 0])[:, idx, :]
            data_loss = (pred_sino - y).abs().mean() / y.abs().mean().clamp_min(1e-6)
            ref_loss = pixel_loss(pred, x)
            tv = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean() + (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
            loss = 0.25 * data_loss + ref_loss + 0.01 * tv
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            train_loss += loss.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_dl:
                x = x.to(device); full = projector.forward(x)
                sparse = torch.zeros_like(full); sparse[:, idx, :] = full[:, idx, :]
                fbp = projector.fbp(sparse); fbp = fbp / fbp.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                pred = model(fbp)
                val_loss += pixel_loss(pred, x).item()
        row = {'epoch': epoch, 'train_loss': train_loss / max(1, len(train_dl)), 'val_l1': val_loss / max(1, len(val_dl))}
        history.append(row); print(row)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        epoch_checkpoint = {'model': model.state_dict(), 'config': vars(args), 'epoch': epoch, 'val_l1': row['val_l1']}
        torch.save(epoch_checkpoint, out / f'epoch_{epoch:02d}.pt')
        if row['val_l1'] < best:
            best = row['val_l1']
            torch.save(epoch_checkpoint, out / 'best.pt')
    print(f'Best checkpoint: {out / "best.pt"}')


if __name__ == '__main__': main()
