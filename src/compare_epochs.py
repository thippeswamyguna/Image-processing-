"""Evaluate matching epoch checkpoints on the held-out test split."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/chest_xray')
    parser.add_argument('--checkpoint-dir', default='outputs')
    parser.add_argument('--output-dir', default='outputs/epoch_comparison')
    parser.add_argument('--epochs', type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    checkpoint_dir, output_dir = Path(args.checkpoint_dir), Path(args.output_dir)
    history = json.loads((checkpoint_dir / 'history.json').read_text())
    history_by_epoch = {item['epoch']: item for item in history}
    rows = []
    for epoch in args.epochs:
        checkpoint = checkpoint_dir / f'epoch_{epoch:02d}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {checkpoint}; train with at least {epoch} epochs first.')
        result_dir = output_dir / f'epoch_{epoch:02d}'
        command = [sys.executable, '-m', 'src.evaluate', '--data-root', args.data_root,
                   '--checkpoint', str(checkpoint), '--device', args.device,
                   '--output', str(result_dir), '--view-counts', '10']
        print(f'Evaluating epoch {epoch} on test split...', flush=True)
        subprocess.run(command, check=True)
        metrics = json.loads((result_dir / 'test_metrics.json').read_text())
        train_row = history_by_epoch[epoch]
        rows.append({
            'epoch': epoch,
            'train_loss': train_row['train_loss'],
            'val_l1': train_row['val_l1'],
            'psnr_10view': metrics['learned_views10_psnr'],
            'ssim_10view': metrics['learned_views10_ssim'],
            'n_test': metrics['n_test'],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / 'comparison.json'
    destination.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print(f'Comparison written to {destination}')


if __name__ == '__main__':
    main()
