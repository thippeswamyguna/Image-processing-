import argparse
import json
from pathlib import Path
import shutil
from .data import XrayDataset


def find_dataset_root(source: Path) -> Path:
    """Find the extracted root that directly contains train/val/test."""
    candidates = [source, *source.glob('**/chest_xray'), *source.glob('**/chest-xray-pneumonia')]
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ('train', 'val', 'test')):
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find train/val/test beneath KaggleHub path: {source}")


def connect_dataset(source: Path, root: Path) -> None:
    source = find_dataset_root(source.expanduser())
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        root.unlink()
    elif root.exists():
        raise FileExistsError(f"Destination already exists: {root}. Remove it or choose another --data-root.")
    try:
        root.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, root)
    print(f'Dataset connected: {source} -> {root}')


def main():
    ap = argparse.ArgumentParser(description="Validate dataset and create image manifest")
    ap.add_argument('--source', help='KaggleHub cache path printed by dataset_download')
    ap.add_argument('--data-root', default='data/chest_xray')
    ap.add_argument('--image-size', type=int, default=128)
    args = ap.parse_args()
    root = Path(args.data_root)
    if args.source:
        connect_dataset(Path(args.source), root)
    manifest = {}
    for split in ('train', 'val', 'test'):
        ds = XrayDataset(root, split, args.image_size)
        manifest[split] = [str(p.relative_to(root)) for p in ds.paths]
        print(f'{split}: {len(ds)} images')
    out = root / 'manifest.json'
    out.write_text(json.dumps(manifest, indent=2))
    print(f'Manifest: {out}')


if __name__ == '__main__':
    main()
