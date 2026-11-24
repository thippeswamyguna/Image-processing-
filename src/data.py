"""Chest X-ray image dataset loader."""
from __future__ import annotations
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale


class XrayDataset(Dataset):
    def __init__(self, root: str | Path, split: str, size: int):
        self.root = Path(root) / split
        if not self.root.is_dir():
            raise FileNotFoundError(f"Missing split folder: {self.root}")
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.paths:
            raise FileNotFoundError(f"No image files found under {self.root}")
        self.transform = Compose([Grayscale(num_output_channels=1), Resize((size, size)), ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            return self.transform(im.convert("RGB")), str(self.paths[index])
