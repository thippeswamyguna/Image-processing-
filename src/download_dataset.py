"""Download the course dataset using KaggleHub and connect it to this project.

Credentials are read by KaggleHub from the user's Kaggle configuration. Never
paste a token into this source file or commit credential files to git.
"""
import argparse
from pathlib import Path
from .prepare_data import connect_dataset


def main():
    ap = argparse.ArgumentParser(description="Download via KaggleHub and link the dataset into this project")
    ap.add_argument('--data-root', default='data/chest_xray')
    args = ap.parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt") from exc
    cached = Path(kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia'))
    destination = Path(args.data_root)
    connect_dataset(cached, destination)
    print(f'Dataset is ready at: {destination.resolve()}')


if __name__ == '__main__':
    main()
