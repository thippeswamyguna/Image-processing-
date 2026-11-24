# Physics-Informed Neural Networks for CT Reconstruction from Sparse Views

**Student:** Kolakanuru Thippeswamy · **Roll No / ID:** 48 / `20241CAI0055`

This project explores sparse-view reconstruction in a controlled **synthetic 2D experiment**. The Kaggle chest X-rays are treated as reference images; simulated parallel-beam Radon projections are reduced to sparse angles. Filtered backprojection (FBP) is compared with a physics-informed residual U-Net trained with image-reference, projection-consistency, and total-variation losses.

> These X-rays are not CT slices or calibrated attenuation measurements. This demonstration does not reconstruct clinical CT volumes and is not for diagnosis.

## Project website

The repository includes a responsive project overview website in `index.html`. Once GitHub Pages is enabled for this repository, it presents the project overview, original-versus-reconstruction examples, measured results, epoch comparison, and download links.

The interactive image-upload demo is `app.py`. To run it locally, install the requirements and start Streamlit:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

It loads the included trained checkpoint and accepts a user image. No dataset download is needed to try the inference demo. For a hosted interactive app, deploy this repository on [Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=thippeswamyguna%2FImage-processing-&branch=main&mainModule=app.py).

## Results

The selected model checkpoint is epoch 17, chosen by the lowest validation L1 during a 20-epoch CPU run. Evaluation used all 624 images in the supplied test split.

| Sparse views | FBP PSNR (dB) | FBP SSIM | Learned PSNR (dB) | Learned SSIM |
|---:|---:|---:|---:|---:|
| 4 | 8.53 | 0.217 | 17.02 | 0.527 |
| 5 | 8.77 | 0.234 | 18.30 | 0.579 |
| 10 | 10.40 | 0.342 | 22.51 | 0.780 |

Epoch scores improved through epoch 15 and dipped slightly at epoch 20. The validation-selected epoch 17 checkpoint is kept as `outputs/best.pt`. See `outputs/epoch_comparison/comparison.json` for the 5/10/15/20 comparison.

## Reconstruct an uploaded image

The Streamlit demo preprocesses an uploaded image to grayscale at 64 × 64, simulates 30 projection angles, retains 4–10 views, and displays the original reference, sparse sinogram, FBP image, and trained reconstruction.

## Dataset and optional retraining

Dataset: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Raw dataset files are intentionally not committed. Kaggle access may require an account and acceptance of dataset terms. Extract it to `data/chest_xray/` with `train`, `val`, and `test` subfolders before training.

```bash
python -m src.train --data-root data/chest_xray --epochs 20 --image-size 64 --views 30 --sparse-views 10 --batch-size 8 --max-train 512 --max-val 16 --device cpu --output outputs
python -m src.evaluate --data-root data/chest_xray --checkpoint outputs/best.pt --device cpu
```

The included `outputs/best.pt` and example result figure allow the website and demo to run without the dataset.

## Main files

- `index.html` — responsive static project website for GitHub Pages
- `app.py` — interactive Streamlit inference demo
- `src/` — projector, residual U-Net, training, data preparation, and evaluation
- `outputs/best.pt` — selected trained checkpoint
- `outputs/reconstruction_examples.png` — held-out original and reconstructed examples
- `deliverables/report/Project_Report.pdf` and `.docx` — report
- `deliverables/presentation/Project_Presentation.pptx` — presentation
- `deliverables/Working_Model_Code_and_Results.zip` — code, weights, metrics, and results package

## Limitations

The model performs reconstruction of synthetic Radon projections from 2D chest radiographs, not real CT scanner projections. The forward model uses simplified parallel-beam geometry. Training used 512 images at 64 × 64 on CPU, and the supplied validation set is small. Metrics are task-specific and do not establish clinical image quality or safety.
