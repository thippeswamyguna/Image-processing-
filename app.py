"""Interactive demo for synthetic sparse-view X-ray reconstruction."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor

from src.model import ReconNet
from src.projector import ParallelBeam, sparse_indices

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "outputs" / "best.pt"

st.set_page_config(page_title="Sparse-View CT Reconstruction", page_icon="◉", layout="wide")
st.markdown("""
<style>
:root { --ink:#17212b; --muted:#657381; --line:#e3e8ed; --wash:#f5f7f9; --accent:#176b87; }
.block-container {max-width:1180px; padding-top:2rem;}
.hero {padding:2.4rem 2.5rem; background:linear-gradient(120deg,#f2f6f8,#fff); border:1px solid var(--line); border-radius:20px; margin-bottom:1.2rem;}
.hero h1 {color:var(--ink); font-size:2.6rem; margin:0 0 .55rem 0; line-height:1.12;}
.hero p {color:var(--muted); font-size:1.08rem; max-width:850px; margin:0;}
.kicker {letter-spacing:.12em; text-transform:uppercase; font-size:.75rem; color:var(--accent); font-weight:700;}
.panel {background:#fff; border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.25rem;}
.small-muted {color:var(--muted); font-size:.9rem;}
</style>
<div class="hero"><div class="kicker">Interactive research demo</div>
<h1>Reconstruct from fewer views</h1>
<p>Upload a chest X-ray and compare its reference image, simulated sparse-view projection, filtered backprojection, and trained residual U-Net output.</p></div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the trained model…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ReconNet().to(device).eval()
    model.load_state_dict(ckpt["model"])
    projector = ParallelBeam(int(cfg["image_size"]), int(cfg["views"]), device)
    return model, projector, device, ckpt

with st.expander("Project overview", expanded=True):
    a,b,c = st.columns(3)
    a.metric("Held-out test images", "624")
    b.metric("Selected checkpoint", "Epoch 17")
    c.metric("Training projection views", "10 / 30")
    st.write("A residual U-Net refines a ramp-filtered backprojection. Training combines image-reference loss, measured-projection consistency, and total-variation regularization. See the report for methodology, literature review, ablations, and limitations.")

left,right = st.columns([1,1], gap="large")
with left:
    st.subheader("Try an image")
    uploaded = st.file_uploader("Upload a grayscale or color X-ray", type=["png","jpg","jpeg","bmp","tif","tiff"])
    views = st.slider("Sparse projection views", min_value=4, max_value=10, value=10, step=1,
                      help="The model was trained using 10 views. Lower counts explore the reported view-count ablation.")
    run = st.button("Run reconstruction", type="primary", use_container_width=True, disabled=uploaded is None)
    st.caption("Images are resized to 64 × 64 pixels for this educational synthetic experiment.")
with right:
    st.subheader("What the model learns")
    st.write("The forward projector simulates parallel-beam Radon measurements. Only a subset of angles is used to create the sparse sinogram. The network then refines the FBP starting image.")
    st.markdown("**Pipeline**  \nReference X-ray → sparse sinogram → filtered backprojection → residual U-Net")

if uploaded is not None and run:
    with st.spinner("Projecting and reconstructing the image…"):
        pil = Image.open(uploaded).convert("RGB")
        size = 64
        x = Compose([Grayscale(num_output_channels=1), Resize((size, size)), ToTensor()])(pil).unsqueeze(0)
        model, projector, device, ckpt = load_model()
        x = x.to(device)
        idx = sparse_indices(projector.views, views, device)
        with torch.inference_mode():
            full = projector.forward(x)
            sparse = torch.zeros_like(full)
            sparse[:, idx, :] = full[:, idx, :]
            fbp = projector.fbp(sparse)
            fbp = fbp / fbp.amax(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
            pred = model(fbp)
        original = x[0,0].detach().cpu().numpy()
        fbp_img = fbp[0,0].detach().cpu().numpy()
        pred_img = pred[0,0].detach().cpu().numpy()
        sino_img = sparse[0].detach().cpu().numpy().T
    st.divider()
    st.subheader("Reconstruction results")
    cols=st.columns(4)
    panels=[("Original reference",original), (f"Sparse sinogram · {views} views",sino_img), ("Filtered backprojection",fbp_img), ("Trained residual U-Net",pred_img)]
    for col,(title,arr) in zip(cols,panels):
        with col:
            st.markdown(f"**{title}**")
            fig,ax=plt.subplots(figsize=(3.0,3.0))
            ax.imshow(arr,cmap="gray",aspect="auto" if "sinogram" in title else "equal",vmin=0 if "sinogram" not in title else None,vmax=1 if "sinogram" not in title else None)
            ax.axis("off"); fig.tight_layout(pad=0)
            st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.success(f"Reconstruction complete using checkpoint epoch {ckpt['epoch']} on {device.type.upper()}.")
    st.info("This output is a synthetic reconstruction of a 2D radiograph. It is not a clinical CT image or a diagnostic result.")

st.divider()
st.subheader("Held-out examples from the completed run")
st.image(str(ROOT / "outputs" / "reconstruction_examples.png"), use_container_width=True,
         caption="Reference X-rays, simulated 10-view sinograms, filtered backprojection, and trained model outputs from the held-out test set.")
st.subheader("Measured test performance")
metrics = {"Views": ["4", "5", "10"], "FBP PSNR (dB)": ["8.53", "8.77", "10.40"], "FBP SSIM": ["0.217", "0.234", "0.342"], "Model PSNR (dB)": ["17.02", "18.30", "22.51"], "Model SSIM": ["0.527", "0.579", "0.780"]}
st.table(metrics)
st.caption("Mean metrics over 624 test images. These results apply only to this synthetic 2D experiment and do not establish clinical CT quality.")

with st.expander("Epoch comparison: 5, 10, 15, and 20"):
    st.write("Scores improved through epoch 15, then dipped slightly at epoch 20. The selected epoch 17 checkpoint was chosen by validation loss, not test scores.")
    st.dataframe({"Epochs":[5,10,15,20],"Validation L1":[0.06246,0.05537,0.05731,0.05620],"Test PSNR (dB)":[21.11,21.92,22.24,21.97],"Test SSIM":[0.713,0.751,0.780,0.777]},hide_index=True,use_container_width=True)

with st.expander("Limitations and data"):
    st.write("The Kaggle chest X-rays are 2D radiographs, not CT slices with calibrated attenuation values. Applying a Radon transform creates a synthetic inverse problem. The model uses a simplified parallel-beam projector and 64 × 64 images. The dataset is not included in this repository; this interactive demo accepts an image upload and needs only the trained checkpoint.")
    st.markdown("Dataset: [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)")

st.markdown("---")
st.caption("Project by Kolakanuru Thippeswamy · Roll No. 48 · 20241CAI0055")
