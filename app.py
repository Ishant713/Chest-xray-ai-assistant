import streamlit as st
import numpy as np
import torch
import cv2
import pandas as pd
from inference import load_model, predict, build_cam
from utils import load_image_any, to_display, overlay_heatmap

st.set_page_config(page_title="AI Assistant for Radiology Images", layout="wide")
st.title("AI Assistant for Radiology Images")
st.caption("Educational demo. Not for clinical use.")


# -----------------------------
# Helper: center-crop + resize to square
# -----------------------------
def preprocess_square(img_arr: np.ndarray, size: int = 224) -> np.ndarray:
    img_arr = img_arr.astype(np.float32)
    h, w = img_arr.shape[:2]
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img_arr = img_arr[start_h : start_h + min_dim, start_w : start_w + min_dim]
    img_arr = cv2.resize(img_arr, (size, size))
    return img_arr


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Upload")
    file = st.file_uploader("JPG/PNG/DICOM (.dcm)", type=["jpg", "jpeg", "png", "dcm"])
    thresh = st.slider("Probability threshold", 0.05, 0.95, 0.3, 0.05)
    cam_layer = st.text_input("Grad-CAM target layer", "features.norm5")
    cam_on = st.toggle("Show Grad-CAM", value=True)
    st.divider()
    st.header("Model")
    device_opt = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    score_mode = st.selectbox("Score mode", ["sigmoid", "softmax"], index=0)

# -----------------------------
# Load model once (cached in session state)
# -----------------------------
if "model" not in st.session_state:
    with st.spinner("Loading model…"):
        st.session_state["model"] = load_model()

if file is None:
    st.info("Upload a chest X-ray image to begin.")
    st.stop()

# -----------------------------
# Read file & preprocess
# -----------------------------
file.seek(0)
data = file.read()
img_arr, info = load_image_any(data, file.name)

# Crop + resize to 224×224 for the model
img_arr = preprocess_square(img_arr, size=224)

# Debug stats
st.write(
    "DEBUG stats → shape:", img_arr.shape,
    "| min:", float(np.min(img_arr)),
    "| max:", float(np.max(img_arr)),
    "| mean:", float(np.mean(img_arr)),
)

# Build uint8 display image
disp = to_display(img_arr)
if isinstance(disp, np.ndarray) and disp.dtype != np.uint8:
    disp = (
        255 * (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    ).astype(np.uint8)

st.image(disp, caption="Input", width=500)

# -----------------------------
# Device & model
# -----------------------------
model = st.session_state["model"]
use_cuda = device_opt == "cuda" or (
    device_opt == "auto" and torch.cuda.is_available()
)
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device).eval()

# -----------------------------
# Predictions
# -----------------------------
probs, labels, logits_cpu, input_tensor = predict(model, img_arr, device, score_mode)

df = pd.DataFrame({"label": labels, "probability": probs})
df = df.sort_values("probability", ascending=False).reset_index(drop=True)

st.subheader("Predictions")

# Highlight rows above threshold — solid green bg + black text for readability
def highlight_above_thresh(row):
    if row["probability"] >= thresh:
        style = "background-color: #1a7a3c; color: #000000; font-weight: bold"
    else:
        style = "background-color: transparent; color: #ffffff"
    return [style] * len(row)

st.dataframe(
    df.style.apply(highlight_above_thresh, axis=1).format({"probability": "{:.3f}"}),
    use_container_width=True,
)

# -----------------------------
# Grad-CAM
# -----------------------------
if cam_on:
    try:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        cam = build_cam(model, target_layer_name=cam_layer, device=device)

        # Use the top predicted class as the GradCAM target
        top_idx = int(torch.argmax(logits_cpu).item())
        targets = [ClassifierOutputTarget(top_idx)]

        # GradCAM expects the tensor on the same device as the model
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)

        heatmap = overlay_heatmap(disp, grayscale_cam)
        st.subheader(f"Grad-CAM  (class: {labels[top_idx]})")
        st.image(heatmap, width=500)

    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")

# -----------------------------
# Metadata
# -----------------------------
with st.expander("Image metadata"):
    st.json(info)