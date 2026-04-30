import numpy as np
from PIL import Image
import pydicom
import io
import cv2
from pytorch_grad_cam import GradCAM
import torch

def load_image_any(data_bytes, filename):
    name = filename.lower()
    if name.endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(data_bytes))
        arr = ds.pixel_array.astype("float32")
        arr = apply_windowing(ds, arr)

        # ---- Orientation check ----
        flip = False
        try:
            if hasattr(ds, "PatientOrientation") and len(ds.PatientOrientation) > 0:
                if ds.PatientOrientation[0].upper() == "R":  # Right-side first → flip
                    flip = True
            elif hasattr(ds, "ImageOrientationPatient"):
                iop = ds.ImageOrientationPatient
                if float(iop[0]) < 0:  # negative x-direction cosine → flip
                    flip = True
        except Exception:
            pass

        if flip:
            arr = np.fliplr(arr)

        info = {
            "modality": str(getattr(ds, "Modality", "")),
            "patient_id": str(getattr(ds, "PatientID", "")),
            "study_date": str(getattr(ds, "StudyDate", "")),
            "rows": int(ds.Rows),
            "cols": int(ds.Columns),
            "flipped": flip
        }
        arr = normalize_to_uint8(arr)
        return arr, info
    else:
        im = Image.open(io.BytesIO(data_bytes)).convert("L")
        arr = np.array(im)
        info = {"mode": "L", "shape": list(arr.shape), "flipped": False}
        return arr, info

def apply_windowing(ds, arr):
    try:
        center = ds.WindowCenter
        width = ds.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = float(center[0])
        if isinstance(width, pydicom.multival.MultiValue):
            width = float(width[0])
        low, high = center - width / 2, center + width / 2
        arr = np.clip(arr, low, high)
    except Exception:
        pass
    return arr

def normalize_to_uint8(arr):
    arr = arr.astype("float32")
    mn, mx = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if mx - mn < 1e-5:
        mn, mx = arr.min(), arr.max()
    arr = np.clip((arr - mn) / (mx - mn + 1e-8), 0, 1)
    arr = (arr * 255).astype("uint8")
    return arr

def preprocess_for_model(img_arr, to_rgb=False):
    """Prepares image for model input"""
    if img_arr.ndim == 3:
        img_arr = np.mean(img_arr, axis=2)  # grayscale
    img_arr = img_arr.astype(np.float32)
    img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr) + 1e-8)

    # If model expects 3 channels (ImageNet-pretrained), stack into RGB
    if to_rgb:
        img_arr = np.stack([img_arr, img_arr, img_arr], axis=0)  # (3, H, W)
    else:
        img_arr = img_arr[None, :, :]  # (1, H, W)
    return img_arr

def to_display(arr):
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

def overlay_heatmap(img_gray_or_rgb, cam_gray):
    cam_gray = (cam_gray - cam_gray.min()) / (cam_gray.max() - cam_gray.min() + 1e-8)
    if img_gray_or_rgb.ndim == 2:
        base = cv2.cvtColor(img_gray_or_rgb, cv2.COLOR_GRAY2RGB)
    else:
        base = img_gray_or_rgb
    cam_resized = cv2.resize(cam_gray.astype("float32"), (base.shape[1], base.shape[0]))
    heat = cv2.applyColorMap((cam_resized * 255).astype("uint8"), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (0.5 * base + 0.5 * heat).astype("uint8")
    return out

def build_cam(model, device, target_layer_name="features.denseblock4.denselayer16.conv2"):
    target = model
    for part in target_layer_name.split("."):
        target = getattr(target, part)
    cam = GradCAM(model=model, target_layers=[target], use_cuda=device.type == "cuda")
    return cam

def get_cam_image(cam, img_arr, device, info=None, to_rgb=False):
    input_tensor = preprocess_for_model(img_arr, to_rgb=to_rgb)
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    # Auto-use orientation info if provided
    if info is not None and info.get("flipped", False):
        grayscale_cam = np.fliplr(grayscale_cam)

    return overlay_heatmap(to_display(img_arr), grayscale_cam)
