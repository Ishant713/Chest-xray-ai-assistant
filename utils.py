import numpy as np
from PIL import Image
import pydicom
import io
import cv2
import torch


def load_image_any(data_bytes, filename):
    name = filename.lower()
    if name.endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(data_bytes))
        arr = ds.pixel_array.astype("float32")
        arr = apply_windowing(ds, arr)

        flip = False
        try:
            if hasattr(ds, "PatientOrientation") and len(ds.PatientOrientation) > 0:
                if ds.PatientOrientation[0].upper() == "R":
                    flip = True
            elif hasattr(ds, "ImageOrientationPatient"):
                iop = ds.ImageOrientationPatient
                if float(iop[0]) < 0:
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
    if img_arr.ndim == 3:
        img_arr = np.mean(img_arr, axis=2)
    img_arr = img_arr.astype(np.float32)
    img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr) + 1e-8)
    if to_rgb:
        img_arr = np.stack([img_arr, img_arr, img_arr], axis=0)
    else:
        img_arr = img_arr[None, :, :]
    return img_arr


def to_display(arr):
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(img_gray_or_rgb, cam_gray):
    """
    Overlay a GradCAM++ heatmap on a chest x-ray.

    Pipeline:
      1. Squeeze cam to (H, W), normalize to [0, 1]
      2. Upsample with INTER_CUBIC (smoother than LINEAR for 32× magnification)
      3. Gaussian blur to smooth block artifacts from 7×7 → 224×224 upsampling
      4. Blend 55% x-ray + 45% heatmap so anatomy stays visible under the overlay
    """
    cam_gray = np.squeeze(cam_gray).astype(np.float32)

    # Normalize CAM to [0, 1]
    cam_min, cam_max = cam_gray.min(), cam_gray.max()
    if cam_max - cam_min > 1e-8:
        cam_gray = (cam_gray - cam_min) / (cam_max - cam_min)
    else:
        cam_gray = np.zeros_like(cam_gray)

    # Convert base image to RGB uint8
    if img_gray_or_rgb.ndim == 2:
        base = cv2.cvtColor(img_gray_or_rgb, cv2.COLOR_GRAY2RGB)
    else:
        base = img_gray_or_rgb.copy()
    if base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)

    H, W = base.shape[:2]

    # Upsample with cubic interpolation (better than linear for large scale factors)
    cam_resized = cv2.resize(cam_gray, (W, H), interpolation=cv2.INTER_CUBIC)

    # Gaussian blur to smooth the block artifacts from 7×7 → 224×224 upsampling.
    # Kernel size chosen as ~1/8 of the image width, always odd.
    k = max(3, (W // 8) | 1)   # e.g. 224//8=28, next odd = 29
    cam_resized = cv2.GaussianBlur(cam_resized, (k, k), sigmaX=0)

    # Re-normalize after blur (blur can slightly compress the range)
    c_min, c_max = cam_resized.min(), cam_resized.max()
    if c_max - c_min > 1e-8:
        cam_resized = (cam_resized - c_min) / (c_max - c_min)

    # Apply JET colormap and blend
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # 55% base image + 45% heatmap keeps anatomy clearly visible
    out = 0.55 * base.astype(np.float32) + 0.45 * heat.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)