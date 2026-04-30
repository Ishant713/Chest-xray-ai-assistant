import torch
import torch.nn.functional as F
import numpy as np
import torchxrayvision as xrv
from utils import preprocess_for_model

# ---- PATCH torch.load to allow old torchxrayvision checkpoints ----
_old_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False  # needed for xrv pretrained .pt files
    return _old_torch_load(*args, **kwargs)


torch.load = _patched_load

# Allow DenseNet to unpickle safely
torch.serialization.add_safe_globals([xrv.models.DenseNet])

# Fix: convert to plain Python list so it's always indexable / iterable
LABELS = list(xrv.datasets.default_pathologies)


def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def _normalize_for_xrv(img_arr: np.ndarray) -> np.ndarray:
    """
    torchxrayvision DenseNet expects a float32 array scaled to [-1024, 1024].
    This function handles the three common input cases:
      • [0, 255]   uint8 or float  → scale to [-1024, 1024]
      • [0, 1]     float           → scale to [-1024, 1024]
      • already near [-1024,1024]  → leave alone
    Output shape: (1, H, W)  (single-channel, channel-first)
    """
    img = img_arr.astype(np.float32)

    # Collapse to single channel if RGB/RGBA
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = img[..., :3].mean(axis=2)          # (H, W)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]

    # Determine range and rescale to [-1024, 1024]
    vmin, vmax = img.min(), img.max()
    if vmax <= 1.0 and vmin >= 0.0:
        # [0, 1] range
        img = img * 2048.0 - 1024.0
    elif vmax <= 255.0 and vmin >= 0.0:
        # [0, 255] range
        img = (img / 255.0) * 2048.0 - 1024.0
    # else: already in HU / [-1024, 1024] range — leave as-is

    img = np.expand_dims(img, axis=0)            # (1, H, W)
    return img


def predict(model, img_arr, device, score_mode="sigmoid"):
    # Use our own robust normalization instead of relying on utils.preprocess_for_model
    x = _normalize_for_xrv(img_arr)
    # x shape: (1, H, W) → add batch dim → (1, 1, H, W)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        logits = model(x_tensor)  # (1, num_classes)

    if score_mode == "softmax":
        probs = F.softmax(logits, dim=-1)
    else:
        probs = torch.sigmoid(logits)

    probs_np = probs.detach().cpu().numpy().flatten()
    logits_cpu = logits.detach().cpu()  # keep as tensor for GradCAM argmax

    # Return x_tensor on its original device so GradCAM can use it
    return probs_np, LABELS, logits_cpu, x_tensor


def build_cam(model, target_layer_name="features.norm5", device=None):
    try:
        from pytorch_grad_cam import GradCAM
    except ImportError:
        raise RuntimeError(
            "Grad-CAM not installed. Add 'pytorch-grad-cam' to requirements.txt."
        )

    # Navigate to the target layer by dotted path
    target = model
    for part in target_layer_name.split("."):
        target = getattr(target, part)

    cam = GradCAM(model=model, target_layers=[target])
    return cam