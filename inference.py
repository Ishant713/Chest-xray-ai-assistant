import torch
import torch.nn.functional as F
import numpy as np
import torchxrayvision as xrv

# ---- PATCH torch.load to allow old torchxrayvision checkpoints ----
_old_torch_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_torch_load(*args, **kwargs)

torch.load = _patched_load
torch.serialization.add_safe_globals([xrv.models.DenseNet])

LABELS = list(xrv.datasets.default_pathologies)

OP_THRESHOLDS = {
    'Atelectasis':              0.07422872,
    'Consolidation':            0.03829084,
    'Infiltration':             0.09814756,
    'Pneumothorax':             0.00981185,
    'Edema':                    0.02360107,
    'Emphysema':                0.00224904,
    'Fibrosis':                 0.01006072,
    'Effusion':                 0.10324661,
    'Pneumonia':                0.05681074,
    'Pleural_Thickening':       0.02679165,
    'Cardiomegaly':             0.05031816,
    'Nodule':                   0.02398586,
    'Mass':                     0.01939503,
    'Hernia':                   0.04288977,
    'Lung Lesion':              0.05336962,
    'Fracture':                 0.03597581,
    'Lung Opacity':             0.20204692,
    'Enlarged Cardiomediastinum': 0.05015312,
}


def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def _normalize_for_xrv(img_arr: np.ndarray) -> np.ndarray:
    """
    Official xrv formula: (2*(img/maxval) - 1) * 1024 → float32 in [-1024, 1024].
    Output shape: (1, H, W).
    """
    img = img_arr.astype(np.float32)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = img[..., :3].mean(axis=2)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]

    vmin, vmax = img.min(), img.max()
    if vmax <= 1.0 and vmin >= 0.0:
        img = (2.0 * img - 1.0) * 1024.0
    elif vmax <= 255.0 and vmin >= 0.0:
        img = (2.0 * (img / 255.0) - 1.0) * 1024.0

    return np.expand_dims(img, axis=0)   # (1, H, W)


def _forward_logits(model, x_tensor):
    """
    Raw logits, bypassing torchxrayvision's sigmoid + op_norm.
    Pipeline: features → ReLU → AdaptiveAvgPool → flatten → classifier
    """
    feat   = model.features(x_tensor)
    out    = F.relu(feat, inplace=False)
    out    = F.adaptive_avg_pool2d(out, (1, 1))
    out    = out.view(feat.size(0), -1)
    logits = model.classifier(out)
    return logits, feat


def predict(model, img_arr, device, score_mode="sigmoid"):
    x = _normalize_for_xrv(img_arr)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = _forward_logits(model, x_tensor)

    if score_mode == "softmax":
        probs = F.softmax(logits, dim=-1)
    else:
        probs = torch.sigmoid(logits)

    probs_np   = probs.detach().cpu().numpy().flatten()
    logits_cpu = logits.detach().cpu()
    return probs_np, LABELS, logits_cpu, x_tensor


class _LogitWrapper(torch.nn.Module):
    """
    Wraps xrv DenseNet so GradCAM++ differentiates through raw logits,
    not the op_norm-rescaled output from model.forward().
    """
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        logits, _ = _forward_logits(self.m, x)
        return logits


def build_cam(model, target_layer_name="features.denseblock4", device=None):
    """
    Build a GradCAM++ object for the given model and target layer.

    Why GradCAM++ instead of GradCAM:
      GradCAM averages gradients globally — for diffuse or multi-region findings
      (Pleural_Thickening, Effusion, Lung Opacity) this produces noisy maps.
      GradCAM++ uses second-order gradients to weight each spatial location
      independently, giving sharper and more anatomically correct heatmaps.

    Why denseblock4 (7×7) as default:
      denseblock3 (14×14) sounds higher-resolution but it activates strongly on
      image EDGES and background because it is only half-way through the network
      and not yet semantically selective for pathology. The result is hot zones
      on image borders and shoulders rather than the lung region.
      denseblock4 is the deepest convolutional block — maximally class-discriminative.
      The lower spatial resolution (7×7) is compensated by Gaussian smoothing
      in overlay_heatmap (utils.py) which produces a clean, anatomically-aligned map.
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
    except ImportError:
        raise RuntimeError("Grad-CAM not installed. Add 'grad-cam' to requirements.txt.")

    target = model
    for part in target_layer_name.split("."):
        target = getattr(target, part)

    wrapped = _LogitWrapper(model)
    cam = GradCAMPlusPlus(model=wrapped, target_layers=[target])
    return cam