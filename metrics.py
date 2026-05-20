import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F


@dataclass
class _RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int) -> None:
        if n <= 0:
            return
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def mean(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count


class MetricsAggregator:
    def __init__(self) -> None:
        self._means: Dict[str, _RunningMean] = {}

    def update(self, values: Dict[str, Any], n: int) -> None:
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    value = value.mean()
                value = float(value.detach().cpu().item())
            else:
                value = float(value)
            self._means.setdefault(key, _RunningMean()).update(value, n=n)

    def as_dict(self) -> Dict[str, float]:
        return {k: v.mean for k, v in self._means.items()}


class CSVLogger:
    def __init__(self, path: str, fieldnames: Optional[List[str]] = None) -> None:
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write_row(self, row: Dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        else:
            for k in row.keys():
                if k not in self.fieldnames:
                    self.fieldnames.append(k)

        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class OptionalMetricBackends:
    """Lazy-load optional heavy models once per process."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._lpips = None
        self._clip = None
        self._clip_backend = None
        self._warned: Set[str] = set()

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(msg)

    def get_lpips(self):
        if self._lpips is not None:
            return self._lpips
        try:
            from taming.modules.autoencoder.lpips import LPIPS  # type: ignore

            self._lpips = LPIPS().to(self.device).eval()
            return self._lpips
        except Exception as e:
            self._warn_once("lpips", f"[metrics] LPIPS unavailable ({e}); writing NaN.")
            self._lpips = False
            return None

    def get_clip(self):
        if self._clip is not None:
            return self._clip

        # Prefer open_clip, fall back to clip (OpenAI)
        try:
            import open_clip  # type: ignore

            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device=str(self.device),
            )
            model.eval()
            self._clip_backend = "open_clip"
            self._clip = model
            return self._clip
        except Exception:
            pass

        try:
            import clip  # type: ignore

            model, _ = clip.load("ViT-B/32", device=str(self.device), jit=False)
            model.eval()
            self._clip_backend = "clip"
            self._clip = model
            return self._clip
        except Exception as e:
            self._warn_once("clip", f"[metrics] CLIP unavailable ({e}); writing NaN.")
            self._clip = False
            return None

    @property
    def clip_backend(self) -> Optional[str]:
        if self._clip is False:
            return None
        return self._clip_backend


class CLIPScore(torch.nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        super().__init__()
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
        )

    def preprocess_tensor(self, x):
        # Input: [-1, 1] tensor [B, 3, H, W]
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, img_orig, img_recon):
        # Both inputs must be in [-1, 1]
        f_orig  = F.normalize(self.model.encode_image(self.preprocess_tensor(img_orig)),  dim=-1)
        f_recon = F.normalize(self.model.encode_image(self.preprocess_tensor(img_recon)), dim=-1)
        return (f_orig * f_recon).sum(dim=-1).mean()  # scalar in [0, 1]


def _to_0_1(img: torch.Tensor) -> torch.Tensor:
    return img.clamp(0.0, 1.0)


def psnr_from_images(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = _to_0_1(x)
    y = _to_0_1(y)
    mse = torch.mean((x * 255.0 - y * 255.0) ** 2)
    mse = torch.clamp(mse, min=eps)
    return 10.0 * torch.log10((255.0 * 255.0) / mse)


def ed_from_images(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean per-image RMSE (0..255 range), resolution-independent."""
    x = _to_0_1(x)
    y = _to_0_1(y)
    b = x.shape[0]
    diff = (x * 255.0 - y * 255.0).reshape(b, -1)
    return torch.sqrt(torch.mean(diff ** 2, dim=1)).mean()


def ssim_from_images(x: torch.Tensor, y: torch.Tensor, ssim_module) -> torch.Tensor:
    # ssim_module should output (B,) or scalar. We average.
    x = _to_0_1(x)
    y = _to_0_1(y)
    out = ssim_module(x, y)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.numel() != 1:
        out = out.mean()
    return out


def lpips_from_images(x: torch.Tensor, y: torch.Tensor, backend: OptionalMetricBackends) -> Optional[torch.Tensor]:
    model = backend.get_lpips()
    if model is None:
        return None

    # LPIPS expects [-1, 1]
    x = _to_0_1(x) * 2.0 - 1.0
    y = _to_0_1(y) * 2.0 - 1.0
    with torch.no_grad():
        val = model(x, y)
        if val.numel() != 1:
            val = val.mean()
        return val


def _clip_normalize(images_0_1: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images_0_1.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images_0_1.device).view(1, 3, 1, 1)
    return (images_0_1 - mean) / std


def clip_score_from_images(x: torch.Tensor, y: torch.Tensor, backend: OptionalMetricBackends) -> Optional[torch.Tensor]:
    model = backend.get_clip()
    if model is None:
        return None

    x = _to_0_1(x)
    y = _to_0_1(y)

    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    y = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False)
    x = _clip_normalize(x)
    y = _clip_normalize(y)

    with torch.no_grad():
        x_feat = model.encode_image(x)
        y_feat = model.encode_image(y)

        x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
        y_feat = y_feat / y_feat.norm(dim=-1, keepdim=True)
        sim = (x_feat * y_feat).sum(dim=-1).mean()
        return sim


def get_model_efficiency(model, input_size=(1, 3, 256, 256), device="cuda"):
    """
    Calculate MACs, Parameters, and Inference Latency.
    """
    import time
    try:
        from thop import profile
    except ImportError:
        print("[metrics] thop not installed, MACs/Params calculation skipped.")
        profile = None

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size).to(device)

    results = {}

    # 1. Parameters (Manual count for precision)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["params"] = params

    # 2. MACs
    if profile is not None:
        try:
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
            results["macs"] = macs
        except Exception as e:
            print(f"[metrics] Error calculating MACs: {e}")
            results["macs"] = float("nan")
    else:
        results["macs"] = float("nan")

    # 3. Latency
    # Warmup
    try:
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

            if "cuda" in str(device):
                torch.cuda.synchronize()
            
            start_time = time.time()
            iterations = 50
            for _ in range(iterations):
                _ = model(dummy_input)
            
            if "cuda" in str(device):
                torch.cuda.synchronize()
            
            latency = (time.time() - start_time) / iterations * 1000  # in ms
            results["latency_ms"] = latency
    except Exception as e:
        print(f"[metrics] Error calculating latency: {e}")
        results["latency_ms"] = float("nan")

    return results
