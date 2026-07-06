"""Standalone EB-JEPA module (PyTorch 2.8).

Self-contained reimplementation of the image example of EB-JEPA
(arXiv:2602.03604, ref impl: github.com/facebookresearch/eb_jepa,
Apache-2.0). EB-JEPA is Meta FAIR's lightweight *energy-based* JEPA: it
prevents representation collapse with explicit VICReg-style variance +
covariance regularization (the "energy") instead of an EMA teacher /
stop-gradient. This makes the training objective a sum of three terms with
two interpretable coefficients (`std_coeff`, `cov_coeff`).

    z  = Proj(Enc(x )) ,  z' = Proj(Enc(x'))     # two augmented views
    p  = Pred(z)                                  # JEPA predictor
    L_inv = MSE(p, z')                            # prediction / invariance
    L_var = std_coeff * [ var_hinge(z) + var_hinge(z') ]   # anti-collapse
    L_cov = cov_coeff * [ cov_pen(z)  + cov_pen(z')  ]      # decorrelation
    L     = L_inv + L_var + L_cov

`var_hinge(Z) = mean_j relu(1 - sqrt(Var(Z_j)+eps))` keeps every embedding
dimension informative; `cov_pen(Z) = sum_{i!=j} Cov(Z)_{ij}^2 / D`
decorrelates them. Both branches carry gradient (energy-based, no EMA /
stop-gradient). Defaults follow the image config: `std_coeff=1.0`,
`cov_coeff=80.0` (paper image example uses cov ~ 80).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("EBJEPA")


# =============================================================================
# ResNet encoder (CIFAR stem: 3x3 conv, no maxpool)
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetEncoder(nn.Module):
    """Compact ResNet with a CIFAR-style stem. Output: global-avg-pooled
    feature of size `width*8*expansion`."""

    def __init__(self, layers=(2, 2, 2, 2), width: int = 64, in_chans: int = 3):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(in_chans, width, 3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, layers[0], stride=1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.out_dim = width * 8 * BasicBlock.expansion

    def _make_layer(self, planes: int, n: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)       # (B, out_dim)


# =============================================================================
# MLP heads
# =============================================================================

def _mlp(sizes: list[int], last_bn: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1],
                                bias=(i == len(sizes) - 2)))
        if i < len(sizes) - 2:
            layers += [nn.BatchNorm1d(sizes[i + 1]), nn.ReLU(inplace=True)]
        elif last_bn:
            layers.append(nn.BatchNorm1d(sizes[i + 1], affine=False))
    return nn.Sequential(*layers)


# =============================================================================
# Config
# =============================================================================

@dataclass
class EBJEPAConfig:
    in_chans: int = 3
    resnet_layers: tuple = (2, 2, 2, 2)       # ResNet-18
    resnet_width: int = 64
    proj_hidden: int = 2048
    proj_out: int = 2048
    pred_hidden: int = 512                    # predictor bottleneck

    # VICReg energy coefficients (invariance weight fixed to 1.0)
    std_coeff: float = 1.0
    cov_coeff: float = 80.0
    var_gamma: float = 1.0                    # target std (hinge target)
    eps: float = 1e-4


# =============================================================================
# Model
# =============================================================================

class EBJEPA(nn.Module):
    """Energy-based JEPA (image). Encoder + expander projector + predictor,
    trained with the VICReg energy loss (no EMA / stop-gradient).

    Inference: `encode(x)` returns the frozen backbone representation (the
    pre-projector feature), the standard linear-probe / kNN target.
    """

    def __init__(self, config: Optional[EBJEPAConfig] = None):
        super().__init__()
        cfg = config or EBJEPAConfig()
        self.config = cfg
        self.encoder = ResNetEncoder(tuple(cfg.resnet_layers), cfg.resnet_width,
                                     cfg.in_chans)
        d = self.encoder.out_dim
        self.projector = _mlp([d, cfg.proj_hidden, cfg.proj_hidden,
                               cfg.proj_out], last_bn=True)
        self.predictor = _mlp([cfg.proj_out, cfg.pred_hidden, cfg.proj_out],
                              last_bn=False)

    # -------------------------------------------------------------------------
    # VICReg energy terms
    # -------------------------------------------------------------------------

    def _var_hinge(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + self.config.eps)
        return F.relu(self.config.var_gamma - std).mean()

    def _cov_pen(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / d

    def compute_loss(self, view1: torch.Tensor, view2: torch.Tensor) -> dict:
        """One training step over two augmented views (B, C, H, W) each."""
        z1 = self.projector(self.encoder(view1))
        z2 = self.projector(self.encoder(view2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Prediction / invariance (symmetric, energy-based: no stop-grad).
        inv = 0.5 * (F.mse_loss(p1, z2) + F.mse_loss(p2, z1))
        var = self._var_hinge(z1) + self._var_hinge(z2)
        cov = self._cov_pen(z1) + self._cov_pen(z2)

        loss = (inv
                + self.config.std_coeff * var
                + self.config.cov_coeff * cov)
        return {"loss": loss, "inv_loss": inv.detach(),
                "var_loss": var.detach(), "cov_loss": cov.detach()}

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone representation (pre-projector) — the downstream feature."""
        return self.encoder(x)

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    _STRIP_PREFIXES = ("module.", "_orig_mod.", "model.", "backbone.")
    _KEEP_PREFIXES = ("encoder.", "projector.", "predictor.")

    @classmethod
    def _unwrap(cls, obj) -> dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state_dict", "model", "weights"):
                if key in obj and isinstance(obj[key], dict):
                    return cls._unwrap(obj[key])
            if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise ValueError(
            f"Could not find a tensor state-dict in checkpoint ({type(obj)})")

    @classmethod
    def _filter(cls, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for raw_k, v in sd.items():
            k = raw_k
            changed = True
            while changed:
                changed = False
                for p in cls._STRIP_PREFIXES:
                    if k.startswith(p):
                        k, changed = k[len(p):], True
            if any(k.startswith(p) for p in cls._KEEP_PREFIXES):
                out[k] = v
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Optional[EBJEPAConfig] = None,
        map_location: str = "cpu",
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> "EBJEPA":
        """Build an `EBJEPA` and load weights. Aborts (per paper/CLAUDE.md)
        if not a single tensor loads."""
        model = cls(config)
        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            blob = torch.load(path, map_location=map_location, weights_only=False)
        sd = cls._filter(cls._unwrap(blob))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        total = len(sd)
        applied = total - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"EBJEPA.from_checkpoint({path!r}): 0/{total} keys applied "
                "— checkpoint/architecture mismatch, aborting.")
        if not missing and not unexpected:
            logger.info("EBJEPA loaded successfully (%d/%d keys)", applied, total)
        else:
            logger.warning(
                "EBJEPA partial load: %d/%d keys (missing=%d, unexpected=%d)",
                applied, total, len(missing), len(unexpected))
            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"strict=True but missing={missing} unexpected={unexpected}")
        if device is not None:
            model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
