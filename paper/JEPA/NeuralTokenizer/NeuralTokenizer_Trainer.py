"""JEPA-Neural-Tokenizer trainers (PyTorch 2.8).

Two trainers matching the paper's two stages, plus the GAN machinery used by
Stage 2:

  * `Stage1Trainer` — masked-latent JEPA SSL. The student encoder sees the
    waveform features with a contiguous **block mask** (ρ=0.5) applied in
    feature space; a small predictor regresses the **EMA-teacher** features
    at the masked frames (stop-grad, instance-normed targets).

  * `Stage2Trainer` — reconstruction. The Stage-1 encoder is **frozen**;
    `proj_in -> FSQ -> proj_out -> HiFi-GAN decoder` reconstruct the
    waveform, trained with L1 + multi-resolution STFT + GAN (multi-period +
    multi-scale discriminators, feature matching).

The deployable modules (`JEPAEncoder`, `FSQ`, `MixedRadixCodec`,
`HiFiGANGenerator`) live in `NeuralTokenizer.py` and are imported here so a
checkpoint saved by either trainer loads with
`NeuralTokenizer.from_checkpoint(...)`.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from NeuralTokenizer import (  # noqa: E402
    FSQ,
    HiFiGANGenerator,
    JEPAEncoder,
    MixedRadixCodec,
    NeuralTokenizerConfig,
)

logger = logging.getLogger("NeuralTokenizer_Trainer")


# =============================================================================
# Stage 1 — masked-latent JEPA SSL
# =============================================================================

class _SelfBlock(nn.Module):
    """Pre-norm self-attention block (JEPA predictor)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class Stage1Trainer(nn.Module):
    """Student encoder + EMA teacher + masked-latent predictor."""

    def __init__(self, config: NeuralTokenizerConfig):
        super().__init__()
        self.config = config
        self.encoder = JEPAEncoder(config)
        d = self.encoder.out_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = nn.ModuleList([
            _SelfBlock(d, config.pred_heads) for _ in range(config.pred_layers)])
        self.pred_norm = nn.LayerNorm(d)

        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    # ---- EMA teacher ----
    @torch.no_grad()
    def ema_step(self, momentum: float = None):
        m = self.config.ema_decay if momentum is None else momentum
        for ps, pt in zip(self.encoder.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)

    # ---- block masking (contiguous spans covering ~mask_ratio of frames) ----
    def _sample_mask(self, B: int, T: int, device) -> torch.Tensor:
        cfg = self.config
        s_min = max(1, cfg.mask_min_span)
        s_max = max(s_min, T // 4)
        target = max(1, int(cfg.mask_ratio * T))
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            guard = 0
            while int(mask[b].sum()) < target and guard < 100:
                span = random.randint(s_min, s_max)
                start = random.randint(0, max(0, T - span))
                mask[b, start:start + span] = True
                guard += 1
            if bool(mask[b].all()):                  # keep >=1 visible
                mask[b, random.randint(0, T - 1)] = False
            if not bool(mask[b].any()):              # keep >=1 masked
                mask[b, random.randint(0, T - 1)] = True
        return mask

    def forward(self, wav: torch.Tensor) -> dict:
        feats = self.encoder.frontend_features(wav)          # (B, T, d)
        B, T, d = feats.shape
        mask = self._sample_mask(B, T, wav.device)           # (B, T)
        m = mask.unsqueeze(-1)
        f_masked = torch.where(m, self.mask_token, feats)
        h = self.encoder.encode_features(f_masked)           # student (B,T,d)
        for blk in self.predictor:
            h = blk(h)
        pred = self.pred_norm(h)

        with torch.no_grad():
            tf = self.teacher.frontend_features(wav)
            target = F.layer_norm(self.teacher.encode_features(tf), (d,))

        err = ((pred - target) ** 2).mean(-1)                # (B, T)
        wsum = mask.float().sum().clamp_min(1.0)
        loss = (err * mask.float()).sum() / wsum
        return {"loss": loss, "mask_frac": mask.float().mean().detach()}


# =============================================================================
# Stage 2 — GAN discriminators + losses
# =============================================================================

class _PeriodDisc(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        ch = [1, 32, 128, 256, 256]
        self.convs = nn.ModuleList([
            nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), padding=(2, 0))
            for i in range(len(ch) - 1)])
        self.post = nn.Conv2d(256, 1, (3, 1), padding=(1, 0))

    def forward(self, x):                                    # x (B, 1, L)
        B, C, L = x.shape
        pad = (self.period - L % self.period) % self.period
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")
        x = x.view(B, C, (L + pad) // self.period, self.period)
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discs = nn.ModuleList([_PeriodDisc(p) for p in periods])

    def forward(self, x):
        return [d(x) for d in self.discs]                    # [(logits, fmap), ...]


class _ScaleDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 64, 15, 1, padding=7),
            nn.Conv1d(64, 128, 41, 4, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 4, groups=16, padding=20),
            nn.Conv1d(256, 256, 5, 1, padding=2),
        ])
        self.post = nn.Conv1d(256, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3):
        super().__init__()
        self.discs = nn.ModuleList([_ScaleDisc() for _ in range(n_scales)])
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        outs = []
        for i, d in enumerate(self.discs):
            outs.append(d(x))
            x = self.pool(x)
        return outs


def discriminator_loss(real_outs, fake_outs) -> torch.Tensor:
    loss = 0.0
    for (dr, _), (df, _) in zip(real_outs, fake_outs):
        loss = loss + ((dr - 1.0) ** 2).mean() + (df ** 2).mean()
    return loss


def generator_adv_loss(fake_outs) -> torch.Tensor:
    loss = 0.0
    for df, _ in fake_outs:
        loss = loss + ((df - 1.0) ** 2).mean()
    return loss


def feature_matching_loss(real_outs, fake_outs) -> torch.Tensor:
    loss = 0.0
    for (_, fr), (_, ff) in zip(real_outs, fake_outs):
        for a, b in zip(fr, ff):
            loss = loss + F.l1_loss(b, a)
    return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, ffts=(2048, 1024, 512, 256, 128),
                 hops=(512, 256, 128, 64, 32)):
        super().__init__()
        self.ffts = ffts
        self.hops = hops
        for n in ffts:
            self.register_buffer(f"win_{n}", torch.hann_window(n),
                                 persistent=False)

    def _stft(self, x, n_fft, hop):
        win = getattr(self, f"win_{n_fft}")
        spec = torch.stft(x, n_fft, hop, win_length=n_fft, window=win,
                          return_complex=True)
        return spec.abs().clamp_min(1e-7)

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_hat, x = x_hat.squeeze(1), x.squeeze(1)
        total = x.new_zeros(())
        for n_fft, hop in zip(self.ffts, self.hops):
            if x.shape[-1] < n_fft:                          # skip too-short res
                continue
            s_hat, s = self._stft(x_hat, n_fft, hop), self._stft(x, n_fft, hop)
            sc = torch.linalg.norm(s - s_hat) / torch.linalg.norm(s).clamp_min(1e-7)
            mag = F.l1_loss(s_hat.log(), s.log())
            total = total + sc + mag
        return total


# =============================================================================
# Stage 2 — generator wrapper (frozen encoder + FSQ + decoder + discriminators)
# =============================================================================

class Stage2Trainer(nn.Module):
    """Frozen Stage-1 encoder -> proj_in -> FSQ -> proj_out -> HiFi-GAN, with
    MPD + MSD discriminators and the reconstruction losses."""

    def __init__(self, config: NeuralTokenizerConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.encoder = JEPAEncoder(config)
        self.proj_in = nn.Linear(d, config.fsq_dim)
        self.fsq = FSQ(config.fsq_dim, config.fsq_level)
        self.codec = MixedRadixCodec(config.fsq_dim, config.pack_group,
                                     config.fsq_level)
        self.proj_out = nn.Linear(config.fsq_dim, d)
        self.decoder = HiFiGANGenerator(
            d, tuple(reversed(config.conv_strides)), config.decoder_channels())

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.stft = MultiResolutionSTFTLoss()
        self.freeze_encoder()

    def freeze_encoder(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def load_stage1_encoder(self, sd: dict) -> tuple[int, int]:
        """Load `encoder.*` weights from a Stage-1 checkpoint; abort if none."""
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        if not enc_sd:                                       # maybe already bare
            enc_sd = {k: v for k, v in sd.items()
                      if not k.startswith(("teacher.", "predictor.",
                                           "mask_token"))}
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        applied = len(enc_sd) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                "Stage2Trainer.load_stage1_encoder: 0 keys applied — aborting.")
        self.freeze_encoder()
        return applied, len(enc_sd)

    def generator_parameters(self):
        mods = (self.proj_in, self.fsq, self.proj_out, self.decoder)
        return [p for m in mods for p in m.parameters() if p.requires_grad]

    def discriminator_parameters(self):
        return list(self.mpd.parameters()) + list(self.msd.parameters())

    def generator_forward(self, wav: torch.Tensor):
        """wav (B,1,L) -> (x_hat (B,1,L'), x_real (B,1,L')) length-aligned."""
        with torch.no_grad():
            feats = self.encoder(wav)                        # frozen (B,T,d)
        z_norm, _ = self.fsq(self.proj_in(feats))
        z = self.proj_out(z_norm).transpose(1, 2)            # (B, d, T)
        x_hat = self.decoder(z)                              # (B, 1, L')
        L = min(x_hat.shape[-1], wav.shape[-1])
        return x_hat[..., :L], wav[..., :L]

    def discriminator_loss(self, x_real, x_fake) -> torch.Tensor:
        d_real = self.mpd(x_real) + self.msd(x_real)
        d_fake = self.mpd(x_fake.detach()) + self.msd(x_fake.detach())
        return discriminator_loss(d_real, d_fake)

    def generator_loss(self, x_real, x_fake, use_gan: bool) -> dict:
        cfg = self.config
        l1 = F.l1_loss(x_fake, x_real)
        stft = self.stft(x_fake, x_real)
        loss = l1 + cfg.lambda_stft * stft
        adv = fm = x_real.new_zeros(())
        if use_gan:
            d_real = self.mpd(x_real) + self.msd(x_real)
            d_fake = self.mpd(x_fake) + self.msd(x_fake)
            adv = generator_adv_loss(d_fake)
            fm = feature_matching_loss(d_real, d_fake)
            loss = loss + cfg.lambda_gan * (adv + fm)
        return {"loss": loss, "l1": l1.detach(), "stft": stft.detach(),
                "adv": adv.detach(), "fm": fm.detach()}
