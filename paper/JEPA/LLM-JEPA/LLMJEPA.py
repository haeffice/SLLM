"""Standalone LLM-JEPA module (PyTorch 2.8 / transformers).

Self-contained reimplementation of LLM-JEPA (arXiv:2509.14252, ref impl:
github.com/rbalestr-lab/llm-jepa, Apache-2.0). LLM-JEPA augments the
standard next-token cross-entropy objective with a JEPA term computed
entirely in embedding space over a pair of "views" of the same example
(e.g. a natural-language `Text` and its `Code`/SQL/regex):

    Enc(s)        := last-layer hidden state at the last non-pad token of s
    Pred(Enc(T))  := the SAME LLM (tied weights) run on `T` followed by k
                     learned [predictor_i] tokens; the embedding is taken
                     at the last predictor token (auto-regressive +
                     self-attention => a tied-weights predictor).
    L_JEPA        := 1 - cos( Pred(Enc(Text)) , Enc(Code) )        (default)
    L             := gamma * L_LM  +  lambda * L_JEPA

Defaults follow the reference impl: `gamma=1.0`, `lambda=0.1`, `k=1`,
cosine JEPA distance, optional LoRA (peft).

`model_name="__tiny__"` builds a tiny random LlamaForCausalLM + a byte
tokenizer entirely offline (used by the CPU smoke test); any other value
is an ordinary `transformers` repo id / local path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("LLMJEPA")

PRED_TOKEN_TMPL = "<|predictor_{i}|>"


# =============================================================================
# Offline byte tokenizer (only for model_name="__tiny__" smoke test)
# =============================================================================

class ByteTokenizer:
    """Minimal byte-level tokenizer, HF-compatible on the surface used here.

    Vocab: 256 raw bytes + [pad]/[eos] + appended special tokens. Enough for
    the train/eval pipeline (batch `__call__`, padding, `eos_token_id`,
    `pad_token_id`, `add_special_tokens`, `convert_tokens_to_ids`, `decode`).
    """

    def __init__(self):
        self.pad_token_id = 256
        self.eos_token_id = 257
        self._specials: dict[str, int] = {}
        self._next = 258

    def add_special_tokens(self, mapping: dict) -> int:
        added = 0
        for tok in mapping.get("additional_special_tokens", []):
            if tok not in self._specials:
                self._specials[tok] = self._next
                self._next += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._specials[tok]

    def __len__(self) -> int:
        return self._next

    @property
    def vocab_size(self) -> int:
        return self._next

    def _encode_one(self, text: str, add_eos: bool) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = None
            for tok in self._specials:                 # greedy special match
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                ids.append(self._specials[matched])
                i += len(matched)
            else:
                ids.append(text[i].encode("utf-8")[0] & 0xFF)
                i += 1
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, text, padding=True, truncation=True, max_length=512,
                 add_eos=False, return_tensors="pt", add_special_tokens=True,
                 **_):
        batch = [text] if isinstance(text, str) else list(text)
        seqs = [self._encode_one(t, add_eos)[:max_length] for t in batch]
        n = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:
            pad = n - len(s)
            input_ids.append(s + [self.pad_token_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True) -> str:
        out = []
        for i in list(ids):
            i = int(i)
            if i < 256:
                out.append(chr(i))
            elif not skip_special_tokens:
                out.append("·")
        return "".join(out)


# =============================================================================
# Config
# =============================================================================

@dataclass
class LLMJEPAConfig:
    model_name: str = "__tiny__"
    num_predictors: int = 1                 # k appended [predictor_i] tokens
    front_pred: bool = False                # predictor tokens before Text
    jepa_objective: str = "cos"             # cos | l2 | mse | infonce
    lbd: float = 0.1                        # JEPA loss weight (lambda)
    gamma: float = 1.0                      # LM loss weight
    infonce_temp: float = 0.07
    max_length: int = 512
    torch_dtype: str = "float32"            # float32 | bfloat16 | float16

    # LoRA (peft); enabled when use_lora=True
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # tiny offline model dims (model_name == "__tiny__" only)
    tiny_hidden: int = 64
    tiny_layers: int = 2
    tiny_heads: int = 4
    tiny_inter: int = 128


# =============================================================================
# Tiny offline model builder
# =============================================================================

def _build_tiny(cfg: LLMJEPAConfig):
    from transformers import LlamaConfig, LlamaForCausalLM
    tok = ByteTokenizer()
    lc = LlamaConfig(
        vocab_size=258 + cfg.num_predictors,    # bytes+pad+eos+predictors
        hidden_size=cfg.tiny_hidden,
        intermediate_size=cfg.tiny_inter,
        num_hidden_layers=cfg.tiny_layers,
        num_attention_heads=cfg.tiny_heads,
        num_key_value_heads=cfg.tiny_heads,
        max_position_embeddings=cfg.max_length + cfg.num_predictors + 4,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = LlamaForCausalLM(lc)
    return model, tok


# =============================================================================
# LLM-JEPA model
# =============================================================================

class LLMJEPA(nn.Module):
    """LLM-JEPA training/inference wrapper around a causal-LM backbone.

    The backbone (`self.backbone`) is the deployable model — after training
    it is an ordinary `AutoModelForCausalLM` (plus k learned predictor-token
    embeddings) and can be used / saved as such. The JEPA objective only
    adds the tied-weights predictor *usage*; there are no extra weights
    besides the k predictor-token rows already inside the embedding matrix.
    """

    def __init__(self, config: Optional[LLMJEPAConfig] = None):
        super().__init__()
        cfg = config or LLMJEPAConfig()
        self.config = cfg

        dtype = getattr(torch, cfg.torch_dtype)
        if cfg.model_name == "__tiny__":
            self.backbone, self.tokenizer = _build_tiny(cfg)
            self.backbone = self.backbone.to(dtype)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.backbone = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, torch_dtype=dtype, trust_remote_code=True)

        # Register the k predictor tokens and grow the embedding table.
        self.pred_tokens = [PRED_TOKEN_TMPL.format(i=i)
                            for i in range(cfg.num_predictors)]
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.pred_tokens})
        if added or self.backbone.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
            self.backbone.resize_token_embeddings(len(self.tokenizer))
        self.pred_token_ids = [self.tokenizer.convert_tokens_to_ids(t)
                               for t in self.pred_tokens]

        if cfg.use_lora:
            self._apply_lora(cfg)

        logger.info(
            "LLMJEPA ready: backbone=%s, k=%d predictor tokens %s, "
            "objective=%s, lambda=%.4g, gamma=%.4g, lora=%s",
            cfg.model_name, cfg.num_predictors, self.pred_token_ids,
            cfg.jepa_objective, cfg.lbd, cfg.gamma, cfg.use_lora,
        )

    # -------------------------------------------------------------------------
    def _apply_lora(self, cfg: LLMJEPAConfig):
        from peft import LoraConfig, TaskType, get_peft_model
        lc = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        self.backbone = get_peft_model(self.backbone, lc)
        self.backbone.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    @staticmethod
    def _last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
        """Index of the last non-pad (real) token per row."""
        return attention_mask.long().sum(dim=1) - 1

    def _hidden_last_layer(self, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        return out.hidden_states[-1]                    # (B, L, H)

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        """Enc(s): last-layer hidden state at the last non-pad token."""
        h = self._hidden_last_layer(input_ids, attention_mask)
        idx = self._last_token_index(attention_mask)
        return h[torch.arange(h.shape[0], device=h.device), idx, :]

    def predict(self, text_ids: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:
        """Pred(Enc(Text)): append k predictor tokens to Text, run the SAME
        backbone, return the hidden state at the last predictor token."""
        B = text_ids.shape[0]
        dev = text_ids.device
        ptoks = torch.tensor(self.pred_token_ids, device=dev,
                             dtype=text_ids.dtype).unsqueeze(0).expand(B, -1)
        pmask = torch.ones_like(ptoks)
        if self.config.front_pred:
            ids = torch.cat([ptoks, text_ids], dim=1)
            msk = torch.cat([pmask, text_mask], dim=1)
            pred_pos = torch.full((B,), len(self.pred_token_ids) - 1,
                                  device=dev)
        else:
            ids = torch.cat([text_ids, ptoks], dim=1)
            msk = torch.cat([text_mask, pmask], dim=1)
            pred_pos = self._last_token_index(text_mask) + len(self.pred_token_ids)
        h = self._hidden_last_layer(ids, msk)
        return h[torch.arange(B, device=dev), pred_pos, :]

    # -------------------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------------------

    def _jepa_loss(self, pred: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        obj = self.config.jepa_objective
        target = target.detach()                        # stop-grad target
        if obj == "cos":
            return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        if obj == "l2":
            return torch.linalg.norm(pred - target, ord=2, dim=-1).mean()
        if obj == "mse":
            return ((pred - target) ** 2).mean()
        if obj == "infonce":
            p = F.normalize(pred, dim=-1)
            t = F.normalize(target, dim=-1)
            logits = (p @ t.t()) / self.config.infonce_temp
            labels = torch.arange(p.shape[0], device=p.device)
            return F.cross_entropy(logits, labels)
        raise ValueError(f"unknown jepa_objective {obj!r}")

    def forward(
        self,
        lm_input_ids: torch.Tensor,
        lm_attention_mask: torch.Tensor,
        lm_labels: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        **_,
    ) -> dict:
        """One training step.

        lm_*  : Text+Code concatenation for the standard causal-LM loss
                (`lm_labels` already masks the Text portion with -100).
        text_*: Text view only          -> Enc(Text) via predictor.
        code_*: Code view only          -> Enc(Code) target.
        """
        lm_out = self.backbone(
            input_ids=lm_input_ids, attention_mask=lm_attention_mask,
            labels=lm_labels, use_cache=False,
        )
        lm_loss = lm_out.loss

        pred = self.predict(text_input_ids, text_attention_mask)
        with torch.no_grad():
            code_emb = self.encode(code_input_ids, code_attention_mask)
        jepa_loss = self._jepa_loss(pred, code_emb)

        total = self.config.gamma * lm_loss + self.config.lbd * jepa_loss
        return {"loss": total, "lm_loss": lm_loss.detach(),
                "jepa_loss": jepa_loss.detach()}

    # -------------------------------------------------------------------------
    # Inference helper (eval + frozen) — sentence embedding extraction
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> torch.Tensor:
        """Frozen Enc(.) for a list of strings (eval mode, no grad)."""
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.config.max_length,
                             return_tensors="pt")
        dev = next(self.backbone.parameters()).device
        return self.encode(enc["input_ids"].to(dev),
                           enc["attention_mask"].to(dev))

    # -------------------------------------------------------------------------
    # Save / load
    # -------------------------------------------------------------------------

    def save_backbone(self, path: str):
        """Persist the deployable backbone (+ tokenizer) as a normal HF dir.
        With LoRA this saves the adapter."""
        self.backbone.save_pretrained(path)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)
        logger.info("saved backbone -> %s", path)
