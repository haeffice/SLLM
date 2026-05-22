"""LLM-JEPA fine-tuning (PyTorch 2.8 / transformers).

Fine-tunes a causal-LM backbone with the LLM-JEPA objective
(arXiv:2509.14252): standard next-token cross-entropy on the Text->Code
sequence PLUS a JEPA term that predicts Enc(Code) from the tied-weights
predictor applied to Enc(Text).

Stack (per `paper/CLAUDE.md`):
    * `transformers`-based model class (`LLMJEPA`, wraps AutoModelForCausalLM)
    * `transformers.Trainer` training loop
    * `torch.utils.data.Dataset` / `DataLoader` view-pair pipeline
    * `accelerate` + `torchrun` for multi-GPU (CPU-compatible)
    * logs: per-module trainable params; first batch's first sample
      UNTOKENIZED prompt before the model is fed; step / train_loss /
      lm_loss / jepa_loss / valid_loss / lr every `logging_steps`;
      checkpoints every `save_steps` with the step number in the filename;
      abort if an init checkpoint fails to load.

Run via `run_train_LLMJEPA.sh config.yaml` (the shell enforces the
"no pre-existing checkpoints" guard). All arguments live in `config.yaml`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.optimization import get_cosine_schedule_with_warmup

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from LLMJEPA import LLMJEPA, LLMJEPAConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_llmjepa")


# =============================================================================
# Dataset — view-pair JSONL  ->  (LM seq, Text view, Code view)
# =============================================================================

def _ids(tokenizer, text: str, max_length: int) -> list[int]:
    enc = tokenizer(text, padding=False, truncation=True,
                    max_length=max_length, add_special_tokens=False,
                    return_tensors="pt")
    return enc["input_ids"][0].tolist()


class ViewPairDataset(Dataset):
    """JSONL with one example per line. Accepted schemas:

        {"text": "<NL>", "code": "<SQL/regex/code>"}
        {"messages": [{"role": "user", "content": "<NL>"},
                       {"role": "assistant", "content": "<code>"}]}

    Produces, per item, token-id lists for: the LM sequence
    (Text+Code, Text masked with -100 in labels), the Text-only view,
    and the Code-only view. Plus the raw Text/Code strings for logging.
    """

    def __init__(self, path: str, tokenizer, max_length: int):
        self.rows: list[tuple[str, str]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "messages" in obj:
                    msgs = obj["messages"]
                    text = next(m["content"] for m in msgs if m["role"] == "user")
                    code = next(m["content"] for m in msgs if m["role"] == "assistant")
                else:
                    text, code = obj["text"], obj["code"]
                self.rows.append((str(text), str(code)))
        if not self.rows:
            raise ValueError(f"Empty dataset: {path}")
        self.tok = tokenizer
        self.max_length = max_length
        self.eos = getattr(tokenizer, "eos_token_id", None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        text, code = self.rows[idx]
        t_ids = _ids(self.tok, text, self.max_length)
        c_ids = _ids(self.tok, code, self.max_length)
        if self.eos is not None:
            c_ids = c_ids + [self.eos]
        lm_ids = t_ids + c_ids
        lm_labels = [-100] * len(t_ids) + list(c_ids)
        return {
            "lm_ids": lm_ids, "lm_labels": lm_labels,
            "text_ids": t_ids if t_ids else [self.eos or 0],
            "code_ids": c_ids,
            "text_str": text, "code_str": code,
        }


def make_collate(pad_id: int):
    def _pad(seqs: list[list[int]], fill: int):
        n = max(len(s) for s in seqs)
        ids = [s + [fill] * (n - len(s)) for s in seqs]
        msk = [[1] * len(s) + [0] * (n - len(s)) for s in seqs]
        return torch.tensor(ids), torch.tensor(msk)

    def collate(batch: list[dict]) -> dict:
        lm_ids, lm_mask = _pad([b["lm_ids"] for b in batch], pad_id)
        lab, _ = _pad([b["lm_labels"] for b in batch], -100)
        t_ids, t_mask = _pad([b["text_ids"] for b in batch], pad_id)
        c_ids, c_mask = _pad([b["code_ids"] for b in batch], pad_id)
        return {
            "lm_input_ids": lm_ids, "lm_attention_mask": lm_mask,
            "lm_labels": lab,
            "text_input_ids": t_ids, "text_attention_mask": t_mask,
            "code_input_ids": c_ids, "code_attention_mask": c_mask,
            "text_str": [b["text_str"] for b in batch],
            "code_str": [b["code_str"] for b in batch],
        }
    return collate


# =============================================================================
# transformers wrapper
# =============================================================================

class LLMJEPAHFConfig(PretrainedConfig):
    model_type = "llm_jepa"

    def __init__(self, jepa_kwargs: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.jepa_kwargs = jepa_kwargs or {}


class LLMJEPAHFModel(PreTrainedModel):
    config_class = LLMJEPAHFConfig

    def __init__(self, config: LLMJEPAHFConfig):
        super().__init__(config)
        self.jepa = LLMJEPA(LLMJEPAConfig(**config.jepa_kwargs))
        self.post_init()

    def _init_weights(self, module):  # LLMJEPA self-initialises (HF backbone).
        pass

    def forward(self, **batch) -> dict:
        batch.pop("text_str", None)
        batch.pop("code_str", None)
        return self.jepa(**batch)


# =============================================================================
# Callbacks
# =============================================================================

def _unwrap(model) -> LLMJEPAHFModel:
    return model.module if hasattr(model, "module") else model


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        j = _unwrap(model).jepa
        logger.info("==== trainable parameters ====")
        logger.info("  %-28s %15s", "backbone (LLM)", f"{_count(j.backbone):,}")
        logger.info("  %-28s %15s", "TOTAL trainable", f"{_count(j):,}")
        logger.info("==============================")


class FirstSampleLogCallback(TrainerCallback):
    """Log the first batch's first sample as the UNTOKENIZED prompt."""

    def __init__(self):
        self.done = False

    def mark(self, text: list[str], code: list[str]):
        if self.done:
            return
        self.done = True
        logger.info(
            "==== first batch / first sample (pre-feed, untokenized) ====\n"
            "  Text view : %s\n  Code view : %s",
            text[0] if text else "<unknown>",
            code[0] if code else "<unknown>",
        )


class LossLogCallback(TrainerCallback):
    """Surface the latest lm_loss / jepa_loss in the periodic train log."""

    def __init__(self):
        self.lm = None
        self.jepa = None

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None:
            if self.lm is not None:
                logs["lm_loss"] = round(float(self.lm), 6)
            if self.jepa is not None:
                logs["jepa_loss"] = round(float(self.jepa), 6)


class SaveBackboneCallback(TrainerCallback):
    """On every HF checkpoint, also dump the deployable backbone into a
    step-numbered dir (loadable with AutoModelForCausalLM / peft)."""

    def on_save(self, args, state, control, model=None, **kw):
        if not state.is_world_process_zero:
            return
        out = os.path.join(args.output_dir,
                           f"backbone_step{state.global_step}")
        _unwrap(model).jepa.save_backbone(out)
        logger.info("saved backbone checkpoint -> %s", out)


# =============================================================================
# Trainer subclass — first-sample logging + lm/jepa loss surfacing
# =============================================================================

class LLMJEPAHFTrainer(Trainer):
    def __init__(self, *a, first_cb=None, loss_cb=None, **kw):
        super().__init__(*a, **kw)
        self._first_cb = first_cb
        self._loss_cb = loss_cb

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._first_cb is not None:
            self._first_cb.mark(inputs.get("text_str"), inputs.get("code_str"))
        outputs = model(**inputs)
        if self._loss_cb is not None:
            self._loss_cb.lm = outputs.get("lm_loss")
            self._loss_cb.jepa = outputs.get("jepa_loss")
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            decay, no_decay = [], []
            for _, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 else decay).append(p)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.args.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.optimizer, self.lr_scheduler


# =============================================================================
# Config + main
# =============================================================================

@dataclass
class _Cfg:
    raw: dict

    def sec(self, k: str) -> dict:
        return self.raw.get(k, {}) or {}


def _build_jepa_kwargs(c: _Cfg) -> dict:
    m = c.sec("model")
    kw: dict = {}
    for k in ("model_name", "num_predictors", "front_pred", "jepa_objective",
              "lbd", "gamma", "infonce_temp", "max_length", "torch_dtype",
              "use_lora", "lora_rank", "lora_alpha", "lora_dropout",
              "lora_target_modules", "tiny_hidden", "tiny_layers",
              "tiny_heads", "tiny_inter"):
        if k in m:
            kw[k] = m[k]
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = _Cfg(yaml.safe_load(f))
    d, t, o = cfg.sec("data"), cfg.sec("train"), cfg.sec("optim")

    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    jepa_kwargs = _build_jepa_kwargs(cfg)
    logger.info("LLMJEPA kwargs: %s", jepa_kwargs)
    model = LLMJEPAHFModel(LLMJEPAHFConfig(jepa_kwargs=jepa_kwargs))

    init_from = t.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location="cpu", weights_only=True)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.jepa.load_state_dict(sd, strict=False)
        if len(sd) - len(unexpected) == 0:
            raise RuntimeError(
                f"init_from={init_from!r}: 0/{len(sd)} keys applied — aborting.")
        logger.info("init_from %s: applied %d/%d keys",
                    init_from, len(sd) - len(unexpected), len(sd))

    tok = model.jepa.tokenizer
    max_len = int(jepa_kwargs.get("max_length", 512))
    train_ds = ViewPairDataset(d["train_file"], tok, max_len)
    eval_ds = (ViewPairDataset(d["valid_file"], tok, max_len)
               if d.get("valid_file") else None)
    collate = make_collate(tok.pad_token_id)

    use_cpu = not torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        max_steps=int(o.get("max_steps", 1000)),
        per_device_train_batch_size=int(o.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(o.get("per_device_eval_batch_size",
                                             o.get("per_device_train_batch_size", 4))),
        gradient_accumulation_steps=int(o.get("gradient_accumulation_steps", 1)),
        learning_rate=float(o.get("learning_rate", 2.0e-5)),
        weight_decay=float(o.get("weight_decay", 0.0)),
        adam_beta1=float(o.get("adam_beta1", 0.9)),
        adam_beta2=float(o.get("adam_beta2", 0.999)),
        warmup_steps=int(o.get("warmup_steps", 0)),
        logging_steps=int(t.get("logging_steps", 10)),
        save_steps=int(t.get("save_steps", 200)),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=int(t.get("eval_steps", 200)),
        dataloader_num_workers=int(t.get("num_workers", 0)),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        label_names=["lm_labels"],
        use_cpu=use_cpu,
        ddp_backend=("gloo" if use_cpu else None),
        max_grad_norm=float(o.get("max_grad_norm", 1.0)),
    )

    first_cb, loss_cb = FirstSampleLogCallback(), LossLogCallback()
    trainer = LLMJEPAHFTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate,
        first_cb=first_cb, loss_cb=loss_cb,
        callbacks=[ParamCountCallback(), first_cb, loss_cb,
                   SaveBackboneCallback()],
    )

    trainer.train()
    model.jepa.save_backbone(os.path.join(targs.output_dir, "final"))
    logger.info("training complete; final backbone -> %s/final", targs.output_dir)


if __name__ == "__main__":
    main()
