"""Dynamic recommended-spec preflight.

Given a local HF model directory and the host it would run on, estimate the
memory footprint and decide whether the host can serve it on the vLLM CPU
backend, emitting a PASS / WARN / FAIL report plus recommended env values.

No ``vllm`` / ``torch`` import — usable standalone (``preflight_cli.py``),
from the FastAPI lifespan, and on the AVX2-only dev box.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
from dataclasses import dataclass, field

from cpu_topology import HostInfo, probe_host

log = logging.getLogger("be.preflight")

_GIB = 1024 ** 3

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

# bytes per parameter by dtype / quantization
_DTYPE_BYTES = {
    "bfloat16": 2.0, "bf16": 2.0,
    "float16": 2.0, "fp16": 2.0, "half": 2.0,
    "float32": 4.0, "fp32": 4.0, "float": 4.0,
    "int8": 1.0, "w8a8": 1.0, "compressed-tensors": 1.0,
    "fp8": 1.0,
    "awq": 0.6, "gptq": 0.6, "int4": 0.6,
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class Recommendations:
    tp_size: int = 1
    kvcache_space_gib: int = 0
    omp_threads_bind: str = ""
    reserved_cpu: int = 1
    ld_preload: str = ""


@dataclass
class PreflightReport:
    overall: str = PASS
    checks: list[Check] = field(default_factory=list)
    estimates: dict = field(default_factory=dict)
    host: dict = field(default_factory=dict)
    recommendations: Recommendations = field(default_factory=Recommendations)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))
        if _RANK[status] > _RANK[self.overall]:
            self.overall = status


# --------------------------------------------------------------------------
# model weight estimation
# --------------------------------------------------------------------------
def _load_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        log.warning("cannot read %s: %s", path, e)
        return {}


def _dtype_bytes(name: str) -> float:
    return _DTYPE_BYTES.get(name.lower().replace("-", ""), 2.0) if name else 2.0


def _params_from_config(c: dict) -> int:
    """Rough parameter count for a standard transformer decoder."""
    h = c.get("hidden_size") or c.get("n_embd") or 0
    layers = c.get("num_hidden_layers") or c.get("n_layer") or 0
    vocab = c.get("vocab_size") or 0
    ffn = c.get("intermediate_size") or (4 * h)
    n_heads = c.get("num_attention_heads") or c.get("n_head") or 1
    n_kv = c.get("num_key_value_heads") or n_heads
    if not (h and layers and vocab):
        return 0
    head_dim = c.get("head_dim") or (h // max(n_heads, 1))
    # attention: q (h*h) + k,v (h * n_kv*head_dim each) + o (h*h)
    attn = 2 * h * h + 2 * h * (n_kv * head_dim)
    # MLP: gated/SwiGLU (3 matrices) vs classic (2 matrices)
    act = str(c.get("hidden_act", "")).lower()
    gated = "glu" in act or "silu" in act or "swish" in act
    mlp = (3 if gated else 2) * h * ffn
    per_layer = attn + mlp
    embed = vocab * h * (1 if c.get("tie_word_embeddings") else 2)
    return per_layer * layers + embed


def estimate_weight_bytes(model_dir: str, cfg_dtype: str, cfg_quant: str):
    """Return (bytes, method, quality)."""
    # 1) index file total_size
    for idx in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(model_dir, idx)
        if os.path.isfile(p):
            try:
                with open(p) as fh:
                    total = json.load(fh).get("metadata", {}).get("total_size")
                if total:
                    return int(total), "index.total_size", "high"
            except (OSError, ValueError):
                pass

    # 2) sum shard files
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not shards:
        shards = glob.glob(os.path.join(model_dir, "*.bin"))
    if shards:
        total = sum(os.path.getsize(s) for s in shards)
        if total:
            return total, "sum_shards", "high"

    # 3) estimate from config.json
    c = _load_config(model_dir)
    params = _params_from_config(c)
    if params:
        quant = cfg_quant or (
            (c.get("quantization_config") or {}).get("quant_method", "")
        )
        dtype = cfg_dtype or str(c.get("torch_dtype", "bfloat16"))
        bpp = _dtype_bytes(quant) if quant else _dtype_bytes(dtype)
        return int(params * bpp), "config_estimate", "low"

    return 0, "unknown", "none"


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------
def _largest_divisor_leq(n: int, cap: int) -> int:
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_ld_preload() -> str:
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",
        "/usr/lib64/libtcmalloc_minimal.so.4",
    ]
    iomp = [
        "/usr/lib/x86_64-linux-gnu/libiomp5.so",
        "/opt/intel/oneapi/compiler/latest/lib/libiomp5.so",
    ]
    found = [p for p in candidates if os.path.exists(p)]
    found += [p for p in iomp if os.path.exists(p)]
    return ":".join(found)


def run_preflight(
    model_dir: str,
    *,
    dtype: str = "bfloat16",
    quantization: str = "",
    tp_size: int = 0,
    max_num_seqs: int = 256,
    max_model_len: int = 0,
    host: HostInfo | None = None,
) -> PreflightReport:
    host = host or probe_host()
    rep = PreflightReport()

    if not model_dir or not os.path.isdir(model_dir):
        rep.add("model_path", FAIL, f"model dir not found: {model_dir!r}")
        rep.host = _host_dict(host)
        return rep

    c = _load_config(model_dir)
    layers = c.get("num_hidden_layers") or c.get("n_layer") or 0
    n_heads = c.get("num_attention_heads") or c.get("n_head") or 1
    n_kv = c.get("num_key_value_heads") or n_heads
    h = c.get("hidden_size") or c.get("n_embd") or 0
    head_dim = c.get("head_dim") or (h // max(n_heads, 1))
    model_len = max_model_len or c.get("max_position_embeddings") or 4096

    weight_bytes, method, quality = estimate_weight_bytes(model_dir, dtype, quantization)
    if weight_bytes == 0:
        rep.add("model_size", FAIL, "could not determine model size (no weights/config)")

    # KV cache (bf16) worst-case at full concurrency
    kv_per_tok = 2 * layers * n_kv * head_dim * 2
    kv_total = kv_per_tok * model_len * max_num_seqs

    # recommended TP = NUMA nodes, reduced to divide the layer count
    nodes = host.numa_count
    rec_tp = tp_size or (
        _largest_divisor_leq(layers, nodes) if layers else max(nodes, 1)
    )
    rec_tp = max(rec_tp, 1)

    act_overhead = max(2 * _GIB, int(0.1 * weight_bytes / rec_tp))
    per_rank = weight_bytes / rec_tp + kv_total / rec_tp + act_overhead
    host_need = weight_bytes + kv_total + rec_tp * act_overhead
    host_need = int(host_need * 1.05)

    rep.estimates = {
        "weight_bytes": weight_bytes,
        "method": method,
        "estimate_quality": quality,
        "kv_total_bytes": kv_total,
        "per_rank_bytes": int(per_rank),
        "host_need_bytes": host_need,
        "model_len": model_len,
        "num_layers": layers,
    }

    # --- ISA ------------------------------------------------------------
    isa = host.isa
    if not isa.get("avx512f"):
        rep.add("isa", FAIL,
                "no avx512f — prebuilt vLLM CPU wheel will not run "
                "(this is expected on the AVX2-only dev box)")
    elif not (isa.get("amx_bf16") or isa.get("avx512_bf16")):
        rep.add("isa", WARN, "avx512f present but no amx_bf16/avx512_bf16 — slow bf16")
    else:
        rep.add("isa", PASS, "amx_bf16/avx512_bf16 available")

    # --- dtype ----------------------------------------------------------
    d = dtype.lower()
    if d in ("float16", "fp16", "half"):
        rep.add("dtype", FAIL, "float16 is unstable on torch CPU — use bfloat16")
    elif d in ("float32", "fp32"):
        rep.add("dtype", WARN, "float32 doubles RAM — prefer bfloat16")
    else:
        rep.add("dtype", PASS, f"dtype={dtype}")

    # --- total RAM ------------------------------------------------------
    if host.mem_total < host_need:
        rep.add("ram_total", FAIL,
                f"need ~{host_need/_GIB:.1f} GiB, host has "
                f"{host.mem_total/_GIB:.1f} GiB")
    elif host.mem_total < host_need * 1.15:
        rep.add("ram_total", WARN,
                f"tight: need ~{host_need/_GIB:.1f} GiB, host has "
                f"{host.mem_total/_GIB:.1f} GiB (<15% headroom)")
    else:
        rep.add("ram_total", PASS,
                f"need ~{host_need/_GIB:.1f} GiB, host has "
                f"{host.mem_total/_GIB:.1f} GiB")

    # --- per-rank vs smallest NUMA node (exitcode-9 OOM guard) ----------
    node_ram = host.smallest_node_ram
    if per_rank > node_ram:
        rep.add("per_rank_numa", FAIL,
                f"per-rank ~{per_rank/_GIB:.1f} GiB > smallest NUMA node "
                f"{node_ram/_GIB:.1f} GiB — TP worker will be OOM-killed "
                f"(exitcode 9). Increase TP / quantize / lower "
                f"max_num_seqs/max_model_len")
    elif per_rank > node_ram * 0.9:
        rep.add("per_rank_numa", WARN,
                f"per-rank ~{per_rank/_GIB:.1f} GiB is 90-100% of NUMA node "
                f"{node_ram/_GIB:.1f} GiB")
    else:
        rep.add("per_rank_numa", PASS,
                f"per-rank ~{per_rank/_GIB:.1f} GiB fits NUMA node "
                f"{node_ram/_GIB:.1f} GiB (tp={rec_tp})")

    # --- python / gcc ---------------------------------------------------
    py = host.python_version
    if not (3, 10) <= py <= (3, 13):
        rep.add("python", FAIL, f"Python {py[0]}.{py[1]} outside 3.10-3.13")
    else:
        rep.add("python", PASS, f"Python {py[0]}.{py[1]}")
    if host.gcc_version and host.gcc_version < (12, 3):
        rep.add("gcc", WARN,
                f"gcc {host.gcc_version[0]}.{host.gcc_version[1]} < 12.3 "
                f"(only matters for from-source builds)")

    # --- quantization vs ISA -------------------------------------------
    q = (quantization or "").lower()
    if q in ("awq", "gptq") and not isa.get("avx512f"):
        rep.add("quant", FAIL, f"{q} CPU kernels need AVX512, host is AVX2-only")

    # --- recommendations ------------------------------------------------
    reserved = 1
    binds: list[str] = []
    for n in host.numa_nodes[:rec_tp]:
        usable = sorted(n.cpus)[: max(len(n.cpus) - reserved, 1)]
        if usable:
            binds.append(f"{usable[0]}-{usable[-1]}")
    rep.recommendations = Recommendations(
        tp_size=rec_tp,
        kvcache_space_gib=math.ceil(kv_total / _GIB * 1.1) if kv_total else 0,
        omp_threads_bind="|".join(binds),
        reserved_cpu=reserved,
        ld_preload=_find_ld_preload(),
    )
    if not rep.recommendations.ld_preload:
        rep.add("ld_preload", WARN,
                "libtcmalloc_minimal / libiomp5 not found — install "
                "google-perftools and Intel OpenMP for best CPU performance")

    rep.host = _host_dict(host)
    return rep


def _host_dict(h: HostInfo) -> dict:
    return {
        "mem_total_gib": round(h.mem_total / _GIB, 1),
        "mem_available_gib": round(h.mem_available / _GIB, 1),
        "numa_nodes": [
            {"id": n.node_id, "ram_gib": round(n.ram_bytes / _GIB, 1),
             "cpus": len(n.cpus)}
            for n in h.numa_nodes
        ],
        "numa_source": h.numa_source,
        "isa": {k: v for k, v in h.isa.items() if v},
        "logical_cores": h.logical_cores,
        "python": f"{h.python_version[0]}.{h.python_version[1]}",
        "gcc": (".".join(map(str, h.gcc_version)) if h.gcc_version else None),
    }
