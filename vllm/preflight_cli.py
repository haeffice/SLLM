#!/usr/bin/env python3
"""Standalone recommended-spec check.

    python preflight_cli.py <model_dir> [--dtype bfloat16] [--quantization awq]
                                        [--tp N] [--max-num-seqs 256]
                                        [--max-model-len 0] [--json]

Exit code: 0 = PASS, 1 = WARN, 2 = FAIL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from preflight import FAIL, PASS, WARN, run_preflight

_EXIT = {PASS: 0, WARN: 1, FAIL: 2}
_COLOR = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
_RESET = "\033[0m"


def _fmt(report) -> str:
    lines: list[str] = []
    e = report.estimates
    if e:
        lines.append(
            f"model: {e.get('weight_bytes', 0) / 1024**3:.1f} GiB weights "
            f"(method={e.get('method')}, quality={e.get('estimate_quality')}), "
            f"{e.get('num_layers')} layers, ctx={e.get('model_len')}"
        )
        lines.append(
            f"       KV {e.get('kv_total_bytes', 0) / 1024**3:.1f} GiB, "
            f"per-rank ~{e.get('per_rank_bytes', 0) / 1024**3:.1f} GiB, "
            f"host needs ~{e.get('host_need_bytes', 0) / 1024**3:.1f} GiB"
        )
    h = report.host
    if h:
        nodes = ", ".join(
            f"n{n['id']}:{n['ram_gib']}G/{n['cpus']}cpu" for n in h["numa_nodes"]
        )
        lines.append(
            f"host:  {h['mem_total_gib']} GiB RAM, [{nodes}] ({h['numa_source']}), "
            f"ISA={','.join(h['isa']) or 'none'}, py{h['python']}"
        )
    lines.append("")
    for c in report.checks:
        col = _COLOR.get(c.status, "")
        lines.append(f"  {col}[{c.status:4}]{_RESET} {c.name:14} {c.detail}")
    r = report.recommendations
    lines.append("")
    lines.append("recommended env (set in run.sh BEFORE uvicorn starts):")
    lines.append(f"  VLLM_BE_TP_SIZE={r.tp_size}")
    lines.append(f"  VLLM_CPU_KVCACHE_SPACE={r.kvcache_space_gib}")
    lines.append(f"  VLLM_CPU_OMP_THREADS_BIND={r.omp_threads_bind or '(n/a)'}")
    lines.append(f"  VLLM_CPU_NUM_OF_RESERVED_CPU={r.reserved_cpu}")
    lines.append(f"  LD_PRELOAD={r.ld_preload or '(not found)'}")
    lines.append("")
    col = _COLOR.get(report.overall, "")
    lines.append(f"OVERALL: {col}{report.overall}{_RESET}")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="vLLM CPU recommended-spec preflight")
    p.add_argument("model_dir")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantization", default="")
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=0)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    report = run_preflight(
        a.model_dir,
        dtype=a.dtype,
        quantization=a.quantization,
        tp_size=a.tp,
        max_num_seqs=a.max_num_seqs,
        max_model_len=a.max_model_len,
    )
    if a.json:
        print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    else:
        print(_fmt(report))
    return _EXIT[report.overall]


if __name__ == "__main__":
    sys.exit(main())
