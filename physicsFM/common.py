"""공용 유틸 — 설정 로드(--set 오버라이드), 로깅, git 커밋 조회.

요약 흐름: 모든 CLI(generate_rollouts/train/rollout)가 같은 config.yaml을 읽고
"--set a.b=v" 형태의 오버라이드를 받는다. 값은 yaml.safe_load 로 파싱해
숫자/불리언/리스트가 자연스럽게 변환되게 한다.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """config.yaml 로드 + "a.b=v" 오버라이드 적용."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"--set 형식은 key.sub=value: {item!r}")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def git_commit(repo_dir: str | Path) -> str:
    """해당 디렉터리의 git HEAD 해시 (실패 시 'unknown')."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
