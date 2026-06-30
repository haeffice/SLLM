"""Physics BE(/predict, /health) HTTP 클라이언트.

base_url 기본값은 PHYSICS_BE_URL 환경변수 → 없으면 127.0.0.1:9003.
predict()는 메쉬 파일 바이트를 받아 Base64로 감싸 전송하고, 변형된 메쉬
파일 바이트를 디코드해 돌려준다 (FE는 Base64를 의식하지 않는다).
"""

from __future__ import annotations

import base64
import os

import requests

DEFAULT_URL = os.environ.get("PHYSICS_BE_URL", "http://127.0.0.1:9003")


class PhysicsClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        health_timeout: float = 3.0,
    ):
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        # health는 매 Simulate마다 호출되므로 짧게 — 미연결 시 GUI가 길게 멈추지 않게.
        self.health_timeout = health_timeout

    def health(self) -> dict:
        """모델 로드 상태 dict. (200=ok, 503=loading/failed — body는 동일 구조)"""
        r = requests.get(f"{self.base_url}/health", timeout=self.health_timeout)
        return r.json()

    def predict(
        self,
        mesh_bytes: bytes,
        file_format: str,
        action: dict,
        model: str | None = None,
    ) -> bytes:
        """메쉬 + action → 변형된 메쉬 파일 바이트. 실패 시 RuntimeError."""
        payload = {
            "mesh_base64": base64.b64encode(mesh_bytes).decode("ascii"),
            "file_format": file_format,
            "action": action,
        }
        params = {"model": model} if model else None
        r = requests.post(
            f"{self.base_url}/predict", json=payload, params=params, timeout=self.timeout
        )
        if r.status_code != 200:
            raise RuntimeError(f"predict failed [{r.status_code}]: {_detail(r)}")
        return base64.b64decode(r.json()["result_mesh_base64"])


def _detail(resp: "requests.Response") -> str:
    """에러 응답에서 사람이 읽을 메시지 추출 (detail → error → raw text)."""
    try:
        body = resp.json()
        return body.get("detail") or body.get("error") or str(body)
    except ValueError:
        return resp.text
