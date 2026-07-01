"""End-to-end smoke test: 샘플 그리드 메쉬를 만들어 /predict에 보내고
변형 결과를 검증한다. 서버가 떠 있어야 한다 (./run.sh).

    python smoke_test.py            # http://127.0.0.1:9003 기본
    PHYSICS_BE_URL=... python smoke_test.py

stdlib(urllib)만 사용 — BE 의존성 외 추가 설치 불필요. utils.mesh_handler를
재사용해 메쉬를 만들고/결과를 다시 파싱하므로 writer/reader 경로도 함께 탄다.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

from utils.mesh_handler import load_mesh_from_bytes, write_mesh_to_bytes

BE_URL = os.environ.get("PHYSICS_BE_URL", "http://127.0.0.1:9003").rstrip("/")
FILE_FORMAT = "vtk"


def make_grid(n: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """z=0 평면 위 n×n 정점 + 삼각형 face. (단위 정사각형 [0,1]^2)"""
    xs = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(xs, xs)
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)])
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b, c, d = a + 1, a + n, a + n + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts.astype(np.float64), np.asarray(faces, dtype=np.int64)


def _post_predict(payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BE_URL}/predict", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    n = 11
    verts, faces = make_grid(n)
    center = (n // 2) * n + (n // 2)  # 중앙 정점 = 충격점
    corner = 0  # (0,0) — 반경 밖이라 안 움직여야 함

    mesh_bytes = write_mesh_to_bytes(verts, faces, FILE_FORMAT)
    payload = {
        "mesh_base64": base64.b64encode(mesh_bytes).decode("ascii"),
        "file_format": FILE_FORMAT,
        "action": {"impact_node": center, "force": [0.0, 0.0, -0.3], "radius": 0.4},
    }

    # --- /health -------------------------------------------------------------
    try:
        with urllib.request.urlopen(f"{BE_URL}/health", timeout=10) as r:
            health = json.loads(r.read().decode("utf-8"))
        print(f"[health] {health['status']} — models={list(health['models'])}")
    except urllib.error.URLError as e:
        print(f"FAIL: cannot reach {BE_URL} — is the server running? ({e})")
        return 1

    # --- /predict ------------------------------------------------------------
    try:
        out = _post_predict(payload)
    except urllib.error.HTTPError as e:
        print(f"FAIL: /predict HTTP {e.code} — {e.read().decode('utf-8', 'replace')}")
        return 1

    assert out.get("success") is True, f"success != True: {out}"
    assert out.get("result_mesh_base64"), "missing result_mesh_base64"
    print(f"[predict] success, model={out.get('model_id')}, N={out.get('num_vertices')}")

    # --- 결과 검증 -----------------------------------------------------------
    result_bytes = base64.b64decode(out["result_mesh_base64"])
    new_verts, _ = load_mesh_from_bytes(result_bytes, FILE_FORMAT)
    assert new_verts.shape == verts.shape, f"shape changed: {new_verts.shape} != {verts.shape}"

    center_dz = float(new_verts[center, 2] - verts[center, 2])
    corner_dz = float(new_verts[corner, 2] - verts[corner, 2])
    print(f"[verify] center Δz={center_dz:+.4f} (expect ~-0.3), corner Δz={corner_dz:+.4f} (expect ≪ center)")

    # 모델-불문 검증: 충격점은 힘(-0.3)만큼 눌리고, 반경 밖 노드는 훨씬 덜 움직인다.
    # (dummy=선형 hard-cutoff → corner 0; metal_dent=가우시안 → 작은 tail.)
    assert abs(center_dz - (-0.3)) < 1e-3, f"impact node not deformed as expected: {center_dz}"
    assert abs(corner_dz) < 0.1 * abs(center_dz), f"node outside radius moved too much: {corner_dz}"

    print("OK ✓ end-to-end /predict deformation verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
