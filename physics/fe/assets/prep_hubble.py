"""허블 우주망원경 자산 베이커 (오프라인 prep 스크립트 — 앱에서 import하지 않음).

NASA-3D-Resources의 "Hubble 25th Anniversary Model" 부품 STL 5개(전시 받침대
Base 제외)를 내려받아 meshio로 파싱한 뒤, 단일 geometry-only OBJ(hubble.obj)로
병합하고 부품별 정점 범위·임계값을 hubble.components.json에 bake한다.

출처/라이선스: NASA public domain — "These assets are free and without copyright"
  https://github.com/nasa/NASA-3D-Resources (README.md)
  커밋 고정: 11ebb4ee043715aefbba6aeec8a61746fad67fa7

인덱스 일관성이 이 자산의 핵심 계약이다:
  - OBJ의 v-라인 순서 = 부품 병합 순서. meshio는 v-라인 순서 그대로 points를
    쌓으므로, components.json의 vertex_range [start, end)는 FE/BE 어느 쪽
    meshio 파싱에서도 같은 정점을 가리킨다.
  - 그룹 마커(o/g 라인)는 쓰지 않는다 — 파서 상호작용 없이 주석(#)으로만 표기.
  - 스크립트가 쓰고 난 뒤 meshio로 재파싱해 점 좌표/면이 병합 배열과 일치하는지
    검증한다(assert). 검증 실패 시 자산을 커밋하면 안 된다.

좌표 정규화: 병합 후 중심을 원점으로, bbox 대각선을 2.0으로 맞춘다 — 내장
판(size 2.0)/캔(height 2.0)과 스케일이 같아 기존 force/radius 기본값이 그대로
말이 된다. 임계값은 정규화된 메쉬 단위의 절대값으로 bake한다.

사용법:
  python prep_hubble.py            # 다운로드 → ./hubble.obj, ./hubble.components.json
  python prep_hubble.py --src-dir <dir>  # 이미 받아둔 STL 디렉토리 재사용
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.parse
import urllib.request

import meshio
import numpy as np

COMMIT = "11ebb4ee043715aefbba6aeec8a61746fad67fa7"
BASE_URL = (
    "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/"
    f"{COMMIT}/3D%20Printing/Hubble%2025th%20Anniversary%20Model/"
)

# 병합 순서 = vertex_range 순서. threshold_fraction은 정규화 후 bbox 대각선(2.0)
# 대비 비율 — 취약 부품일수록 낮다. 수치는 데모용 상대치(실측 물성 아님).
#
# functional 블록 = fe/analysis.py가 type으로 디스패치하는 기능 영향 분석 상수.
# 출처(데모 근사, 실제 허블 스펙 기반):
#   - antenna: HGA 빔폭 "over four degrees", S-band 2287.5 MHz(λ≈13.1 cm)
#     — NASA SM3A Media Reference Guide (asd.gsfc.nasa.gov, HST Systems)
#   - solar_panel: SA3 GaAs 어레이 발전량 ≈5,000 W — NASA Hubble electrical-power
#   - optical_tube: 지향 안정도 0.007 arcsec RMS/24h — NASA Hubble pointing-control;
#     광학계 f/24, 관측 λ≈550 nm
#   - yield_strain: Al 6061-T6 ε_y=σ_y/E≈0.40% (MatWeb); 태양전지 셀은 취성
#     반도체 웨이퍼라 더 낮게 0.2% (데모 근사)
PARTS = [
    {
        "file": "Main body.stl",
        "id": "main_body",
        "name": "본체 경통",
        "material": "알루미늄 외피",
        "fragility": "low",
        "threshold_fraction": 0.05,
        "notes": "구조 강성이 높은 원통 경통 — 임계값이 가장 높다.",
        "functional": {
            "type": "optical_tube",
            "pointing_budget_arcsec": 0.007,
            "obs_wavelength_nm": 550,
            "f_number": 24,
        },
    },
    {
        "file": "Solar panels.stl",
        "id": "solar_panels",
        "name": "태양전지판",
        "material": "실리콘 셀 패널",
        "fragility": "high",
        "threshold_fraction": 0.015,
        "notes": "얇은 패널 — 점 충격에 셀 균열이 쉽게 발생, 임계값 최저.",
        "yield_strain": 0.002,
        "functional": {"type": "solar_panel", "p0_w": 5000},
    },
    {
        "file": "Radio dishes.stl",
        "id": "radio_dishes",
        "name": "고이득 안테나",
        "material": "접시형 반사판",
        "fragility": "high",
        "threshold_fraction": 0.02,
        "notes": "접시 곡면 변형 시 지향성 상실 — 임계값 낮음.",
        "functional": {"type": "antenna", "beam_deg": 4.0, "wavelength_m": 0.131},
    },
    {
        "file": "Cover hatch.stl",
        "id": "cover_hatch",
        "name": "개구부 해치",
        "material": "힌지 결합 덮개",
        "fragility": "medium",
        "threshold_fraction": 0.035,
        "notes": "힌지 이음새가 약점 — 변형 시 개폐 불능.",
    },
    {
        "file": "Body coupler.stl",
        "id": "body_coupler",
        "name": "동체 커플러",
        "material": "결합 링",
        "fragility": "medium",
        "threshold_fraction": 0.035,
        "notes": "경통-후방 장비부 결합 링 — 이음새 어긋남 주의.",
    },
]

# 실물 환산 배율: 허블 전장 13.2 m ↔ 정규화 모델 (bbox 대각선 2.0) — 데모 근사.
REAL_SCALE_M_PER_UNIT = 6.9
# 기본 항복 변형률 (Al 6061-T6) — 부품별 yield_strain으로 덮어쓸 수 있음.
DEFAULT_YIELD_STRAIN = 0.004

NORMALIZED_DIAG = 2.0  # 정규화 목표 bbox 대각선 (내장 plate/can과 동일 스케일)

# 시나리오 기본 충격 반경(정규화 단위). 0.15면 기본 충격점(태양전지판 끝단)에서
# 취약 부품(전지판·안테나)만 FAIL하고 본체는 OK — 부품별 차이가 선명한 데모 기본값.
DEFAULT_RADIUS = 0.15


def fetch_part(filename: str, src_dir: str | None, dst_dir: str) -> str:
    """부품 STL을 src_dir에서 복사 경로로 쓰거나 NASA raw URL에서 내려받는다."""
    if src_dir:
        path = os.path.join(src_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--src-dir에 {filename!r} 없음: {path}")
        return path
    url = BASE_URL + urllib.parse.quote(filename)
    path = os.path.join(dst_dir, filename)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def load_part(path: str) -> tuple[np.ndarray, np.ndarray]:
    """STL → (points (N,3) float64, tris (M,3) int64). 삼각형 외 cell은 거부."""
    mesh = meshio.read(path)
    tris = [c.data for c in mesh.cells if c.type == "triangle"]
    others = sorted({c.type for c in mesh.cells} - {"triangle"})
    if others:
        raise ValueError(f"{path}: 삼각형이 아닌 cell {others} — STL 자산 가정 위반")
    if not tris:
        raise ValueError(f"{path}: triangle cell 없음")
    return (
        np.asarray(mesh.points, dtype=np.float64),
        np.concatenate([np.asarray(t, dtype=np.int64) for t in tris], axis=0),
    )


def write_obj(path: str, points: np.ndarray, faces: np.ndarray, ranges) -> None:
    """geometry-only OBJ 기록. o/g 라인 없음(주석만) — meshio 파싱 형태 불변.

    좌표는 repr(최단 round-trip 표현)로 기록 — 텍스트 왕복 후에도 float64가
    비트 단위로 보존된다 (아래 재파싱 검증이 엄격 일치를 확인).
    """
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("# Hubble Space Telescope - NASA 25th Anniversary Model (public domain)\n")
        f.write(f"# source: {BASE_URL} (commit {COMMIT})\n")
        f.write("# generated by prep_hubble.py - DO NOT edit by hand; re-bake instead.\n")
        f.write("# normalized: centered at origin, bbox diagonal = 2.0\n")
        for part_id, start, end in ranges:
            f.write(f"# part: {part_id} vertices [{start}, {end})\n")
        for x, y, z in points.tolist():  # tolist → 순수 float (np scalar repr 방지)
            f.write(f"v {x!r} {y!r} {z!r}\n")
        for a, b, c in faces + 1:  # OBJ는 1-based
            f.write(f"f {a} {b} {c}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src-dir", default=None, help="이미 받아둔 부품 STL 디렉토리 (미지정 시 다운로드)")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    all_points, all_faces, ranges = [], [], []
    offset = 0
    with tempfile.TemporaryDirectory() as tmp:
        for part in PARTS:
            path = fetch_part(part["file"], args.src_dir, tmp)
            points, faces = load_part(path)
            all_points.append(points)
            all_faces.append(faces + offset)
            ranges.append((part["id"], offset, offset + len(points)))
            print(f"  {part['id']}: {len(points)} verts, {len(faces)} tris")
            offset += len(points)

    points = np.concatenate(all_points, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    # 정규화: 중심 원점, bbox 대각선 NORMALIZED_DIAG. float32 왕복 대비 반올림.
    lo, hi = points.min(0), points.max(0)
    diag = float(np.linalg.norm(hi - lo))
    points = (points - (lo + hi) / 2.0) * (NORMALIZED_DIAG / diag)

    obj_path = os.path.join(out_dir, "hubble.obj")
    write_obj(obj_path, points, faces, ranges)

    # 검증: 기록한 OBJ를 meshio로 재파싱 → 병합 배열과 정확히 일치해야 한다.
    reread = meshio.read(obj_path)
    re_tris = np.concatenate(
        [np.asarray(c.data, dtype=np.int64) for c in reread.cells if c.type == "triangle"]
    )
    assert reread.points.shape == points.shape, "재파싱 점 개수 불일치"
    assert np.array_equal(re_tris, faces), "재파싱 face 불일치"
    assert np.allclose(reread.points, points, rtol=0, atol=0), "재파싱 좌표 불일치"

    # 임계값 bake + 기본 충격점: 태양전지판에서 원점에서 가장 먼 정점(패널 끝단).
    sp_start, sp_end = next((s, e) for pid, s, e in ranges if pid == "solar_panels")
    sp_slice = points[sp_start:sp_end]
    default_impact = sp_start + int(np.argmax(np.linalg.norm(sp_slice, axis=1)))

    components = []
    for part, (pid, start, end) in zip(PARTS, ranges):
        comp = {
            "id": pid,
            "name": part["name"],
            "material": part["material"],
            "fragility": part["fragility"],
            "notes": part["notes"],
            "damage_threshold": round(part["threshold_fraction"] * NORMALIZED_DIAG, 6),
            "warn_ratio": 0.7,
            "yield_strain": part.get("yield_strain", DEFAULT_YIELD_STRAIN),
            "vertex_range": [start, end],
        }
        if "functional" in part:
            comp["functional"] = part["functional"]
        components.append(comp)
    doc = {
        "schema_version": 1,
        "asset": "hubble.obj",
        "display_name": "허블 우주망원경",
        "source": f"{BASE_URL} (commit {COMMIT}, NASA public domain)",
        "normalized": f"centered at origin, bbox diagonal = {NORMALIZED_DIAG}",
        "real_scale_m_per_unit": REAL_SCALE_M_PER_UNIT,
        "default_impact_node": int(default_impact),
        "default_radius": DEFAULT_RADIUS,
        "components": components,
    }
    json_path = os.path.join(out_dir, "hubble.components.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote {obj_path}: {len(points)} verts, {len(faces)} tris (verified round-trip)")
    print(f"wrote {json_path}: {len(components)} components, default_impact_node={default_impact}")


if __name__ == "__main__":
    main()
