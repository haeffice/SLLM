"""physicsFM 메쉬 레지스트리 — 메쉬 + 노드 메타데이터 로더 (순수 numpy).

요약 흐름: mesh_id → (절차 메쉬 생성 | OBJ 파싱 + sidecar JSON) → edges/lumped mass/
노드 메타(node_type·component_id) 계산 → Mesh 데이터클래스. generate_rollouts.py가
이 레지스트리를 통해 학습(plate41/can48x24/smartphone)·평가(hubble) 메쉬를 얻는다.

절차 메쉬(plate41, can48x24)는 physics/fe/app.py의 내장 시나리오 메쉬를 그대로
미러링해 FE 데모와 기하가 동일하다. 자산 메쉬(smartphone, hubble)는
physics/fe/assets/<name>.obj('v'/'f' 라인, 1-based, 다각형은 팬 삼각분할)와
<name>.components.json sidecar(부품별 fragility/vertex 셀렉터)를 읽는다.

노드 메타:
  node_type(n)     = FRAGILITY_CLASSES[부품 fragility]  # 미지정 노드 = unknown(0)
  component_id(n)  = components.json 순서 인덱스        # 미지정 노드 = -1
  node_mass(n)     = Σ_{인접 삼각형} 면적/3, Σ=1 정규화  # 면적 가중 lumped mass
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 취약도 클래스 → node_type 인덱스 (sidecar components[].fragility 유래).
FRAGILITY_CLASSES = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
NUM_NODE_TYPES = 4

# 학습/평가 분할 (config.yaml data.train_meshes / data.eval_meshes와 일치).
TRAIN_MESHES = ["plate41", "can48x24", "smartphone"]
EVAL_MESHES = ["hubble"]

# 자산(OBJ + sidecar JSON) 위치 — physics FE 데모와 동일 자산을 공유.
PHYSICS_FE_ASSETS = Path(__file__).resolve().parent.parent / "physics" / "fe" / "assets"

# 절차 메쉬 파라미터 — physics/fe/app.py의 내장 시나리오 상수 미러.
PLATE_N = 41  # 금속 판: n×n 그리드
PLATE_SIZE = 2.0
CAN_NTHETA = 48  # 금속 캔: 원주 분할
CAN_NH = 24  # 높이 분할
CAN_RADIUS = 0.6
CAN_HEIGHT = 2.0


# ---- 절차 메쉬 (physics/fe/app.py 미러) --------------------------------------

def make_plate_mesh(n: int = PLATE_N, size: float = PLATE_SIZE):
    """중앙 정렬 n×n 금속 판(z=0 그리드) → (vertices (N,3), faces (M,3)).

    physics/fe/app.py의 make_plate_mesh 미러 — FE 데모와 기하 동일.
    """
    xs = np.linspace(-size / 2, size / 2, n)
    gx, gy = np.meshgrid(xs, xs)
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)]).astype(np.float64)
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b, c, d = a + 1, a + n, a + n + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=np.int64)


def make_can_mesh(n_theta=CAN_NTHETA, n_h=CAN_NH, radius=CAN_RADIUS, height=CAN_HEIGHT):
    """금속 캔(원통 측벽) → (vertices (n_h*n_theta,3), faces). 정점 index = j*n_theta+i.

    physics/fe/app.py의 make_can_mesh 미러 — FE 데모와 기하 동일.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    hs = np.linspace(-height / 2, height / 2, n_h)
    TH, H = np.meshgrid(th, hs)  # (n_h, n_theta)
    verts = np.column_stack([
        (radius * np.cos(TH)).ravel(),
        (radius * np.sin(TH)).ravel(),
        H.ravel(),
    ]).astype(np.float64)
    faces = []
    for j in range(n_h - 1):
        for i in range(n_theta):
            i2 = (i + 1) % n_theta  # 원주 wrap
            a = j * n_theta + i
            b = j * n_theta + i2
            c = (j + 1) * n_theta + i
            d = (j + 1) * n_theta + i2
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=np.int64)


# ---- OBJ 파서 (최소 구현 — 'v'/'f' 라인만, meshio/trimesh 불필요) -------------

def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """OBJ 파일 → (vertices (N,3) f64, faces (M,3) i64).

    'f i j k' 또는 'f i/j/k ...' 형식(1-based) 지원. 4각형 이상 다각형은
    팬 삼각분할([0,1,2],[0,2,3],...). 음수 인덱스는 OBJ 규약대로 뒤에서 상대.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0] not in ("v", "f"):
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            else:
                idx = [int(tok.split("/")[0]) for tok in parts[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]  # 1-based → 0-based
                for k in range(1, len(idx) - 1):  # 팬 삼각분할
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


# ---- 그래프/질량 헬퍼 ---------------------------------------------------------

def faces_to_edges(faces: np.ndarray) -> np.ndarray:
    """삼각형 faces (M,3) → 무방향 유니크 edges (E,2) i64.

    세 변 [0,1],[1,2],[2,0]을 이어붙이고 행별 정렬 후 np.unique(axis=0).
    """
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def area_lumped_mass(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """면적 가중 lumped mass (N,) f64, Σ=1.

    각 삼각형 면적의 1/3을 세 꼭짓점에 누적(np.add.at). 퇴화 삼각형(면적 0)은
    기여 0으로 허용. 전체 면적이 0인 퇴화 메쉬는 균등 질량으로 폴백.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # (M,)
    mass = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(mass, faces.ravel(), np.repeat(areas / 3.0, 3))
    total = mass.sum()
    if total <= 0.0:
        return np.full(len(vertices), 1.0 / len(vertices))
    return mass / total


# ---- Mesh 데이터클래스 + 로더 --------------------------------------------------

@dataclass
class Mesh:
    """메쉬 + 노드 메타데이터 번들 (레지스트리 반환 단위)."""

    mesh_id: str
    vertices: np.ndarray      # (N,3) f64
    faces: np.ndarray         # (M,3) i64
    edges: np.ndarray         # (E,2) i64 무방향 유니크
    node_mass: np.ndarray     # (N,) f64, Σ=1 — 면적 가중 lumped mass
    node_type: np.ndarray     # (N,) i8 — FRAGILITY_CLASSES 인덱스 (미지정=unknown)
    component_id: np.ndarray  # (N,) i16 — components.json 순서 인덱스, 미지정 -1
    unit_scale_m: float       # 메쉬 단위 → 미터 (sidecar real_scale_m_per_unit)
    component_table: dict     # component_id -> {"name","material","fragility",...}
    source: str               # "procedural" | OBJ 상대경로


def _assemble(mesh_id: str, vertices: np.ndarray, faces: np.ndarray,
              node_type: np.ndarray, component_id: np.ndarray,
              unit_scale_m: float, component_table: dict, source: str) -> Mesh:
    """공통 파생값(edges, node_mass) 계산 후 Mesh 조립."""
    return Mesh(
        mesh_id=mesh_id,
        vertices=vertices,
        faces=faces,
        edges=faces_to_edges(faces),
        node_mass=area_lumped_mass(vertices, faces),
        node_type=node_type,
        component_id=component_id,
        unit_scale_m=unit_scale_m,
        component_table=component_table,
        source=source,
    )


def _procedural_mesh(mesh_id: str, vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """절차 메쉬: 부품 정의 없음 → 전 노드 unknown(0)/-1, 스케일 1.0."""
    n = len(vertices)
    return _assemble(
        mesh_id, vertices, faces,
        node_type=np.zeros(n, dtype=np.int8),
        component_id=np.full(n, -1, dtype=np.int16),
        unit_scale_m=1.0,
        component_table={},
        source="procedural",
    )


def _asset_mesh(mesh_id: str) -> Mesh:
    """자산 메쉬: OBJ + sidecar components.json → 노드 메타 bake."""
    obj_path = PHYSICS_FE_ASSETS / f"{mesh_id}.obj"
    sidecar_path = PHYSICS_FE_ASSETS / f"{mesh_id}.components.json"
    vertices, faces = load_obj(obj_path)
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))

    n = len(vertices)
    node_type = np.zeros(n, dtype=np.int8)  # 기본 unknown(0)
    component_id = np.full(n, -1, dtype=np.int16)
    component_table: dict = {}
    for cid, comp in enumerate(meta.get("components", [])):
        component_table[cid] = {
            "id": comp.get("id", f"component_{cid}"),
            "name": comp.get("name", ""),
            "material": comp.get("material", ""),
            "fragility": comp.get("fragility", "unknown"),
            "damage_threshold": comp.get("damage_threshold"),
            "warn_ratio": comp.get("warn_ratio"),
            "yield_strain": comp.get("yield_strain"),
        }
        if "vertex_range" in comp:  # [start, end)
            start, end = comp["vertex_range"]
            sel = np.arange(int(start), int(end), dtype=np.int64)
        elif "vertex_indices" in comp:
            sel = np.asarray(comp["vertex_indices"], dtype=np.int64)
        else:
            continue  # rule 셀렉터는 v0에서 미지원 — 해당 노드는 미지정(-1) 유지
        sel = sel[(sel >= 0) & (sel < n)]
        component_id[sel] = cid
        node_type[sel] = FRAGILITY_CLASSES.get(comp.get("fragility", "unknown"), 0)

    return _assemble(
        mesh_id, vertices, faces,
        node_type=node_type,
        component_id=component_id,
        unit_scale_m=float(meta.get("real_scale_m_per_unit", 1.0)),
        component_table=component_table,
        source=str(obj_path.relative_to(Path(__file__).resolve().parent.parent)),
    )


def load_mesh(mesh_id: str) -> Mesh:
    """레지스트리 진입점: mesh_id → Mesh. 알 수 없는 id는 ValueError."""
    if mesh_id == "plate41":
        return _procedural_mesh(mesh_id, *make_plate_mesh())
    if mesh_id == "can48x24":
        return _procedural_mesh(mesh_id, *make_can_mesh())
    if mesh_id in ("smartphone", "hubble"):
        return _asset_mesh(mesh_id)
    raise ValueError(
        f"알 수 없는 mesh_id: {mesh_id!r} (지원: {TRAIN_MESHES + EVAL_MESHES})"
    )


# ---- CLI ----------------------------------------------------------------------

def main():
    """레지스트리 점검 CLI — --list로 전 메쉬 표 출력 + sanity assert."""
    parser = argparse.ArgumentParser(description="physicsFM 메쉬 레지스트리 점검")
    parser.add_argument("--list", action="store_true", help="레지스트리 표 출력")
    args = parser.parse_args()

    if not args.list:
        parser.print_help()
        return

    header = (f"{'mesh_id':<12} {'N':>6} {'M':>6} {'E':>6} {'Σmass':>8} "
              f"{'unit_scale_m':>13} {'부품':>4} {'부품노드':>8}")
    print(header)
    print("-" * len(header))
    for mesh_id in TRAIN_MESHES + EVAL_MESHES:
        m = load_mesh(mesh_id)
        n = len(m.vertices)
        # sanity: faces/edges 인덱스 범위, 질량 정규화, 메타 배열 크기
        assert m.faces.min() >= 0 and m.faces.max() < n, f"{mesh_id}: faces 인덱스 범위 밖"
        assert m.edges.min() >= 0 and m.edges.max() < n, f"{mesh_id}: edges 인덱스 범위 밖"
        assert abs(m.node_mass.sum() - 1.0) < 1e-9, f"{mesh_id}: Σmass != 1"
        assert m.node_type.shape == (n,) and m.component_id.shape == (n,)
        assert m.node_type.max() < NUM_NODE_TYPES
        covered = int((m.component_id >= 0).sum())  # 부품이 지정된 노드 수
        print(f"{m.mesh_id:<12} {n:>6} {len(m.faces):>6} {len(m.edges):>6} "
              f"{m.node_mass.sum():>8.4f} {m.unit_scale_m:>13.6f} "
              f"{len(m.component_table):>4} {covered:>8}")
    print("sanity OK: faces/edges 인덱스 범위, Σmass=1, node_type < NUM_NODE_TYPES")


if __name__ == "__main__":
    main()
