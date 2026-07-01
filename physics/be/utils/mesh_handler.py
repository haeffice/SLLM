"""meshio 기반 메쉬 직렬화 유틸리티.

FE가 보낸 메쉬 파일 바이트를 (vertices, faces)로 파싱하고, 변형된 결과를
다시 같은 포맷의 파일 바이트로 직렬화한다. meshio는 경로 기반 I/O라서
임시 파일을 거치되, NamedTemporaryFile은 Windows에서 열린 채 다시 열 수
없으므로 mkstemp → close(fd) → meshio가 경로로 read/write → remove 패턴을
쓴다 (cross-platform).

포맷: meshio가 지원하는 모든 확장자를 받는다. file_format(확장자)을 임시파일
suffix로 붙인 뒤 meshio가 확장자로 포맷을 "자동 추론"하게 한다 — 확장자와
meshio 포맷명이 다른 경우(.bdf→nastran, .inp→abaqus, .msh→gmsh, .mesh→medit
등)도 그대로 읽힌다. (모호한 .msh는 write 시 gmsh로 명시 — _WRITE_FORMAT_OVERRIDE.)

face 추출 우선순위:
  체적 cell(tetra/hexahedron/wedge/pyramid, 2차 요소 포함)이 있으면 그
  "경계면(boundary surface)"을 우선. 없으면(순수 표면 메쉬) triangle/quad cell.
  → 체적+태그표면이 섞인 FE 메쉬에서 부분 태그 표면이 전체 경계를 가리지 않게.

face 추출 로직은 fe/app.py에 동일하게 복제돼 있다 — FE/BE가 같은 (points, faces)를
만들어야 impact_node 인덱스가 일치하므로, 한쪽만 수정하면 안 된다.

요약 흐름:
  bytes → (임시파일) → meshio.read(자동추론) → points(vertices) + 표면 삼각형(faces)
  (vertices, faces) → meshio.Mesh → (임시파일) → write(추론/override) → bytes
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import meshio
import numpy as np

# face cell 컬럼 수 → meshio cell type. 표면 삼각형(3)이 기본, 사각형(4)도 허용.
_FACE_CELL_TYPES = {3: "triangle", 4: "quad"}

# 체적(volume) cell → 국소 face 정의(경계면 추출용). VTK/meshio 노드 순서 기준.
_VOLUME_FACE_TEMPLATES: dict[str, list[tuple[int, ...]]] = {
    "tetra": [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
    "hexahedron": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "wedge": [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "pyramid": [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}

# 2차(고차) 요소 → (선형 base type, corner 노드 수). meshio/VTK 순서상 corner
# 노드가 항상 앞쪽 N개라 data[:, :N] 슬라이스로 선형 요소처럼 처리한다.
_QUADRATIC_SURFACE = {
    "triangle6": ("triangle", 3), "triangle7": ("triangle", 3),
    "quad8": ("quad", 4), "quad9": ("quad", 4),
}
_QUADRATIC_VOLUME = {
    "tetra10": ("tetra", 4),
    "hexahedron20": ("hexahedron", 8), "hexahedron24": ("hexahedron", 8),
    "hexahedron27": ("hexahedron", 8),
    "wedge12": ("wedge", 6), "wedge15": ("wedge", 6), "wedge18": ("wedge", 6),
    "pyramid13": ("pyramid", 5), "pyramid14": ("pyramid", 5),
}

# 확장자→포맷이 모호한 write 케이스의 명시적 writer. '.msh'는 meshio가
# ['ansys','gmsh']로 매핑해 file_format 미지정 시 첫 번째(ansys)를 골라
# 문서화된 .msh→gmsh 및 read-side 폴백과 어긋난다. None은 자동추론과 동일.
_WRITE_FORMAT_OVERRIDE = {"msh": "gmsh"}


def _normalize_format(file_format: str) -> str:
    """'.VTK' / 'OBJ' 같은 입력을 meshio 확장자 'vtk' / 'obj'로 정규화."""
    return file_format.strip().lstrip(".").lower()


def _tris_from_quads(quads: np.ndarray) -> np.ndarray:
    """(M,4) quad → (2M,3) triangle. quad [a,b,c,d] → [a,b,c] + [a,c,d]."""
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)


def _surface_faces(mesh: "meshio.Mesh") -> Optional[np.ndarray]:
    """triangle/quad(및 2차 triangle6/quad8/9) 표면 cell → (M,3). 없으면 None."""
    tris: list[np.ndarray] = []
    for cell in mesh.cells:
        ctype = cell.type
        data = np.asarray(cell.data, dtype=np.int64)
        if ctype in _QUADRATIC_SURFACE:  # 2차 → corner 노드로 축약
            base, n = _QUADRATIC_SURFACE[ctype]
            ctype, data = base, data[:, :n]
        if ctype == "triangle":
            tris.append(data)
        elif ctype == "quad":
            tris.append(_tris_from_quads(data))
    if not tris:
        return None
    return np.concatenate(tris, axis=0)


def _boundary_faces(mesh: "meshio.Mesh") -> Optional[np.ndarray]:
    """체적 cell의 경계면을 (M,3) 삼각형으로 추출. 체적 cell이 없으면 None.

    내부 face는 이웃 cell과 공유되어 짝수 번 등장하고, 경계면은 홀수 번
    등장한다. 정점을 정렬한 키(방향 무관)로 개수를 세어 홀수인 face만 남긴다
    (출력은 원래 정점 순서 유지 → normal 방향 보존). 2차 요소는 corner로 축약,
    중복 cell은 제거한다.

    가정: conformal 메쉬(한 face를 최대 2 cell이 공유). hanging-node 등
    비정합 메쉬(큰 face vs 두 half-face)는 sorted 키가 달라 큰 내부 face가
    가짜 경계로 남을 수 있다 — counting 방식의 본질적 한계.
    """
    tri_groups: list[np.ndarray] = []
    quad_groups: list[np.ndarray] = []
    for cell in mesh.cells:
        tmpl = _VOLUME_FACE_TEMPLATES.get(cell.type)
        data = np.asarray(cell.data, dtype=np.int64)
        if tmpl is None:  # 2차 체적 요소 → 선형 base + corner 슬라이스
            ho = _QUADRATIC_VOLUME.get(cell.type)
            if ho is None:
                continue
            tmpl = _VOLUME_FACE_TEMPLATES[ho[0]]
            data = data[:, : ho[1]]
        data = np.unique(data, axis=0)  # 중복 cell 제거(구멍 방지; 순서 무관)
        for local in tmpl:
            group = data[:, list(local)]  # (N_cells, len(local))
            (tri_groups if len(local) == 3 else quad_groups).append(group)

    if not tri_groups and not quad_groups:
        return None

    tris: list[np.ndarray] = []
    for groups in (tri_groups, quad_groups):
        if not groups:
            continue
        faces = np.concatenate(groups, axis=0)  # (F, k)
        keys = np.sort(faces, axis=1)  # 방향 무관 키
        _, inv, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        boundary = faces[counts[np.ravel(inv)] % 2 == 1]  # 홀수 등장 = 경계면
        if boundary.shape[1] == 4:
            boundary = _tris_from_quads(boundary)
        if boundary.size:
            tris.append(boundary)
    if not tris:
        return None
    return np.concatenate(tris, axis=0)


def _extract_faces(mesh: "meshio.Mesh") -> np.ndarray:
    """렌더/변형용 표면 삼각형 (M,3) 추출.

    체적 cell이 있으면 경계면을 우선 추출(태그된 부분 표면 cell이 전체 경계를
    가리지 않게). 체적 cell이 없으면 triangle/quad 표면 cell을 쓴다. 둘 다
    없으면 ValueError (router에서 400 매핑).
    """
    faces = _boundary_faces(mesh)  # 체적 cell 없으면 None
    if faces is None:
        faces = _surface_faces(mesh)
    if faces is None:
        present = sorted({c.type for c in mesh.cells})
        raise ValueError(
            "no triangle/quad surface or tetra/hexahedron/wedge/pyramid "
            f"volume cells found; mesh cell types = {present}"
        )
    return faces


def load_mesh_from_bytes(
    data: bytes, file_format: str = "vtk"
) -> Tuple[np.ndarray, np.ndarray]:
    """메쉬 파일 바이트를 (vertices (N,3) float64, faces (M,3) int64)로 파싱.

    file_format은 확장자 — 임시파일 suffix로 붙여 meshio가 포맷을 자동 추론한다.

    Raises:
        ValueError: 디코딩 실패 또는 표면/경계 face 부재 시 (router에서 400 매핑).
    """
    fmt = _normalize_format(file_format)
    fd, path = tempfile.mkstemp(suffix=f".{fmt}")
    try:
        # fdopen으로 전체 버퍼를 한 번에 쓰고(부분 write 방지) 컨텍스트 종료 시
        # fd를 확실히 닫는다 — 예외가 나도 fd/임시파일이 새지 않고, Windows에서
        # 핸들이 열린 채 os.remove가 실패하는 문제를 막는다.
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            # file_format 미지정 → meshio가 suffix(확장자)로 포맷 자동 추론.
            mesh = meshio.read(path)
        except Exception as e:  # meshio가 던지는 예외 종류가 포맷별로 다양함
            raise ValueError(f"failed to decode {fmt} mesh: {e}") from e
    finally:
        os.remove(path)

    vertices = np.asarray(mesh.points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        raise ValueError("mesh has no points")
    # 2D 포맷은 (N,2)로 올 수 있음 → z=0 패딩해 (N,3) 보장.
    if vertices.shape[1] == 2:
        vertices = np.hstack([vertices, np.zeros((vertices.shape[0], 1))])
    elif vertices.shape[1] != 3:
        raise ValueError(f"points must be (N,2|3), got {vertices.shape}")

    faces = _extract_faces(mesh)
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def write_mesh_to_bytes(
    vertices: np.ndarray, faces: np.ndarray, file_format: str = "vtk"
) -> bytes:
    """(vertices, faces)를 file_format 메쉬 파일 바이트로 직렬화.

    cell type은 faces의 컬럼 수로 결정 (3→triangle, 4→quad). file_format은
    확장자 — suffix로 meshio가 writer를 추론하되, 모호한 확장자(.msh)는
    _WRITE_FORMAT_OVERRIDE로 명시한다.
    """
    fmt = _normalize_format(file_format)
    points = np.asarray(vertices, dtype=np.float64)
    cells_arr = np.asarray(faces, dtype=np.int64)
    if cells_arr.ndim != 2 or cells_arr.shape[1] not in _FACE_CELL_TYPES:
        raise ValueError(
            f"faces must be (M,3) or (M,4), got {cells_arr.shape}"
        )
    cell_type = _FACE_CELL_TYPES[cells_arr.shape[1]]
    mesh = meshio.Mesh(points=points, cells=[(cell_type, cells_arr)])

    fd, path = tempfile.mkstemp(suffix=f".{fmt}")
    try:
        os.close(fd)
        try:
            # 미지정(None)이면 suffix로 자동 추론, 모호한 것만 명시 override.
            mesh.write(path, file_format=_WRITE_FORMAT_OVERRIDE.get(fmt))
        except Exception as e:
            raise ValueError(f"failed to encode {fmt} mesh: {e}") from e
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)
