"""meshio 기반 메쉬 직렬화 유틸리티.

FE가 보낸 메쉬 파일 바이트를 (vertices, faces)로 파싱하고, 변형된 결과를
다시 같은 포맷의 파일 바이트로 직렬화한다. meshio는 경로 기반 I/O라서
임시 파일을 거치되, NamedTemporaryFile은 Windows에서 열린 채 다시 열 수
없으므로 mkstemp → close(fd) → meshio가 경로로 read/write → remove 패턴을
쓴다 (cross-platform).

요약 흐름:
  bytes → (임시파일) → meshio.read → points(vertices) + 표면 삼각형(faces)
  (vertices, faces) → meshio.Mesh → (임시파일) → write → bytes
"""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import meshio
import numpy as np

# face cell 컬럼 수 → meshio cell type. 표면 삼각형(3)이 기본, 사각형(4)도 허용.
_FACE_CELL_TYPES = {3: "triangle", 4: "quad"}


def _normalize_format(file_format: str) -> str:
    """'.VTK' / 'OBJ' 같은 입력을 meshio가 쓰는 'vtk' / 'obj'로 정규화."""
    return file_format.strip().lstrip(".").lower()


def _extract_faces(mesh: "meshio.Mesh") -> np.ndarray:
    """meshio cells에서 표면 삼각형 (M, 3) 연결정보를 추출.

    triangle 블록은 그대로, quad 블록은 두 삼각형으로 쪼개 합친다.
    표면 cell이 없으면(예: tetra만 있는 체적 메쉬) ValueError.
    """
    tris: list[np.ndarray] = []
    for cell in mesh.cells:
        data = np.asarray(cell.data, dtype=np.int64)
        if cell.type == "triangle":
            tris.append(data)
        elif cell.type == "quad":
            # quad [a,b,c,d] → [a,b,c] + [a,c,d]
            tris.append(data[:, [0, 1, 2]])
            tris.append(data[:, [0, 2, 3]])
    if not tris:
        raise ValueError(
            "no surface faces found — provide a triangulated surface mesh "
            "(triangle/quad cells)"
        )
    return np.concatenate(tris, axis=0)


def load_mesh_from_bytes(
    data: bytes, file_format: str = "vtk"
) -> Tuple[np.ndarray, np.ndarray]:
    """메쉬 파일 바이트를 (vertices (N,3) float64, faces (M,3) int64)로 파싱.

    Raises:
        ValueError: 디코딩 실패 또는 표면 face 부재 시 (router에서 400 매핑).
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
            mesh = meshio.read(path, file_format=fmt)
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

    cell type은 faces의 컬럼 수로 결정 (3→triangle, 4→quad).
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
            mesh.write(path, file_format=fmt)
        except Exception as e:
            raise ValueError(f"failed to encode {fmt} mesh: {e}") from e
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)
