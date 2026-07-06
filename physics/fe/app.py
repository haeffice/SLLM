"""Physics Impact Simulator — Windows 데스크톱 FE (PySide6 + PyVista).

흐름:
  1. 시작 시 기본 시나리오(허블 우주망원경 — NASA public domain 자산)가 로드된다.
     Scenario 드롭다운(허블/판/캔) 또는 "Load mesh…"로 로컬 파일도 열 수 있다.
  2. 3D 뷰포트에서 노드를 클릭해 impact_node를 고른다.
  3. force(X/Y/Z) · radius · scale을 입력하고 Simulate.
  4. 모델 연결 시 → BE /simulate가 준 (T,N,3) 프레임을, 미연결 시 → 로컬 metal_dent
     궤적을 재생 애니메이션한다. 변위 크기를 히트맵('turbo')으로 칠하고, ▶/⏸·Loop·
     타임라인 슬라이더로 제어. (모드 배너로 LIVE/DUMMY 명시.)
  5. 시뮬 직후 부품별 충격 분석(analysis.py)이 좌측 [분석] 탭에 뜬다 — 부품별
     최대 변위/임계값/상태(OK·WARN·FAIL) 표 + 3D 마커(부품별 최대 충격 위치,
     임계 초과 정점 빨간 점). 부품 정의는 자산 sidecar JSON에서 온다.
  6. 좌측 하단 입력창으로 분석 결과에 대해 질문하면 [챗] 탭에 답이 뜬다 — BE
     /chat(LLM 또는 rule-based) 경유, 서버 미연결 시 로컬 chat_fallback 즉답.

레이아웃: [좌: 분석/챗 탭 패널] [중: 3D 뷰포트] [우: 컨트롤 패널].

노드 인덱스 일관성: FE도 BE와 동일하게 meshio로 메쉬를 파싱해 vertices/faces를
얻고, 그 배열 순서대로 렌더링한다. 피킹으로 고른 점은 vertices 배열의 인덱스로
환산되며, Simulate 시 "원본 파일 바이트"를 그대로 보내므로 BE가 같은 순서로
재파싱 → impact_node가 정확히 일치한다. 허블 자산의 components.json vertex_range도
같은 이유로 FE/BE 어느 쪽에서든 유효하다.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")  # qtpy가 PySide6를 고르도록

import html
import sys

import meshio
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtGui, QtWidgets

import analysis
import free_fall_sim
from api_client import PhysicsClient
from chat_fallback import fallback_answer


def _base_path() -> str:
    """리소스(VERSION, assets/) 해석 기준 — PyInstaller 번들은 sys._MEIPASS,
    개발 환경은 스크립트 디렉토리."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def _asset_path(*parts: str) -> str:
    return os.path.join(_base_path(), *parts)


def _read_version() -> str:
    """버전 문자열을 읽는다 (CI가 빌드마다 patch를 올린다)."""
    try:
        with open(os.path.join(_base_path(), "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "dev"


APP_VERSION = _read_version()

# /health status → 상태 점 색 (repo FE 규약: 초록 ready / 보라 loading / 빨강 오류)
STATUS_COLORS = {
    "ok": "#2e7d32",
    "loading": "#6a1b9a",
    "failed": "#c62828",
    "down": "#c62828",
    "unknown": "#9e9e9e",
}

# Simulate 동작 모드 배너 색 (배경, 글자)
MODE_STYLES = {
    "live": ("#e8f5e9", "#2e7d32"),     # BE 연결됨
    "loading": ("#f3e5f5", "#6a1b9a"),  # BE 로딩 중 → 로컬 더미
    "dummy": ("#fff3e0", "#e65100"),    # 모델 미연결 → 로컬 더미
}

# 분석 상태(OK/WARN/FAIL/N/A) → (배경, 글자). 분석 패널 배지·표 상태 셀·3D 마커 공용.
ANALYSIS_STATUS_STYLES = {
    "OK": ("#e8f5e9", "#2e7d32"),
    "WARN": ("#fff3e0", "#e65100"),
    "FAIL": ("#ffebee", "#c62828"),
    "N/A": ("#f5f5f5", "#9e9e9e"),
}

# 기본 시나리오 자산 (fe/assets/ — prep_*.py가 생성).
ASSET_MESH = "hubble.obj"
ASSET_COMPONENTS = "hubble.components.json"

# 자산 시나리오 목록 (kind, mesh 파일, 콤보 표시명) — 파일이 있는 것만 콤보에 뜬다.
# 첫 항목이 시작 기본 시나리오.
SCENARIO_ASSETS = [
    ("hubble", "hubble.obj", "허블 우주망원경 (hubble)"),
    ("smartphone", "smartphone.obj", "스마트폰 (smartphone)"),
]

# 낙하 자세 공통 프리셋 (표시명, Euler [rx,ry,rz]°). 자산 sidecar의
# drop_orientations가 이 뒤에 추가된다.
DROP_ORIENTATION_PRESETS = [
    ("그대로", (0.0, 0.0, 0.0)),
    ("모서리 (대각 45°)", (45.0, 35.264, 0.0)),
    ("측면 (90°)", (90.0, 0.0, 0.0)),
]

# 종료 시 제한 시간 내 못 끝낸 워커 스레드 보관 — QThread가 실행 중에 파괴되면
# 크래시하므로 프로세스 종료까지 참조를 유지한다 (closeEvent._drain_worker).
_ORPHAN_WORKERS: list = []

# 기능 영향 분석의 실화 앵커 (상세 패널 하단 표시).
FUNCTIONAL_ANCHORS = {
    "solar_panel": "실제 허블도 태양전지판 열 플러터 지터로 SM1(1993)에서 어레이를 교체했다.",
    "antenna": "허블 HGA는 빔폭 ~4°의 S-band 접시로 TDRS에 1 Mbps 과학 데이터를 보낸다.",
    "optical_tube": "허블 주경은 2.2 μm 연마 오차만으로 임무가 마비됐던 광학계다.",
    "screen": "실제 스마트폰도 면·모서리 반복 낙하 시험으로 커버글라스 파손율을 검증한다.",
}

# 변형률 판정 → 분석 상태 색 매핑.
_STRAIN_STATUS = {"소성 변형": "FAIL", "항복 근접": "WARN", "탄성": "OK"}


def _functional_brief(c: "analysis.ComponentResult") -> str:
    """분석 표 '기능 영향' 열 — 부품 functional type별 대표 지표 한 줄."""
    extras = c.extras or {}
    func = extras.get("functional") or {}
    ftype = func.get("type")
    if ftype == "antenna":
        return f"-{func.get('loss_total_db', 0):.3g} dB"
    if ftype == "solar_panel":
        return f"전력 {func.get('power_frac', 1.0) * 100:.0f}%"
    if ftype == "optical_tube":
        s = func.get("strehl", 1.0)
        return f"S={s:.2g}" if s >= 0.001 else "S<0.001"
    if ftype == "screen":
        return f"불능 {func.get('dead_area_frac', 0) * 100:.0f}%"
    rigid = extras.get("rigid")
    if rigid and rigid.get("rotation_deg", 0) > 0.01:
        return f"회전 {rigid['rotation_deg']:.2g}°"
    return "—"


def _status_span(text, status: str) -> str:
    _, fg = ANALYSIS_STATUS_STYLES.get(status, ANALYSIS_STATUS_STYLES["N/A"])
    return f'<b style="color:{fg};">{html.escape(str(text))}</b>'


def _detail_html(c: "analysis.ComponentResult") -> str:
    """행 선택 상세 — 구조 지표 + 기능/정착/변형률 breakdown (QTextBrowser HTML)."""
    extras = c.extras or {}
    lines = [
        f"<b>{html.escape(c.name)}</b> ({html.escape(c.component_id)}) — "
        + _status_span(c.status, c.status),
        f"최대 변위 {c.max_disp:.4g} (임계값 {c.threshold:.4g}의 {c.ratio:.2f}배)"
        f" @ 노드 #{c.max_disp_node}",
        f"정점 {c.num_vertices}개 · 임계 초과 {len(c.over_indices)}개 · "
        f"충격 점수 {c.max_score:.1f}/100",
    ]
    if c.material:
        lines.append(f"재질: {html.escape(c.material)}")
    rigid = extras.get("rigid")
    if rigid:
        lines.append(
            f"강체 회전 {rigid['rotation_deg']:.3g}° · 영구 뒤틀림 RMS "
            f"{rigid['residual_rms_mm']:.3g} mm(실스케일)"
        )

    func = extras.get("functional") or {}
    ftype = func.get("type")
    if ftype == "solar_panel":
        lines.append(
            "<br><b>[기능] 발전량</b> — 잔여 "
            + _status_span(f"{func['power_frac'] * 100:.1f}%", func["verdict"])
            + f"<br>법선 기울기 {func['tilt_deg']:.3g}° (cos {func['cos_factor']:.3f}) · "
            f"손상 셀 면적 {func['dead_area_frac'] * 100:.1f}%"
            f"<br>손실 {func['power_lost_w']:.0f} W / {func['p0_w']:.0f} W"
        )
    elif ftype == "antenna":
        lines.append(
            "<br><b>[기능] 통신 링크</b> — 손실 "
            + _status_span(f"{func['loss_total_db']:.3g} dB", func["verdict"])
            + f"<br>지향 이탈 {func['pointing_deg']:.3g}° / 빔폭 {func['beam_deg']:.3g}°"
            f" (지향 {func['loss_point_db']:.3g} dB)"
            f"<br>표면 오차 {func['surface_err_mm']:.3g} mm (Ruze {func['loss_ruze_db']:.3g} dB)"
        )
    elif ftype == "optical_tube":
        s = func["strehl"]
        s_txt = f"{s:.3g}" if s >= 0.001 else "<0.001"
        lines.append(
            "<br><b>[기능] 광학 상 품질</b> — 스트렐 비 "
            + _status_span(s_txt, func["verdict"])
            + f"<br>파면 오차 {func['wfe_um']:.3g} μm (Maréchal 근사, S≥0.8=회절한계)"
            f"<br>광축 기울기 {func['axis_tilt_arcsec']:.3g}″ "
            f"(지향 예산 {func['budget_arcsec']:.3g}″의 {func['budget_ratio']:.3g}배)"
        )
    elif ftype == "screen":
        lines.append(
            "<br><b>[기능] 표시/터치</b> — 불능 면적 "
            + _status_span(f"{func['dead_area_frac'] * 100:.1f}%", func["verdict"])
            + "<br>임계 초과 셀 면적 비율 (커버글라스 균열/터치 손상 추정)"
        )

    settling = extras.get("settling") or {}
    if settling.get("available"):
        line = (
            f"<br><b>[진동 정착]</b> 구간 {settling['settle_frac'] * 100:.0f}% 시점 정착 · "
            f"잔류 진동 {settling['residual_ratio'] * 100:.1f}%"
        )
        if settling.get("oscillatory"):
            if settling.get("zeta") is not None:
                line += f" · 감쇠비 ζ={settling['zeta']:.3g} ({settling['cycles']} 사이클/구간)"
            else:  # 진동은 검출됐으나 진폭이 유지/증가 → 감쇠비 산정 불가
                line += f" · 감쇠 미검출(진폭 유지/증가, {settling.get('cycles', '?')} 사이클/구간)"
        else:
            line += " · 과감쇠(잔류 진동 미검출)"
        lines.append(line)

    strain = extras.get("strain") or {}
    if strain.get("available"):
        lines.append(
            "<br><b>[변형률]</b> 피크 "
            f"{strain['peak'] * 100:.3g}% / 잔류 {strain['residual'] * 100:.3g}% "
            f"(항복 {strain['yield_strain'] * 100:.2g}%) → "
            + _status_span(strain["verdict"], _STRAIN_STATUS.get(strain["verdict"], "N/A"))
        )

    anchor = FUNCTIONAL_ANCHORS.get(ftype)
    if anchor:
        lines.append(f'<br><i style="color:#888;">{anchor}</i>')
    if c.notes:
        lines.append(f'<span style="color:#777;">{html.escape(c.notes)}</span>')
    return '<div style="font-size:12px;">' + "<br>".join(lines) + "</div>"

MESH_FILTER = (
    "Mesh (*.vtk *.vtu *.obj *.stl *.ply *.off *.msh *.bdf *.nas *.fem "
    "*.inp *.mesh *.med *.node *.vol *.e *.exo *.dat *.ugrid *.su2 *.xdmf);;"
    "All files (*)"
)

# 내장 시나리오 메쉬 (BE/모델 없이 FE 데모/인터랙션 확인용).
PLATE_N = 41  # 금속 판: n×n 그리드
PLATE_SIZE = 2.0
CAN_NTHETA = 48  # 금속 캔: 원주 분할
CAN_NH = 24  # 높이 분할
CAN_RADIUS = 0.6
CAN_HEIGHT = 2.0


def make_plate_mesh(n: int = PLATE_N, size: float = PLATE_SIZE):
    """중앙 정렬 n×n 금속 판(z=0 그리드) → (vertices (N,3), faces (M,3))."""
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
    """금속 캔(원통 측벽) → (vertices (n_h*n_theta,3), faces). 정점 index = j*n_theta+i."""
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


def dump_mesh(vertices: np.ndarray, faces: np.ndarray, file_format: str = "vtk") -> bytes:
    """(vertices, faces) → 메쉬 파일 바이트. 더미 메쉬의 orig_bytes(BE 전송용) 생성.

    BE utils.mesh_handler.write_mesh_to_bytes의 FE측 대응 (meshio temp-file 직렬화).
    """
    import tempfile

    mesh = meshio.Mesh(
        points=np.asarray(vertices, dtype=np.float64),
        cells=[("triangle", np.asarray(faces, dtype=np.int64))],
    )
    fd, path = tempfile.mkstemp(suffix=f".{file_format}")
    try:
        os.close(fd)
        mesh.write(path, file_format=file_format)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


# --- metal_dent 오프라인 반응: be/models/metal_dent/model.py와 동일 수식 --------
_MD_FRAMES = 60
_MD_MAX_FRAMES = 240
_MD_RADIUS_FRACTION = 0.25
_MD_TAU_RISE = 0.22
_MD_TAU_RING = 0.16
_MD_RING_AMP = 0.35
_MD_RING_CYCLES = 1.5
_MD_RING_HZ = 2.5


def metal_dent_simulate(vertices, faces, action) -> np.ndarray:
    """(T,N,3) 변형 궤적 — BE MetalDentSimulator와 동일 수식(오프라인 반응).

    잘못된 impact_node는 raise 대신 clamp(오프라인은 관대하게). frames[0]=원본.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    n = vertices.shape[0]
    impact_node = max(0, min(int(action.get("impact_node", 0)), n - 1))
    force = np.asarray(action.get("force", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    force = force[:3] if force.size >= 3 else np.zeros(3)
    scale = float(action.get("scale", 1.0))
    frames = max(2, min(int(action.get("frames", _MD_FRAMES)), _MD_MAX_FRAMES))

    fmag = float(np.linalg.norm(force))
    fdir = force / fmag if fmag > 0 else force
    depth = scale * fmag

    radius = action.get("radius")
    if radius is None:
        diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
        sigma = _MD_RADIUS_FRACTION * diag if diag > 0 else 1.0
    else:
        sigma = float(radius)
    if sigma <= 0:
        sigma = 1.0

    d = np.linalg.norm(vertices - vertices[impact_node], axis=1)
    env = np.exp(-((d / sigma) ** 2))
    ts = np.linspace(0.0, 1.0, frames)
    # rise(0)=0, rise(1)=1 정규화 — BE metal_dent_trajectory와 동일해야 함.
    rise = (1.0 - np.exp(-ts / _MD_TAU_RISE)) / (1.0 - np.exp(-1.0 / _MD_TAU_RISE))
    k = 2.0 * np.pi * _MD_RING_CYCLES / sigma
    omega = 2.0 * np.pi * _MD_RING_HZ
    phase = k * d[None, :] - omega * ts[:, None]
    ring = _MD_RING_AMP * np.sin(phase) * rise[:, None] * np.exp(-ts[:, None] / _MD_TAU_RING)
    amp = env[None, :] * (rise[:, None] + ring)
    disp = depth * amp[..., None] * fdir[None, None, :]
    return vertices[None, :, :] + disp


# --- face 추출: be/utils/mesh_handler.py와 "동일 규칙"으로 유지한다 ----------
# FE/BE가 같은 (points, faces)를 만들어야 impact_node 인덱스가 일치하므로,
# 아래 5개 함수/상수는 BE와 byte 단위로 동일하게 두어야 한다(한쪽만 고치면 안 됨).
_VOLUME_FACE_TEMPLATES = {
    "tetra": [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
    "hexahedron": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "wedge": [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "pyramid": [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}
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


def _tris_from_quads(quads):
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)


def _surface_faces(mesh):
    """triangle/quad(및 2차 triangle6/quad8/9) 표면 cell → (M,3). 없으면 None."""
    tris = []
    for cell in mesh.cells:
        ctype = cell.type
        data = np.asarray(cell.data, dtype=np.int64)
        if ctype in _QUADRATIC_SURFACE:
            base, n = _QUADRATIC_SURFACE[ctype]
            ctype, data = base, data[:, :n]
        if ctype == "triangle":
            tris.append(data)
        elif ctype == "quad":
            tris.append(_tris_from_quads(data))
    return np.concatenate(tris, axis=0) if tris else None


def _boundary_faces(mesh):
    """체적 cell 경계면(홀수 번 등장하는 face)을 (M,3) 삼각형으로 추출. 없으면 None.

    2차 요소는 corner로 축약, 중복 cell 제거. conformal(한 face 최대 2 cell) 가정.
    """
    tri_groups, quad_groups = [], []
    for cell in mesh.cells:
        tmpl = _VOLUME_FACE_TEMPLATES.get(cell.type)
        data = np.asarray(cell.data, dtype=np.int64)
        if tmpl is None:
            ho = _QUADRATIC_VOLUME.get(cell.type)
            if ho is None:
                continue
            tmpl = _VOLUME_FACE_TEMPLATES[ho[0]]
            data = data[:, : ho[1]]
        data = np.unique(data, axis=0)  # 중복 cell 제거(구멍 방지)
        for local in tmpl:
            group = data[:, list(local)]
            (tri_groups if len(local) == 3 else quad_groups).append(group)
    if not tri_groups and not quad_groups:
        return None
    tris = []
    for groups in (tri_groups, quad_groups):
        if not groups:
            continue
        faces = np.concatenate(groups, axis=0)
        keys = np.sort(faces, axis=1)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        boundary = faces[counts[np.ravel(inv)] % 2 == 1]  # 홀수 등장 = 경계면
        if boundary.shape[1] == 4:
            boundary = _tris_from_quads(boundary)
        if boundary.size:
            tris.append(boundary)
    return np.concatenate(tris, axis=0) if tris else None


def _extract_faces(mesh):
    """체적 cell 있으면 경계면 우선, 없으면 표면 cell. 둘 다 없으면 ValueError."""
    faces = _boundary_faces(mesh)  # 체적 cell 없으면 None
    if faces is None:
        faces = _surface_faces(mesh)
    if faces is None:
        present = sorted({c.type for c in mesh.cells})
        raise ValueError(f"no surface or volume cells in mesh; types = {present}")
    return faces


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """meshio로 (vertices (N,3), faces (M,3), file_format) 추출.

    meshio가 확장자로 포맷을 자동 추론하므로 meshio가 지원하는 모든 확장자를
    연다. face 추출 규칙이 BE와 동일해 정점 순서·impact_node 인덱스가 일치한다.
    """
    mesh = meshio.read(path)  # 확장자로 포맷 자동 추론
    verts = np.asarray(mesh.points, dtype=np.float64)
    if verts.shape[1] == 2:
        verts = np.hstack([verts, np.zeros((verts.shape[0], 1))])
    faces = _extract_faces(mesh)
    file_format = os.path.splitext(path)[1].lstrip(".").lower() or "vtk"
    return verts, faces, file_format


def to_polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """(N,3)+(M,3) → pyvista PolyData (VTK face 형식 [3,i,j,k,...])."""
    faces_pv = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces]
    ).ravel()
    return pv.PolyData(vertices, faces_pv)


class SimulateWorker(QtCore.QThread):
    """blocking /simulate 호출을 GUI 스레드 밖에서 실행 (뷰포트 멈춤 방지)."""

    finished_ok = QtCore.Signal(object)  # (faces (M,3), frames (T,N,3))
    failed = QtCore.Signal(str)

    def __init__(self, client, mesh_bytes, file_format, action, model):
        super().__init__()
        self.client = client
        self.mesh_bytes = mesh_bytes
        self.file_format = file_format
        self.action = action
        self.model = model

    def run(self):
        try:
            faces, frames = self.client.simulate(
                self.mesh_bytes, self.file_format, self.action, self.model
            )
            self.finished_ok.emit((faces, frames))
        except Exception as e:  # 네트워크/서버 오류 모두 메시지로 환원
            self.failed.emit(str(e))


class ChatWorker(QtCore.QThread):
    """blocking /chat 호출을 GUI 스레드 밖에서 실행 (LLM 지연에도 뷰포트 유지).

    SimulateWorker와 같은 패턴. 전송 중 입력을 비활성화해 in-flight 1개만
    허용하므로 _sim_gen 같은 세대 카운터는 불필요하다 (답변은 질문 시점의
    분석 요약에 대한 텍스트라 메쉬 교체와 경합하지 않는다).
    """

    finished_ok = QtCore.Signal(object)  # /chat 응답 dict
    failed = QtCore.Signal(str)

    def __init__(self, client, question, analysis_summary, history):
        super().__init__()
        self.client = client
        self.question = question
        self.analysis_summary = analysis_summary
        self.history = history

    def run(self):
        try:
            self.finished_ok.emit(
                self.client.chat(self.question, self.analysis_summary, self.history)
            )
        except Exception as e:  # 네트워크/서버 오류 모두 메시지로 환원
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Physics Impact Simulator v{APP_VERSION}")
        self.resize(1600, 820)

        self.client = PhysicsClient()
        self.worker: SimulateWorker | None = None
        self.chat_worker: ChatWorker | None = None

        # 현재 로드된 메쉬 상태
        self.orig_bytes: bytes | None = None
        self.file_format = "vtk"
        self.vertices: np.ndarray | None = None
        self.faces: np.ndarray | None = None
        self.impact_node: int | None = None
        self.components_def: dict | None = None  # 자산 sidecar JSON (없으면 None)
        self._be_ready = False  # BE 모델 준비 완료 → Simulate가 BE로 감
        self._server_reachable = False  # /health 응답 여부 — 챗 라우팅(BE vs 로컬)용
        self._pick_actor = None

        # 분석/챗 상태
        self.analysis: analysis.AnalysisResult | None = None
        self._overlay_actors: list = []  # 분석 3D 마커 — plotter.clear() 시 함께 사라짐
        self._last_action: dict = {}
        self.chat_history: list[dict] = []  # [{"role","content"}] — /chat에 최근 6개 전송
        self._pending_question: str | None = None  # in-flight 챗 질문 (단일 보장)

        # 애니메이션 상태
        self.frames: np.ndarray | None = None  # (T,N,3)
        self.disp: np.ndarray | None = None  # (T,N) 변위 크기(히트맵)
        self.vmax = 1.0
        self.anim_poly = None  # pv.PolyData (프레임마다 in-place 갱신)
        self.anim_idx = 0
        self.playing = False
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self._next_frame)
        self._sim_gen = 0  # 시뮬 요청 세대 — mesh 교체 시 오래된 worker 결과 무시

        self._build_ui()
        self._on_sim_mode_changed(0)  # 초기 모드(자유 낙하)에 맞춰 컨트롤 그룹 표시
        self.refresh_health()
        # 시작 시 기본 시나리오(허블 자산, 없으면 금속 판)를 띄워 BE 없이도
        # 인터랙션을 확인할 수 있게.
        self._load_scenario(self._scenario_kinds[0])

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # 좌: 분석/챗 탭 패널 (+ 하단 채팅 입력)
        root.addWidget(self._build_left_panel())

        # 중: 3D 뷰포트
        self.plotter = QtInteractor(central)
        self.plotter.set_background("white")
        self.plotter.show_axes()
        root.addWidget(self.plotter.interactor, stretch=1)

        # 우: 컨트롤 패널
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(320)
        form = QtWidgets.QVBoxLayout(panel)
        root.addWidget(panel)

        # 서버 상태 + 모델
        self.status_dot = QtWidgets.QLabel("●")
        self.status_text = QtWidgets.QLabel("connecting…")
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(self.status_dot)
        srow.addWidget(self.status_text, stretch=1)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_health)
        srow.addWidget(refresh_btn)
        form.addLayout(srow)

        # Simulate 동작 모드 배너 (LIVE=BE / DUMMY=로컬). plotter.clear() 영향 없음.
        self.mode_banner = QtWidgets.QLabel("")
        self.mode_banner.setWordWrap(True)
        form.addWidget(self.mode_banner)

        # 시나리오 선택 — 허블 자산이 있으면 첫 항목(기본), 없으면 판/캔만
        # (개발 체크아웃에서 자산 미생성/패키징 누락 시에도 앱은 동작해야 한다).
        self._scenario_kinds: list[str] = []
        items: list[str] = []
        for kind, mesh, label in SCENARIO_ASSETS:
            if os.path.isfile(_asset_path("assets", mesh)):
                self._scenario_kinds.append(kind)
                items.append(label)
        self._scenario_kinds += ["plate", "can"]
        items += ["금속 판 (plate)", "금속 캔 (can)"]
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.addItems(items)
        # activated(사용자 선택)로 연결 — currentIndexChanged와 달리 같은 항목을
        # 다시 골라도 발화한다 (Load mesh 후 원래 시나리오로 복귀 가능).
        # 프로그램적 로드(시작/폴백)는 _load_scenario를 직접 부른다.
        self.scenario_combo.activated.connect(self._on_scenario_changed)
        form.addWidget(self._labeled("Scenario", self.scenario_combo))

        self.model_combo = QtWidgets.QComboBox()
        form.addWidget(self._labeled("Model", self.model_combo))

        # Mode: 자유 낙하(기본) / 충격 — 아래 컨트롤 그룹을 전환한다.
        self.sim_mode_combo = QtWidgets.QComboBox()
        self.sim_mode_combo.addItems(["자유 낙하 (free fall)", "충격 (impact)"])
        self.sim_mode_combo.currentIndexChanged.connect(self._on_sim_mode_changed)
        form.addWidget(self._labeled("Mode", self.sim_mode_combo))

        # 메쉬 로드
        load_btn = QtWidgets.QPushButton("Load mesh…")
        load_btn.clicked.connect(self.on_load)
        form.addWidget(load_btn)
        self.mesh_label = QtWidgets.QLabel("(no mesh loaded)")
        self.mesh_label.setWordWrap(True)
        form.addWidget(self.mesh_label)

        # --- 자유 낙하 컨트롤 그룹 ---------------------------------------------
        self.freefall_group = QtWidgets.QWidget()
        ff = QtWidgets.QVBoxLayout(self.freefall_group)
        ff.setContentsMargins(0, 0, 0, 0)
        self.drop_height = self._spin(1.0, minimum=0.01)
        self.drop_orient_combo = QtWidgets.QComboBox()  # _load_scenario가 채움
        self.restitution = self._spin(0.30, minimum=0.0, maximum=0.95)
        self.restitution.setSingleStep(0.05)
        ff.addWidget(self._labeled("낙하 높이", self.drop_height))
        ff.addWidget(self._labeled("낙하 자세", self.drop_orient_combo))
        ff.addWidget(self._labeled("반발계수", self.restitution))
        form.addWidget(self.freefall_group)

        # --- 충격 컨트롤 그룹 (노드 피킹 + force + radius) ----------------------
        self.impact_group = QtWidgets.QWidget()
        ig = QtWidgets.QVBoxLayout(self.impact_group)
        ig.setContentsMargins(0, 0, 0, 0)
        self.node_label = QtWidgets.QLabel("Impact node: — (click a node)")
        ig.addWidget(self.node_label)
        self.fx = self._spin(0.0)
        self.fy = self._spin(0.0)
        self.fz = self._spin(-0.3)
        ig.addWidget(self._labeled("Force X", self.fx))
        ig.addWidget(self._labeled("Force Y", self.fy))
        ig.addWidget(self._labeled("Force Z", self.fz))
        self.radius = self._spin(0.0, minimum=0.0)
        self.radius.setSpecialValueText("auto")  # 0 → 서버가 bbox 비례로 자동
        ig.addWidget(self._labeled("Radius (0=auto)", self.radius))
        form.addWidget(self.impact_group)

        # scale은 두 모드 공용
        self.scale = self._spin(1.0, minimum=0.0)
        form.addWidget(self._labeled("Scale", self.scale))

        # 액션 버튼
        self.simulate_btn = QtWidgets.QPushButton("Simulate")
        self.simulate_btn.clicked.connect(self.on_simulate)
        form.addWidget(self.simulate_btn)
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self.on_reset)
        form.addWidget(reset_btn)

        # 재생 컨트롤 (애니메이션)
        prow = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        prow.addWidget(self.play_btn)
        self.loop_check = QtWidgets.QCheckBox("Loop")
        self.loop_check.setChecked(True)
        prow.addWidget(self.loop_check)
        form.addLayout(prow)

        self.timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.valueChanged.connect(self._on_timeline)
        form.addWidget(self._labeled("Timeline", self.timeline))

        self.speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed.setRange(5, 60)
        self.speed.setValue(30)  # fps
        self.speed.valueChanged.connect(self._on_speed)
        form.addWidget(self._labeled("Speed (fps)", self.speed))

        self.frame_label = QtWidgets.QLabel("frame —/—")
        self.frame_label.setStyleSheet("color:#555;")
        form.addWidget(self.frame_label)

        form.addStretch(1)
        self.log_label = QtWidgets.QLabel("")
        self.log_label.setWordWrap(True)
        self.log_label.setStyleSheet("color:#555;")
        form.addWidget(self.log_label)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        """좌측 패널: [분석]/[챗] 탭 + 탭 밖 하단 고정 채팅 입력줄."""
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(340)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        self.left_tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.left_tabs, stretch=1)

        # 탭 0: 분석 — 종합 판정 배지 + 부품별 표 + 선택 부품 상세
        analysis_tab = QtWidgets.QWidget()
        alay = QtWidgets.QVBoxLayout(analysis_tab)
        self.analysis_summary = QtWidgets.QLabel()
        self.analysis_summary.setWordWrap(True)
        alay.addWidget(self.analysis_summary)
        self.analysis_table = QtWidgets.QTableWidget(0, 5)
        self.analysis_table.setHorizontalHeaderLabels(
            ["부품", "최대 변위", "임계값", "상태", "기능 영향"]
        )
        # 열은 내용 크기(부품명 잘림 방지), 넘치면 스크롤 — 스크롤바 UI는 숨기되
        # 휠/드래그 스크롤은 동작한다.
        self.analysis_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.analysis_table.setTextElideMode(QtCore.Qt.ElideNone)
        self.analysis_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.analysis_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.analysis_table.verticalHeader().setVisible(False)
        self.analysis_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.analysis_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.analysis_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.analysis_table.itemSelectionChanged.connect(self._on_analysis_row_selected)
        alay.addWidget(self.analysis_table)  # stretch 없음 — 높이는 내용 맞춤(_fit_table_height)
        # 행 선택 상세 — 표 바로 아래가 메인 영역 (기능/정착/변형률 breakdown HTML)
        self.analysis_detail = QtWidgets.QTextBrowser()
        self.analysis_detail.setOpenExternalLinks(False)
        self.analysis_detail.setPlaceholderText("행을 선택하면 부품 상세가 표시됩니다.")
        self.analysis_detail.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.analysis_detail.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.analysis_detail.setStyleSheet("QTextBrowser{border:1px solid #ddd; color:#333;}")
        alay.addWidget(self.analysis_detail, stretch=1)
        # ChatGPT/Gemini식 프롬프트 입력 — 분석 탭 하단(메인 진입점). 전송 시 챗 탭 전환.
        alay.addWidget(self._make_chat_input(
            "분석 결과에 대해 무엇이든 물어보세요…"))
        self.left_tabs.addTab(analysis_tab, "분석")

        # 탭 1: 챗 — 대화 뷰 + 자체 입력(후속 질문으로 멀티턴 이어가기)
        chat_tab = QtWidgets.QWidget()
        clay = QtWidgets.QVBoxLayout(chat_tab)
        self.chat_view = QtWidgets.QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        clay.addWidget(self.chat_view, stretch=1)
        clay.addWidget(self._make_chat_input("메시지 입력…"))
        self.left_tabs.addTab(chat_tab, "챗")

        self._reset_analysis_panel()
        return panel

    # 두 탭 공용 입력 위젯을 각각 관리 — 어느 쪽에서 보내든 같은 흐름을 탄다.
    def _make_chat_input(self, placeholder: str) -> QtWidgets.QWidget:
        """ChatGPT식 프롬프트 입력 한 벌(둥근 입력창 + 전송 버튼)을 만들어 반환.

        여러 탭에 각각 배치하므로 위젯 참조를 리스트에 모아 활성/비활성·클리어를
        일괄 처리한다."""
        if not hasattr(self, "chat_inputs"):
            self.chat_inputs: list[QtWidgets.QLineEdit] = []
            self.chat_send_btns: list[QtWidgets.QPushButton] = []
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 6, 0, 0)
        row.setSpacing(6)
        line = QtWidgets.QLineEdit()
        line.setPlaceholderText(placeholder)
        line.setMinimumHeight(38)
        line.setStyleSheet(
            "QLineEdit{border:1px solid #cfcfcf; border-radius:19px; padding:0 14px;"
            " background:#fafafa;} QLineEdit:focus{border:1px solid #2e7d32;}"
        )
        line.returnPressed.connect(lambda w=line: self.on_chat_send(w))
        btn = QtWidgets.QPushButton("전송")
        btn.setMinimumHeight(38)
        btn.setStyleSheet(
            "QPushButton{border:none; border-radius:19px; padding:0 16px;"
            " background:#2e7d32; color:white; font-weight:bold;}"
            " QPushButton:disabled{background:#bdbdbd;}"
        )
        btn.clicked.connect(lambda _=False, w=line: self.on_chat_send(w))
        row.addWidget(line, stretch=1)
        row.addWidget(btn)
        self.chat_inputs.append(line)
        self.chat_send_btns.append(btn)
        return box

    # ---- 분석 패널/오버레이 --------------------------------------------------
    def _reset_analysis_panel(self):
        """메쉬 교체 시 분석 상태 초기화 (챗 히스토리는 유지 — 답변은 질문
        시점의 분석 기준이라 그대로 읽어도 유효)."""
        self.analysis = None
        self.analysis_summary.setText("시뮬레이션 후 분석 결과가 표시됩니다.")
        self.analysis_summary.setStyleSheet(
            "background:#f5f5f5; color:#555; padding:6px; border-radius:4px;"
        )
        self.analysis_table.setRowCount(0)
        self._fit_table_height()
        self.analysis_detail.clear()

    def _update_analysis_panel(self, result: analysis.AnalysisResult):
        bg, fg = ANALYSIS_STATUS_STYLES.get(result.verdict, ANALYSIS_STATUS_STYLES["N/A"])
        mode_txt = "LIVE" if result.sim_mode == "live" else "DUMMY"
        sim_txt = "낙하" if result.action.get("mode") == "free_fall" else "충격"
        extras = ""
        if result.single_frame:
            extras += " · 단일 프레임(속도 지표 없음)"
        if result.fallback_whole_mesh:
            extras += " · 부품 정의 없음(전체 메쉬)"
        gm = result.global_motion or {}
        posture = ""
        if gm.get("available") and gm.get("rotation_deg", 0) > 1.0:
            posture = f"\n낙하 후 자세 변화 {gm['rotation_deg']:.1f}° (변형과 별개)"
        self.analysis_summary.setText(
            f"종합 판정: {result.verdict} — {result.scenario} · {sim_txt} · {mode_txt}{extras}\n"
            f"최대 변형 {result.overall_max_disp:.4g} @ 노드 #{result.overall_max_node}{posture}"
        )
        self.analysis_summary.setStyleSheet(
            f"background:{bg}; color:{fg}; padding:6px; border-radius:4px; font-weight:bold;"
        )

        prev_row = self.analysis_table.currentRow()
        self.analysis_table.setRowCount(len(result.components))
        for i, c in enumerate(result.components):
            cells = [
                c.name, f"{c.max_disp:.4g}", f"{c.threshold:.4g}", c.status,
                _functional_brief(c),
            ]
            for j, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if j == 3:
                    sbg, sfg = ANALYSIS_STATUS_STYLES.get(c.status, ANALYSIS_STATUS_STYLES["N/A"])
                    item.setBackground(QtGui.QColor(sbg))
                    item.setForeground(QtGui.QColor(sfg))
                self.analysis_table.setItem(i, j, item)
        self._fit_table_height()
        # 재-Simulate로 행 수가 같으면 Qt가 선택을 유지한 채 시그널을 안 쏜다 —
        # 유지된 선택은 새 분석 내용으로 상세를 직접 갱신, 아니면 초기화.
        if 0 <= prev_row < len(result.components):
            self.analysis_table.selectRow(prev_row)
            self.analysis_detail.setHtml(_detail_html(result.components[prev_row]))
        else:
            self.analysis_table.clearSelection()
            self.analysis_detail.clear()

    def _fit_table_height(self):
        """표 높이를 내용(헤더+행)에 맞춘다 — 상세 영역이 표 바로 아래에 오도록."""
        h = self.analysis_table.horizontalHeader().height() + 2 * self.analysis_table.frameWidth()
        for r in range(self.analysis_table.rowCount()):
            h += self.analysis_table.rowHeight(r)
        self.analysis_table.setFixedHeight(min(h, 280))

    def _on_analysis_row_selected(self):
        if self.analysis is None:
            return
        row = self.analysis_table.currentRow()
        if not 0 <= row < len(self.analysis.components):
            return
        self.analysis_detail.setHtml(_detail_html(self.analysis.components[row]))

    def _show_analysis_overlays(self, result: analysis.AnalysisResult):
        """분석 3D 마커를 애니메이션 메쉬 위에 추가.

        위치는 정착 상태(frames[-1]) 기준 — 애니메이션이 그 상태로 끝나므로
        루프 재생 중 표면이 일시적으로 마커와 어긋나는 것은 감수한다(데모 절충).
        스칼라는 쓰지 않는다(단색만) — anim 메쉬의 |displacement| 스칼라바와
        충돌을 피하기 위해. 모든 마커는 pickable=False로 피킹을 방해하지 않는다.
        """
        if result is None or self.frames is None:
            return
        settled = self.frames[-1]

        # 임계 초과 정점 (전 부품 합산) — 빨간 점
        over_groups = [c.over_indices for c in result.components if len(c.over_indices)]
        if over_groups:
            over = np.unique(np.concatenate(over_groups))
            actor = self.plotter.add_points(
                settled[over], color="#c62828", point_size=7,
                render_points_as_spheres=True, name="analysis_over",
                pickable=False, reset_camera=False,
            )
            self._overlay_actors.append(actor)

        # 부품별 최대 충격 위치 마커+라벨 (상태별 색). 라벨 텍스트는 ASCII
        # component_id를 쓴다 — VTK 기본 폰트에 한글 글리프가 없어 깨진다.
        for status in ("FAIL", "WARN", "OK"):
            comps = [c for c in result.components if c.status == status and c.max_disp_node >= 0]
            if not comps:
                continue
            _, color = ANALYSIS_STATUS_STYLES[status]
            pts = settled[[c.max_disp_node for c in comps]]
            self._overlay_actors.append(self.plotter.add_points(
                pts, color=color, point_size=13, render_points_as_spheres=True,
                name=f"analysis_max_{status}", pickable=False, reset_camera=False,
            ))
            self._overlay_actors.append(self.plotter.add_point_labels(
                pts, [f"{c.component_id}: {c.status}" for c in comps],
                name=f"analysis_labels_{status}", show_points=False,
                font_size=12, text_color=color, shape_color="white",
                shape_opacity=0.75, always_visible=True,
            ))
        self.plotter.render()

    def _clear_overlays(self, already_cleared: bool = False):
        """분석 마커 제거. plotter.clear()가 이미 전체 액터를 지운 경우
        리스트만 리셋한다 (clear 지점마다 호출해 stale 참조를 막는다)."""
        if not already_cleared:
            for actor in self._overlay_actors:
                self.plotter.remove_actor(actor)
        self._overlay_actors = []

    # ---- 챗 ------------------------------------------------------------------
    def on_chat_send(self, source: QtWidgets.QLineEdit | None = None):
        # 분석/챗 두 탭의 입력 중 트리거한 쪽(또는 내용 있는 첫 입력)의 텍스트.
        line = source if source is not None else next(
            (w for w in self.chat_inputs if w.text().strip()), None
        )
        question = line.text().strip() if line is not None else ""
        if not question:
            return
        if self.chat_worker is not None and self.chat_worker.isRunning():
            return  # 입력 비활성화로 도달하지 않지만 방어
        for w in self.chat_inputs:
            w.clear()
        self.left_tabs.setCurrentIndex(1)  # 챗 탭으로 전환 (대화가 이어지는 곳)
        self._append_chat("user", question)

        summary = self.analysis.to_summary_json() if self.analysis is not None else {}
        # 서버 도달 가능하면 /chat(LLM 또는 서버측 폴백), 아니면 로컬 폴백 즉답.
        # 챗은 mesh 모델 준비(_be_ready)와 무관 — 서버 생존 여부만 본다.
        if self._server_reachable:
            self._pending_question = question
            self._set_chat_enabled(False)
            self.chat_worker = ChatWorker(
                self.client, question, summary, self.chat_history[-6:]
            )
            self.chat_worker.finished_ok.connect(self._on_chat_reply)
            self.chat_worker.failed.connect(self._on_chat_failed)
            self.chat_worker.start()
        else:
            answer = fallback_answer(question, summary)
            self._append_chat("assistant", answer, tag="규칙·로컬")
            self._push_chat_history(question, answer)

    def _on_chat_reply(self, resp: dict):
        self._set_chat_enabled(True)
        question, self._pending_question = self._pending_question, None
        answer = str(resp.get("answer", "")).strip() or "(빈 응답)"
        tag = "LLM" if resp.get("mode") == "llm" else "규칙"
        self._append_chat("assistant", answer, tag=tag)
        if resp.get("error"):
            self._log(f"chat: LLM 실패로 규칙 폴백 ({resp['error'][:80]})")
        if question:
            self._push_chat_history(question, answer)

    def _on_chat_failed(self, msg: str):
        # BE 도달 실패 → 로컬 폴백으로 강등. 요약은 워커가 들고 있는 "질문 시점"
        # 스냅샷을 쓴다 — in-flight 중 메쉬 교체로 self.analysis가 바뀌어도 무관.
        self._set_chat_enabled(True)
        question, self._pending_question = self._pending_question, None
        summary = self.chat_worker.analysis_summary if self.chat_worker is not None else {}
        answer = fallback_answer(question or "", summary)
        self._append_chat("assistant", answer, tag="규칙·로컬")
        self._log(f"chat: 서버 호출 실패 → 로컬 폴백 ({msg[:80]})")
        if question:
            self._push_chat_history(question, answer)

    def _set_chat_enabled(self, enabled: bool):
        for w in self.chat_inputs:
            w.setEnabled(enabled)
        for b in self.chat_send_btns:
            b.setEnabled(enabled)
            b.setText("전송" if enabled else "…")

    def _append_chat(self, role: str, text: str, tag: str | None = None):
        body = html.escape(text).replace("\n", "<br>")
        if role == "user":
            prefix = '<b style="color:#1a5276;">나</b>'
        else:
            tag_html = (
                f' <span style="color:#e65100; font-size:11px;">[{tag}]</span>' if tag else ""
            )
            prefix = f'<b style="color:#2e7d32;">답변</b>{tag_html}'
        self.chat_view.append(f'<div style="margin:4px 0;">{prefix}<br>{body}</div>')

    def _push_chat_history(self, question: str, answer: str):
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})
        del self.chat_history[:-20]  # 메모리 상한 (서버 전송은 최근 6개만)

    def _labeled(self, text: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = QtWidgets.QLabel(text)
        lab.setFixedWidth(110)
        lay.addWidget(lab)
        lay.addWidget(widget, stretch=1)
        return box

    @staticmethod
    def _spin(value: float, minimum: float = -1e6, maximum: float = 1e6):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(minimum, maximum)
        s.setDecimals(3)
        s.setSingleStep(0.1)
        s.setValue(value)
        return s

    # ---- 서버 상태 ----------------------------------------------------------
    def refresh_health(self):
        try:
            health = self.client.health()
        except Exception as e:
            self._set_status("down", f"server unreachable: {e}")
            self.model_combo.clear()
            self._be_ready = False
            self._server_reachable = False  # 챗도 로컬 폴백으로
            self._set_mode("dummy", "⚠ DUMMY 모드 · 모델 미연결 — Simulate는 로컬 더미 반응")
            return

        # 응답이 왔으면(503 포함) 서버 자체는 살아있다 — 챗은 /chat로 보낸다.
        self._server_reachable = True
        status = health.get("status", "unknown")
        self._set_status(status, f"{status} @ {self.client.base_url}")

        # ready 모델 우선 채우기 (없으면 전체)
        models = health.get("models", {})
        ready = [mid for mid, m in models.items() if m.get("status") == "ready"]
        ids = ready or list(models)
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(ids)
        if current in ids:
            self.model_combo.setCurrentText(current)
        elif health.get("default_model") in ids:
            self.model_combo.setCurrentText(health["default_model"])

        # ready 모델이 하나라도 있으면 Simulate는 BE로 (LIVE), 아니면 로컬 더미.
        self._be_ready = bool(ready)
        if self._be_ready:
            self._set_mode("live", f"● LIVE · BE 연결됨 (model: {self.model_combo.currentText()})")
        elif status == "loading":
            self._set_mode("loading", "BE 모델 로딩 중 · 지금은 로컬 더미 반응")
        else:
            self._set_mode("dummy", "⚠ DUMMY 모드 · 모델 미준비 — Simulate는 로컬 더미 반응")

    def _set_status(self, status: str, text: str):
        color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
        self.status_dot.setStyleSheet(f"color:{color}; font-size:16px;")
        self.status_text.setText(text)

    def _set_mode(self, kind: str, text: str):
        bg, fg = MODE_STYLES.get(kind, MODE_STYLES["dummy"])
        self.mode_banner.setStyleSheet(
            f"background:{bg}; color:{fg}; padding:4px 6px; border-radius:4px; font-weight:bold;"
        )
        self.mode_banner.setText(text)

    # ---- 메쉬 로드/렌더 -----------------------------------------------------
    def on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open mesh", "", MESH_FILTER)
        if not path:
            return
        try:
            with open(path, "rb") as f:
                self.orig_bytes = f.read()
            self.vertices, self.faces, self.file_format = load_mesh(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
            return

        self._stop_play()
        self._invalidate_sim()  # 진행 중 worker 결과 무효화 + Simulate 버튼 복구
        self.frames = None
        self.impact_node = None
        self.components_def = None  # 사용자 메쉬는 부품 정의 없음 → 전체 메쉬 분석
        self._reset_analysis_panel()
        self.node_label.setText("Impact node: — (click a node)")
        self._configure_drop_controls(None)  # 사용자 메쉬는 자산 프리셋 없음
        self.mesh_label.setText(
            f"{os.path.basename(path)}  —  {len(self.vertices)} nodes, "
            f"{len(self.faces)} faces  [{self.file_format}]"
        )
        self._render_original(reset_camera=True)
        self._refresh_picking()
        self._reset_timeline()
        self._log("mesh loaded — 모드에 맞춰 조건을 정하고 Simulate.")

    def _on_scenario_changed(self, index: int):
        if 0 <= index < len(self._scenario_kinds):
            self._load_scenario(self._scenario_kinds[index])

    def _load_scenario(self, kind: str):
        """시나리오 메쉬(허블/스마트폰 자산 또는 금속 판/캔)를 로드."""
        self._stop_play()
        self._invalidate_sim()  # 진행 중 worker 결과 무효화 + Simulate 버튼 복구
        self.components_def = None

        asset = next((a for a in SCENARIO_ASSETS if a[0] == kind), None)
        if asset is not None:
            self._load_asset_scenario(*asset)
            return

        default_radius = 0.0  # 0 = auto (bbox 비례)
        if kind == "can":
            self.vertices, self.faces = make_can_mesh()
            center = (CAN_NH // 2) * CAN_NTHETA  # 중간 높이, θ=0
            label = f"금속 캔 (cylinder {CAN_NTHETA}×{CAN_NH})"
        else:  # plate
            self.vertices, self.faces = make_plate_mesh()
            center = (PLATE_N // 2) * PLATE_N + (PLATE_N // 2)
            label = f"금속 판 ({PLATE_N}×{PLATE_N} grid)"
        self.file_format = "vtk"
        self.orig_bytes = dump_mesh(self.vertices, self.faces, self.file_format)
        self.radius.setValue(default_radius)
        self._configure_drop_controls(None)  # 내장 메쉬는 자산 프리셋 없음
        self._finish_scenario_load(center, label)

    def _load_asset_scenario(self, kind: str, mesh_file: str, combo_label: str):
        """자산 메쉬(원본 바이트 그대로 BE 전송 → 인덱스 일치) + components.json 로드.

        파일 손상/유실 시 금속 판으로 강등하고 콤보도 동기화한다."""
        path = _asset_path("assets", mesh_file)
        try:
            with open(path, "rb") as f:
                self.orig_bytes = f.read()
            self.vertices, self.faces, self.file_format = load_mesh(path)
        except Exception as e:
            fallback = "plate"
            self.scenario_combo.blockSignals(True)
            self.scenario_combo.setCurrentIndex(self._scenario_kinds.index(fallback))
            self.scenario_combo.blockSignals(False)
            self._load_scenario(fallback)
            self._log(f"{combo_label} 자산 로드 실패({e}) — 금속 판으로 대체")
            return

        self.components_def = analysis.load_components(
            _asset_path("assets", os.path.splitext(mesh_file)[0] + ".components.json")
        )
        center, default_radius = 0, 0.0
        if self.components_def is not None:
            try:  # sidecar 값 타입 오류도 기본값으로 강등
                center = int(self.components_def.get("default_impact_node") or 0)
                default_radius = float(self.components_def.get("default_radius") or 0.0)
            except (TypeError, ValueError):
                center, default_radius = 0, 0.0
        center = min(max(0, center), len(self.vertices) - 1)
        self.radius.setValue(default_radius)
        self._configure_drop_controls(self.components_def)
        self._finish_scenario_load(center, combo_label.split(" (")[0])

    def _configure_drop_controls(self, components_def: dict | None):
        """낙하 자세 콤보(공통 프리셋 + 자산 프리셋)와 기본 낙하 높이를 세팅.

        메쉬 스케일이 자산마다 달라(정규화 여부) 높이는 자산 JSON 우선, 없으면
        bbox 대각선의 0.5배를 기본으로 한다."""
        self.drop_orient_combo.clear()
        presets = list(DROP_ORIENTATION_PRESETS)
        default_name = None
        default_height = None
        if isinstance(components_def, dict):
            for item in components_def.get("drop_orientations") or []:
                euler = item.get("euler_deg")
                if not (isinstance(euler, (list, tuple)) and len(euler) == 3):
                    continue
                try:  # 원소 타입 오류도 '자산 손상 시 강등' 계약대로 건너뛴다
                    vals = tuple(float(v) for v in euler)
                except (TypeError, ValueError):
                    continue
                presets.append((str(item.get("name", "자산 자세")), vals))
            default_name = components_def.get("default_drop_orientation")
            try:
                default_height = float(components_def.get("default_drop_height"))
            except (TypeError, ValueError):
                default_height = None
        for name, euler in presets:
            self.drop_orient_combo.addItem(name, userData=euler)
        if default_name:
            idx = self.drop_orient_combo.findText(default_name)
            if idx >= 0:
                self.drop_orient_combo.setCurrentIndex(idx)
        if default_height is None:
            diag = float(np.linalg.norm(self.vertices.max(0) - self.vertices.min(0))) or 1.0
            default_height = round(0.5 * diag, 2)
        self.drop_height.setValue(default_height)

    def _finish_scenario_load(self, center: int, label: str):
        """시나리오 로드 공통 마무리 — 렌더/피킹/충격점/타임라인."""
        self.frames = None
        self._reset_analysis_panel()
        self.mesh_label.setText(
            f"{label} — {len(self.vertices)} nodes, {len(self.faces)} faces"
        )
        self._render_original(reset_camera=True)
        self.impact_node = min(max(0, center), len(self.vertices) - 1)
        self._refresh_picking()  # 충격 모드에서만 픽 활성 + 하이라이트
        self.node_label.setText(f"Impact node: {self.impact_node} (기본값)")
        self._reset_timeline()
        verb = "충격점을 바꾸고" if self._current_mode() == "impact" else "낙하 조건을 정하고"
        self._log(f"{label} 로드됨 — {verb} Simulate.")

    def _render_original(self, reset_camera: bool = False):
        """원본 메쉬만 렌더 (애니메이션/pick highlight/분석 마커 제거)."""
        self.plotter.clear()
        self.anim_poly = None
        self._pick_actor = None
        self._clear_overlays(already_cleared=True)  # clear()가 이미 액터를 지움
        poly = to_polydata(self.vertices, self.faces)
        self.plotter.add_mesh(
            poly, color="#9ecae1", show_edges=True, edge_color="#3182bd",
            name="original", pickable=True,
        )
        if reset_camera:
            self.plotter.reset_camera()

    def _current_mode(self) -> str:
        """'impact' | 'free_fall' — Mode 콤보 현재값."""
        return "impact" if self.sim_mode_combo.currentIndex() == 1 else "free_fall"

    def _disable_picking(self):
        # 결과 재생 중·낙하 모드에서는 클릭이 카메라 조작 전용 — 픽 마커를 안 찍는다.
        try:
            self.plotter.disable_picking()
        except Exception:
            pass

    def _enable_picking(self):
        # 표면 위 클릭 지점 → 최근접 정점 인덱스로 환산.
        # pyvista 최신 버전은 중복 enable을 PyVistaPickingError로 막는다 —
        # (리셋/시나리오 로드마다 재호출되므로) 기존 피킹을 먼저 해제한다.
        self._disable_picking()
        self.plotter.enable_point_picking(
            callback=self._on_pick, show_message=False, left_clicking=True,
            use_picker=True,
        )

    def _refresh_picking(self):
        """충격 모드 + 원본 표시(frames 없음)일 때만 픽 활성화 + 충격점 하이라이트.

        낙하 모드는 충격점 개념이 없어 픽을 끈다. 결과 재생 중에도 끈다."""
        if self._current_mode() == "impact" and self.frames is None and self.vertices is not None:
            self._enable_picking()
            if self.impact_node is not None:
                self._highlight_node(self.impact_node)
        else:
            self._disable_picking()
            # 낙하 모드/결과 재생 중엔 충격점 마커가 무의미 — 남아 있으면 제거.
            if self._pick_actor is not None:
                self.plotter.remove_actor(self._pick_actor)
                self._pick_actor = None

    def _on_sim_mode_changed(self, _index: int):
        impact = self._current_mode() == "impact"
        self.impact_group.setVisible(impact)
        self.freefall_group.setVisible(not impact)
        self._refresh_picking()

    def _on_pick(self, point, *args):
        if self.vertices is None or point is None:
            return
        if self.frames is not None:
            return  # 픽은 Simulate 전(원본 표시 상태)에만 — 결과 재생 중 방어
        point = np.asarray(point, dtype=np.float64).reshape(-1)[:3]
        # 원본 메쉬 표면 정점만 후보로 (체적 메쉬 내부 정점으로 스냅 방지).
        if self.faces is not None and len(self.faces):
            surf = np.unique(self.faces)
            idx = int(surf[np.argmin(np.linalg.norm(self.vertices[surf] - point, axis=1))])
        else:
            idx = int(np.argmin(np.linalg.norm(self.vertices - point, axis=1)))
        self.impact_node = idx
        self.node_label.setText(f"Impact node: {idx}  @ {self.vertices[idx].round(3).tolist()}")
        self._highlight_node(idx)

    def _highlight_node(self, idx: int):
        if self._pick_actor is not None:
            self.plotter.remove_actor(self._pick_actor)
        diag = float(np.linalg.norm(self.vertices.max(0) - self.vertices.min(0))) or 1.0
        sphere = pv.Sphere(radius=0.02 * diag, center=self.vertices[idx])
        self._pick_actor = self.plotter.add_mesh(sphere, color="#e6550d", name="impact")

    # ---- 시뮬레이션 / 애니메이션 -------------------------------------------
    def _invalidate_sim(self):
        """진행 중 시뮬 결과 무효화 + Simulate 버튼 복구.

        메쉬 교체/리셋 시 stale worker 결과는 gen 체크로 무시되므로, 버튼을
        바로 되살려도 안전하다 (안 살리면 이전 요청 타임아웃까지 잠긴다)."""
        self._sim_gen += 1
        self.simulate_btn.setEnabled(True)

    def on_simulate(self):
        if self.vertices is None or self.orig_bytes is None:
            QtWidgets.QMessageBox.information(self, "No mesh", "Load a mesh first.")
            return

        mode = self._current_mode()
        if mode == "free_fall":
            action = {
                "mode": "free_fall",
                "drop_height": self.drop_height.value(),
                "restitution": self.restitution.value(),
                "scale": self.scale.value(),
                "frames": 90,  # 명시 전송 — BE MAX_SIM_POINTS 힌트가 정확해진다
                "orientation": list(self.drop_orient_combo.currentData() or (0.0, 0.0, 0.0)),
            }
            model_wanted = "free_fall"
        else:  # impact
            if self.impact_node is None:
                QtWidgets.QMessageBox.information(self, "No node", "충격점을 클릭해 고르세요.")
                return
            action = {
                "mode": "impact",
                "impact_node": self.impact_node,
                "force": [self.fx.value(), self.fy.value(), self.fz.value()],
                "scale": self.scale.value(),
            }
            if self.radius.value() > 0:
                action["radius"] = self.radius.value()
            # 충격 모드는 model 콤보 선택을 쓰되, free_fall이면 metal_dent로 대체.
            model_wanted = self.model_combo.currentText() or None
            if model_wanted == "free_fall":
                model_wanted = "metal_dent"
        self._last_action = action  # 분석 요약(action 표시)용

        self._stop_play()
        # BE ready + 요청 모델이 BE에 있으면 /simulate, 아니면 로컬 미러.
        be_has_model = model_wanted is None or self._model_available(model_wanted)
        if self._be_ready and be_has_model:
            self.simulate_btn.setEnabled(False)
            self._log(f"simulating (BE /simulate, {model_wanted or 'default'})…")
            self._sim_gen += 1
            gen = self._sim_gen
            self.worker = SimulateWorker(
                self.client, self.orig_bytes, self.file_format, action, model_wanted
            )
            self.worker.finished_ok.connect(lambda p, g=gen: self._on_frames_ready(g, p))
            self.worker.failed.connect(lambda m, g=gen: self._on_sim_failed(g, m))
            self.worker.start()
        else:
            if self._be_ready and not be_has_model:
                self._log(f"BE에 {model_wanted} 모델 없음 — 로컬 미러로 대체")
            try:
                frames = self._local_simulate(mode, action)
            except Exception as e:
                self._log(f"failed: {e}")
                QtWidgets.QMessageBox.critical(self, "Simulate failed", str(e))
                return
            self._setup_animation(self.faces, frames, dummy=True)

    def _model_available(self, model_id: str) -> bool:
        return self.model_combo.findText(model_id) >= 0

    def _local_simulate(self, mode: str, action: dict) -> np.ndarray:
        """오프라인 폴백 — 모드별 로컬 미러 (BE 모델과 동일 수식)."""
        if mode == "free_fall":
            return free_fall_sim.free_fall_trajectory(self.vertices, action)
        return metal_dent_simulate(self.vertices, self.faces, action)

    def _on_frames_ready(self, gen, payload):
        self.simulate_btn.setEnabled(True)
        if gen != self._sim_gen:  # mesh가 바뀐 뒤 도착한 오래된 결과 → 무시
            return
        faces, frames = payload
        self._setup_animation(faces, frames, dummy=False)

    def _on_sim_failed(self, gen, msg: str):
        self.simulate_btn.setEnabled(True)
        if gen != self._sim_gen:
            return
        self._log(f"failed: {msg}")
        QtWidgets.QMessageBox.critical(self, "Simulate failed", msg)

    def _setup_animation(self, faces, frames, dummy: bool):
        """(T,N,3) 프레임을 변위 히트맵 메쉬로 렌더하고 재생을 시작한다."""
        frames = np.asarray(frames, dtype=np.float64)
        self.frames = frames
        self.faces = np.asarray(faces)
        # 변위 기준점: T==1이면 frames[0]이 이미 변형 상태 → 원본 대비
        # (analysis.compute_analysis와 동일 규칙 — 히트맵/분석 수치 일관).
        if len(frames) == 1 and self.vertices is not None \
                and self.vertices.shape == frames.shape[1:]:
            rest = np.asarray(self.vertices, dtype=np.float64)
        else:
            rest = frames[0]
        # 자유 낙하는 전역 강체(낙하/바운스)를 제거한 변형장으로 색칠한다 — 물체는
        # 눈에 보이게 낙하하되(렌더는 원시 frames) 색은 dent만 나타낸다. 충격 모드는
        # 물체가 고정돼 있어 원시 변위를 그대로 쓴다(국소 대변형 보존).
        free_fall = self._last_action.get("mode") == "free_fall"
        rigid_removed = None
        if free_fall:
            rigid_removed = analysis.remove_global_rigid(frames, rest)
            self.disp = np.linalg.norm(rigid_removed[0] - rest[None], axis=2)
        else:
            self.disp = np.linalg.norm(frames - rest[None], axis=2)  # (T,N)
        self.vmax = float(self.disp.max())
        if self.vmax < 1e-9:  # 순수 강체 낙하 등 — fp 노이즈를 증폭하지 않는다
            self.vmax = 1.0

        self.plotter.clear()
        self._pick_actor = None
        self._clear_overlays(already_cleared=True)  # clear()가 이미 액터를 지움
        if free_fall:
            self._add_floor_plane(frames)
        self.anim_poly = to_polydata(frames[0], self.faces)
        self.anim_poly["displacement"] = self.disp[0]
        self.plotter.add_mesh(
            self.anim_poly, scalars="displacement", cmap="turbo",
            clim=[0.0, self.vmax], show_edges=False, name="anim",
            scalar_bar_args={"title": "|displacement|"}, reset_camera=False,
        )
        # 카메라는 로드 시 이미 맞춰졌으므로 Simulate마다 리셋하지 않는다(시점 보존).
        # 결과 재생 중에는 피킹 해제 — 회전/확대 클릭에 픽 마커가 찍히지 않게.
        # 충격점을 바꾸려면 Reset(픽 재활성화) 후 선택한다.
        self._disable_picking()

        T = len(frames)
        self.timeline.setEnabled(T > 1)
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, max(0, T - 1))
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.anim_idx = 0
        self._render_frame(0)

        if dummy:
            tag = " (로컬 free_fall)" if free_fall else " (로컬 metal_dent)"
        else:
            tag = ""
        self._log(f"simulate 완료{tag} — {T} frames, max|변형|={self.vmax:.4f}")

        # 부품별 충격 분석 — LIVE/DUMMY 동일 경로(frames는 이미 FE 메모리에 있다).
        # 낙하 모드는 히트맵과 같은 강체 제거 결과를 공유해 이중 Kabsch를 피한다.
        self.analysis = analysis.compute_analysis(
            frames, self.vertices, self.faces, self._last_action,
            self.components_def, "dummy" if dummy else "live",
            remove_rigid=free_fall, rigid_removed=rigid_removed,
        )
        self._update_analysis_panel(self.analysis)
        self._show_analysis_overlays(self.analysis)
        if T > 1:
            self._start_play()

    def _add_floor_plane(self, frames: np.ndarray):
        """낙하 시나리오의 바닥 평면(반투명) — 접촉 지점을 직관적으로 보이게."""
        z_floor = float(frames[:, :, 2].min())
        lo, hi = self.vertices.min(0), self.vertices.max(0)
        span = float(np.linalg.norm(hi[:2] - lo[:2])) or 1.0
        cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
        plane = pv.Plane(center=(cx, cy, z_floor), direction=(0, 0, 1),
                         i_size=1.6 * span, j_size=1.6 * span)
        actor = self.plotter.add_mesh(
            plane, color="#bdbdbd", opacity=0.35, name="floor",
            pickable=False, reset_camera=False,
        )
        self._overlay_actors.append(actor)

    def _render_frame(self, i: int):
        if self.frames is None or self.anim_poly is None:
            return
        i = int(np.clip(i, 0, len(self.frames) - 1))
        self.anim_idx = i
        self.anim_poly.points = self.frames[i]
        self.anim_poly["displacement"] = self.disp[i]
        self.plotter.render()
        self.timeline.blockSignals(True)
        self.timeline.setValue(i)
        self.timeline.blockSignals(False)
        self.frame_label.setText(f"frame {i + 1}/{len(self.frames)}")

    def _next_frame(self):
        if self.frames is None:
            self._stop_play()
            return
        i = self.anim_idx + 1
        if i >= len(self.frames):
            if self.loop_check.isChecked():
                i = 0
            else:
                self._stop_play()
                return
        self._render_frame(i)

    def _interval_ms(self) -> int:
        return int(1000 / max(1, self.speed.value()))

    def _start_play(self):
        if self.frames is None or len(self.frames) <= 1:
            return
        if self.anim_idx >= len(self.frames) - 1:
            self._render_frame(0)  # 끝에서 다시 누르면 처음부터
        self.playing = True
        self.play_btn.setText("⏸ Pause")
        self.anim_timer.start(self._interval_ms())

    def _stop_play(self):
        self.playing = False
        self.play_btn.setText("▶ Play")
        self.anim_timer.stop()

    def _toggle_play(self):
        self._stop_play() if self.playing else self._start_play()

    def _on_timeline(self, value: int):
        if self.frames is None:
            return
        self._stop_play()
        self._render_frame(int(value))

    def _on_speed(self, _value: int):
        if self.playing:
            self.anim_timer.start(self._interval_ms())

    def _reset_timeline(self):
        self._stop_play()
        self.timeline.setEnabled(False)
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, 0)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.frame_label.setText("frame —/—")

    def on_reset(self):
        if self.vertices is None:
            return
        self._stop_play()
        self._invalidate_sim()  # 진행 중 worker 결과 무효화 + Simulate 버튼 복구
        self.frames = None
        self._render_original(reset_camera=False)
        self._refresh_picking()  # 충격 모드에서만 픽 재활성 + 하이라이트
        self._reset_timeline()
        self._log("reset to original.")

    def _log(self, msg: str):
        self.log_label.setText(msg)

    def _drain_worker(self, worker, timeout_ms: int = 3000):
        """워커 종료를 상한 시간까지만 대기. 무한 wait()는 네트워크 타임아웃
        (시뮬 30s/챗 60s)만큼 창 닫기를 얼리므로, 초과 시 시그널만 끊고
        모듈 참조에 보관해(파괴 방지) 프로세스 종료에 맡긴다."""
        if worker is None or not worker.isRunning():
            return
        if not worker.wait(timeout_ms):
            try:
                worker.finished_ok.disconnect()
                worker.failed.disconnect()
            except (TypeError, RuntimeError):
                pass  # 이미 끊겼거나 연결 없음
            _ORPHAN_WORKERS.append(worker)  # GC로 QThread가 파괴되지 않게 유지

    def closeEvent(self, event):
        # 애니메이션 타이머·시뮬/챗 스레드를 정리하고 VTK 렌더 윈도우를 명시적으로
        # 닫는다 — Windows 종료 중 implicit finalize 시의 access violation/hang 방지.
        self.anim_timer.stop()
        self._drain_worker(self.worker)
        self._drain_worker(self.chat_worker)
        self.plotter.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
