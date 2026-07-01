"""Physics Impact Simulator — Windows 데스크톱 FE (PySide6 + PyVista).

흐름:
  1. 시작 시 내장 시나리오 메쉬(금속 판)가 로드된다. Scenario 드롭다운(판/캔) 또는
     "Load mesh…"로 로컬 파일(meshio 지원 확장자)을 열 수도 있다.
  2. 3D 뷰포트에서 노드를 클릭해 impact_node를 고른다.
  3. force(X/Y/Z) · radius · scale을 입력하고 Simulate.
  4. 모델 연결 시 → BE /simulate가 준 (T,N,3) 프레임을, 미연결 시 → 로컬 metal_dent
     궤적을 재생 애니메이션한다. 변위 크기를 히트맵('turbo')으로 칠하고, ▶/⏸·Loop·
     타임라인 슬라이더로 제어. (모드 배너로 LIVE/DUMMY 명시.)

노드 인덱스 일관성: FE도 BE와 동일하게 meshio로 메쉬를 파싱해 vertices/faces를
얻고, 그 배열 순서대로 렌더링한다. 피킹으로 고른 점은 vertices 배열의 인덱스로
환산되며, Simulate 시 "원본 파일 바이트"를 그대로 보내므로 BE가 같은 순서로
재파싱 → impact_node가 정확히 일치한다.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")  # qtpy가 PySide6를 고르도록

import sys

import meshio
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtWidgets

from api_client import PhysicsClient


def _read_version() -> str:
    """버전 문자열을 읽는다. PyInstaller 패키징 시 번들된 VERSION(sys._MEIPASS),
    개발 환경에선 스크립트 옆 VERSION을 사용 (CI가 빌드마다 patch를 올린다)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(base, "VERSION"), encoding="utf-8") as f:
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Physics Impact Simulator v{APP_VERSION}")
        self.resize(1280, 760)

        self.client = PhysicsClient()
        self.worker: SimulateWorker | None = None

        # 현재 로드된 메쉬 상태
        self.orig_bytes: bytes | None = None
        self.file_format = "vtk"
        self.vertices: np.ndarray | None = None
        self.faces: np.ndarray | None = None
        self.impact_node: int | None = None
        self._be_ready = False  # BE 모델 준비 완료 → Simulate가 BE로 감
        self._pick_actor = None

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
        self.refresh_health()
        # 시작 시 기본 시나리오(금속 판)를 띄워 BE 없이도 인터랙션을 확인할 수 있게.
        self._load_scenario("plate")

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # 좌: 3D 뷰포트
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

        # 시나리오 선택 (내장 데모 메쉬)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.addItems(["금속 판 (plate)", "금속 캔 (can)"])
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_changed)
        form.addWidget(self._labeled("Scenario", self.scenario_combo))

        self.model_combo = QtWidgets.QComboBox()
        form.addWidget(self._labeled("Model", self.model_combo))

        # 메쉬 로드
        load_btn = QtWidgets.QPushButton("Load mesh…")
        load_btn.clicked.connect(self.on_load)
        form.addWidget(load_btn)
        self.mesh_label = QtWidgets.QLabel("(no mesh loaded)")
        self.mesh_label.setWordWrap(True)
        form.addWidget(self.mesh_label)

        # impact node
        self.node_label = QtWidgets.QLabel("Impact node: — (click a node)")
        form.addWidget(self.node_label)

        # force
        self.fx = self._spin(0.0)
        self.fy = self._spin(0.0)
        self.fz = self._spin(-0.3)
        form.addWidget(self._labeled("Force X", self.fx))
        form.addWidget(self._labeled("Force Y", self.fy))
        form.addWidget(self._labeled("Force Z", self.fz))

        # radius / scale
        self.radius = self._spin(0.0, minimum=0.0)
        self.radius.setSpecialValueText("auto")  # 0 → 서버가 bbox 비례로 자동
        self.scale = self._spin(1.0, minimum=0.0)
        form.addWidget(self._labeled("Radius (0=auto)", self.radius))
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
            self._set_mode("dummy", "⚠ DUMMY 모드 · 모델 미연결 — Simulate는 로컬 더미 반응")
            return

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
        self._sim_gen += 1  # 진행 중 worker 결과 무효화
        self.frames = None
        self.impact_node = None
        self.node_label.setText("Impact node: — (click a node)")
        self.mesh_label.setText(
            f"{os.path.basename(path)}  —  {len(self.vertices)} nodes, "
            f"{len(self.faces)} faces  [{self.file_format}]"
        )
        self._render_original(reset_camera=True)
        self._enable_picking()
        self._reset_timeline()
        self._log("mesh loaded — click a node to set impact point.")

    def _on_scenario_changed(self, index: int):
        self._load_scenario("can" if index == 1 else "plate")

    def _load_scenario(self, kind: str):
        """내장 시나리오 메쉬(금속 판/캔)를 로드. 기본 충격점을 미리 선택."""
        self._stop_play()
        self._sim_gen += 1  # 진행 중 worker 결과 무효화
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
        self.frames = None

        self.mesh_label.setText(
            f"{label} — {len(self.vertices)} nodes, {len(self.faces)} faces"
        )
        self._render_original(reset_camera=True)
        self._enable_picking()
        self.impact_node = center
        self.node_label.setText(f"Impact node: {center} (기본값)")
        self._highlight_node(center)
        self._reset_timeline()
        self._log(f"{label} 로드됨 — 노드를 클릭해 충격점을 바꾸고 Simulate.")

    def _render_original(self, reset_camera: bool = False):
        """원본 메쉬만 렌더 (애니메이션/pick highlight 제거)."""
        self.plotter.clear()
        self.anim_poly = None
        self._pick_actor = None
        poly = to_polydata(self.vertices, self.faces)
        self.plotter.add_mesh(
            poly, color="#9ecae1", show_edges=True, edge_color="#3182bd",
            name="original", pickable=True,
        )
        if reset_camera:
            self.plotter.reset_camera()

    def _enable_picking(self):
        # 표면 위 클릭 지점 → 최근접 정점 인덱스로 환산.
        self.plotter.enable_point_picking(
            callback=self._on_pick, show_message=False, left_clicking=True,
            use_picker=True,
        )

    def _on_pick(self, point, *args):
        if self.vertices is None or point is None:
            return
        point = np.asarray(point, dtype=np.float64).reshape(-1)[:3]
        # 클릭 좌표는 "현재 표시 중인 프레임" 표면 위 → 그 좌표로 최근접 노드를 찾는다
        # (전역 인덱스는 프레임 무관하게 동일 → BE impact_node 일치). 표면 정점만 후보로
        # (체적 메쉬 내부 정점으로 스냅 방지).
        base = self.frames[self.anim_idx] if self.frames is not None else self.vertices
        if self.faces is not None and len(self.faces):
            surf = np.unique(self.faces)
            idx = int(surf[np.argmin(np.linalg.norm(base[surf] - point, axis=1))])
        else:
            idx = int(np.argmin(np.linalg.norm(base - point, axis=1)))
        self.impact_node = idx
        self.node_label.setText(f"Impact node: {idx}  @ {self.vertices[idx].round(3).tolist()}")
        self._highlight_node(idx, center=base[idx])

    def _highlight_node(self, idx: int, center=None):
        if self._pick_actor is not None:
            self.plotter.remove_actor(self._pick_actor)
        diag = float(np.linalg.norm(self.vertices.max(0) - self.vertices.min(0))) or 1.0
        c = self.vertices[idx] if center is None else center
        sphere = pv.Sphere(radius=0.02 * diag, center=c)
        self._pick_actor = self.plotter.add_mesh(sphere, color="#e6550d", name="impact")

    # ---- 시뮬레이션 / 애니메이션 -------------------------------------------
    def on_simulate(self):
        if self.vertices is None or self.orig_bytes is None:
            QtWidgets.QMessageBox.information(self, "No mesh", "Load a mesh first.")
            return
        if self.impact_node is None:
            QtWidgets.QMessageBox.information(self, "No node", "Click a node to set the impact point.")
            return

        action = {
            "impact_node": self.impact_node,
            "force": [self.fx.value(), self.fy.value(), self.fz.value()],
            "scale": self.scale.value(),
        }
        if self.radius.value() > 0:
            action["radius"] = self.radius.value()

        self._stop_play()
        # BE ready면 /simulate, 아니면 로컬 metal_dent. 연결 상태는 startup/Refresh 기준 —
        # GUI 스레드에서 동기 health GET을 돌리지 않아 미연결 시에도 멈추지 않는다.
        if self._be_ready:
            model = self.model_combo.currentText() or None
            self.simulate_btn.setEnabled(False)
            self._log("simulating (BE /simulate)…")
            self._sim_gen += 1
            gen = self._sim_gen
            self.worker = SimulateWorker(
                self.client, self.orig_bytes, self.file_format, action, model
            )
            self.worker.finished_ok.connect(lambda p, g=gen: self._on_frames_ready(g, p))
            self.worker.failed.connect(lambda m, g=gen: self._on_sim_failed(g, m))
            self.worker.start()
        else:
            try:
                frames = metal_dent_simulate(self.vertices, self.faces, action)
            except Exception as e:
                self._log(f"failed: {e}")
                QtWidgets.QMessageBox.critical(self, "Simulate failed", str(e))
                return
            self._setup_animation(self.faces, frames, dummy=True)

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
        frames = np.asarray(frames)
        self.frames = frames
        self.faces = np.asarray(faces)
        self.disp = np.linalg.norm(frames - frames[0], axis=2)  # (T,N)
        self.vmax = float(self.disp.max()) or 1.0

        self.plotter.clear()
        self._pick_actor = None
        self.anim_poly = to_polydata(frames[0], self.faces)
        self.anim_poly["displacement"] = self.disp[0]
        self.plotter.add_mesh(
            self.anim_poly, scalars="displacement", cmap="turbo",
            clim=[0.0, self.vmax], show_edges=False, name="anim",
            scalar_bar_args={"title": "|displacement|"}, reset_camera=False,
        )
        # 카메라는 로드 시 이미 맞춰졌으므로 Simulate마다 리셋하지 않는다(시점 보존).
        self._enable_picking()  # 재픽 가능(현재 표시 프레임 기준)

        T = len(frames)
        self.timeline.setEnabled(T > 1)
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, max(0, T - 1))
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.anim_idx = 0
        self._render_frame(0)

        tag = " (로컬 metal_dent)" if dummy else ""
        self._log(f"simulate 완료{tag} — {T} frames, max|disp|={self.vmax:.4f}")
        if T > 1:
            self._start_play()

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
        self._sim_gen += 1  # 진행 중 worker 결과 무효화
        self.frames = None
        self._render_original(reset_camera=False)
        if self.impact_node is not None:
            self._highlight_node(self.impact_node)
        self._enable_picking()
        self._reset_timeline()
        self._log("reset to original.")

    def _log(self, msg: str):
        self.log_label.setText(msg)

    def closeEvent(self, event):
        # 애니메이션 타이머·시뮬 스레드를 정리하고 VTK 렌더 윈도우를 명시적으로
        # 닫는다 — Windows 종료 중 implicit finalize 시의 access violation/hang 방지.
        self.anim_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()
        self.plotter.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
