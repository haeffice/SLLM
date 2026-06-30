"""Physics Impact Simulator — Windows 데스크톱 FE (PySide6 + PyVista).

흐름:
  1. 시작 시 내장 더미 메쉬가 로드된다 (모델 연결 없이 인터랙션을 확인하기 위함).
     "Load mesh…"로 로컬 파일(.vtk/.obj/.stl/.ply/.off)을 열 수도 있다.
  2. 3D 뷰포트에서 노드를 클릭해 impact_node를 고른다.
  3. force(X/Y/Z) · radius · scale을 입력하고 Simulate.
  4. 모델 연결 시 → BE /predict가 돌려준 변형 메쉬(solid)를 표시.
     모델 미연결 시 → 로컬 더미 반응(선형 감쇠 변형)을 계산해 표시 (UI에 DUMMY 모드 명시).

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

MESH_FILTER = "Mesh (*.vtk *.obj *.stl *.ply *.off);;All files (*)"

# 내장 더미 메쉬 (BE/모델 없이 FE 인터랙션 확인용) + 로컬 변형 파라미터.
DUMMY_GRID_N = 21
DUMMY_GRID_SIZE = 2.0
DUMMY_RADIUS_FRACTION = 0.3  # BE models/dummy/model.py와 동일


def make_dummy_mesh(n: int = DUMMY_GRID_N, size: float = DUMMY_GRID_SIZE):
    """중앙 정렬 n×n 그리드 시트(z=0) → (vertices (N,3), faces (M,3)).

    BE smoke_test.make_grid와 동일한 삼각화. 아래로 누르면 dent가 또렷이 보인다.
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


def local_dummy_deform(vertices: np.ndarray, faces: np.ndarray, action: dict) -> np.ndarray:
    """BE models/dummy/model.py의 선형 감쇠 변형을 미러링한 오프라인 반응.

    모델 미연결 시 동일 수식으로 dent를 만들어 인터랙션을 검증할 수 있게 한다
    (impact_node에서 force, radius 밖은 0으로 선형 감쇠).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    num_nodes = vertices.shape[0]
    impact_node = max(0, min(int(action.get("impact_node", 0)), num_nodes - 1))
    force = np.asarray(action.get("force", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = float(action.get("scale", 1.0))

    radius = action.get("radius")
    if radius is None:
        diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
        radius = DUMMY_RADIUS_FRACTION * diag if diag > 0 else 1.0
    radius = float(radius) or 1.0

    origin = vertices[impact_node]
    dist = np.linalg.norm(vertices - origin, axis=1)
    weight = np.clip(1.0 - dist / radius, 0.0, 1.0)
    return vertices + scale * weight[:, None] * force[None, :]


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """meshio로 (vertices (N,3), faces (M,3), file_format) 추출.

    BE utils.mesh_handler와 동일 규칙(triangle 그대로, quad는 2분할)이라
    정점 순서·인덱스가 BE와 일치한다.
    """
    mesh = meshio.read(path)
    verts = np.asarray(mesh.points, dtype=np.float64)
    if verts.shape[1] == 2:
        verts = np.hstack([verts, np.zeros((verts.shape[0], 1))])

    tris: list[np.ndarray] = []
    for cell in mesh.cells:
        data = np.asarray(cell.data, dtype=np.int64)
        if cell.type == "triangle":
            tris.append(data)
        elif cell.type == "quad":
            tris.append(data[:, [0, 1, 2]])
            tris.append(data[:, [0, 2, 3]])
    if not tris:
        raise ValueError("no surface faces (triangle/quad) in mesh")
    faces = np.concatenate(tris, axis=0)

    file_format = os.path.splitext(path)[1].lstrip(".").lower() or "vtk"
    return verts, faces, file_format


def to_polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """(N,3)+(M,3) → pyvista PolyData (VTK face 형식 [3,i,j,k,...])."""
    faces_pv = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces]
    ).ravel()
    return pv.PolyData(vertices, faces_pv)


class PredictWorker(QtCore.QThread):
    """blocking predict를 GUI 스레드 밖에서 실행 (뷰포트 멈춤 방지)."""

    finished_ok = QtCore.Signal(bytes)
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
            out = self.client.predict(
                self.mesh_bytes, self.file_format, self.action, self.model
            )
            self.finished_ok.emit(out)
        except Exception as e:  # 네트워크/서버 오류 모두 메시지로 환원
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Physics Impact Simulator v{APP_VERSION}")
        self.resize(1280, 760)

        self.client = PhysicsClient()
        self.worker: PredictWorker | None = None

        # 현재 로드된 메쉬 상태
        self.orig_bytes: bytes | None = None
        self.file_format = "vtk"
        self.vertices: np.ndarray | None = None
        self.faces: np.ndarray | None = None
        self.impact_node: int | None = None
        self._be_ready = False  # BE 모델 준비 완료 → Simulate가 BE로 감
        self._pick_actor = None
        self._deformed_actor = None

        self._build_ui()
        self.refresh_health()
        # 시작 시 내장 더미 메쉬를 띄워 BE 없이도 인터랙션을 확인할 수 있게 한다.
        self._load_dummy_mesh()

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
        self.fz = self._spin(-1.0)
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

        self.impact_node = None
        self.node_label.setText("Impact node: — (click a node)")
        self.mesh_label.setText(
            f"{os.path.basename(path)}  —  {len(self.vertices)} nodes, "
            f"{len(self.faces)} faces  [{self.file_format}]"
        )
        self._render_original(reset_camera=True)
        self._enable_picking()
        self._log("mesh loaded — click a node to set impact point.")

    def _load_dummy_mesh(self):
        """내장 더미 메쉬를 로드 — BE 없이도 인터랙션을 확인할 수 있게 한다.

        중앙 노드를 기본 impact_node로 미리 선택해 바로 Simulate가 가능하다.
        orig_bytes도 직렬화해 두므로, BE 연결 시 동일 경로로 전송된다.
        """
        self.vertices, self.faces = make_dummy_mesh()
        self.file_format = "vtk"
        self.orig_bytes = dump_mesh(self.vertices, self.faces, self.file_format)

        self.mesh_label.setText(
            f"내장 더미 메쉬 ({DUMMY_GRID_N}×{DUMMY_GRID_N} grid) — "
            f"{len(self.vertices)} nodes, {len(self.faces)} faces"
        )
        self._render_original(reset_camera=True)
        self._enable_picking()
        center = (DUMMY_GRID_N // 2) * DUMMY_GRID_N + (DUMMY_GRID_N // 2)
        self.impact_node = center
        self.node_label.setText(f"Impact node: {center} (dummy 기본값)")
        self._highlight_node(center)
        self._log("내장 더미 메쉬 로드됨 — 노드를 클릭해 충격점을 바꾸고 Simulate.")

    def _render_original(self, reset_camera: bool = False):
        """원본 메쉬만 렌더 (deformed/pick highlight 제거)."""
        self.plotter.clear()
        self._deformed_actor = None
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

    # ---- 시뮬레이션 ---------------------------------------------------------
    def on_simulate(self):
        if self.orig_bytes is None:
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

        # 최신 연결 상태를 반영한 뒤 분기 — ready면 BE, 아니면 로컬 더미 반응.
        self.refresh_health()
        if self._be_ready:
            model = self.model_combo.currentText() or None
            self.simulate_btn.setEnabled(False)
            self._log("simulating (BE)…")
            self.worker = PredictWorker(
                self.client, self.orig_bytes, self.file_format, action, model
            )
            self.worker.finished_ok.connect(self._on_predict_ok)
            self.worker.failed.connect(self._on_predict_failed)
            self.worker.start()
        else:
            # 모델 미연결 → 로컬에서 동일 수식으로 변형을 계산해 표시 (DUMMY).
            new_verts = local_dummy_deform(self.vertices, self.faces, action)
            self._show_deformed(new_verts, self.faces, dummy=True)

    def _on_predict_ok(self, result_bytes: bytes):
        self.simulate_btn.setEnabled(True)
        try:
            verts_def, faces_def, _ = self._parse_result(result_bytes)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Bad result", str(e))
            return
        self._show_deformed(verts_def, faces_def, dummy=False)

    def _show_deformed(self, verts_def: np.ndarray, faces_def: np.ndarray, dummy: bool):
        """원본(wireframe)+deformed(solid) 오버레이. BE/로컬 두 경로 공통.

        deformed는 picking 대상에서 제외해 클릭이 항상 원본 정점(self.vertices)
        인덱스로 환산되게 하고, clear()로 사라진 picking을 다시 켠다.
        """
        self._render_original()  # _deformed_actor=None으로 리셋 + 원본 재렌더
        self._deformed_actor = self.plotter.add_mesh(
            to_polydata(verts_def, faces_def),
            color="#fc9272", show_edges=True, edge_color="#de2d26",
            opacity=0.9, name="deformed", pickable=False,
        )
        self._enable_picking()
        tag = " (로컬 더미 반응)" if dummy else ""
        # 정점 수가 보존된 경우에만 변위를 계산 (포맷에 따라 dedup될 가능성 방어).
        if self.vertices is not None and verts_def.shape == self.vertices.shape:
            max_disp = float(np.abs(verts_def - self.vertices).max())
            self._log(f"done{tag} — max|displacement| = {max_disp:.4f}")
        else:
            self._log(f"done{tag} — deformed mesh rendered")

    def _on_predict_failed(self, msg: str):
        self.simulate_btn.setEnabled(True)
        self._log(f"failed: {msg}")
        QtWidgets.QMessageBox.critical(self, "Simulate failed", msg)

    def _parse_result(self, result_bytes: bytes):
        """결과 바이트를 file_format으로 임시 파싱 (load_mesh 재사용 위해 임시파일)."""
        import tempfile

        fd, path = tempfile.mkstemp(suffix=f".{self.file_format}")
        try:
            with os.fdopen(fd, "wb") as f:  # 전체 write + 확실한 close (Windows 안전)
                f.write(result_bytes)
            return load_mesh(path)
        finally:
            os.remove(path)

    def on_reset(self):
        if self.vertices is None:
            return
        self._render_original(reset_camera=False)
        if self.impact_node is not None:
            self._highlight_node(self.impact_node)
        self._enable_picking()
        self._log("reset to original.")

    def _log(self, msg: str):
        self.log_label.setText(msg)

    def closeEvent(self, event):
        # 진행 중인 predict 스레드를 정리하고 VTK 렌더 윈도우를 명시적으로
        # 닫는다 — Windows에서 인터프리터 종료 중 implicit finalize 시 발생하는
        # access violation / hang을 방지 (pyvistaqt 권장 lifecycle).
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
