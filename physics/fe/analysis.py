"""시뮬레이션 프레임 → 부품별 충격 분석 (순수 numpy — Qt/pyvista 의존 없음).

frames (T,N,3)는 LIVE(BE /simulate)든 DUMMY(로컬 metal_dent)든 FE 메모리에
존재하므로, 분석은 FE에서 일관되게 계산한다 → 두 모드의 수치가 동일하다.

부품 정의는 자산 sidecar JSON(assets/<name>.components.json, prep 스크립트가
bake)에서 온다. 부품별 정점 집합은 vertex_range([start,end)) / vertex_indices /
rule(bbox|sphere) 중 하나로 지정한다. 정의가 없거나 아무 부품도 매칭되지 않는
메쉬(내장 판/캔, 사용자 로드 메쉬)는 "전체 메쉬" 의사 부품 + 자동 임계값(bbox
대각선 5%)으로 강등해 모든 시나리오에서 분석이 동작한다.

지표 (모두 메쉬 단위, 결정적):
  peak_disp(n) = max_t |frames[t,n] - rest[n]|      # 최대 변위
  peak_vel(n)  = max_t |frames[t+1,n] - frames[t,n]| # 프레임 차분 속도 프록시
  score(n)     = 100·(0.7·peak_disp/max + 0.3·peak_vel/max)  # 상대 충격 점수
  status       = FAIL(ratio≥1) | WARN(ratio≥warn_ratio) | OK, ratio=max_disp/임계값

T==1 엣지 케이스: dummy처럼 단일 프레임을 주는 모델은 frames[0]이 이미 변형
상태이므로 rest는 원본 vertices를 쓴다 (frames[0] 기준이면 변위가 전부 0).
이때 속도 프록시는 정의 불가 → 0 + single_frame 플래그.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# 부품 정의가 없을 때 전체 메쉬 의사 부품에 적용하는 자동 임계값 (bbox 대각선 비율).
WHOLE_MESH_THRESHOLD_FRACTION = 0.05
DEFAULT_WARN_RATIO = 0.7

# 충격 점수 가중치 (변위 vs 속도 프록시).
_SCORE_W_DISP = 0.7
_SCORE_W_VEL = 0.3


def load_components(path: str) -> dict | None:
    """자산 sidecar JSON 로드. 없거나 형식이 어긋나면 None (분석은 전체 메쉬로 강등)."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("components"), list):
        return None
    return doc


def resolve_component_indices(comp: dict, vertices: np.ndarray) -> np.ndarray:
    """부품 정의 → 전역 정점 인덱스 (int64, 1D). 매칭 없으면 빈 배열.

    우선순위: vertex_range > vertex_indices > rule. 범위 밖 인덱스는 조용히
    버린다(자산 재-export 후 stale 인덱스 방어) — num_vertices로 드러난다.
    sidecar 값 타입 오류(null/문자열 등)도 크래시 대신 빈 배열로 강등 —
    전 부품이 비면 compute_analysis가 전체 메쉬 의사 부품으로 폴백한다.
    """
    try:
        return _resolve_indices(comp, vertices)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.int64)


def _resolve_indices(comp: dict, vertices: np.ndarray) -> np.ndarray:
    n = len(vertices)
    rng = comp.get("vertex_range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        start, end = max(0, int(rng[0])), min(n, int(rng[1]))
        return np.arange(start, max(start, end), dtype=np.int64)

    idx = comp.get("vertex_indices")
    if isinstance(idx, (list, tuple)):
        arr = np.asarray(idx, dtype=np.int64).ravel()
        return arr[(arr >= 0) & (arr < n)]

    rule = comp.get("rule")
    if isinstance(rule, dict):
        if rule.get("type") == "bbox":
            lo = np.asarray(rule.get("min", []), dtype=np.float64)
            hi = np.asarray(rule.get("max", []), dtype=np.float64)
            if lo.shape == (3,) and hi.shape == (3,):
                mask = np.all((vertices >= lo) & (vertices <= hi), axis=1)
                return np.nonzero(mask)[0].astype(np.int64)
        elif rule.get("type") == "sphere":
            center = np.asarray(rule.get("center", []), dtype=np.float64)
            radius = rule.get("radius")
            if center.shape == (3,) and isinstance(radius, (int, float)) and radius > 0:
                mask = np.linalg.norm(vertices - center, axis=1) <= float(radius)
                return np.nonzero(mask)[0].astype(np.int64)
    return np.empty(0, dtype=np.int64)


@dataclass
class ComponentResult:
    component_id: str
    name: str
    material: str
    fragility: str
    notes: str
    threshold: float
    warn_ratio: float
    num_vertices: int
    max_disp: float
    max_disp_node: int  # 전역 정점 인덱스 (-1 = N/A)
    max_vel: float
    max_score: float
    ratio: float  # max_disp / threshold
    status: str  # "OK" | "WARN" | "FAIL" | "N/A"
    over_indices: np.ndarray = field(repr=False)  # 임계 초과 전역 정점 (오버레이용)


@dataclass
class AnalysisResult:
    scenario: str
    sim_mode: str  # "live" | "dummy"
    action: dict
    num_frames: int
    num_nodes: int
    single_frame: bool
    fallback_whole_mesh: bool
    peak_disp: np.ndarray = field(repr=False)  # (N,) 오버레이용 — 요약 JSON 미포함
    peak_vel: np.ndarray = field(repr=False)  # (N,)
    overall_max_disp: float = 0.0
    overall_max_node: int = 0
    verdict: str = "OK"
    components: list[ComponentResult] = field(default_factory=list)

    def to_summary_json(self) -> dict:
        """per-node 배열을 뺀 압축 요약 — /chat payload 및 chat_fallback 입력."""
        return {
            "scenario": self.scenario,
            "sim_mode": self.sim_mode,
            "action": self.action,
            "num_frames": self.num_frames,
            "num_nodes": self.num_nodes,
            "single_frame": self.single_frame,
            "fallback_whole_mesh": self.fallback_whole_mesh,
            "overall": {
                "max_disp": _round4(self.overall_max_disp),
                "node": self.overall_max_node,
                "verdict": self.verdict,
            },
            "components": [
                {
                    "id": c.component_id,
                    "name": c.name,
                    "material": c.material,
                    "max_disp": _round4(c.max_disp),
                    "threshold": _round4(c.threshold),
                    "ratio": _round4(c.ratio),
                    "status": c.status,
                    # N/A 부품의 sentinel(-1)은 요약에서 None으로 — "노드 #-1" 방지.
                    "max_node": c.max_disp_node if c.max_disp_node >= 0 else None,
                    "score": _round4(c.max_score),
                    "notes": c.notes,
                }
                for c in self.components
            ],
        }


def _round4(value: float) -> float:
    """4 유효자리 반올림 (JSON 크기 억제 + 표시 일관성)."""
    return float(f"{float(value):.4g}")


def _status_for(ratio: float, warn_ratio: float, empty: bool) -> str:
    if empty:
        return "N/A"
    if ratio >= 1.0:
        return "FAIL"
    if ratio >= warn_ratio:
        return "WARN"
    return "OK"


def compute_analysis(
    frames: np.ndarray,
    vertices: np.ndarray,
    action: dict,
    components_def: dict | None,
    sim_mode: str,
) -> AnalysisResult:
    """(T,N,3) 프레임 → 부품별 충격 분석. GUI 스레드에서 돌 만큼 가볍다(수 회의
    (T,N) 벡터 연산 — 로컬 metal_dent 시뮬 자체보다 저렴)."""
    frames = np.asarray(frames, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    T, N = frames.shape[0], frames.shape[1]
    single_frame = T == 1

    # 변위 기준점: 단일 프레임 결과는 frames[0]이 이미 변형 상태 → 원본 대비.
    rest = vertices if (single_frame and vertices.shape == frames.shape[1:]) else frames[0]

    peak_disp = np.linalg.norm(frames - rest[None], axis=2).max(axis=0)  # (N,)
    if single_frame:
        peak_vel = np.zeros(N, dtype=np.float64)
    else:
        peak_vel = np.linalg.norm(np.diff(frames, axis=0), axis=2).max(axis=0)  # (N,)

    max_disp = float(peak_disp.max()) or 1.0
    max_vel = float(peak_vel.max()) or 1.0
    score = 100.0 * (
        _SCORE_W_DISP * peak_disp / max_disp + _SCORE_W_VEL * peak_vel / max_vel
    )  # (N,)

    # --- 부품 정의 해석 (없거나 전부 미매칭 → 전체 메쉬 의사 부품) --------------
    comp_defs = (components_def or {}).get("components") or []
    resolved = [(c, resolve_component_indices(c, rest)) for c in comp_defs]
    fallback_whole_mesh = not any(len(idx) for _, idx in resolved)
    if fallback_whole_mesh:
        diag = float(np.linalg.norm(rest.max(0) - rest.min(0))) or 1.0
        resolved = [(
            {
                "id": "whole_mesh",
                "name": "전체 메쉬",
                "material": "",
                "fragility": "unknown",
                "notes": "부품 정의 없음 — bbox 대각선 5%를 자동 임계값으로 사용.",
                "damage_threshold": WHOLE_MESH_THRESHOLD_FRACTION * diag,
            },
            np.arange(N, dtype=np.int64),
        )]

    components: list[ComponentResult] = []
    for comp, idx in resolved:
        threshold = float(comp.get("damage_threshold", 0.0)) or 1.0
        warn_ratio = float(comp.get("warn_ratio", DEFAULT_WARN_RATIO))
        empty = len(idx) == 0
        if empty:
            c_disp = c_vel = c_score = ratio = 0.0
            c_node = -1
            over = np.empty(0, dtype=np.int64)
        else:
            local = np.argmax(peak_disp[idx])
            c_node = int(idx[local])
            c_disp = float(peak_disp[c_node])
            c_vel = float(peak_vel[idx].max())
            c_score = float(score[idx].max())
            ratio = c_disp / threshold
            over = idx[peak_disp[idx] >= threshold]
        components.append(ComponentResult(
            component_id=str(comp.get("id", "?")),
            name=str(comp.get("name", comp.get("id", "?"))),
            material=str(comp.get("material", "")),
            fragility=str(comp.get("fragility", "")),
            notes=str(comp.get("notes", "")),
            threshold=threshold,
            warn_ratio=warn_ratio,
            num_vertices=int(len(idx)),
            max_disp=c_disp,
            max_disp_node=c_node,
            max_vel=c_vel,
            max_score=c_score,
            ratio=ratio,
            status=_status_for(ratio, warn_ratio, empty),
            over_indices=over,
        ))

    statuses = [c.status for c in components]
    verdict = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "OK"
    scenario = str((components_def or {}).get("display_name", "")) or "(사용자 메쉬)"

    return AnalysisResult(
        scenario=scenario,
        sim_mode=sim_mode,
        action=dict(action),
        num_frames=int(T),
        num_nodes=int(N),
        single_frame=single_frame,
        fallback_whole_mesh=fallback_whole_mesh,
        peak_disp=peak_disp,
        peak_vel=peak_vel,
        overall_max_disp=float(peak_disp.max()),
        overall_max_node=int(np.argmax(peak_disp)),
        verdict=verdict,
        components=components,
    )
