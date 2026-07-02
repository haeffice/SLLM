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

부품별 추가 분석 (extras — 자산 functional 메타 주도, 상수 출처는 prep_hubble.py):
  rigid      Kabsch 강체 분해 — 회전각·잔차 RMS(영구 뒤틀림)
  functional type별 기능 환산: antenna(지향 12(θ/θ₃dB)²dB + Ruze 685.8(ε/λ)²dB),
             solar_panel(cosθ × 손상 면적 → 잔여 발전량 W),
             optical_tube(광축 arcsec vs 예산 + Maréchal 스트렐 비)
  settling   진동 정착 — ±2% 밴드 정착 시점(구간 %), 로그 감쇠율→감쇠비 ζ,
             지배 진동수(사이클/구간), 잔류 진폭 비 (T>1일 때)
  strain     edge 신장률 — 피크/잔류 vs 항복 변형률 → 탄성/항복 근접/소성 변형

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

# 기본 항복 변형률 (Al 6061-T6: σ_y/E ≈ 0.40%) — 부품 yield_strain으로 덮어씀.
DEFAULT_YIELD_STRAIN = 0.004
# 진동 정착 밴드 (제어공학 표준 ±2%).
_SETTLE_BAND = 0.02


# ---- 강체 분해 / 기하 헬퍼 ---------------------------------------------------

def _kabsch_rot(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """중심화된 (M,3) 쌍의 최적 강체 회전 (Kabsch). x1 ≈ x0 @ R.T.

    H = X0ᵀX1, SVD H=UΣVᵀ, R = V·diag(1,1,det(VUᵀ))·Uᵀ (반사 보정).
    """
    u, _, vt = np.linalg.svd(x0.T @ x1)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _kabsch(rest: np.ndarray, deformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """최적 강체 회전 적합. (M,3) 쌍 → (R (3,3), 잔차 (M,)).

    잔차 = 강체 운동(회전+병진)을 제거하고 남는 순수 변형 크기.
    """
    x0, x1 = rest - rest.mean(0), deformed - deformed.mean(0)
    rot = _kabsch_rot(x0, x1)
    residuals = np.linalg.norm(x1 - x0 @ rot.T, axis=1)
    return rot, residuals


def remove_global_rigid(frames: np.ndarray, rest: np.ndarray) -> tuple[np.ndarray, dict]:
    """프레임별 전역 최적 강체(회전+병진)를 제거해 rest 좌표계로 정렬한 프레임.

    자유 낙하처럼 전역 강체 운동(낙하·바운스·회전)이 지배하는 궤적에서, 변위/속도
    지표가 이동이 아니라 "변형"만 재도록 한다. aligned[t] = (frames[t]−c_t)@R_t + c0
    (R_t는 rest→frames[t] 회전; 역사상은 @R_t). 순수 강체 궤적이면 aligned[t] ≡ rest.

    Returns:
        (aligned (T,N,3), global_motion) — global_motion은 마지막 프레임의 전역
        회전각(도)·병진 크기 {"available", "rotation_deg", "translation"}.
        T==1 또는 N<3이면 정렬 없이 원본과 {"available": False}를 돌려준다.
    """
    frames = np.asarray(frames, dtype=np.float64)
    rest = np.asarray(rest, dtype=np.float64)
    T, N = frames.shape[0], frames.shape[1]
    if T <= 1 or N < 3:
        return frames, {"available": False}

    c0 = rest.mean(0)
    x0 = rest - c0
    aligned = np.empty_like(frames)
    rot_last = np.eye(3)
    for t in range(T):
        c_t = frames[t].mean(0)
        rot = _kabsch_rot(x0, frames[t] - c_t)
        aligned[t] = (frames[t] - c_t) @ rot + c0
        if t == T - 1:
            rot_last = rot
    translation = float(np.linalg.norm(frames[-1].mean(0) - c0))
    return aligned, {
        "available": True,
        "rotation_deg": _rotation_angle_deg(rot_last),
        "translation": translation,
    }


def _rotation_angle_deg(rot: np.ndarray) -> float:
    """회전 행렬 → 회전각(도)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))


def _principal_axes(pts: np.ndarray) -> np.ndarray:
    """정점 집합의 주축. 반환 (3,3) — 열 0=최대 분산(장축), 열 2=최소 분산(법선)."""
    centered = pts - pts.mean(0)
    _, vecs = np.linalg.eigh(centered.T @ centered)  # 고유값 오름차순
    return vecs[:, ::-1]


def _axis_tilt_deg(axis: np.ndarray, rot: np.ndarray) -> float:
    """축 벡터가 강체 회전으로 기울어진 각도(도). 방향 부호는 무시."""
    tilted = rot @ axis
    return float(np.degrees(np.arccos(np.clip(abs(float(axis @ tilted)), 0.0, 1.0))))


def _component_faces(faces: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    """세 정점이 모두 부품에 속한 face만 추출. faces 없으면 None."""
    if faces is None or len(faces) == 0:
        return None
    f = np.asarray(faces, dtype=np.int64)
    inside = f[np.all(mask[f], axis=1)]
    return inside if len(inside) else None


# ---- 추가 분석: 기능 영향 / 진동 정착 / 변형률 --------------------------------

def _functional_metrics(functional: dict, idx, rest, settled, rot, residuals,
                        faces_in, peak_disp, threshold, scale_m) -> dict:
    """부품 functional.type별 기능 영향 환산 (상수 출처: assets/prep_hubble.py 주석).

    - antenna: 지향 손실 12(θ/θ₃dB)² dB (링크버짓 표준 근사) + Ruze 표면오차 손실
      685.8(ε/λ)² dB. 잔차 크기 RMS를 표면 오차로 근사(보수적 상한).
    - solar_panel: 법선 기울기 cosθ 손실 × 임계 초과 면적(셀 손상) 비율.
    - optical_tube: 광축 기울기(arcsec) vs 지향 예산 + 파면 오차 → 스트렐 비
      S=exp[−(2πσ/λ)²] (Maréchal 근사; S≥0.8=회절한계).
    """
    ftype = functional.get("type")
    axes = _principal_axes(rest[idx])
    out: dict = {"type": ftype}

    if ftype == "antenna":
        theta = _axis_tilt_deg(axes[:, 2], rot)  # 접시 법선(최소 분산 축) 이탈
        beam = float(functional.get("beam_deg", 4.0))
        wavelength = float(functional.get("wavelength_m", 0.131))
        loss_point = 12.0 * (theta / beam) ** 2
        eps_m = float(np.sqrt(np.mean(residuals**2))) * scale_m
        loss_ruze = 685.8 * (eps_m / wavelength) ** 2
        total = loss_point + loss_ruze
        out.update({
            "pointing_deg": theta, "beam_deg": beam,
            "loss_point_db": loss_point, "loss_ruze_db": loss_ruze,
            "surface_err_mm": eps_m * 1e3, "loss_total_db": total,
            "verdict": "OK" if total < 1.0 else "WARN" if total <= 3.0 else "FAIL",
        })

    elif ftype == "solar_panel":
        tilt = _axis_tilt_deg(axes[:, 2], rot)  # 패널 법선 기울기
        cos_factor = float(np.cos(np.radians(tilt)))
        frac_dead = _dead_area_frac(rest, faces_in, peak_disp, threshold)
        p0 = float(functional.get("p0_w", 5000.0))
        power_frac = max(0.0, cos_factor * (1.0 - frac_dead))
        out.update({
            "tilt_deg": tilt, "cos_factor": cos_factor, "dead_area_frac": frac_dead,
            "power_frac": power_frac, "power_lost_w": p0 * (1.0 - power_frac), "p0_w": p0,
            "verdict": "OK" if power_frac > 0.9 else "WARN" if power_frac >= 0.6 else "FAIL",
        })

    elif ftype == "screen":
        # 표시/터치 불능 면적 = 임계 초과 정점을 포함한 셀 면적 비율.
        frac = _dead_area_frac(rest, faces_in, peak_disp, threshold)
        out.update({
            "dead_area_frac": frac,
            "verdict": "OK" if frac < 0.05 else "WARN" if frac < 0.20 else "FAIL",
        })

    elif ftype == "optical_tube":
        tilt_arcsec = _axis_tilt_deg(axes[:, 0], rot) * 3600.0  # 광축=장축
        budget = float(functional.get("pointing_budget_arcsec", 0.007))
        # 스트렐: 축좌표 앞/뒤 15% 링 분리 → 디포커스 + 주경부(뒤쪽) 잔류 변형.
        axis = axes[:, 0]
        z = (rest[idx] - rest[idx].mean(0)) @ axis
        lo, hi = np.quantile(z, 0.15), np.quantile(z, 0.85)
        disp_axis = (settled[idx] - rest[idx]) @ axis
        rear, front = z <= lo, z >= hi
        dz_m = abs(float(disp_axis[front].mean() - disp_axis[rear].mean())) * scale_m
        f_number = float(functional.get("f_number", 24.0))
        sigma_defocus = dz_m / (8.0 * f_number**2 * 2.0 * np.sqrt(3.0))
        sigma_mirror = 2.0 * float(np.sqrt(np.mean(residuals[rear] ** 2))) * scale_m
        sigma_wfe = float(np.hypot(sigma_mirror, sigma_defocus))
        if sigma_wfe < 1e-8:  # 10 nm 미만 = 수치 노이즈 — 광학 영향 없음으로 처리
            sigma_wfe = 0.0
        wavelength = float(functional.get("obs_wavelength_nm", 550.0)) * 1e-9
        strehl = float(np.exp(-((2.0 * np.pi * sigma_wfe / wavelength) ** 2)))
        out.update({
            "axis_tilt_arcsec": tilt_arcsec, "budget_arcsec": budget,
            "budget_ratio": tilt_arcsec / budget if budget > 0 else 0.0,
            "wfe_um": sigma_wfe * 1e6, "strehl": strehl,
            "verdict": "OK" if strehl >= 0.8 else "WARN" if strehl >= 0.3 else "FAIL",
        })
    return out


def _dead_area_frac(rest: np.ndarray, faces_in: np.ndarray | None,
                    peak_disp: np.ndarray, threshold: float) -> float:
    """임계 초과 정점을 포함한 face의 면적 비율 (solar_panel/screen 공용)."""
    if faces_in is None:
        return 0.0
    v0, v1, v2 = rest[faces_in[:, 0]], rest[faces_in[:, 1]], rest[faces_in[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = float(areas.sum())
    if total_area <= 0:
        return 0.0
    dead = np.any(peak_disp[faces_in] >= threshold, axis=1)
    return float(areas[dead].sum() / total_area)


def _settling_metrics(comp_frames: np.ndarray) -> dict:
    """부품 진동 정착 지표 — 시간축은 정규화 구간(0..1) 기준으로 정직하게 표기.

    s(t)=정착 상태 대비 평균 잔차. 정착 시점=±2% 밴드에 최종 진입한 프레임.
    감쇠비 ζ=δ/√(4π²+δ²) (로그 감쇠율 δ, 연속 양(+) 피크 필요 — 부족 시 None).
    """
    T = len(comp_frames)
    if T < 2:
        return {"available": False}
    settled = comp_frames[-1]
    s = np.linalg.norm(comp_frames - settled[None], axis=2).mean(axis=1)  # (T,)
    peak = float(s.max())
    if peak <= 1e-9:  # 순수 강체 궤적의 정렬 잔여 노이즈로 통계를 만들지 않는다
        return {"available": False}

    above = np.nonzero(s > _SETTLE_BAND * peak)[0]
    settle_frac = float((above[-1] + 1) / (T - 1)) if len(above) else 0.0
    settle_frac = min(settle_frac, 1.0)

    # 부호 있는 진동: 부품 평균은 링잉 위상이 공간적으로 상쇄되므로, 진폭이 가장
    # 큰 정점(프로브)의 궤적을 그 주방향에 투영해 감쇠 피크를 잡는다.
    amp = np.linalg.norm(comp_frames - settled[None], axis=2)  # (T,M)
    probe = int(np.argmax(amp.max(axis=0)))
    disp = comp_frames[:, probe, :] - settled[probe]  # (T,3)
    r = disp @ _principal_axes(disp)[:, 0]

    # 단조 정착 추세가 링잉보다 크면 피크가 묻힌다 — 지배 주기 길이의 이동평균으로
    # 추세를 제거한 진동 성분에서 감쇠율을 잰다 (순수 사인엔 영향 없음).
    spec0 = np.abs(np.fft.rfft(r - r.mean()))
    cycles0 = int(np.argmax(spec0[1:]) + 1) if len(spec0) > 1 else 1
    win = min(T, max(3, round(T / max(cycles0, 1))))
    r_osc = r - np.convolve(r, np.ones(win) / win, mode="same")
    spec = np.abs(np.fft.rfft(r_osc - r_osc.mean()))
    cycles = int(np.argmax(spec[1:]) + 1) if len(spec) > 1 else 0  # 구간당 사이클 수

    diff = np.diff(r_osc)
    ext = np.nonzero(diff[:-1] * diff[1:] < 0)[0] + 1  # 극값 인덱스
    pos_peaks = [float(r_osc[i]) for i in ext if r_osc[i] > 0]
    # 양(+) 피크 2개 미만 = 정착이 진동을 지배(오버슈트 없음) → 과감쇠 거동으로
    # 보고하고 ζ/사이클은 생략 (metal_dent처럼 단조 정착하는 궤적의 정직한 답).
    oscillatory = len(pos_peaks) >= 2
    zeta = None
    if oscillatory:
        ratios = [pos_peaks[i] / pos_peaks[i + 1] for i in range(len(pos_peaks) - 1)
                  if pos_peaks[i + 1] > 0]
        if ratios:
            delta = float(np.mean(np.log(ratios)))
            if delta > 0:
                zeta = float(delta / np.sqrt(4.0 * np.pi**2 + delta**2))

    tail = np.abs(r[int(0.9 * T):])
    r_max = float(np.abs(r).max())
    residual_ratio = float(tail.max() / r_max) if (len(tail) and r_max > 0) else 0.0
    return {
        "available": True, "settle_frac": settle_frac, "oscillatory": oscillatory,
        "zeta": zeta, "cycles": cycles if oscillatory else None,
        "residual_ratio": residual_ratio,
    }


def _strain_metrics(faces_in: np.ndarray | None, rest: np.ndarray,
                    frames: np.ndarray, yield_strain: float) -> dict:
    """edge 신장률 ε=ΔL/L0 — 크기와 무관한 재료 손상 지표.

    잔류(settled) ε ≥ 항복 변형률이면 소성 변형(영구 손상), 피크만 넘었으면
    항복 근접 후 복귀, 둘 다 아니면 탄성.
    """
    if faces_in is None:
        return {"available": False}
    edges = np.unique(np.sort(np.concatenate(
        [faces_in[:, [0, 1]], faces_in[:, [1, 2]], faces_in[:, [2, 0]]]), axis=1), axis=0)
    length0 = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    valid = length0 > 0
    edges, length0 = edges[valid], length0[valid]
    if len(edges) == 0:
        return {"available": False}
    # 프레임별 순회로 O(E) 메모리 유지 — (T,E,3)를 통째로 실체화하면 대형 사용자
    # 메쉬(전체 메쉬 의사 부품, E≈1.5×face 수)에서 수 GB로 폭증한다.
    peak = residual = 0.0
    for t in range(len(frames)):
        delta = frames[t, edges[:, 0], :] - frames[t, edges[:, 1], :]
        eps_t = np.abs(np.linalg.norm(delta, axis=1) - length0) / length0
        eps_max = float(eps_t.max())
        peak = max(peak, eps_max)
        if t == len(frames) - 1:
            residual = eps_max
    verdict = ("소성 변형" if residual >= yield_strain
               else "항복 근접" if peak >= yield_strain else "탄성")
    return {
        "available": True, "peak": peak, "residual": residual,
        "yield_strain": yield_strain, "verdict": verdict,
    }


def _compute_extras(comp, idx, rest, settled, frames, faces, peak_disp,
                    threshold, scale_m, single_frame) -> dict:
    """부품별 추가 분석 묶음: 강체 분해 + 기능 영향 + 진동 정착 + 변형률."""
    extras: dict = {}
    if len(idx) < 3:
        return extras
    mask = np.zeros(len(rest), dtype=bool)
    mask[idx] = True
    faces_in = _component_faces(faces, mask)

    rot, residuals = _kabsch(rest[idx], settled[idx])
    extras["rigid"] = {
        "rotation_deg": _rotation_angle_deg(rot),
        "residual_rms": float(np.sqrt(np.mean(residuals**2))),
        "residual_rms_mm": float(np.sqrt(np.mean(residuals**2))) * scale_m * 1e3,
    }
    functional = comp.get("functional")
    if isinstance(functional, dict) and functional.get("type"):
        try:
            extras["functional"] = _functional_metrics(
                functional, idx, rest, settled, rot, residuals,
                faces_in, peak_disp, threshold, scale_m,
            )
        except (ValueError, TypeError, IndexError, ZeroDivisionError):
            pass  # sidecar 상수 이상(0 분모 포함) — 기능 환산만 생략 (구조 분석은 유지)
    if not single_frame:
        extras["settling"] = _settling_metrics(frames[:, idx, :])
    yield_strain = comp.get("yield_strain")
    try:
        yield_strain = float(yield_strain) if yield_strain else DEFAULT_YIELD_STRAIN
    except (TypeError, ValueError):
        yield_strain = DEFAULT_YIELD_STRAIN
    extras["strain"] = _strain_metrics(faces_in, rest, frames, yield_strain)
    return extras


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
    extras: dict = field(default_factory=dict, repr=False)  # 강체/기능/정착/변형률


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
    # 전역 강체 운동(낙하 후 자세 변화 등) — 변형 지표와 분리해 별도 보고.
    global_motion: dict = field(default_factory=dict)

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
            "global_motion": _round_tree(self.global_motion),
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
                    "extras": _round_tree(c.extras),
                }
                for c in self.components
            ],
        }


def _round4(value: float) -> float:
    """4 유효자리 반올림 (JSON 크기 억제 + 표시 일관성)."""
    return float(f"{float(value):.4g}")


def _round_tree(obj):
    """중첩 dict/list의 float를 4 유효자리로 반올림 — extras 요약용."""
    if isinstance(obj, dict):
        return {k: _round_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_tree(v) for v in obj]
    if isinstance(obj, float):
        return _round4(obj)
    return obj


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
    faces: np.ndarray | None,
    action: dict,
    components_def: dict | None,
    sim_mode: str,
    remove_rigid: bool = False,
    rigid_removed: tuple[np.ndarray, dict] | None = None,
) -> AnalysisResult:
    """(T,N,3) 프레임 → 부품별 충격 분석. GUI 스레드에서 돌 만큼 가볍다
    (벡터 연산 + 프레임/부품당 소형 SVD — 로컬 시뮬 자체보다 저렴).

    remove_rigid=True(자유 낙하)이면 모든 변위/속도/정착 지표를 **전역 강체 제거
    후의 변형장** 기준으로 잰다 (remove_global_rigid) — 낙하·바운스 이동이 변위로
    새지 않게. 충격 모드(remove_rigid=False, 기본)는 물체가 고정돼 있으므로 원시
    프레임을 그대로 쓴다 — 국소 대변형이 강체로 흡수되는 것을 막는다. 낙하 모드는
    호출자가 계산한 (aligned, global_motion)을 rigid_removed로 넘겨 재계산을 아낀다
    (app.py 히트맵과 공유).

    faces는 변형률(edge 신장률)·손상 면적 계산에 쓰인다 — None이면 해당
    지표만 생략(available=False)."""
    frames = np.asarray(frames, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    T, N = frames.shape[0], frames.shape[1]
    single_frame = T == 1

    # 변위 기준점: 단일 프레임 결과는 frames[0]이 이미 변형 상태 → 원본 대비.
    rest = vertices if (single_frame and vertices.shape == frames.shape[1:]) else frames[0]

    if not remove_rigid:
        frames_a, global_motion = frames, {"available": False}
    elif rigid_removed is not None and rigid_removed[0].shape == frames.shape:
        frames_a, global_motion = np.asarray(rigid_removed[0], dtype=np.float64), rigid_removed[1]
    else:
        frames_a, global_motion = remove_global_rigid(frames, rest)

    peak_disp = np.linalg.norm(frames_a - rest[None], axis=2).max(axis=0)  # (N,)
    if single_frame:
        peak_vel = np.zeros(N, dtype=np.float64)
    else:
        # 변형 속도 프록시 — 낙하/바운스 강체 속도가 아니라 변형장의 변화율.
        peak_vel = np.linalg.norm(np.diff(frames_a, axis=0), axis=2).max(axis=0)  # (N,)

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

    settled = frames_a[-1]  # 정렬된 정착 상태 — 부품 Kabsch가 국소 변형만 재도록
    try:
        scale_m = float((components_def or {}).get("real_scale_m_per_unit") or 1.0)
    except (TypeError, ValueError):
        scale_m = 1.0

    components: list[ComponentResult] = []
    for comp, idx in resolved:
        threshold = float(comp.get("damage_threshold", 0.0)) or 1.0
        warn_ratio = float(comp.get("warn_ratio", DEFAULT_WARN_RATIO))
        empty = len(idx) == 0
        if empty:
            c_disp = c_vel = c_score = ratio = 0.0
            c_node = -1
            over = np.empty(0, dtype=np.int64)
            extras: dict = {}
        else:
            local = np.argmax(peak_disp[idx])
            c_node = int(idx[local])
            c_disp = float(peak_disp[c_node])
            c_vel = float(peak_vel[idx].max())
            c_score = float(score[idx].max())
            ratio = c_disp / threshold
            over = idx[peak_disp[idx] >= threshold]
            extras = _compute_extras(
                comp, idx, rest, settled, frames_a, faces, peak_disp,
                threshold, scale_m, single_frame,
            )
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
            extras=extras,
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
        global_motion=global_motion,
    )
