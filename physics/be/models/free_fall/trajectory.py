"""자유 낙하(drop test) 궤적 — 순수 함수 (BE 모델과 FE 오프라인 미러가 공유).

이 파일은 BE(be/models/free_fall/trajectory.py)와 FE(fe/free_fall_sim.py)에
바이트 단위로 동일하게 존재한다 — chat_fallback 미러와 같은 규약으로, 한쪽만
고치면 안 된다(LIVE/오프라인 낙하 거동이 달라짐). 수정 시 두 파일을 함께 갱신할 것.

물체를 낙하 높이 h에서 바닥(z=0 평면)으로 떨어뜨린다 (정규화 t∈[0,1], 단일 중력):
  1) 강체 낙하: 포물선, 첫 접촉 t_c1 = min(0.45, 0.85/denom)
  2) 감쇠 바운스: k번째 높이 h·e^{2k}, 지속 2·t_c1·e^k (같은 중력에서 물리적으로
     일관), h·e^{2k} < 0.02·h 이면 생략, 최대 2회 — 마지막 접촉은 t ≤ 0.85
  3) 다중 접촉: 낙하 자세 기준 최저 밴드(dz ≤ 0.05·z-extent — bbox 대각선이 아니라
     z-extent 기준: 납작한 자세에서도 밴드가 과대해지지 않는다) 정점을 xy 격자
     (0.12·bbox 대각선)로 클러스터 → 가중치 상위 최대 6개 접촉 사이트.
     dent 깊이는 사이트 지분(w̄_j)으로 분배 — 모서리 낙하는 한 곳 깊게,
     면 낙하는 여러 곳 얕게 (충격 에너지 분배).
  4) 사이트별 응답 = metal_dent식 가우시안 env × (rise + 감쇠 링잉), +z 방향
     (바닥이 몸체를 밀어올리는 압축). 접촉 이벤트 k의 지분 s_k = e^k/Σe^m.
     rise는 이벤트별 (1−E_k) 정규화 → frames[-1]에서 정확히 정착 깊이.

결정적(동일 입력=동일 출력). frames[0] = 공중(변형 0), frames[-1] = 바닥 정착+영구 dent.
알려진 근사: 정착 시 dent 부위가 z=0보다 dent 깊이만큼 떠 보일 수 있으나
(몸체가 그만큼 다시 가라앉지 않음) 데모 스케일(≤0.08 vs diag 2)에서는 보이지 않는다.

action 키:
  drop_height  낙하 높이 (메쉬 단위, > 0, 기본 1.0) — 최저 정점의 초기 높이
  restitution  반발계수 e ∈ [0, 1) (기본 0.3)
  orientation  낙하 자세 [rx, ry, rz]° — 메쉬 중심 기준 회전 (기본 [0,0,0]).
               내부 복사본에 적용 → 원본 바이트/정점 인덱스 순서 불변.
  scale        변형 배율 (기본 1.0)
  frames       프레임 수 (기본 90, 최대 240)
  radius       사이트 dent 반경 σ 명시값 (기본 0.10·bbox 대각선)
"""

from __future__ import annotations

import numpy as np

DEFAULT_FRAMES = 90
MAX_FRAMES = 240
_FALL_FRAC_MAX = 0.45  # 첫 낙하가 타임라인에서 차지하는 최대 비율
_CONTACT_END = 0.85  # 마지막 접촉 시점 상한 — 이후는 링다운/정착 구간
_MAX_BOUNCES = 2
_MIN_BOUNCE_H_FRAC = 0.02  # h·e^{2k}가 이 비율보다 작은 바운스는 생략
_BAND_FRACTION = 0.05  # 접촉 밴드: dz ≤ 이 비율 × z-extent
_BUCKET_FRACTION = 0.12  # xy 클러스터 격자 = 이 비율 × bbox 대각선
_MAX_SITES = 6  # 접촉 사이트 상한
_SITE_RADIUS_FRACTION = 0.10  # dent σ 기본 = 이 비율 × bbox 대각선
_DEPTH_COEFF = 0.12  # 깊이 보정 — 화면면 낙하(h=1)에서 액정 파손이 뚜렷하도록
_TAU_RISE = 0.06  # dent 형성 시상수 (충격은 metal_dent보다 급격)
_TAU_RING = 0.12  # 링잉 감쇠 시상수
_RING_AMP = 0.35
_RING_CYCLES = 1.5
_RING_HZ = 3.0


def _num(value, cast, name: str):
    """int/float 변환. 타입 불일치도 ValueError로 통일(router 400 매핑)."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be {cast.__name__}, got {value!r}")


def _rotation_matrix(euler_deg) -> np.ndarray:
    """낙하 자세 Euler [rx, ry, rz]° → R = Rz @ Ry @ Rx."""
    try:
        rx, ry, rz = (np.radians(float(a)) for a in euler_deg)
    except (TypeError, ValueError):
        raise ValueError(f"orientation must be 3 numbers (deg), got {euler_deg!r}")
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _contact_sites(vrot: np.ndarray, diag: float) -> tuple[np.ndarray, np.ndarray]:
    """최저 밴드 정점 클러스터링 → (centers (J,3), shares (J,), Σshares=1). 결정적.

    밴드 정점의 가중치 w = 1 − dz/tol (낮을수록 크게). xy 격자 버킷별 가중
    무게중심을 사이트로, 가중 합을 지분으로 삼는다. 최소 1개 사이트 보장
    (최저 정점은 dz=0으로 항상 밴드에 든다).
    """
    z = vrot[:, 2]
    z_min = float(z.min())
    z_ext = float(z.max()) - z_min
    tol = max(_BAND_FRACTION * z_ext, 1e-9)
    dz = z - z_min
    band = dz <= tol
    pts = vrot[band]
    w = np.clip(1.0 - dz[band] / tol, 0.0, 1.0)

    bucket = max(_BUCKET_FRACTION * diag, 1e-9)
    xy = pts[:, :2]
    keys = np.floor((xy - xy.min(0)) / bucket).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    weight = np.bincount(inv, weights=w, minlength=len(uniq))
    centers = np.stack(
        [np.bincount(inv, weights=w * pts[:, k], minlength=len(uniq)) for k in range(3)],
        axis=1,
    )
    keep = weight > 0  # 밴드 경계(w=0)로만 이뤄진 버킷 제거
    uniq, weight, centers = uniq[keep], weight[keep], centers[keep]
    centers = centers / weight[:, None]
    # 가중치 내림차순, 격자 키 사전순 타이브레이크 → 결정적 순서
    order = np.lexsort((uniq[:, 1], uniq[:, 0], -weight))[:_MAX_SITES]
    weight, centers = weight[order], centers[order]
    return centers, weight / weight.sum()


def free_fall_trajectory(vertices: np.ndarray, action: dict) -> np.ndarray:
    """(T, N, 3) 자유 낙하 변형 궤적. BE 모델과 FE 오프라인 반응이 공유하는 순수 함수."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {vertices.shape}")
    n = vertices.shape[0]
    if n == 0:
        raise ValueError("empty vertices")

    height = _num(action.get("drop_height", 1.0), float, "drop_height")
    if height <= 0:
        raise ValueError(f"drop_height must be > 0, got {height}")
    rest_e = _num(action.get("restitution", 0.3), float, "restitution")
    if not 0.0 <= rest_e < 1.0:
        raise ValueError(f"restitution must be in [0, 1), got {rest_e}")
    scale = _num(action.get("scale", 1.0), float, "scale")
    frames = _num(action.get("frames", DEFAULT_FRAMES), int, "frames")
    frames = max(2, min(frames, MAX_FRAMES))
    rot = _rotation_matrix(action.get("orientation", (0.0, 0.0, 0.0)))

    # 낙하 자세 적용 + 바닥 기준 정렬 (최저 정점 z=0).
    center = vertices.mean(0)
    vrot = (vertices - center) @ rot.T + center
    vrot = vrot - np.array([0.0, 0.0, float(vrot[:, 2].min())])
    diag = float(np.linalg.norm(vrot.max(0) - vrot.min(0))) or 1.0

    radius = action.get("radius")
    sigma = _num(radius, float, "radius") if radius is not None else _SITE_RADIUS_FRACTION * diag
    if sigma <= 0:
        raise ValueError(f"radius must be > 0, got {sigma}")

    centers, shares = _contact_sites(vrot, diag)
    v0 = np.sqrt(2.0 * height / diag)  # 무차원 충격 속도 계수
    depths = scale * _DEPTH_COEFF * diag * v0 * shares  # (J,) 사이트별 정착 깊이

    # --- 타임라인: 접촉 이벤트 시점 E_0..E_K --------------------------------
    bounce_ks = [
        k for k in range(1, _MAX_BOUNCES + 1)
        if rest_e ** (2 * k) >= _MIN_BOUNCE_H_FRAC
    ]
    denom = 1.0 + 2.0 * sum(rest_e ** k for k in bounce_ks)
    t_c1 = min(_FALL_FRAC_MAX, _CONTACT_END / denom)
    events = [t_c1]
    durations = []
    for k in bounce_ks:
        durations.append(2.0 * t_c1 * rest_e ** k)
        events.append(events[-1] + durations[-1])
    ev_weight = np.array([rest_e ** i for i in range(len(events))])
    ev_weight = ev_weight / ev_weight.sum()  # 접촉 이벤트별 dent 지분

    ts = np.linspace(0.0, 1.0, frames)

    # --- 강체 z 오프셋 (T,) ---------------------------------------------------
    z_rigid = np.zeros(frames)
    falling = ts < events[0]
    z_rigid[falling] = height * (1.0 - (ts[falling] / t_c1) ** 2)
    for idx, k in enumerate(bounce_ks):
        start, dur = events[idx], durations[idx]
        m = (ts >= start) & (ts < start + dur)
        tau = (ts[m] - start) / dur
        z_rigid[m] = 4.0 * height * rest_e ** (2 * k) * tau * (1.0 - tau)

    # --- 다중 접촉 dent + 링잉 중첩 (T,N) ------------------------------------
    disp = np.zeros((frames, n))
    for center_j, depth_j in zip(centers, depths):
        d = np.linalg.norm(vrot - center_j, axis=1)  # (N,)
        env = np.exp(-((d / sigma) ** 2))
        phase_d = 2.0 * np.pi * _RING_CYCLES * d / sigma
        for t_ev, w_ev in zip(events, ev_weight):
            active = ts >= t_ev
            if not active.any():
                continue
            u = ts[active] - t_ev
            # rise(t_ev)=0, rise(1)=1 — 이벤트별 정규화로 t=1에 정확히 지분만큼 정착.
            rise = (1.0 - np.exp(-u / _TAU_RISE)) / (1.0 - np.exp(-(1.0 - t_ev) / _TAU_RISE))
            ring = (
                _RING_AMP
                * np.sin(phase_d[None, :] - 2.0 * np.pi * _RING_HZ * u[:, None])
                * rise[:, None]
                * np.exp(-u[:, None] / _TAU_RING)
            )
            disp[active] += (depth_j * w_ev) * env[None, :] * (rise[:, None] + ring)

    frames_arr = np.repeat(vrot[None, :, :], frames, axis=0)
    frames_arr[:, :, 2] += z_rigid[:, None] + disp
    return frames_arr
