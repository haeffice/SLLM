"""free_fall 궤적 + 전역 강체 제거 오프라인 자가 테스트 (서버 불필요).

    python test_free_fall.py        # be/ 디렉토리에서, .venv 활성 상태

numpy만 사용. 미러 파일 바이트 일치까지 확인한다 (fe/가 체크아웃에 있어야 함).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from models.free_fall.trajectory import (
    _MAX_SITES,
    _contact_sites,
    free_fall_trajectory,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.join(os.path.dirname(_HERE), "fe")
sys.path.insert(0, _FE_DIR)

import analysis  # fe/analysis.py — remove_global_rigid 검증용


def make_plate(n: int = 15, size: float = 2.0) -> np.ndarray:
    xs = np.linspace(-size / 2, size / 2, n)
    gx, gy = np.meshgrid(xs, xs)
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)]).astype(np.float64)


def main() -> int:
    plate = make_plate()

    # --- 1. 결정성 + 기본 불변식 --------------------------------------------
    action = {"drop_height": 1.0, "restitution": 0.3, "frames": 60}
    f1 = free_fall_trajectory(plate, action)
    f2 = free_fall_trajectory(plate, action)
    assert np.array_equal(f1, f2), "결정성 위반"
    assert f1.shape == (60, len(plate), 3)
    # frames[0] = 공중 (최저 정점 z = h), 변형 0 (xy 불변, z 균일 오프셋)
    assert abs(float(f1[0, :, 2].min()) - 1.0) < 1e-12
    assert np.allclose(f1[0, :, :2], plate[:, :2])
    z_span0 = float(f1[0, :, 2].max() - f1[0, :, 2].min())
    assert z_span0 < 1e-12, "t=0에 변형 존재"
    # frames[-1] = 정착: 강체 오프셋 0 + dent는 +z (평면 판은 전면 접촉)
    assert float(f1[-1, :, 2].min()) >= -1e-9, "정착 프레임이 바닥 아래로 뚫림"
    assert float(f1[-1, :, 2].max()) > 0.001, "정착 dent 없음"
    print("1. 결정성/공중/정착 불변식 OK")

    # --- 2. e=0 → 바운스 없음, 강체 하강 단조 (scale=0로 dent 제거) ----------
    f0 = free_fall_trajectory(plate, {"drop_height": 1.0, "restitution": 0.0,
                                      "frames": 60, "scale": 0.0})
    zc = f0[:, :, 2].mean(axis=1)
    assert np.all(np.diff(zc) <= 1e-12), "e=0인데 상승 구간 존재"
    assert abs(zc[-1]) < 1e-12
    print("2. e=0 단조 하강 OK")

    # --- 3. 잘못된 입력 → ValueError -----------------------------------------
    for bad in (
        {"drop_height": 0.0},
        {"drop_height": -1},
        {"restitution": 1.0},
        {"restitution": -0.1},
        {"orientation": [0, 0]},
        {"orientation": "flat"},
        {"radius": 0},
        {"drop_height": "high"},
    ):
        try:
            free_fall_trajectory(plate, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ValueError 미발생: {bad}")
    print("3. 입력 검증 OK")

    # --- 4. 접촉 클러스터: 평면 낙하=다중 사이트, 모서리 낙하=단일 지배 ------
    diag = float(np.linalg.norm(plate.max(0) - plate.min(0)))
    centers_flat, shares_flat = _contact_sites(plate - [0, 0, plate[:, 2].min()], diag)
    assert len(centers_flat) == _MAX_SITES, f"평면 접촉 사이트 {len(centers_flat)}"
    assert shares_flat.max() < 0.5, "평면 낙하가 한 사이트에 집중됨"

    f_corner = free_fall_trajectory(plate, {"orientation": [45.0, 35.264, 0.0],
                                            "frames": 4, "scale": 0.0})
    corner_rot = f_corner[-1]
    diag_c = float(np.linalg.norm(corner_rot.max(0) - corner_rot.min(0)))
    centers_c, shares_c = _contact_sites(corner_rot, diag_c)
    assert shares_c[0] > 0.6, f"모서리 낙하 지배 사이트 지분 {shares_c[0]:.2f}"
    print(f"4. 접촉 클러스터 OK (평면 {len(centers_flat)}사이트 max지분 "
          f"{shares_flat.max():.2f} / 모서리 지배 지분 {shares_c[0]:.2f})")

    # --- 5. 전역 강체 제거: 순수 강체 궤적 → 변형 0 --------------------------
    rng = np.random.default_rng(7)
    body = rng.normal(size=(300, 3))
    T = 40
    frames_rigid = np.empty((T, 300, 3))
    for t in range(T):
        ang = 0.02 * t
        rot = np.array([[np.cos(ang), -np.sin(ang), 0],
                        [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        frames_rigid[t] = body @ rot.T + np.array([0.0, 0.0, 2.0 - 0.05 * t])
    aligned, motion = analysis.remove_global_rigid(frames_rigid, frames_rigid[0])
    err = float(np.abs(aligned - frames_rigid[0]).max())
    assert err < 1e-8, f"강체 제거 잔차 {err}"
    assert motion["available"] and motion["translation"] > 1.0
    res = analysis.compute_analysis(frames_rigid, frames_rigid[0], None, {}, None,
                                    "dummy", remove_rigid=True)
    assert res.verdict == "OK" and res.overall_max_disp < 1e-8, \
        f"순수 강체 낙하가 손상으로 판정됨: {res.verdict}, {res.overall_max_disp}"
    # 강체 제거를 끄면(충격 모드) 낙하 이동이 그대로 변위로 잡혀야 한다 (대조).
    res_raw = analysis.compute_analysis(frames_rigid, frames_rigid[0], None, {}, None, "dummy")
    assert res_raw.overall_max_disp > 1.0, "remove_rigid=False인데 강체 이동이 안 잡힘"
    print(f"5. 강체 제거 불변식 OK (잔차 {err:.1e}, 회전 {motion['rotation_deg']:.1f}°, "
          f"raw disp {res_raw.overall_max_disp:.2f})")

    # --- 6. 낙하 궤적 전체 분석: 변위가 dent 스케일(≪ 낙하 높이) -------------
    res_ff = analysis.compute_analysis(f1, f1[0], None, action, None, "dummy",
                                       remove_rigid=True)
    assert res_ff.overall_max_disp < 0.3, \
        f"낙하 이동이 변위로 새어 들어옴: {res_ff.overall_max_disp}"
    assert res_ff.overall_max_disp > 0.005, "dent가 분석에 안 잡힘"
    print(f"6. 낙하 분석 정합 OK (max 변형 {res_ff.overall_max_disp:.4f} ≪ h=1.0)")

    # --- 7. screen 기능 판정 경계 ---------------------------------------------
    for frac, expected in ((0.04, "OK"), (0.10, "WARN"), (0.30, "FAIL")):
        # _dead_area_frac를 직접 흉내: 임계로 나뉘는 합성 peak_disp
        grid = make_plate(11, 1.0)
        faces = []
        n = 11
        for j in range(n - 1):
            for i in range(n - 1):
                a = j * n + i
                faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
        faces = np.asarray(faces)
        peak = np.zeros(len(grid))
        # 앞에서부터 대략 frac 비율의 face가 dead가 되도록 정점 마킹
        k = int(frac * len(grid))
        peak[:k] = 1.0
        out = analysis._functional_metrics(
            {"type": "screen"}, np.arange(len(grid)), grid, grid,
            np.eye(3), np.zeros(len(grid)), faces, peak, 0.5, 1.0,
        )
        assert out["verdict"] == expected, (frac, out)
    print("7. screen 판정 경계 OK")

    # --- 8. 미러 바이트 일치 ---------------------------------------------------
    for a, b in (
        (os.path.join(_HERE, "models", "free_fall", "trajectory.py"),
         os.path.join(_FE_DIR, "free_fall_sim.py")),
        (os.path.join(_HERE, "utils", "chat_fallback.py"),
         os.path.join(_FE_DIR, "chat_fallback.py")),
    ):
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read(), f"미러 불일치: {a} vs {b}"
    print("8. 미러 바이트 일치 OK")

    print("OK ✓ free_fall 오프라인 자가 테스트 전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
