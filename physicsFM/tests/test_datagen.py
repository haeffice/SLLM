"""generate_rollouts.py 데이터 생성 경로 통합 테스트 (D1/D5/D6 검증).

요약 흐름:
  1) scale=0 패리티(D6): 변형 0 → 생성기 frames = vrot + z_rigid.
     compute_vrot/compute_timeline 재유도 수식이 physics/be 원본 생성기와
     바이트 수준으로 일치함을 증명한다 (재유도 드리프트 방지의 핵심).
  2) dt 물리(D1): dt_s = T_phys/(frames−1), 낙하 구간 z_rigid 가 물리 시간
     τ = ts·T_phys 에서 정확히 뉴턴 포물선 h − ½gτ² 이 된다.
  3) 초소형 end-to-end 생성 → inspect 불변식 + 분할 서로소 분할/frame_kind
     꼬리 hold(D5)/ke[0]=0/f32/derive_eos 단조·말단 1 검증.
  4) 결정성: 같은 seed 두 번 생성 → r00000 positions 동일.
pytest와 `python tests/test_datagen.py` 양쪽에서 실행 가능.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

import generate_rollouts as gr
from dataset import derive_eos
from meshes import load_mesh

# config.yaml data 섹션 구조를 그대로 축소한 초소형 설정 (plate41 6롤아웃).
TINY_CFG = {
    "data": {
        "h5": "data/rollouts.h5",
        "train_meshes": ["plate41"],
        "eval_meshes": [],
        "rollouts_per_mesh": 6,
        "eval_rollouts_per_mesh": 0,
        "dt_target_s": 0.012,
        "g": 9.81,
        "drop_height": [0.5, 2.0],
        "restitution": [0.0, 0.65],
        "scale": [0.5, 1.5],
        "hold_frac": 0.3,
        "seed": 7,
    }
}


def test_scale0_parity():
    """D6: scale=0 → frames == vrot + z_rigid — 재유도 타임라인/자세 정렬이
    physics/be 생성기와 바이트 수준으로 일치."""
    traj = gr.load_trajectory_module()
    mesh = load_mesh("plate41")
    action = {"drop_height": 1.3, "restitution": 0.4,
              "orientation": [30.0, -20.0, 45.0], "scale": 0.0, "frames": 97}

    frames_arr = traj.free_fall_trajectory(mesh.vertices, action)
    assert frames_arr.shape == (97, len(mesh.vertices), 3)

    vrot, _ = gr.compute_vrot(traj, mesh.vertices, action)
    _, t_c1, events, z_rigid = gr.compute_timeline(traj, 1.3, 0.4, 97)
    assert 0.0 < t_c1 <= traj._FALL_FRAC_MAX
    assert events[-1] <= traj._CONTACT_END + 1e-12

    # 변형 0 → 궤적 = 정렬 자세 + 강체 z 오프셋 (그 이상도 이하도 아님)
    expected = np.repeat(vrot[None, :, :], 97, axis=0)
    expected[:, :, 2] += z_rigid[:, None]
    assert np.allclose(frames_arr - expected, 0.0, atol=1e-9), \
        "scale=0 패리티 실패: 재유도 vrot/z_rigid 가 생성기와 불일치"
    # 성분별 재확인: z == vrot_z + z_rigid, xy 는 전 프레임 상수
    assert np.allclose(frames_arr[:, :, 2], vrot[None, :, 2] + z_rigid[:, None], atol=1e-9)
    assert np.array_equal(frames_arr[:, :, :2],
                          np.repeat(vrot[None, :, :2], 97, axis=0))


def test_dt_physics():
    """D1: dt_s = T_phys/(frames−1), 낙하 구간 z_rigid = h − ½g·(ts·T_phys)²."""
    traj = gr.load_trajectory_module()
    g, dt_target = 9.81, 0.012
    for h, e in [(0.5, 0.0), (1.0, 0.3), (2.0, 0.6)]:
        _, t_c1, _, _ = gr.compute_timeline(traj, h, e, 2)
        frames_n, dt_s, t_phys = gr.physical_frames(t_c1, h, g, dt_target, traj)
        assert 2 <= frames_n <= traj.MAX_FRAMES
        assert abs(dt_s - t_phys / (frames_n - 1)) < 1e-12, f"(h={h}, e={e}) dt 정의 불일치"
        # 캡에 걸리지 않는 그리드 — dt 는 목표 근방
        assert abs(dt_s - dt_target) <= 0.5 * dt_target

        ts, t_c1_full, _, z_rigid = gr.compute_timeline(traj, h, e, frames_n)
        assert t_c1_full == t_c1  # t_c1 은 e 만의 함수 — frames 와 무관
        fall = ts < t_c1
        assert fall.any()
        tau = ts[fall] * t_phys  # 정규화 시간 → 물리 시간 (초)
        assert np.allclose(z_rigid[fall], h - 0.5 * g * tau**2, atol=1e-9), \
            f"(h={h}, e={e}) 낙하 포물선 불일치"


def test_generate_and_inspect():
    """초소형 end-to-end 생성 → inspect 불변식 + 분할/hold/ke/eos 검증."""
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "tiny.h5")
        gr.generate(TINY_CFG, out)
        gr.inspect(out)  # 내장 불변식 검사 — 실패 시 AssertionError

        with h5py.File(out, "r") as h5:
            rids_all = set(h5["rollouts"].keys())
            assert len(rids_all) == 6
            splits = {
                s: {x.decode() if isinstance(x, bytes) else x
                    for x in h5[f"splits/{s}"][:]}
                for s in ("train", "valid", "test")
            }
            # 분할은 서로소이며 전체 rollout 을 정확히 덮는다
            assert splits["train"] | splits["valid"] | splits["test"] == rids_all
            assert sum(len(v) for v in splits.values()) == len(rids_all)
            assert not (splits["train"] & splits["valid"])
            assert not (splits["train"] & splits["test"])
            assert not (splits["valid"] & splits["test"])

            for rid in sorted(rids_all):
                grp = h5[f"rollouts/{rid}"]
                assert grp["positions"].dtype == np.float32
                # hold-extension(D5)은 꼬리에만: frame_kind 0…0 1…1
                fk = grp["frame_kind"][:].astype(int)
                t_gen = int(grp.attrs["frames_generated"])
                assert fk[0] == 0 and np.all(np.diff(fk) >= 0)
                assert np.all(fk[:t_gen] == 0) and np.all(fk[t_gen:] == 1)
                # ke: 초기 정지 → 정확히 0
                ke = grp["ke"][:]
                assert ke[0] == 0.0 and np.all(ke >= 0.0)
                # eos 라벨: 역방향 cummax 파생 → 단조 비감소, 말단 settled=1
                eos = derive_eos(ke, 1e-3)
                assert np.all(np.diff(eos) >= 0.0), f"{rid}: eos 비단조"
                assert eos[-1] == 1.0, f"{rid}: 말단이 settled 가 아님"


def test_determinism():
    """같은 seed 로 두 번 생성 → r00000 positions/action 완전 동일."""
    with tempfile.TemporaryDirectory() as td:
        path_a = str(Path(td) / "a.h5")
        path_b = str(Path(td) / "b.h5")
        gr.generate(TINY_CFG, path_a)
        gr.generate(TINY_CFG, path_b)
        with h5py.File(path_a, "r") as ha, h5py.File(path_b, "r") as hb:
            pos_a = ha["rollouts/r00000/positions"][:]
            pos_b = hb["rollouts/r00000/positions"][:]
            assert pos_a.shape == pos_b.shape
            assert np.array_equal(pos_a, pos_b), "seed 고정에도 positions 이 다름"
            assert (ha["rollouts/r00000"].attrs["action"]
                    == hb["rollouts/r00000"].attrs["action"])


def main():
    """pytest 없이 직접 실행할 때 전체 테스트 수행."""
    tests = [
        test_scale0_parity,
        test_dt_physics,
        test_generate_and_inspect,
        test_determinism,
    ]
    for fn in tests:
        fn()
        print(f"{fn.__name__}: OK")
    print("OK")


if __name__ == "__main__":
    main()
