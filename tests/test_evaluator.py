"""ADE/FDE 평가 지표 테스트 (손계산과 일치해야 함)."""
import math

from trajectory.evaluator import ade, fde


def test_ade_fde_match_hand_computation():
    # 예측 스텝 (t, x, z, sigma)
    pred = [(4.0, 1.0, 0.0, 0.0), (5.0, 2.0, 0.0, 0.0)]
    # 정답 (t, x, z): 첫 점은 정확, 마지막 점은 (3,4) 만큼 어긋남 → 거리 5
    gt = [(4.0, 1.0, 0.0), (5.0, 5.0, 4.0)]

    # 각 스텝 오차: 0.0, 5.0 → ADE = 2.5, FDE = 5.0
    assert math.isclose(ade(pred, gt), 2.5, abs_tol=1e-9)
    assert math.isclose(fde(pred, gt), 5.0, abs_tol=1e-9)
