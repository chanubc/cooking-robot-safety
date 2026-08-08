"""칼만 필터 예측기 테스트.

핵심 성질:
1. 무노이즈 직선 → 등속과 거의 동일한 예측.
2. 노이즈 낀 직선 → 등속보다 ADE(예측 정확도) 낮음 (노이즈 보정 효과).
"""
import numpy as np

from trajectory.types import Track, TrackScene
from trajectory.predictors import ConstantVelocityPredictor, KalmanPredictor
from trajectory.evaluator import ade


def _straight_history(now, n=10, dt=0.1, v=0.4, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    hist = []
    t0 = now - (n - 1) * dt
    for i in range(n):
        t = t0 + i * dt
        x = v * (t - now)          # now 시점에 x=0 통과
        z = 0.0
        if noise > 0:
            x += rng.normal(0, noise)
            z += rng.normal(0, noise)
        hist.append((t, x, z))
    return hist


def _true_future(now, horizon, n_steps, v=0.4):
    return [(now + horizon * i / n_steps, v * (horizon * i / n_steps), 0.0)
            for i in range(1, n_steps + 1)]


def test_kalman_matches_constant_velocity_when_no_noise():
    now, horizon, n = 3.0, 2.0, 5
    hist = _straight_history(now, noise=0.0)
    scene = TrackScene(now=now, horizon=horizon, agents=[Track(1, hist)], map=None)

    k = KalmanPredictor(n_steps=n).predict(scene).per_agent[1][0].steps
    gt = _true_future(now, horizon, n)

    # 무노이즈면 칼만도 참값에 매우 가까움
    assert ade(k, gt) < 0.05


def test_kalman_beats_constant_velocity_under_noise():
    now, horizon, n = 3.0, 2.0, 5
    hist = _straight_history(now, noise=0.08, seed=42)
    scene = TrackScene(now=now, horizon=horizon, agents=[Track(1, hist)], map=None)
    gt = _true_future(now, horizon, n)

    cv = ConstantVelocityPredictor(n_steps=n).predict(scene).per_agent[1][0].steps
    kf = KalmanPredictor(n_steps=n).predict(scene).per_agent[1][0].steps

    assert ade(kf, gt) < ade(cv, gt)


def test_kalman_reports_growing_uncertainty():
    now, horizon, n = 3.0, 2.0, 5
    hist = _straight_history(now, noise=0.05, seed=1)
    scene = TrackScene(now=now, horizon=horizon, agents=[Track(1, hist)], map=None)
    steps = KalmanPredictor(n_steps=n).predict(scene).per_agent[1][0].steps
    sigmas = [s[3] for s in steps]
    assert sigmas[0] > 0.0                # 불확실성 제공
    assert sigmas[-1] >= sigmas[0]        # 멀수록 불확실성 증가
