"""등속(constant velocity) 예측기 테스트."""
import math

from trajectory.types import Track, TrackScene
from trajectory.predictors import ConstantVelocityPredictor


def test_straight_line_predicts_exactly_on_the_line():
    # 사람이 +x 방향으로 0.4 m/s 등속 이동 (z 고정)
    hist = [(2.4, 0.8, 3.0), (2.6, 0.88, 3.0), (2.8, 0.96, 3.0), (3.0, 1.04, 3.0)]
    scene = TrackScene(now=3.0, horizon=2.0, agents=[Track(id=1, history=hist)], map=None)

    pred = ConstantVelocityPredictor(n_steps=5).predict(scene)

    modes = pred.per_agent[1]
    assert len(modes) == 1              # 단일 궤적 모델 → Mode 1개
    assert modes[0].prob == 1.0
    steps = modes[0].steps
    assert len(steps) == 5

    # 마지막 시점은 now+horizon = 5.0s, x = 1.04 + 0.4*2.0 = 1.84
    t_last, x_last, z_last, sigma_last = steps[-1]
    assert math.isclose(t_last, 5.0, abs_tol=1e-6)
    assert math.isclose(x_last, 1.84, abs_tol=1e-2)
    assert math.isclose(z_last, 3.0, abs_tol=1e-6)
    assert sigma_last == 0.0            # 등속은 불확실성 미제공
