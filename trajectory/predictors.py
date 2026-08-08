"""궤적 예측기들. 공통 인터페이스: predict(TrackScene) -> Prediction."""
from __future__ import annotations

import math

import numpy as np

from trajectory.types import Mode, Prediction, TrackScene


class ConstantVelocityPredictor:
    """최근 속도가 유지된다고 가정하고 직선 외삽. 학습 불필요, sigma=0."""

    def __init__(self, n_steps: int = 20):
        self.n_steps = n_steps

    def predict(self, scene: TrackScene) -> Prediction:
        pred = Prediction()
        for track in scene.agents:
            pred.per_agent[track.id] = [self._predict_one(track, scene.now, scene.horizon)]
        return pred

    def _predict_one(self, track, now: float, horizon: float) -> Mode:
        hist = track.history
        vx = vz = 0.0
        if len(hist) >= 2:
            t0, x0, z0 = hist[0]
            t1, x1, z1 = hist[-1]
            dt = t1 - t0
            if dt > 1e-9:
                vx = (x1 - x0) / dt
                vz = (z1 - z0) / dt
        _, x_now, z_now = hist[-1]
        steps = []
        for i in range(1, self.n_steps + 1):
            tau = horizon * i / self.n_steps
            steps.append((now + tau, x_now + vx * tau, z_now + vz * tau, 0.0))
        return Mode(prob=1.0, steps=steps)


class KalmanPredictor:
    """등속(constant-velocity) 운동 모델 칼만 필터.

    상태 = [x, vx, z, vz]. 과거 측정을 순차 반영(예측+보정)해 노이즈를 걸러낸 뒤,
    horizon 동안 예측-only로 외삽. sigma는 위치 공분산에서 산출(멀수록 증가).
    학습 파라미터 없음. r=측정노이즈분산, q=가속 스펙트럼밀도.
    """

    def __init__(self, n_steps: int = 20, r: float = 0.08 ** 2, q: float = 0.05):
        self.n_steps = n_steps
        self.r = r
        self.q = q

    def predict(self, scene: TrackScene) -> Prediction:
        pred = Prediction()
        for track in scene.agents:
            pred.per_agent[track.id] = [self._predict_one(track, scene.now, scene.horizon)]
        return pred

    def _F(self, dt):
        return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], float)

    def _Q(self, dt):
        q = self.q
        a, b, c = dt ** 3 / 3, dt ** 2 / 2, dt
        return q * np.array([[a, b, 0, 0], [b, c, 0, 0], [0, 0, a, b], [0, 0, b, c]], float)

    def _predict_one(self, track, now: float, horizon: float) -> Mode:
        hist = track.history
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
        R = self.r * np.eye(2)

        t_prev, x0, z0 = hist[0]
        x = np.array([x0, 0.0, z0, 0.0], float)
        P = np.diag([self.r, 1.0, self.r, 1.0]).astype(float)

        for (t, mx, mz) in hist[1:]:
            dt = max(1e-6, t - t_prev)
            t_prev = t
            F = self._F(dt)
            x = F @ x
            P = F @ P @ F.T + self._Q(dt)
            # 보정
            zmeas = np.array([mx, mz], float)
            y = zmeas - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P

        # horizon 외삽 (예측-only)
        steps = []
        t_cur = now
        xf, Pf = x.copy(), P.copy()
        for i in range(1, self.n_steps + 1):
            t_next = now + horizon * i / self.n_steps
            dt = max(1e-6, t_next - t_cur)
            t_cur = t_next
            F = self._F(dt)
            xf = F @ xf
            Pf = F @ Pf @ F.T + self._Q(dt)
            sigma = math.sqrt(max(0.0, 0.5 * (Pf[0, 0] + Pf[2, 2])))
            steps.append((t_next, float(xf[0]), float(xf[2]), sigma))
        return Mode(prob=1.0, steps=steps)
