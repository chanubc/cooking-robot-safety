"""ISO/TS 15066 SSM 기반 로봇 3단계 판정.

예측기가 뭐든 Prediction만 받는다(교체 가능). 모든 agent×mode×step 중
위험영역에 가장 빨리 진입하는 최악 케이스로 TTC(time-to-contact)를 잡아 판정.
불확실성(sigma)은 유효 반경을 키워 보수적으로 반영.
"""
from __future__ import annotations

import math

from trajectory.types import Map, Prediction


class SSMPolicy:
    def __init__(self, zone: Map, t_stop: float = 1.2, t_slow: float = 3.0):
        z = zone.robot_zone
        self.cx, self.cz, self.r = z["x"], z["z"], z["r"]
        self.t_stop = t_stop
        self.t_slow = t_slow

    def _earliest_entry(self, pred: Prediction, now: float) -> float:
        """위험영역 최초 진입까지 걸리는 최소 시간. 없으면 inf."""
        ttc = math.inf
        for modes in pred.per_agent.values():
            for mode in modes:
                for (t, x, z, sigma) in mode.steps:
                    dist = math.hypot(x - self.cx, z - self.cz)
                    if dist - sigma <= self.r:          # sigma만큼 보수적으로
                        ttc = min(ttc, max(0.0, t - now))
                        break                            # 이 mode의 최초 진입만
        return ttc

    def decide(self, pred: Prediction, now: float) -> str:
        ttc = self._earliest_entry(pred, now)
        if ttc <= self.t_stop:
            return "stop"
        if ttc <= self.t_slow:
            return "slow"
        return "run"
