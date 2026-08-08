"""예측 정확도 지표: ADE(평균 위치오차), FDE(최종시점 오차)."""
from __future__ import annotations

import math


def _dist(px: float, pz: float, gx: float, gz: float) -> float:
    return math.hypot(px - gx, pz - gz)


def ade(pred_steps, gt_points) -> float:
    """Average Displacement Error — 스텝별 위치오차의 평균(미터)."""
    n = min(len(pred_steps), len(gt_points))
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(n):
        _, px, pz, _ = pred_steps[i]
        _, gx, gz = gt_points[i]
        total += _dist(px, pz, gx, gz)
    return total / n


def fde(pred_steps, gt_points) -> float:
    """Final Displacement Error — 마지막 공통 시점의 위치오차(미터)."""
    n = min(len(pred_steps), len(gt_points))
    if n == 0:
        return float("nan")
    _, px, pz, _ = pred_steps[n - 1]
    _, gx, gz = gt_points[n - 1]
    return _dist(px, pz, gx, gz)
