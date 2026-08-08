"""궤적 예측 인터페이스 자료구조.

되돌림 방지를 위해 multi-agent + multimodal + 불확실성을 처음부터 담는다.
등속/칼만 같은 단일 궤적 모델은 Mode 1개(prob=1.0)로 채우면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    """한 사람의 바닥좌표 시계열. history = [(t, x, z), ...]."""
    id: int
    history: list[tuple[float, float, float]]


@dataclass
class Map:
    """선택적 맥락. 로봇 위험영역 등."""
    robot_zone: Optional[dict] = None  # {"x":.., "z":.., "r":..}


@dataclass
class TrackScene:
    """예측기 입력. 한 명이 아니라 전원 + 선택적 맵."""
    now: float
    horizon: float
    agents: list[Track]
    map: Optional[Map] = None


@dataclass
class Mode:
    """미래 궤적 한 갈래. steps = [(t, x, z, sigma), ...]."""
    prob: float
    steps: list[tuple[float, float, float, float]]


@dataclass
class Prediction:
    """예측기 출력. agent id -> 여러 Mode."""
    per_agent: dict[int, list[Mode]] = field(default_factory=dict)
