"""SSMPolicy 테스트: 예측된 미래로 로봇 3단계(run/slow/stop) 판정.

규칙: 모든 agent×mode×step 중 위험영역에 가장 빨리 진입하는 최악 케이스로 TTC.
- 이미 안/곧(<t_stop) 진입 → stop
- t_stop~t_slow 사이 진입 → slow
- 안 들어옴 → run
불확실성(sigma)이 크면 더 보수적으로(진입으로) 판단.
"""
from trajectory.types import Mode, Prediction, Map
from trajectory.ssm import SSMPolicy


ZONE = Map(robot_zone={"x": 0.0, "z": 0.0, "r": 1.0})


def _pred(steps):
    return Prediction(per_agent={1: [Mode(prob=1.0, steps=steps)]})


def policy():
    return SSMPolicy(zone=ZONE, t_stop=1.2, t_slow=3.0)


def test_run_when_person_never_enters_zone():
    # 항상 멀리(z=5)에 머무름
    steps = [(3.0 + t, 5.0, 5.0, 0.0) for t in (0.5, 1.0, 2.0, 3.0)]
    assert policy().decide(_pred(steps), now=3.0) == "run"


def test_stop_when_entry_within_t_stop():
    # 0.5s 뒤 원점(반경 내) 진입 → t_stop(1.2) 이내
    steps = [(3.5, 0.2, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0)]
    assert policy().decide(_pred(steps), now=3.0) == "stop"


def test_slow_when_entry_between_t_stop_and_t_slow():
    # 2.0s 뒤 진입 → t_stop(1.2)~t_slow(3.0) 사이
    steps = [(4.0, 3.0, 0.0, 0.0), (5.0, 0.5, 0.0, 0.0)]
    assert policy().decide(_pred(steps), now=3.0) == "slow"


def test_uncertainty_makes_decision_conservative():
    # 위치상으론 반경 밖(1.3m)이지만 sigma가 커서 진입으로 간주 → 최소 slow
    steps = [(4.5, 1.3, 0.0, 0.5)]
    assert policy().decide(_pred(steps), now=3.0) in ("slow", "stop")
