"""실제 입력→출력 예시. 실행: python -m trajectory.example

사람 1명이 로봇(원점)으로 다가오는 상황을 넣으면, 칼만이 미래 위치를 예측하고
SSM이 정지 판정을 내린다.
"""
from trajectory.types import Track, TrackScene, Map
from trajectory.predictors import KalmanPredictor
from trajectory.ssm import SSMPolicy


def main():
    scene = TrackScene(
        now=3.0, horizon=2.0,
        agents=[Track(id=1, history=[
            (2.4, 2.4, 0.0), (2.6, 2.0, 0.0), (2.8, 1.6, 0.0), (3.0, 1.2, 0.0),
        ])],
        map=Map(robot_zone={"x": 0.0, "z": 0.0, "r": 1.0}),
    )
    pred = KalmanPredictor(n_steps=5).predict(scene)
    steps = pred.per_agent[1][0].steps
    decision = SSMPolicy(scene.map, t_stop=1.2, t_slow=3.0).decide(pred, now=3.0)

    print("INPUT  TrackScene: now=3.0 horizon=2.0 robot_zone=(0,0,r=1.0)")
    print("       agent 1 history (t,x,z):",
          [(t, round(x, 2), z) for t, x, z in scene.agents[0].history])
    print("OUTPUT Prediction (Kalman):")
    for t, x, z, s in steps:
        print(f"       t={t:.1f}  x={x:+.2f}  z={z:+.2f}  sigma={s:.2f}")
    print("SSM    decision:", decision)


if __name__ == "__main__":
    main()
