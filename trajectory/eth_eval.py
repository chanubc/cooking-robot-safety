"""공개 데이터(ETH) 궤적 예측 비교: 등속 vs 칼만 vs 경량 LSTM.

표준 프로토콜: 과거 8스텝(3.2s) 관찰 → 미래 12스텝(4.8s) 예측, dt=0.4s.
obsmat 컬럼: frame, ped_id, x, z(높이=0), y, vx, vz, vy → 지면좌표 (x, y).
지표: ADE(평균 위치오차), FDE(최종 위치오차), 미터.
LSTM은 관찰 변위 시퀀스 → 미래 변위 시퀀스를 학습(작은 모델).
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np

OBS, PRED, DT = 8, 12, 0.4


def load_trajs(path: Path):
    rows = []
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        f, pid, x, _z, y = float(p[0]), int(float(p[1])), float(p[2]), float(p[3]), float(p[4])
        rows.append((f, pid, x, y))
    # ped별 프레임순 정렬
    by_ped = {}
    for f, pid, x, y in rows:
        by_ped.setdefault(pid, []).append((f, x, y))
    # 프레임 간격 추정(최빈 diff)
    frames = sorted({f for f, _, _, _ in rows})
    diffs = np.diff(frames)
    step = int(np.median(diffs)) if len(diffs) else 6
    # 연속 20스텝 윈도우 추출
    windows = []
    for pid, seq in by_ped.items():
        seq.sort()
        fs = [s[0] for s in seq]
        for i in range(len(seq) - (OBS + PRED) + 1):
            block = seq[i:i + OBS + PRED]
            ok = all(abs((block[j + 1][0] - block[j][0]) - step) < 1e-6 for j in range(len(block) - 1))
            if ok:
                windows.append(np.array([[b[1], b[2]] for b in block], float))  # (20,2)
    return windows


def const_vel(obs):
    v = (obs[-1] - obs[0]) / ((OBS - 1) * DT)
    return np.array([obs[-1] + v * DT * (k + 1) for k in range(PRED)])


def kalman(obs):
    dt = DT
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], float)
    q, r = 0.5, 0.05
    Q = q * np.array([[dt**3/3, dt**2/2, 0, 0], [dt**2/2, dt, 0, 0],
                      [0, 0, dt**3/3, dt**2/2], [0, 0, dt**2/2, dt]], float)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
    R = r * np.eye(2)
    x = np.array([obs[0, 0], 0, obs[0, 1], 0], float)
    P = np.diag([r, 1, r, 1]).astype(float)
    for m in obs[1:]:
        x = F @ x; P = F @ P @ F.T + Q
        y = m - H @ x; S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y; P = (np.eye(4) - K @ H) @ P
    out = []
    for _ in range(PRED):
        x = F @ x; out.append([x[0], x[2]])
    return np.array(out)


def ade_fde(pred, gt):
    d = np.linalg.norm(pred - gt, axis=1)
    return d.mean(), d[-1]


def eval_classical(fn, windows):
    a = f = 0.0
    for w in windows:
        obs, gt = w[:OBS], w[OBS:]
        p = fn(obs); da, df = ade_fde(p, gt); a += da; f += df
    n = len(windows)
    return a / n, f / n


def run_lstm(train, test):
    import torch, torch.nn as nn
    torch.manual_seed(0)

    def to_disp(windows):
        X, Y = [], []
        for w in windows:
            obs, gt = w[:OBS], w[OBS:]
            X.append(np.diff(obs, axis=0))          # (7,2) 관찰 변위
            Y.append(gt - obs[-1])                   # (12,2) 마지막관찰 기준 미래 변위
        return np.array(X, np.float32), np.array(Y, np.float32)

    Xtr, Ytr = to_disp(train); Xte, Yte = to_disp(test)
    scale = float(np.std(Xtr)) or 1.0               # 입력/출력 정규화
    Xtr, Ytr, Xte = Xtr / scale, Ytr / scale, Xte / scale
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xtr, device=dev); Ytr = torch.tensor(Ytr, device=dev)

    class Net(nn.Module):
        def __init__(s, h=128):
            super().__init__()
            s.enc = nn.LSTM(2, h, batch_first=True)
            s.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, PRED * 2))
        def forward(s, x):
            _, (hn, _) = s.enc(x)
            return s.head(hn[-1]).view(-1, PRED, 2)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    n = len(Xtr); bs = 128
    net.train()
    for ep in range(400):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(Xtr[idx]), Ytr[idx]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xte, device=dev)).cpu().numpy() * scale
    a = f = 0.0
    for i, w in enumerate(test):
        p = w[OBS - 1] + pred[i]
        da, df = ade_fde(p, w[OBS:]); a += da; f += df
    return a / len(test), f / len(test)


def ensure_data(base: Path):
    """OpenTraj에서 ETH obsmat을 자동 다운로드 (없을 때만)."""
    import urllib.request
    base.mkdir(parents=True, exist_ok=True)
    srcs = {
        "seq_eth.txt": "https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/ETH/seq_eth/obsmat.txt",
        "seq_hotel.txt": "https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/ETH/seq_hotel/obsmat.txt",
    }
    for fn, url in srcs.items():
        p = base / fn
        if not p.exists():
            print(f"downloading {fn} ...", flush=True)
            urllib.request.urlretrieve(url, p)


def main():
    base = Path("datasets/eth")
    ensure_data(base)
    eth = load_trajs(base / "seq_eth.txt")
    hotel = load_trajs(base / "seq_hotel.txt")
    print(f"windows: eth={len(eth)} hotel={len(hotel)}", flush=True)

    def report(title, train, test):
        print(f"\n=== {title} — ADE / FDE (m) ===", flush=True)
        for name, fn in [("const-velocity", const_vel), ("kalman", kalman)]:
            a, f = eval_classical(fn, test)
            print(f"  {name:16s}  ADE={a:.3f}  FDE={f:.3f}", flush=True)
        la, lf = run_lstm(train, test)
        print(f"  {'lstm (small)':16s}  ADE={la:.3f}  FDE={lf:.3f}", flush=True)

    # 1) 교차 장면: train=hotel, test=eth (진짜 일반화, 분포 이동)
    report("cross-scene (train=hotel, test=eth)", hotel, eth)

    # 2) 동일 분포: eth+hotel 섞어 80/20 분할 (LSTM에 최선의 기회)
    allw = eth + hotel
    rng = np.random.default_rng(0); idx = rng.permutation(len(allw))
    cut = int(len(allw) * 0.8)
    tr = [allw[i] for i in idx[:cut]]; te = [allw[i] for i in idx[cut:]]
    report("in-distribution (mixed 80/20 split)", tr, te)


if __name__ == "__main__":
    main()
