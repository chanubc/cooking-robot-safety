"""ATC 데이터 실측: 사람이 실제로 걷는가? (오버헤드 데이터셋과 직접 비교)"""
import numpy as np, csv
from pathlib import Path

CSV=Path("datasets/atc/atc-20121114.csv")
# 컬럼: time, person_id, x[mm], y[mm], z[mm], velocity, angle_motion, facing_angle
OBS_S, PRED_S, DT = 3.2, 4.8, 0.4     # ETH 표준과 동일: 관찰 3.2s -> 예측 4.8s
tracks={}
with open(CSV) as f:
    for i,r in enumerate(csv.reader(f)):
        if len(r)<4: continue
        t,pid,x,y=float(r[0]),int(float(r[1])),float(r[2]),float(r[3])
        tracks.setdefault(pid,[]).append((t,x/1000.0,y/1000.0))   # mm -> m
        if i>6_000_000: break     # 하루 일부만 (충분한 표본)
print(f"읽은 트랙: {len(tracks)}개",flush=True)

# 0.4초 간격으로 리샘플 후 20스텝(8+12) 창 생성
wins=[]
for pid,pts in tracks.items():
    pts.sort()
    if len(pts)<20: continue
    t0=pts[0][0]; grid=np.arange(t0, pts[-1][0], DT)
    if len(grid)<20: continue
    ts=np.array([p[0] for p in pts]); xs=np.array([p[1] for p in pts]); ys=np.array([p[2] for p in pts])
    rx=np.interp(grid,ts,xs); ry=np.interp(grid,ts,ys)
    arr=np.stack([rx,ry],1)
    for i in range(0,len(arr)-20+1,10):
        wins.append(arr[i:i+20])
wins=np.array(wins)
print(f"창(관찰8/예측12) {len(wins)}개",flush=True)

net=np.linalg.norm(wins[:,-1]-wins[:,7],axis=1)          # 4.8초 순변위
path=np.sum(np.linalg.norm(np.diff(wins[:,7:],axis=1),axis=2),axis=1)
print(f"\n4.8초 순변위: 평균 {net.mean():.2f}m  중앙값 {np.median(net):.2f}m")
print(f"  1m 미만 비율 {np.mean(net<1)*100:.1f}%   2m 이상 비율 {np.mean(net>=2)*100:.1f}%")
print(f"직진성(순변위/경로길이) 중앙값 {np.median(net/np.maximum(path,1e-6)):.2f}")
print(f"\n[비교] 오버헤드 구내식당: 순변위 중앙값 2.1px, 53%가 3px 미만")

# 기준선 비교 (순변위 2m 이상 필터)
def zero_vel(o): return np.repeat(o[-1][None,:],12,0)
def const_vel(o):
    v=(o[-1]-o[0])/7; return np.array([o[-1]+v*(k+1) for k in range(12)])
for label,sel in [("전체",np.ones(len(wins),bool)),("순변위 2m 이상",net>=2)]:
    W=wins[sel]
    if len(W)<100: continue
    res={}
    for nm,fn in [("정지",zero_vel),("등속",const_vel)]:
        e=np.array([np.linalg.norm(fn(w[:8])-w[8:],axis=1) for w in W])
        res[nm]=(e.mean(),e[:,-1].mean())
    print(f"\n=== {label} ({len(W)}창) ADE/FDE (m) ===")
    for nm,(a,f) in res.items(): print(f"  {nm:6s} ADE={a:.3f}  FDE={f:.3f}")
    print(f"  등속이 정지 대비 ADE {res['정지'][0]/res['등속'][0]:.1f}배 정확")
