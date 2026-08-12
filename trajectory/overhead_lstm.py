"""오버헤드 YOLO 데이터셋 → IoU 추적으로 ID부여 → (x,y) 궤적 → 등속/칼만/LSTM 테스트.

데이터셋은 24개 연속영상 클립. GT 박스를 프레임간 IoU로 추적해 궤적 생성.
좌표=YOLO 정규화 중심(0~1), 예측은 프레임 스텝 단위, ADE/FDE는 640px 환산.
"""
from __future__ import annotations
import os, re, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path("datasets/overhead-person-v3")
OBS, PRED, IMG = 8, 8, 640
CLIP_RE = re.compile(r'(.+?)[_-](\d{6,8})_jpg\.rf\.')

def load_clip_frames():
    """clip -> {frame_idx: [(cx,cy,w,h),...]}"""
    clips = defaultdict(dict)
    for split in ["train","valid","test"]:
        for lb in glob.glob(str(BASE/split/"labels"/"*.txt")):
            b = os.path.basename(lb); m = CLIP_RE.search(b)
            if not m: continue
            clip, frame = m.group(1), int(m.group(2))
            boxes=[]
            for ln in Path(lb).read_text().splitlines():
                p=ln.split()
                if len(p)>=5: boxes.append(tuple(map(float,p[1:5])))
            clips[clip][frame]=boxes
    return clips

def iou(a,b):
    ax1,ay1,ax2,ay2=a[0]-a[2]/2,a[1]-a[3]/2,a[0]+a[2]/2,a[1]+a[3]/2
    bx1,by1,bx2,by2=b[0]-b[2]/2,b[1]-b[3]/2,b[0]+b[2]/2,b[1]+b[3]/2
    ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=a[2]*a[3]+b[2]*b[3]-inter
    return inter/ua if ua>0 else 0

def track_clip(frames_dict):
    """그리디 IoU 추적 -> {tid: [(frame, cx, cy)]}"""
    tracks=defaultdict(list); active={}; nid=0
    for frame in sorted(frames_dict):
        boxes=frames_dict[frame]; assigned=set()
        # 매칭
        for tid,last in list(active.items()):
            best,bi=0.3,-1
            for i,bx in enumerate(boxes):
                if i in assigned: continue
                v=iou(last,bx)
                if v>best: best,bi=v,i
            if bi>=0:
                assigned.add(bi); active[tid]=boxes[bi]
                tracks[tid].append((frame,boxes[bi][0],boxes[bi][1]))
            else:
                active[tid]=(last[0],last[1],1e-6,1e-6)  # keep last (coast) but low iou
        # 미매칭 -> 신규
        for i,bx in enumerate(boxes):
            if i in assigned: continue
            nid+=1; active[nid]=bx; tracks[nid].append((frame,bx[0],bx[1]))
    return tracks

def build(clips):
    per_clip_polys={}
    for clip,fd in clips.items():
        tr=track_clip(fd); polys=[]
        for tid,pts in tr.items():
            pts=sorted(pts)
            # 연속 구간만
            seg=[pts[0]]
            for i in range(1,len(pts)):
                if pts[i][0]-pts[i-1][0]==1: seg.append(pts[i])
                else:
                    if len(seg)>=OBS+PRED: polys.append(np.array([[x,y] for _,x,y in seg]))
                    seg=[pts[i]]
            if len(seg)>=OBS+PRED: polys.append(np.array([[x,y] for _,x,y in seg]))
        if polys: per_clip_polys[clip]=polys
    return per_clip_polys

def windows(polys):
    X=[]
    for arr in polys:
        for i in range(len(arr)-(OBS+PRED)+1):
            X.append(arr[i:i+OBS+PRED])
    return X

def zero_vel(obs):
    """정지 기준선: 마지막 관측 위치 유지."""
    return np.repeat(obs[-1][None, :], PRED, axis=0)


def const_vel(obs):
    v=(obs[-1]-obs[0])/(OBS-1); return np.array([obs[-1]+v*(k+1) for k in range(PRED)])
def kalman(obs):
    F=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]],float)
    q,r=1e-4,1e-3
    Q=q*np.eye(4); H=np.array([[1,0,0,0],[0,0,1,0]],float); R=r*np.eye(2)
    x=np.array([obs[0,0],0,obs[0,1],0],float); P=np.diag([r,1e-2,r,1e-2])
    for mz in obs[1:]:
        x=F@x;P=F@P@F.T+Q;y=mz-H@x;S=H@P@H.T+R;K=P@H.T@np.linalg.inv(S);x=x+K@y;P=(np.eye(4)-K@H)@P
    out=[]
    for _ in range(PRED): x=F@x;out.append([x[0],x[2]])
    return np.array(out)
def ade_fde(p,g):
    d=np.linalg.norm((p-g)*IMG,axis=1); return d.mean(),d[-1]  # 640px 환산
def eval_cls(fn,wins):
    A=[]; 
    for w in wins:
        a,f=ade_fde(fn(w[:OBS]),w[OBS:]); A.append((a,f))
    A=np.array(A); return A[:,0].mean(),A[:,1].mean()

def run_lstm(train_w,test_w):
    import torch,torch.nn as nn
    torch.manual_seed(0)
    def prep(wins):
        X=[np.diff(w[:OBS],axis=0) for w in wins]; Y=[w[OBS:]-w[OBS-1] for w in wins]
        return np.array(X,np.float32),np.array(Y,np.float32)
    Xtr,Ytr=prep(train_w); Xte,Yte=prep(test_w)
    sc=float(np.std(Xtr)) or 1.0; Xtr,Ytr,Xte=Xtr/sc,Ytr/sc,Xte/sc
    dev="cuda" if torch.cuda.is_available() else "cpu"
    Xtr=torch.tensor(Xtr,device=dev);Ytr=torch.tensor(Ytr,device=dev)
    class Net(nn.Module):
        def __init__(s,h=128):
            super().__init__(); s.enc=nn.LSTM(2,h,batch_first=True)
            s.head=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,PRED*2))
        def forward(s,x):_,(hn,_)=s.enc(x);return s.head(hn[-1]).view(-1,PRED,2)
    net=Net().to(dev);opt=torch.optim.Adam(net.parameters(),1e-3);L=nn.MSELoss()
    n=len(Xtr);bs=256
    for ep in range(300):
        perm=torch.randperm(n,device=dev)
        for i in range(0,n,bs):
            idx=perm[i:i+bs];opt.zero_grad();loss=L(net(Xtr[idx]),Ytr[idx]);loss.backward();opt.step()
    net.eval()
    import torch as T
    with T.no_grad(): pred=net(T.tensor(Xte,device=dev)).cpu().numpy()*sc
    A=[]
    for i,w in enumerate(test_w):
        p=w[OBS-1]+pred[i]; a,f=ade_fde(p,w[OBS:]); A.append((a,f))
    A=np.array(A); return A[:,0].mean(),A[:,1].mean()

def main():
    clips = load_clip_frames()
    polys = build(clips)
    per_clip = {c: p for c, p in polys.items()}
    all_tracks = [(c, arr) for c, ps in per_clip.items() for arr in ps]
    allw = [w for _, arr in all_tracks for w in windows([arr])]
    net = [np.linalg.norm(w[-1] - w[OBS - 1]) * IMG for w in allw]
    print(f"clips={len(per_clip)}  tracks={len(all_tracks)}  windows={len(allw)}", flush=True)
    print(f"8프레임 순변위: 평균 {np.mean(net):.1f}px  중앙값 {np.median(net):.1f}px  "
          f"(3px 미만 {np.mean(np.array(net) < 3) * 100:.0f}%)", flush=True)

    def report(title, tr, te):
        print(f"\n=== {title} — ADE/FDE px ===", flush=True)
        for nm, fn in [("zero-velocity", zero_vel), ("const-velocity", const_vel), ("kalman", kalman)]:
            a, f = eval_cls(fn, te)
            print(f"  {nm:16s} ADE={a:.1f} FDE={f:.1f}", flush=True)
        if tr:
            a, f = run_lstm(tr, te)
            print(f"  {'lstm':16s} ADE={a:.1f} FDE={f:.1f}", flush=True)

    # 트랙 단위 분할 — 같은 트랙의 겹치는 창이 학습/평가에 섞이지 않게
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(all_tracks))
    cut = int(len(all_tracks) * 0.8)
    tr = [w for i in idx[:cut] for w in windows([all_tracks[i][1]])]
    te = [w for i in idx[cut:] for w in windows([all_tracks[i][1]])]
    report(f"track-level split (train {len(tr)} / test {len(te)} windows)", tr, te)

    # 클립 단위 분할
    cl = sorted(per_clip)
    half = cl[:len(cl) // 2]
    trw = [w for c in half for w in windows(per_clip[c])]
    tew = [w for c in cl[len(cl) // 2:] for w in windows(per_clip[c])]
    report(f"cross-clip (train {len(half)} clips / test {len(cl) - len(half)})", trw, tew)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
