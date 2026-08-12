import os, re, glob
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE=Path("datasets/overhead-person-v3"); OUT=Path(__file__).resolve().parent.parent/"assets"; OUT.mkdir(exist_ok=True)
CLIP_RE=re.compile(r'(.+?)[_-](\d{6,8})_jpg\.rf\.'); OBS,PRED=8,8
clips=defaultdict(dict); imgpath=defaultdict(dict)
for split in ["train","valid","test"]:
    for lb in glob.glob(str(BASE/split/"labels"/"*.txt")):
        b=os.path.basename(lb); m=CLIP_RE.search(b)
        if not m: continue
        clip,fr=m.group(1),int(m.group(2))
        clips[clip][fr]=[tuple(map(float,l.split()[1:5])) for l in Path(lb).read_text().splitlines() if len(l.split())>=5]
        imgpath[clip][fr]=str(BASE/split/"images"/(b[:-4]+".jpg"))
def iou(a,b):
    ax1,ay1,ax2,ay2=a[0]-a[2]/2,a[1]-a[3]/2,a[0]+a[2]/2,a[1]+a[3]/2
    bx1,by1,bx2,by2=b[0]-b[2]/2,b[1]-b[3]/2,b[0]+b[2]/2,b[1]+b[3]/2
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1)); inter=iw*ih
    ua=a[2]*a[3]+b[2]*b[3]-inter; return inter/ua if ua>0 else 0
def track(fd):
    tracks=defaultdict(list); active={}; nid=0
    for fr in sorted(fd):
        boxes=fd[fr]; used=set()
        for tid,last in list(active.items()):
            best,bi=0.3,-1
            for i,bx in enumerate(boxes):
                if i in used: continue
                v=iou(last,bx)
                if v>best: best,bi=v,i
            if bi>=0: used.add(bi); active[tid]=boxes[bi]; tracks[tid].append((fr,)+boxes[bi])
        for i,bx in enumerate(boxes):
            if i in used: continue
            nid+=1; active[nid]=bx; tracks[nid].append((fr,)+bx)
    return tracks
def const_vel(o):
    v=(o[-1]-o[0])/(OBS-1); return np.array([o[-1]+v*(k+1) for k in range(PRED)])

clip="cam_1"; fd=clips[clip]; tr=track(fd)
mid=sorted(fd)[len(fd)//2]; bg=cv2.cvtColor(cv2.imread(imgpath[clip][mid]),cv2.COLOR_BGR2RGB); H,W=bg.shape[:2]

# 대표 윈도우: 부드러운 움직임(점프 없음) + 실제 이동 있음
cands=[]
for tid,pts in tr.items():
    arr=np.array([[p[1],p[2]] for p in sorted(pts)])
    if len(arr)<OBS+PRED: continue
    for i in range(len(arr)-(OBS+PRED)+1):
        w=arr[i:i+OBS+PRED]
        step=np.linalg.norm(np.diff(w,axis=0),axis=1)
        if step.max()>0.04: continue           # 프레임간 점프(ID스위치) 제외
        move=np.linalg.norm(w[-1]-w[0])
        if move<0.03: continue                  # 거의 정지 제외
        err=np.linalg.norm((const_vel(w[:OBS])[-1]-w[-1])*640)
        cands.append((err,move,tid,w))
cands.sort(key=lambda x:x[0])
errs=[c[0] for c in cands]
print(f"clean windows={len(cands)}  median FDE={np.median(errs):.1f}px  p90={np.percentile(errs,90):.1f}px")
# 중앙값 근처 2개 + 어려운 케이스 1개(정직하게)
med=len(cands)//2
sel=[cands[med-1], cands[med+1], cands[int(len(cands)*0.95)]]
titles=["typical","typical","hard case (turning)"]
fig,axes=plt.subplots(1,3,figsize=(15,5.2))
for ax,(err,mv,tid,w),tt in zip(axes,sel,titles):
    ax.imshow(bg,alpha=0.45)
    obs,gt=w[:OBS],w[OBS:]; pr=const_vel(obs)
    ax.plot(obs[:,0]*W,obs[:,1]*H,'o-',color="#3b82f6",lw=2.5,ms=5,label="observed (past 8 frames)")
    ax.plot(np.r_[obs[-1,0],gt[:,0]]*W,np.r_[obs[-1,1],gt[:,1]]*H,'--',color="k",lw=2.5,label="actual future")
    ax.plot(np.r_[obs[-1,0],pr[:,0]]*W,np.r_[obs[-1,1],pr[:,1]]*H,'s-',color="#e45756",lw=2,ms=4,label="predicted")
    x0=min(obs[:,0].min(),gt[:,0].min(),pr[:,0].min())*W; x1=max(obs[:,0].max(),gt[:,0].max(),pr[:,0].max())*W
    y0=min(obs[:,1].min(),gt[:,1].min(),pr[:,1].min())*H; y1=max(obs[:,1].max(),gt[:,1].max(),pr[:,1].max())*H
    pad=90; ax.set_xlim(max(0,x0-pad),min(W,x1+pad)); ax.set_ylim(min(H,y1+pad),max(0,y0-pad))
    ax.set_title(f"{tt} — final error {err:.0f}px"); ax.axis("off")
axes[0].legend(loc="upper left",fontsize=8.5)
plt.suptitle("Predicting a person's next moves: past 8 frames -> next 8 frames (constant-velocity)",fontsize=12)
plt.tight_layout(); plt.savefig(OUT/"prediction_example.png",dpi=120); plt.close()
print("saved")
