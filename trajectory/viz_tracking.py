"""오버헤드 데이터셋 -> 추적 -> 궤적 -> 예측 을 실제 이미지로 시각화."""
import os, re, glob
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("datasets/overhead-person-v3")
OUT = Path(__file__).resolve().parent.parent/"assets"; OUT.mkdir(exist_ok=True)
CLIP_RE = re.compile(r'(.+?)[_-](\d{6,8})_jpg\.rf\.')
OBS, PRED = 8, 8

# --- 클립별 프레임 로드 (이미지경로 + 박스) ---
clips = defaultdict(dict); imgpath = defaultdict(dict)
for split in ["train","valid","test"]:
    for lb in glob.glob(str(BASE/split/"labels"/"*.txt")):
        b=os.path.basename(lb); m=CLIP_RE.search(b)
        if not m: continue
        clip,fr=m.group(1),int(m.group(2))
        boxes=[tuple(map(float,l.split()[1:5])) for l in Path(lb).read_text().splitlines() if len(l.split())>=5]
        clips[clip][fr]=boxes
        imgpath[clip][fr]=str(BASE/split/"images"/(b[:-4]+".jpg"))

def iou(a,b):
    ax1,ay1,ax2,ay2=a[0]-a[2]/2,a[1]-a[3]/2,a[0]+a[2]/2,a[1]+a[3]/2
    bx1,by1,bx2,by2=b[0]-b[2]/2,b[1]-b[3]/2,b[0]+b[2]/2,b[1]+b[3]/2
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1)); inter=iw*ih
    ua=a[2]*a[3]+b[2]*b[3]-inter
    return inter/ua if ua>0 else 0

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
            if bi>=0:
                used.add(bi); active[tid]=boxes[bi]; tracks[tid].append((fr,)+boxes[bi])
        for i,bx in enumerate(boxes):
            if i in used: continue
            nid+=1; active[nid]=bx; tracks[nid].append((fr,)+bx)
    return tracks

# 궤적이 풍부한 클립 선택
best_clip=None; best_score=0
for c,fd in clips.items():
    tr=track(fd); score=sum(1 for t in tr.values() if len(t)>=20)
    if score>best_score: best_score, best_clip, best_tr = score, c, tr
print("clip:",best_clip,"long tracks:",best_score)
tr=best_tr; fd=clips[best_clip]
mid=sorted(fd)[len(fd)//2]
img=cv2.imread(imgpath[best_clip][mid]); H,W=img.shape[:2]
bg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# ============ 그림 1: 프레임 -> ID -> 궤적 ============
fig,axes=plt.subplots(1,3,figsize=(15,5.2))
frames3=[sorted(fd)[len(fd)//3], sorted(fd)[len(fd)//3+8], sorted(fd)[len(fd)//3+16]]
colors=plt.cm.tab10(np.linspace(0,1,10))
for ax,fr in zip(axes[:2],frames3[:2]):
    im=cv2.cvtColor(cv2.imread(imgpath[best_clip][fr]),cv2.COLOR_BGR2RGB); ax.imshow(im)
    for tid,pts in tr.items():
        d={p[0]:p[1:] for p in pts}
        if fr in d:
            cx,cy,w,h=d[fr]; c=colors[tid%10]
            ax.add_patch(plt.Rectangle(((cx-w/2)*W,(cy-h/2)*H),w*W,h*H,fill=False,ec=c,lw=2))
            ax.text((cx-w/2)*W,(cy-h/2)*H-4,f"id {tid}",color=c,fontsize=9,weight="bold")
    ax.set_title(f"frame {fr}"); ax.axis("off")
ax=axes[2]; ax.imshow(bg,alpha=0.55)
for tid,pts in tr.items():
    if len(pts)<15: continue
    xs=[p[1]*W for p in pts]; ys=[p[2]*H for p in pts]; c=colors[tid%10]
    ax.plot(xs,ys,'-',color=c,lw=2.2); ax.plot(xs[-1],ys[-1],'o',color=c,ms=6)
ax.set_title("tracked trajectories (x,y over time)"); ax.axis("off")
plt.suptitle("YOLO dataset frames -> IoU matching gives IDs -> each person becomes an (x,y) trajectory",fontsize=12)
plt.tight_layout(); plt.savefig(OUT/"how_trajectories.png",dpi=120); plt.close()
print("saved how_trajectories.png")

# ============ 그림 2: 예측 예시 ============
def const_vel(o):
    v=(o[-1]-o[0])/(OBS-1); return np.array([o[-1]+v*(k+1) for k in range(PRED)])
cands=[]
for tid,pts in tr.items():
    arr=np.array([[p[1],p[2]] for p in sorted(pts)])
    if len(arr)>=OBS+PRED:
        for i in range(0,len(arr)-(OBS+PRED)+1,4):
            w=arr[i:i+OBS+PRED]
            move=np.linalg.norm(w[-1]-w[0])
            cands.append((move,tid,w))
cands.sort(reverse=True,key=lambda x:x[0])
sel=cands[:3]
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,(mv,tid,w) in zip(axes,sel):
    ax.imshow(bg,alpha=0.5)
    obs,gt=w[:OBS],w[OBS:]; pr=const_vel(obs)
    ax.plot(obs[:,0]*W,obs[:,1]*H,'o-',color="#3b82f6",lw=2.5,ms=5,label="observed (8 frames)")
    ax.plot(np.r_[obs[-1,0],gt[:,0]]*W,np.r_[obs[-1,1],gt[:,1]]*H,'--',color="k",lw=2.5,label="actual future")
    ax.plot(np.r_[obs[-1,0],pr[:,0]]*W,np.r_[obs[-1,1],pr[:,1]]*H,'s-',color="#e45756",lw=2,ms=4,label="predicted")
    err=np.linalg.norm((pr[-1]-gt[-1])*640)
    ax.set_title(f"id {tid} — final error {err:.0f}px"); ax.axis("off")
axes[0].legend(loc="upper left",fontsize=9)
plt.suptitle("Predicting where a person goes next: past 8 frames -> next 8 frames",fontsize=12)
plt.tight_layout(); plt.savefig(OUT/"prediction_example.png",dpi=120); plt.close()
print("saved prediction_example.png")
