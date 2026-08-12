"""시나리오 before/after: 예측 시점 프레임 vs 8프레임 뒤 실제 프레임."""
import os,re,glob
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _f in ["C:/Windows/Fonts/malgun.ttf"]:
    if os.path.exists(_f): fm.fontManager.addfont(_f); matplotlib.rcParams["font.family"]="Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.pyplot as plt

BASE=Path("datasets/overhead-person-v3"); OUT=Path(__file__).resolve().parent.parent/"assets"; OUT.mkdir(exist_ok=True)
CLIP_RE=re.compile(r'(.+?)[_-](\d{6,8})_jpg\.rf\.'); OBS,PRED=8,8

clips=defaultdict(dict); imgp=defaultdict(dict)
for sp in ["train","valid","test"]:
    for lb in glob.glob(str(BASE/sp/"labels"/"*.txt")):
        b=os.path.basename(lb); m=CLIP_RE.search(b)
        if not m: continue
        c,f=m.group(1),int(m.group(2))
        clips[c][f]=[tuple(map(float,l.split()[1:5])) for l in Path(lb).read_text().splitlines() if len(l.split())>=5]
        imgp[c][f]=str(BASE/sp/"images"/(b[:-4]+".jpg"))

def iou(a,b):
    ax1,ay1,ax2,ay2=a[0]-a[2]/2,a[1]-a[3]/2,a[0]+a[2]/2,a[1]+a[3]/2
    bx1,by1,bx2,by2=b[0]-b[2]/2,b[1]-b[3]/2,b[0]+b[2]/2,b[1]+b[3]/2
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1)); it=iw*ih
    ua=a[2]*a[3]+b[2]*b[3]-it; return it/ua if ua>0 else 0

def track(fd):
    T=defaultdict(list); act={}; nid=0
    for fr in sorted(fd):
        bx=fd[fr]; used=set()
        for tid,last in list(act.items()):
            best,bi=0.3,-1
            for i,b in enumerate(bx):
                if i in used: continue
                v=iou(last,b)
                if v>best: best,bi=v,i
            if bi>=0: used.add(bi); act[tid]=bx[bi]; T[tid].append((fr,)+bx[bi])
        for i,b in enumerate(bx):
            if i in used: continue
            nid+=1; act[nid]=b; T[nid].append((fr,)+b)
    return T

def cv_pred(o):
    v=(o[-1]-o[0])/(OBS-1); return np.array([o[-1]+v*(k+1) for k in range(PRED)])

clip="cam_1"; fd=clips[clip]; T=track(fd)
best=None
for tid,pts in T.items():
    pts=sorted(pts); arr=np.array([[p[1],p[2]] for p in pts]); frs=[p[0] for p in pts]
    if len(arr)<OBS+PRED: continue
    for i in range(len(arr)-(OBS+PRED)+1):
        if frs[i+OBS+PRED-1]-frs[i]!=OBS+PRED-1: continue
        w=arr[i:i+OBS+PRED]
        if np.linalg.norm(np.diff(w,axis=0),axis=1).max()>0.03: continue
        move=np.linalg.norm(w[-1]-w[0])
        err=np.linalg.norm((cv_pred(w[:OBS])[-1]-w[-1])*640)
        score=move*100-err/20
        if best is None or score>best[0]:
            best=(score,tid,frs[i],w,err,move,[p[3:5] for p in pts[i:i+OBS+PRED]])
_,tid,f0,w,err,move,sizes=best
obs,gt=w[:OBS],w[OBS:]; pr=cv_pred(obs)
f_now=f0+OBS-1; f_after=f0+OBS+PRED-1
v=(obs[-1]-obs[0])/(OBS-1)*640
print(f"id={tid} predict_at_frame={f_now} after_frame={f_after} err={err:.0f}px move={move*640:.0f}px")

im_now=cv2.cvtColor(cv2.imread(imgp[clip][f_now]),cv2.COLOR_BGR2RGB)
im_aft=cv2.cvtColor(cv2.imread(imgp[clip][f_after]),cv2.COLOR_BGR2RGB)
H,W=im_now.shape[:2]
bw,bh=sizes[-1]

# 공통 확대 영역
xs=np.r_[obs[:,0],gt[:,0],pr[:,0]]*W; ys=np.r_[obs[:,1],gt[:,1],pr[:,1]]*H
pad=130; X0,X1=max(0,xs.min()-pad),min(W,xs.max()+pad); Y0,Y1=max(0,ys.min()-pad),min(H,ys.max()+pad)

fig,axes=plt.subplots(1,2,figsize=(14,6))
# --- BEFORE: 예측 시점 ---
ax=axes[0]; ax.imshow(im_now)
ax.add_patch(plt.Rectangle(((obs[-1,0]-bw/2)*W,(obs[-1,1]-bh/2)*H),bw*W,bh*H,fill=False,ec="#3b82f6",lw=3))
ax.plot(obs[:,0]*W,obs[:,1]*H,'o-',color="#3b82f6",lw=3,ms=6,label="관찰한 경로")
ax.annotate("",xy=(pr[-1,0]*W,pr[-1,1]*H),xytext=(obs[-1,0]*W,obs[-1,1]*H),
            arrowprops=dict(arrowstyle="-|>",color="#e45756",lw=3,ls="--"))
ax.plot(pr[-1,0]*W,pr[-1,1]*H,'*',color="#e45756",ms=26,label="예측 위치")
ax.set_title(f"예측 시점 · frame {f_now}",fontsize=13)
ax.legend(loc="lower left",fontsize=9.5)
# --- AFTER: 8프레임 뒤 ---
ax=axes[1]; ax.imshow(im_aft)
ax.plot(pr[-1,0]*W,pr[-1,1]*H,'*',color="#e45756",ms=26,label="예측 위치")
ax.add_patch(plt.Rectangle(((gt[-1,0]-bw/2)*W,(gt[-1,1]-bh/2)*H),bw*W,bh*H,fill=False,ec="k",lw=3))
ax.plot(gt[-1,0]*W,gt[-1,1]*H,'X',color="k",ms=15,label="실제 위치")
ax.plot([pr[-1,0]*W,gt[-1,0]*W],[pr[-1,1]*H,gt[-1,1]*H],'-',color="#7a1c1f",lw=2)
ax.annotate(f"오차 {err:.0f}px",xy=((pr[-1,0]+gt[-1,0])/2*W,(pr[-1,1]+gt[-1,1])/2*H),
            xytext=(10,-55),textcoords="offset points",fontsize=12,color="#7a1c1f",weight="bold",
            arrowprops=dict(arrowstyle="->",color="#7a1c1f"))
ax.set_title(f"8프레임 뒤 · frame {f_after}",fontsize=13)
ax.legend(loc="lower left",fontsize=9.5)
for ax in axes:
    ax.set_xlim(X0,X1); ax.set_ylim(Y1,Y0); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"scenario.png",dpi=125,bbox_inches="tight",pad_inches=0.25); plt.close()
print("saved scenario.png")
