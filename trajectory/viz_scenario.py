"""실제 데이터에서 '이 사람이 이 방향으로 가려던 걸 예측' 시나리오 1개 추출."""
import os,re,glob
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
matplotlib.rcParams["font.family"]="Malgun Gothic"
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
    arr=np.array([[p[1],p[2]] for p in sorted(pts)]); frs=[p[0] for p in sorted(pts)]
    if len(arr)<OBS+PRED: continue
    for i in range(len(arr)-(OBS+PRED)+1):
        w=arr[i:i+OBS+PRED]
        step=np.linalg.norm(np.diff(w,axis=0),axis=1)
        if step.max()>0.03: continue
        move=np.linalg.norm(w[-1]-w[0])
        err=np.linalg.norm((cv_pred(w[:OBS])[-1]-w[-1])*640)
        # 이동 크고 오차 작은(대표적 성공) 케이스
        score=move*100-err/20
        if best is None or score>best[0]: best=(score,tid,frs[i],w,err,move)
_,tid,f0,w,err,move=best
obs,gt=w[:OBS],w[OBS:]; pr=cv_pred(obs)
px=lambda a:(a*640)
v=(obs[-1]-obs[0])/(OBS-1)*640
print(f"clip={clip} id={tid} start_frame={f0}")
print(f"관찰 8프레임 좌표(px): "+" -> ".join(f"({x*640:.0f},{y*640:.0f})" for x,y in obs))
print(f"추정 속도: {v[0]:+.1f}, {v[1]:+.1f} px/frame  (속력 {np.hypot(*v):.1f} px/frame)")
print(f"방향: {'오른쪽' if v[0]>0 else '왼쪽'}{'+아래' if v[1]>0 else '+위'}")
print(f"예측 8프레임 뒤: ({pr[-1][0]*640:.0f},{pr[-1][1]*640:.0f})")
print(f"실제 8프레임 뒤: ({gt[-1][0]*640:.0f},{gt[-1][1]*640:.0f})")
print(f"오차: {err:.0f}px  (이동거리 {move*640:.0f}px)")

mid=sorted(fd)[len(fd)//2]; bg=cv2.cvtColor(cv2.imread(imgp[clip][mid]),cv2.COLOR_BGR2RGB); H,W=bg.shape[:2]
fig,ax=plt.subplots(figsize=(9,6)); ax.imshow(bg,alpha=0.5)
ax.plot(obs[:,0]*W,obs[:,1]*H,'o-',color="#3b82f6",lw=3,ms=7,label=f"① 관찰: 과거 8프레임 (속도 {np.hypot(*v):.0f} px/frame)")
ax.annotate("",xy=(obs[-1,0]*W,obs[-1,1]*H),xytext=(obs[-3,0]*W,obs[-3,1]*H),
            arrowprops=dict(arrowstyle="-|>",color="#3b82f6",lw=3))
ax.plot(np.r_[obs[-1,0],pr[:,0]]*W,np.r_[obs[-1,1],pr[:,1]]*H,'s--',color="#e45756",lw=2.5,ms=6,label="② 예측: 이 방향으로 계속 갈 것")
ax.plot(np.r_[obs[-1,0],gt[:,0]]*W,np.r_[obs[-1,1],gt[:,1]]*H,'-',color="k",lw=2.5,label="③ 실제로 간 경로")
ax.plot(gt[-1,0]*W,gt[-1,1]*H,'*',color="k",ms=18)
ax.plot(pr[-1,0]*W,pr[-1,1]*H,'*',color="#e45756",ms=18)
ax.annotate(f"오차 {err:.0f}px",xy=((pr[-1,0]+gt[-1,0])/2*W,(pr[-1,1]+gt[-1,1])/2*H),
            xytext=(20,-40),textcoords="offset points",fontsize=11,color="#7a1c1f",
            arrowprops=dict(arrowstyle="->",color="#7a1c1f"))
x0=min(obs[:,0].min(),gt[:,0].min(),pr[:,0].min())*W; x1=max(obs[:,0].max(),gt[:,0].max(),pr[:,0].max())*W
y0=min(obs[:,1].min(),gt[:,1].min(),pr[:,1].min())*H; y1=max(obs[:,1].max(),gt[:,1].max(),pr[:,1].max())*H
p=110; ax.set_xlim(max(0,x0-p),min(W,x1+p)); ax.set_ylim(min(H,y1+p),max(0,y0-p))
ax.legend(loc="upper left",fontsize=10); ax.axis("off")
dirtxt=("오른쪽" if v[0]>0 else "왼쪽")
ax.set_title(f"사람 이동 시나리오: {dirtxt}으로 걷던 사람이 계속 갈 것을 예측 (오차 {err:.0f}px)",fontsize=12)
plt.tight_layout(); plt.savefig(OUT/"scenario.png",dpi=125); plt.close()
print("saved scenario.png")
