import sys, numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); import survey_paths as sp
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib"); import aru_py_logger
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
survey, bags_npy, out, label = sys.argv[1:5]
mono=sp.monos(survey); laser=mono/"laser.monolithic"
if not laser.exists(): laser=mono/"monolithics"/"laser.monolithic"
incs=sp._read_increments(mono/"transform_lio.monolithic"); inc_ts=np.asarray([t for t,_ in incs],dtype=np.int64)
ab=np.empty((len(incs),4,4)); ab[0]=np.eye(4)
for k in range(1,len(incs)): ab[k]=ab[k-1]@incs[k][1]
def pose_at(t):
    j=int(np.searchsorted(inc_ts,t))
    if j<=0 or j>=len(incs): return None
    r=float(inc_ts[j]-t)/float(inc_ts[j]-inc_ts[j-1]); part=sp._slerp_partial(incs[j][1],r); return ab[j-1]@(incs[j][1]@np.linalg.inv(part))
bags=np.load(bags_npy)
# window: a 16 m stretch of the first lane (scans 150-450)
lg=aru_py_logger.LaserLogger(str(laser),False); W=[]; traj=[]; n=0
while n<450:
    try: r=lg.read_from_file()
    except Exception: break
    pts=[x for x in r if hasattr(x,"shape")][0]; ts=[x for x in r if isinstance(x,int)][0]; n+=1
    if n<150: continue
    T=pose_at(ts)
    if T is None: continue
    P=pts[:,:3].astype(np.float64); P=P[(np.linalg.norm(P,axis=1)<6)&(np.linalg.norm(P,axis=1)>0.6)]
    Q=(T[:3,:3]@P.T).T+T[:3,3]; W.append(Q[::3]); traj.append(T[:3,3])
W=np.vstack(W); traj=np.array(traj)
# local ground = 5th pct of z in the window; keep points 0.05-1.5 m above it, colour by height
g=np.percentile(W[:,2],5); h=W[:,2]-g; m=(h>0.05)&(h<1.5); W=W[m]; h=h[m]
x0,x1=traj[:,0].min()-4,traj[:,0].max()+4; y0,y1=traj[:,1].min()-4,traj[:,1].max()+4
fig,ax=plt.subplots(figsize=(14,9)); sc=ax.scatter(W[:,0],W[:,1],c=h,s=0.4,cmap="viridis",vmin=0,vmax=1.2); plt.colorbar(sc,ax=ax,label="height above local ground (m)")
ax.plot(traj[:,0],traj[:,1],"w-",lw=1.5,label="sensor path (KISS-ICP)")
inb=(bags[:,0]>x0)&(bags[:,0]<x1)&(bags[:,1]>y0)&(bags[:,1]<y1); ax.scatter(bags[inb,0],bags[inb,1],s=60,facecolors="none",edgecolors="red",lw=1.5,label=f"ledger bag positions ({inb.sum()})")
ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.set_aspect("equal"); ax.legend(loc="upper right"); ax.set_title(f"{label}: accumulated lidar (scans 150-450) in the KISS-ICP world vs ledger bag positions; sensor above ground {np.median(traj[:,2])-g:.2f} m")
plt.tight_layout(); plt.savefig(out,dpi=110); print("saved",out,"points",len(W),"sensor above ground %.2f"%(np.median(traj[:,2])-g))
