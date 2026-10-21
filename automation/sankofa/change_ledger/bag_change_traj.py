import json, sys, numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); import survey_paths as sp
from offline_vtr import monos
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
K="/home/paperspace/data/klapmuts"; D=K+"/dec_2025_ten_rows"; A=K+"/apr_2026_zed"; O=D+"/experimental/azalai/stpg_dec_apr_20261021"
L=json.load(open(D+"/experimental/sankofa/klapmuts_ledger_v5.json")); r=L["registration_A_to_B"]; th=np.radians(r["yaw_deg"]); Rz=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); tt=np.array(r["t"])
ch=np.load(O+"/bag_change_ledger.npz",allow_pickle=True); Ab,Bb=ch["A"],ch["B"]; pairs=ch["pairs"]; moved=ch["moved"]; added=ch["added"].astype(int); removed=ch["removed"].astype(int)
l2c=sp.load_l2c(D); Rt=l2c[:3,:3].T
def traj_laser2d(survey):
    ts,c2w=monos.keyframe_poses(survey); ok=~np.isnan(c2w[:,0,0]); P=(Rt@c2w[ok][:,:3,3].T).T; return P[:,:2]
TD=traj_laser2d(D)@Rz.T+tt     # December LO world -> April frame
TA=traj_laser2d(A)             # April laser world == ledger April frame (checked: bags 0.93 m from the lane path)
fig,ax=plt.subplots(figsize=(12,13))
ax.plot(TD[:,0],TD[:,1],"-",color="#3b6ea5",lw=1.2,alpha=.9,label="December keyframe path (KISS-ICP, via ledger registration)")
ax.plot(TA[:,0],TA[:,1],"-",color="#d35400",lw=1.2,alpha=.8,label="April keyframe path (LIO)")
ax.scatter(Ab[:,0],Ab[:,1],s=12,c="#9aa5ad",label=f"December bags ({len(Ab)})")
pi=pairs[:,1].astype(int); ax.scatter(Bb[pi,0],Bb[pi,1],s=10,c="#27ae60",label=f"April bags matched ({len(pairs)})")
for i,j,dd in moved: i=int(i); j=int(j); ax.annotate("",xy=Bb[j],xytext=Ab[i],arrowprops=dict(arrowstyle="->",color="#e67e22",lw=1.3))
ax.scatter(Bb[moved[:,1].astype(int),0],Bb[moved[:,1].astype(int),1],s=30,c="#e67e22",label=f"moved 0.5-1.5 m ({len(moved)})")
ax.scatter(Bb[added,0],Bb[added,1],s=46,marker="^",c="#c0392b",label=f"April only ({len(added)})")
ax.scatter(Ab[removed,0],Ab[removed,1],s=70,facecolors="none",edgecolors="#2c3e50",lw=1.4,label=f"December only ({len(removed)})")
for k,(x,y) in enumerate(Bb[added]): ax.annotate(str(int(added[k])),(x,y),fontsize=6,color="#c0392b",xytext=(3,3),textcoords="offset points")
for k,(x,y) in enumerate(Ab[removed]): ax.annotate(str(int(removed[k])),(x,y),fontsize=6,color="#2c3e50",xytext=(3,-7),textcoords="offset points")
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper right",fontsize=8); ax.set_xlabel("across rows (m, April frame)"); ax.set_ylabel("along rows (m)")
ax.set_title("Grow bags Dec -> Apr with both keyframe paths (ids = ledger indices)")
plt.tight_layout(); plt.savefig(O+"/bag_change_traj.png",dpi=120); print("saved")
# distance of each class to the nearest lane path (either survey): bags far from any path are suspect
from scipy.spatial import cKDTree
t=cKDTree(np.vstack([TD,TA]))
for name,P in [("matched Apr",Bb[pi]),("April only",Bb[added]),("December only",Ab[removed]),("moved (Apr pos)",Bb[moved[:,1].astype(int)])]:
    d,_=t.query(P); print(f"{name}: distance to nearest path median {np.median(d):.2f} m, > 2.5 m: {int((d>2.5).sum())} of {len(P)}")
