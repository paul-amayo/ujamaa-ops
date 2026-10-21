import numpy as np, sys
sys.path.insert(0,"/home/paperspace/code/aru_sil_core/src/scripts"); import survey_paths as sp
from scipy.spatial import cKDTree
D="/home/paperspace/data/klapmuts/dec_2025_ten_rows"; A="/home/paperspace/data/klapmuts/apr_2026_zed"
tsD=np.asarray(sp.kf_timestamps(D),dtype=np.int64); tsA=np.asarray(sp.kf_timestamps(A),dtype=np.int64)
zd=np.load(D+"/experimental/ins_enu_abs.npz"); za=np.load(A+"/experimental/ins/ins_at_images.npz")
print("dec ins ts range vs kf:", zd["ts_ms"][[0,-1]], tsD[[0,-1]]); print("apr ins ts range vs kf:", za["ts_ms"][[0,-1]], tsA[[0,-1]], "clock_offset_ms", za["clock_offset_ms"])
def at(ts_src, xyz, ts_q, off=0):
    ts_src=ts_src+off; o=np.argsort(ts_src); ts_src, xyz = ts_src[o], xyz[o]
    out=np.column_stack([np.interp(ts_q, ts_src, xyz[:,i]) for i in range(2)])
    inside=(ts_q>=ts_src[0])&(ts_q<=ts_src[-1]); return out, inside
PD,inD=at(zd["ts_ms"],zd["xyz"],tsD)
best=None
for off in [0, int(za["clock_offset_ms"]), -int(za["clock_offset_ms"])]:
    PA,inA=at(za["ts_ms"],za["xyz_enu"],tsA,off); print(f"apr offset {off}: kf inside {inA.sum()}/{len(tsA)}")
    if best is None or inA.sum()>best[2].sum(): best=(off,PA,inA)
off,PA,inA=best; print("using apr offset",off)
# heading from ENU velocity
def heading(P):
    v=np.gradient(P,axis=0); n=np.linalg.norm(v,axis=1,keepdims=True)+1e-9; return v/n
HD, HA = heading(PD), heading(PA)
# translation (+ small yaw) ICP, trimmed; init from centroids
Q=PD[inD]; tree=cKDTree(Q); P=PA[inA]
def cost(yaw,t):
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]); d,_=tree.query(P@R.T+t); return np.median(d)
cands=[]
for yaw in np.deg2rad(np.arange(-3,3.01,0.5)):
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]); t0=Q.mean(0)-(P@R.T).mean(0)
    for dx in np.arange(-8,8.1,1.0):
        for dz in np.arange(-8,8.1,2.0):
            t=t0+[dx,dz]; cands.append((cost(yaw,t),yaw,t))
cands.sort(key=lambda c:c[0]); c,yaw,t=cands[0]
for _ in range(30):
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]); X=P@R.T+t; d,idx=tree.query(X); keep=d<max(2*np.median(d),0.3)
    a,b=P[keep],Q[idx[keep]]; ca,cb=a.mean(0),b.mean(0); U,_,Vt=np.linalg.svd((a-ca).T@(b-cb)); Rn=Vt.T@U.T
    if np.linalg.det(Rn)<0: Rn=Vt.T@np.diag([1,-1])@U.T
    yaw=np.clip(np.arctan2(Rn[1,0],Rn[0,0]),np.radians(-3),np.radians(3)); Rn=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]); t=cb-ca@Rn.T
print(f"ENU alignment apr->dec: yaw {np.degrees(yaw):.2f} deg, t {np.round(t,2)}, median NN {cost(yaw,t):.2f} m; second-best candidate cost {cands[1][0]:.2f} at t {np.round(cands[1][2],1)}")
R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]); PA2=PA@R.T+t; HA2=HA@R.T
# lanes: rotate so lane direction = principal axis
ok=inD; X=PD[ok]-PD[ok].mean(0); U,S,Vt=np.linalg.svd(X,full_matrices=False); e=Vt[0]  # along-lane
across=np.array([-e[1],e[0]])
def lanes(P,H,label):
    s=P@across; al=H@e; out=[]; i=0; n=len(P); along=np.abs(al)>0.8
    while i<n:
        if not along[i]: i+=1; continue
        j=i
        while j<n and along[j] and np.sign(al[j])==np.sign(al[i]): j+=1
        if j-i>40: out.append((round(float(np.median(s[i:j])),1), int(np.sign(np.mean(al[i:j]))), j-i))
        i=j
    print(label,"lanes (across, dir, n):",out); return out
Ld=lanes(PD[inD],HD[inD],"Dec ENU"); La=lanes(PA2[inA],HA2[inA],"Apr ENU->Dec")
for xa,da,na in La:
    m=[(xd,dd) for xd,dd,nd in Ld if abs(xd-xa)<1.5]; print("  Apr lane %5.1f dir %+d <-> Dec %s %s"%(xa,da,m,"OPPOSITE" if m and m[0][1]!=da else ("same" if m else "no Dec lane")))
# per-keyframe correspondence: nearest Dec kf with same heading
treeD=cKDTree(PD[inD]); idxD=np.flatnonzero(inD); jstar=np.full(len(tsA),-1); dist=np.full(len(tsA),np.nan)
same1=opp1=0
for k in np.flatnonzero(inA):
    d,i=treeD.query(PA2[k],k=8); cos=HD[idxD[i]]@HA2[k]; g=(cos>0.5)
    if g.any(): jstar[k]=idxD[i[g][0]]; dist[k]=d[g][0]
    same1+=bool(((d<=1.0)&(cos>0.5)).any()); opp1+=bool(((d<=1.0)&(cos<-0.5)).any())
n=int(inA.sum()); print(f"April kf {n}: same-heading Dec kf within 1 m {same1} ({100*same1/n:.0f}%), opposite within 1 m {opp1} ({100*opp1/n:.0f}%); correspondence found for {int((jstar>=0).sum())}, dist median {np.nanmedian(dist):.2f} m")
np.savez("/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/enu_correspondence_apr_to_dec.npz", jstar=jstar, dist=dist, yaw_deg=np.degrees(yaw), t=t, apr_clock_offset_ms=off, PD_enu=PD, PA_enu_in_dec=PA2)
print("saved enu_correspondence_apr_to_dec.npz")
