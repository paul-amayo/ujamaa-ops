"""Image pairs for the largest height changes: December keyframe seeing the Dec bag, April keyframe seeing its
partner (chosen 2-5 m before the bag, same heading), bag position projected with the lidar ground height."""
import json, sys, numpy as np, cv2
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); import survey_paths as sp
from offline_vtr import monos
K="/home/paperspace/data/klapmuts"; D=K+"/dec_2025_ten_rows"; A=K+"/apr_2026_zed"; O=D+"/experimental/azalai/stpg_dec_apr_20261021"
L=json.load(open(D+"/experimental/sankofa/klapmuts_ledger_v5.json")); r=L["registration_A_to_B"]; th=np.radians(r["yaw_deg"]); Rz=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); tt=np.array(r["t"])
Adec=(np.array(L["A_positions_apr_frame"])-tt)@Rz; B=np.array(L["B_positions"])
hd=np.load(O+"/bag_heights_dec.npz"); ha=np.load(O+"/bag_heights_apr.npz")
l2c=sp.load_l2c(D); R=l2c[:3,:3]; Kmat,W,H=monos.intrinsics(A)
tsA,apr=monos.keyframe_poses(A); tsD,dec=monos.keyframe_poses(D)
stA,stD=monos.survey_streams(A),monos.survey_streams(D); kfA,kfD=monos.image_stamps(stA["kf"]),monos.image_stamps(stD["kf"])
def view(c2w_all, p):
    best=None
    for k in np.flatnonzero(~np.isnan(c2w_all[:,0,0])):
        v=np.linalg.inv(c2w_all[k])@np.array([*p,1.0])
        if 2.0<=v[2]<=5.0 and abs(v[0])<=v[2]*0.7:
            s=abs(v[2]-3.0)+0.2*abs(v[0])
            if best is None or s<best[0]: best=(s,int(k))
    return None if best is None else best[1]
def proj(c2w,p):
    v=np.linalg.inv(c2w)@np.array([*p,1.0]); u=Kmat@v[:3]; return (int(u[0]/u[2]),int(u[1]/u[2])) if v[2]>0.3 else None
ext=json.load(open(O+"/growth_extremes.json")); tiles=[]
for kind in ["shrunk","grown"]:
    for i,j,dh in ext[kind][:3]:
        pD=(l2c@np.array([Adec[i,0],Adec[i,1],float(hd["ground"][i]),1.0]))[:3]; pA=(l2c@np.array([B[j,0],B[j,1],float(ha["ground"][j]),1.0]))[:3]   # full laser->camera extrinsic, rotation and translation
        kd=view(dec,pD); ka=view(apr,pA)
        if kd is None or ka is None: continue
        imd=next(monos.iter_images(stD["kf"],[kfD[kd]],gray=False))[2]; ima=next(monos.iter_images(stA["kf"],[kfA[ka]],gray=False))[2]
        for im,c2w,p in [(imd,dec[kd],pD),(ima,apr[ka],pA)]:
            u=proj(c2w,p)
            if u and 0<=u[0]<W and 0<=u[1]<H: cv2.circle(im,u,44,(0,220,255),3,cv2.LINE_AA)
        cv=np.hstack([imd,ima]); cv2.rectangle(cv,(0,0),(2*W,34),(0,0,0),-1)
        cv2.putText(cv,f"{kind.upper()}: Dec bag {i} (kf {kd}, plant top {hd['height'][i]:.2f} m)  ->  Apr bag {j} (kf {ka}, {ha['height'][j]:.2f} m): {dh:+.2f} m   (lidar 95th pct above local ground; circle = bag base)",(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
        tiles.append(cv2.resize(cv,(1920,540)))
cv2.imwrite("/tmp/growth_evidence.jpg",np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,90]); print("tiles",len(tiles))
