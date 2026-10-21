import numpy as np, csv, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O="/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021"
hd=np.load(O+"/bag_heights_dec.npz"); ha=np.load(O+"/bag_heights_apr.npz"); Hd,Ha=hd["height"],ha["height"]; Nd,Na=hd["n_points"],ha["n_points"]
ch=np.load(O+"/bag_change_ledger.npz",allow_pickle=True); A,B=ch["A"],ch["B"]; pairs=ch["pairs"]
rows=np.load(O+"/bag_rows.npz"); R=rows["rows"]
i=pairs[:,0].astype(int); j=pairs[:,1].astype(int); d=pairs[:,2]
ok=np.isfinite(Hd[i])&np.isfinite(Ha[j])&(d<=0.4)&(Nd[i]>=200)&(Na[j]>=200)
dh=Ha[j][ok]-Hd[i][ok]
print(f"matched pairs with heights in both surveys: {ok.sum()} of {len(pairs)}")
print(f"Dec height median {np.median(Hd[i][ok]):.2f} m, Apr {np.median(Ha[j][ok]):.2f} m; change Apr-Dec: median {np.median(dh):+.2f} m, IQR {np.percentile(dh,25):+.2f}..{np.percentile(dh,75):+.2f}, 10th {np.percentile(dh,10):+.2f}, 90th {np.percentile(dh,90):+.2f}")
# noise floor: repeatability is unknown without a same-day pair; use the spread of the matched distribution's core as the scale and 0.15 m as the call threshold
for thr in (0.10,0.15,0.20):
    print(f"  |dh| > {thr:.2f} m: grown {int((dh>thr).sum())}, shrunk {int((dh<-thr).sum())}, same {int((np.abs(dh)<=thr).sum())}")
# per row (across-row position of the April bag)
rid=np.abs(B[j][ok][:,0][:,None]-R[None,:]).argmin(1)
print("per bag row: n, median dh, grown>0.15, shrunk<-0.15")
for r in range(len(R)):
    m=rid==r
    if m.sum()>=5: print(f"  row x={R[r]:5.1f}: n {m.sum():3d}, median {np.median(dh[m]):+.2f}, grown {int((dh[m]>0.15).sum()):2d}, shrunk {int((dh[m]<-0.15).sum()):2d}")
# figure: map coloured by dh + histogram
fig,ax=plt.subplots(1,2,figsize=(16,10),gridspec_kw={"width_ratios":[3,1.2]})
sc=ax[0].scatter(B[j][ok][:,0],B[j][ok][:,1],c=np.clip(dh,-0.5,0.5),cmap="RdYlGn",s=18,vmin=-0.5,vmax=0.5); plt.colorbar(sc,ax=ax[0],shrink=0.7,label="plant-top height change Apr - Dec (m, lidar 95th pct above local ground)")
ax[0].set_aspect("equal"); ax[0].grid(alpha=.3); ax[0].set_xlabel("across rows (m, April frame)"); ax[0].set_ylabel("along rows (m)"); ax[0].set_title(f"Per-bag height change, {ok.sum()} matched bags")
ax[1].hist(dh,bins=np.arange(-1.0,1.01,0.05),color="#4a7",edgecolor="white"); ax[1].axvline(0,color="k",lw=1); ax[1].set_xlabel("Apr - Dec (m)"); ax[1].set_title(f"median {np.median(dh):+.2f} m; grown>0.15: {int((dh>0.15).sum())}, shrunk<-0.15: {int((dh<-0.15).sum())}")
plt.tight_layout(); plt.savefig(O+"/bag_growth.png",dpi=110)
# write dh into the change ledger csv
recs=list(csv.DictReader(open(O+"/change_ledger_dec_apr.csv"))); byj={int(jj):(float(Ha[jj]),float(Hd[ii]),float(Ha[jj]-Hd[ii])) for ii,jj,o in zip(i,j,ok) if o}
for r in recs:
    jj=r["apr_id"]; 
    if jj!="" and int(jj) in byj: ha_,hd_,dh_=byj[int(jj)]; r["h_dec_m"]=round(hd_,2); r["h_apr_m"]=round(ha_,2); r["dh_m"]=round(dh_,2); r["growth"]="grown" if dh_>0.15 else ("shrunk" if dh_<-0.15 else "same")
    else: r["h_dec_m"]=r["h_apr_m"]=r["dh_m"]=r["growth"]=""
with open(O+"/change_ledger_dec_apr.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
# extremes for evidence
order=np.argsort(dh); ext={"shrunk":[(int(i[ok][q]),int(j[ok][q]),float(dh[q])) for q in order[:6]],"grown":[(int(i[ok][q]),int(j[ok][q]),float(dh[q])) for q in order[-6:]]}
json.dump(ext,open(O+"/growth_extremes.json","w")); print("extremes:",ext)
