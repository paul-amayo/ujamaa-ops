"""Grow-bag change Dec -> Apr on the ledger's global registration (both bag sets in the April frame):
mutual pairs <= 0.5 m (the ledger's own), then among the unmatched a second mutual pass <= 1.5 m = moved,
the rest added (April only) or removed (December only). Noise floor: the matched-pair distance distribution."""
import json, numpy as np
from scipy.spatial import cKDTree
K="/home/paperspace/data/klapmuts"; L=json.load(open(K+"/dec_2025_ten_rows/experimental/sankofa/klapmuts_ledger_v5.json"))
A=np.array(L["A_positions_apr_frame"]); B=np.array(L["B_positions"])
def mutual(P,Q,thr):
    tq=cKDTree(Q); tp=cKDTree(P); dq,iq=tq.query(P); dp,ip=tp.query(Q)
    return [(i,int(iq[i]),float(dq[i])) for i in range(len(P)) if dq[i]<=thr and ip[iq[i]]==i]
pairs=mutual(A,B,0.5); d=np.array([p[2] for p in pairs])
print(f"Dec bags {len(A)}, Apr bags {len(B)}; mutual pairs <= 0.5 m: {len(pairs)} (ledger {L['association']['n_pairs']}); pair distance median {np.median(d):.2f}, 75th {np.percentile(d,75):.2f}, 90th {np.percentile(d,90):.2f} m")
ua=sorted(set(range(len(A)))-{p[0] for p in pairs}); ub=sorted(set(range(len(B)))-{p[1] for p in pairs})
A2=A[ua]; B2=B[ub]; moved=[(ua[i],ub[j],dd) for i,j,dd in mutual(A2,B2,1.5)] if len(ua) and len(ub) else []
added=[j for j in ub if j not in {m[1] for m in moved}]; removed=[i for i in ua if i not in {m[0] for m in moved}]
print(f"unmatched: Dec {len(ua)}, Apr {len(ub)} -> moved 0.5-1.5 m (mutual among the unmatched): {len(moved)}, distances {np.round(sorted([m[2] for m in moved]),2).tolist()[:12]}...")
print(f"added (April only, no Dec bag within 1.5 m): {len(added)}; removed (December only): {len(removed)}")
# within the matched pairs: shifts above the noise floor (90th pct of pair distance) — 'nudged'
thr=float(np.percentile(d,90)); nud=[p for p in pairs if p[2]>0.4]
print(f"matched pairs displaced > 0.4 m (above the ~{thr:.2f} m 90th-percentile noise): {len(nud)}")
# row-level: where are the added/removed? by across-row coordinate (x of the April frame)
def rows_of(P): return np.round(P[:,0]/1.0)*1.0
print("added by April-frame x (m):", sorted(np.round(B[added][:,0],1).tolist()))
print("removed by x (m):", sorted(np.round(A[removed][:,0],1).tolist()))
np.savez(K+"/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/bag_change_ledger.npz", pairs=np.array(pairs), moved=np.array(moved), added=np.array(added), removed=np.array(removed), A=A, B=B)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,11)); ax.scatter(A[:,0],A[:,1],s=12,c="#9aa5ad",label=f"December bags ({len(A)})")
pi=np.array([p[1] for p in pairs]); ax.scatter(B[pi,0],B[pi,1],s=10,c="#27ae60",label=f"April bags matched <= 0.5 m ({len(pairs)})")
if moved:
    for i,j,dd in moved: ax.annotate("",xy=B[j],xytext=A[i],arrowprops=dict(arrowstyle="->",color="#e67e22",lw=1.2))
    ax.scatter(B[[m[1] for m in moved],0],B[[m[1] for m in moved],1],s=28,c="#e67e22",label=f"moved 0.5-1.5 m ({len(moved)})")
if added: ax.scatter(B[added,0],B[added,1],s=40,marker="^",c="#c0392b",label=f"added in April ({len(added)})")
if removed: ax.scatter(A[removed,0],A[removed,1],s=60,facecolors="none",edgecolors="#2c3e50",label=f"removed since December ({len(removed)})")
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper right",fontsize=8); ax.set_xlabel("across rows (m, April frame)"); ax.set_ylabel("along rows (m)")
ax.set_title("Klapmuts grow bags, December 2025 -> April 2026 (ledger v5 registration)")
plt.tight_layout(); plt.savefig(K+"/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/bag_change_ledger.png",dpi=110); print("fig saved")
