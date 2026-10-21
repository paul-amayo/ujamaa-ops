import json, numpy as np, csv
O="/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021"
ch=np.load(O+"/bag_change_ledger.npz",allow_pickle=True); A,B=ch["A"],ch["B"]; pairs=ch["pairs"]; moved=ch["moved"]; added=ch["added"].astype(int); removed=ch["removed"].astype(int)
rows=np.load(O+"/bag_rows.npz"); R=rows["rows"]; lo,hi=rows["along"]
def place(p):
    dr=float(np.abs(p[0]-R).min()); inside=lo+1.0<=p[1]<=hi-1.0
    return ("row" if inside else "row_end") if dr<=0.45 else "off_row", round(dr,2)
recs=[]
for i,j,d in pairs: i=int(i); j=int(j); pl,dr=place(B[j]); recs.append(dict(change="matched" if d<=0.4 else "displaced", dec_id=i, apr_id=j, dist_m=round(float(d),2), x=round(float(B[j,0]),2), y=round(float(B[j,1]),2), placement=pl, row_dist_m=dr))
for i,j,d in moved: i=int(i); j=int(j); pl,dr=place(B[j]); recs.append(dict(change="moved", dec_id=i, apr_id=j, dist_m=round(float(d),2), x=round(float(B[j,0]),2), y=round(float(B[j,1]),2), placement=pl, row_dist_m=dr))
for j in added: pl,dr=place(B[j]); recs.append(dict(change="april_only", dec_id="", apr_id=int(j), dist_m="", x=round(float(B[j,0]),2), y=round(float(B[j,1]),2), placement=pl, row_dist_m=dr))
for i in removed: pl,dr=place(A[i]); recs.append(dict(change="december_only", dec_id=int(i), apr_id="", dist_m="", x=round(float(A[i,0]),2), y=round(float(A[i,1]),2), placement=pl, row_dist_m=dr))
for r in recs: r["verdict"]= "artefact (off row)" if r["placement"]=="off_row" and r["change"] in ("april_only","december_only","moved") else ("headland" if r["placement"]=="row_end" and r["change"]!="matched" else "candidate change" if r["change"]!="matched" else "same")
with open(O+"/change_ledger_dec_apr.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
from collections import Counter
print(Counter((r["change"],r["verdict"]) for r in recs)); print("wrote", O+"/change_ledger_dec_apr.csv", len(recs), "rows")
