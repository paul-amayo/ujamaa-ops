import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
O="/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021"
ch=np.load(O+"/bag_change_ledger.npz",allow_pickle=True); A,B=ch["A"],ch["B"]; pairs=ch["pairs"]; moved=ch["moved"]; added=ch["added"].astype(int); removed=ch["removed"].astype(int)
# bag rows = peaks of the across-row histogram of the matched bags (both surveys)
X=np.concatenate([A[pairs[:,0].astype(int),0],B[pairs[:,1].astype(int),0]]); e=np.arange(X.min()-1,X.max()+1,0.05); h,_=np.histogram(X,bins=e); hs=gaussian_filter1d(h.astype(float),2)
pk,_=find_peaks(hs,height=hs.max()*0.15,distance=12); rows=(e[pk]+e[pk+1])/2; print("bag rows (across-row x, m):",np.round(rows,2).tolist(), "n",len(rows))
along_lo,along_hi=np.percentile(np.concatenate([A[:,1],B[:,1]]),[1,99])
def classify(P):
    dr=np.abs(P[:,0][:,None]-rows[None,:]).min(1); inside=(P[:,1]>=along_lo+1.0)&(P[:,1]<=along_hi-1.0)
    on=(dr<=0.45)&inside; ends=(dr<=0.45)&~inside; off=dr>0.45
    return on,ends,off
for name,P,ids in [("April only",B[added],added),("December only",A[removed],removed),("moved (Apr pos)",B[moved[:,1].astype(int)],moved[:,1].astype(int))]:
    on,ends,off=classify(P); print(f"{name}: on a bag row mid-field {on.sum()}, on a row at the row ends {ends.sum()}, off the rows (lane / outside) {off.sum()}  -> off-row ids: {ids[off].tolist()}")
on,ends,off=classify(B[pairs[:,1].astype(int)]); print(f"matched pairs: on row {on.sum()}, ends {ends.sum()}, off {off.sum()}")
np.savez(O+"/bag_rows.npz",rows=rows,along=[along_lo,along_hi])
