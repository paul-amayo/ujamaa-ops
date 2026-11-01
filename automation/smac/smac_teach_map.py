"""Teach map for the SegMatch descriptor arm: every teach scan above the sensor-frame ground cut, carried into the teach LIO frame, voxelised."""
import numpy as np, re, sys, time
split, n, out, minz, vox = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
def readply(p):
    f=open(p,"rb"); hdr=b""
    while True:
        l=f.readline(); hdr+=l
        if l.strip()==b"end_header": break
    m=int(re.search(rb"element vertex (\d+)",hdr).group(1)); props=re.findall(rb"property (\w+) (\w+)",hdr)
    tm={b"float":"<f4",b"double":"<f8",b"uchar":"u1",b"int":"<i4",b"uint":"<u4",b"float32":"<f4"}
    dt=np.dtype([(nm.decode(), tm[t]) for t,nm in props]); a=np.frombuffer(f.read(m*dt.itemsize),dtype=dt); return np.c_[a["x"],a["y"],a["z"]].astype(np.float64)
keys=set(); t0=time.time(); npts=0
for i in range(1,n+1):
    P=readply(f"{split}/laser/laser_{i}.ply"); P=P[P[:,2]>=minz]; T=np.loadtxt(f"{split}/transforms/transform_{i}.txt")
    W=P@T[:3,:3].T+T[:3,3]; npts+=len(P); keys.update(map(tuple,np.unique(np.floor(W/vox).astype(np.int64),axis=0).tolist()))
    if i%1000==0: print(i, len(keys), f"{time.time()-t0:.0f}s", flush=True)
V=(np.array(sorted(keys),dtype=np.float64)+0.5)*vox
with open(out,"wb") as f:
    f.write(f"ply\nformat binary_little_endian 1.0\ncomment teach map {split} n={n} min_z={minz} voxel={vox}\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n".encode()); f.write(V.astype("<f4").tobytes())
print("written", out, len(V), "voxels from", npts, "points; extent", V.min(0).round(1), V.max(0).round(1))
