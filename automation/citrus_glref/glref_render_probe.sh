#!/bin/bash
# Second render-service instance on :8005 pointed at the CORRECTED (glref stage-1) 05 runs.
# Renders one keyframe (kf_001703 / block_021) under three pose candidates, beside GT.
cd /home/paperspace/code/aru_sil_core
RENDER_BLOCKS_ROOT=/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100 \
RENDER_RUN_GLOB='splat_runs_STAGE1/stage1_bg00_glref/high/*/config.yml' \
HIGH_EMBEDDER_CKPT=/home/paperspace/data/citrus_all/05_13D_Jackal/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth \
RENDER_VRAM_BUDGET=6 setsid nohup pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml python -m uvicorn src.interfaces.splat_viewer.render_service:app --host 127.0.0.1 --port 8005 > /home/paperspace/logs/render_8005.log 2>&1 < /dev/null &
for i in $(seq 1 40); do curl -sf -m3 http://127.0.0.1:8005/healthz >/dev/null 2>&1 && break; sleep 3; done
curl -s -m5 http://127.0.0.1:8005/healthz | python3 -c "import json,sys; d=json.load(sys.stdin); print('[8005] up: trained', d['blocks_trained'], '/', d['blocks_total'])"
/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python - << 'PY'
import asyncio, json, numpy as np, websockets
from PIL import Image
BD="/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100/block_021"
old = {f["file_path"].split("/")[-1]: np.array(f["transform_matrix"]) for f in json.load(open(f"{BD}/transforms_cv_pre_glfix.json"))["frames"]}
cur = {f["file_path"].split("/")[-1]: np.array(f["transform_matrix"]) for f in json.load(open(f"{BD}/transforms.json"))["frames"]}
name = "kf_001703.png"; FLIP = np.diag([1,-1,-1,1.0])
cands = {"A_recorded_cv": old[name], "B_recorded_gl": old[name] @ FLIP, "C_block_refined_gl": cur[name]}
async def main():
    async with websockets.connect("ws://127.0.0.1:8005/ws", max_size=2**24) as ws:
        for k,(tag,M) in enumerate(cands.items()):
            data=b""
            for attempt in range(40):
                await ws.send(json.dumps({"t":"pose","seq":k*100+attempt,"c2w":[float(x) for x in M.flatten()],"w":640,"h":360,"fovy":0.99,"block":21}))
                data=await asyncio.wait_for(ws.recv(),timeout=90)
                while isinstance(data,str): data=await asyncio.wait_for(ws.recv(),timeout=90)
                if len(data)>15000: break
                await asyncio.sleep(1.5)
            open(f"/home/paperspace/logs/glref_{tag}.jpg","wb").write(data[20:]); print(f"[8005] {tag}: {len(data)} B after {attempt+1} sends")
asyncio.run(main())
gt = Image.open("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3/kf_001703.png").convert("RGB").resize((640,360))
tiles=[gt]+[Image.open(f"/home/paperspace/logs/glref_{t}.jpg").convert("RGB").resize((640,360)) for t in cands]
c=Image.new("RGB",(640*2+10,360*2+10),(20,20,20))
for i,im in enumerate(tiles): c.paste(im,((i%2)*650,(i//2)*370))
c.save("/home/paperspace/logs/glref_pose_probe.png"); print("[8005] wrote glref_pose_probe.png: TL=GT, TR=A recorded OpenCV, BL=B recorded->GL, BR=C block refined GL")
PY
pkill -f "render_service:ap[p] --host 127.0.0.1 --port 8005"; echo "[8005] stopped"
