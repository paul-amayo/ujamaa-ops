#!/usr/bin/env python3
"""Start a sites.json H3DGS stage the way launch_api._start_stage does (cwd ARU_DIR, absolute script path, env =
os.environ + the stage's render.env) PLUS the native-identity keys, on a spare port; drive it with native_ws_check.py;
SIGKILL it after (hier_render_service ignores SIGTERM once its port closes). UJAMAA, 2026-10-03.
  python3 native_stage_emulate.py --farm 1 --features F --embedder E --bank B --port 8037 --proj P --chunk 1_0 --frame kf_000025.png --queries tree:5 fruit:10003 row:7 --out DIR"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request
ap = argparse.ArgumentParser()
for k in ('--features', '--embedder', '--bank', '--proj', '--chunk', '--frame', '--out'): ap.add_argument(k, required=True)
ap.add_argument('--farm', type=int, required=True); ap.add_argument('--port', type=int, default=8037); ap.add_argument('--queries', nargs='+', required=True)
a = ap.parse_args()
r = json.load(open('/home/paperspace/code/ujamaa/app/sites.json'))['farms'][a.farm]['surveys'][0]['render']
env = {**os.environ, **{k: str(v) for k, v in (r.get('env') or {}).items()},
       'PORT': str(a.port), 'HIER_FEATURES': a.features, 'HIER_EMBEDDER': a.embedder, 'HIER_TEXT_BANK': a.bank,
       'CUDA_HOME': '/home/paperspace/code/_cuda12'}
env.pop('IDENTITY_URL', None)
env['PATH'] = ':'.join(x for x in env['PATH'].split(':') if '_cuda12' not in x)    # prove the service adds CUDA itself
log = open(f'/home/paperspace/logs/native_stage_emulate_{a.port}.log', 'w')
p = subprocess.Popen(['/home/paperspace/miniconda3/envs/h3dgs/bin/python', '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer/hier_render_service.py'],
                     cwd='/home/paperspace/code/aru_sil_core', env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
t0 = time.time()
try:
    while True:
        try: h = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{a.port}/healthz', timeout=2).read()); break
        except Exception:
            if p.poll() is not None: sys.exit(f'[emulate] service exited rc={p.returncode}; see {log.name}')
            time.sleep(2)
    print(f"[emulate] farm {a.farm} up in {time.time() - t0:.0f}s: identity={h.get('identity')} nodes={h.get('nodes')} resident={h.get('resident_gib')} GiB free={h.get('free_gib')} GiB", flush=True)
    subprocess.run(['/home/paperspace/miniconda3/envs/h3dgs/bin/python', '/home/paperspace/code/automation/h3dgs/native/native_ws_check.py', '--port', str(a.port), '--proj', a.proj,
                    '--chunk', a.chunk, '--frame', a.frame, '--queries', *a.queries, '--out', a.out], cwd='/home/paperspace/code/hierarchical-3d-gaussians')
finally:
    os.killpg(p.pid, signal.SIGKILL); p.wait(); print(f'[emulate] stopped (SIGKILL), rc={p.returncode}', flush=True)
