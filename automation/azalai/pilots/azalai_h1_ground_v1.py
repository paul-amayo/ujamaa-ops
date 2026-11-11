#!/usr/bin/env python3
"""azalai_h1_ground_v1.py (11 Nov 2026) — H1 pilot v1: ground the frozen instruction bank with a local LLM (ollama, Gemma 4 12B,
the v0 model) in two arms that share the model, the prompt structure and this tool-calling loop; the arms differ only in the
fields their registry holds (azalai_h1_tools.py). The LLM decomposes the instruction into tool calls; the registry resolves them.
This script does NOT read the gold: it writes one JSONL record per instruction and arm (every model reply, every tool call and
result, the final answer with result_ids expanded); azalai_h1_score_v1.py scores the records afterwards.

Usage (box):  python3 azalai_h1_ground_v1.py <dir with instruction_bank_v0.csv and registry_v0.json> <out prefix> [--seed 0] [--bank other.csv] [--arms twin,enumerated]
writes <out prefix>_<arm>.jsonl (appends; an instruction already in the file is skipped, so a crashed run resumes)."""
import sys, os, json, csv, time, argparse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azalai_h1_tools import Registry, system_prompt

ap = argparse.ArgumentParser(); ap.add_argument('d'); ap.add_argument('out'); ap.add_argument('--model', default='hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M')
ap.add_argument('--url', default='http://localhost:11434/api/chat'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--bank', default=None); ap.add_argument('--arms', default='twin,enumerated')
a = ap.parse_args()
MAX_CALLS, MAX_TURNS, MAX_BAD = 8, 14, 2
bank = list(csv.DictReader(open(a.bank or f'{a.d}/instruction_bank_v0.csv'))); R = json.load(open(f'{a.d}/registry_v0.json'))
OPT = dict(temperature=0, seed=a.seed, num_ctx=8192, num_predict=500)


def chat(msgs):
    """one model turn. think=False and format=json as in v0 (Gemma 4 otherwise spends the budget in its thinking channel);
    v0's fallback is kept: an empty reply is asked again once without the format constraint and the outermost {...} is taken."""
    def post(fmt):
        body = dict(model=a.model, messages=msgs, stream=False, think=False, options=OPT)
        if fmt: body['format'] = 'json'
        for attempt in range(3):
            try:
                r = urllib.request.urlopen(urllib.request.Request(a.url, json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=600); return json.loads(r.read())
            except Exception as e:
                if attempt == 2: raise
                time.sleep(10)
    j = post(True); txt = j['message'].get('content', ''); fb = False
    if not txt.strip():
        j = post(False); txt = j['message'].get('content', ''); fb = True
        i0, i1 = txt.find('{'), txt.rfind('}'); txt = txt[i0:i1 + 1] if i0 >= 0 and i1 > i0 else txt
    return txt, dict(fallback=fb, done_reason=j.get('done_reason'), eval_count=j.get('eval_count'), prompt_eval_count=j.get('prompt_eval_count'))


def parse(txt):
    """-> ('tool', name, args) | ('final', dict) | None. Accepts the final object with or without the "final" wrapper."""
    try: o = json.loads(txt)
    except Exception: return None
    if not isinstance(o, dict): return None
    if isinstance(o.get('final'), dict): return ('final', o['final'])
    if o.get('tool') == 'final' and isinstance(o.get('args'), dict): return ('final', o['args'])
    if isinstance(o.get('tool'), str): return ('tool', o['tool'], o.get('args') if o.get('args') is not None else {})
    if 'targets' in o or 'reject' in o or 'answer' in o: return ('final', o)
    return None


def ground(arm, ins):
    reg = Registry(R, arm); msgs = [dict(role='system', content=SYS[arm]), dict(role='user', content=f'INSTRUCTION: {ins}')]
    steps = []; calls = bad = 0; final = None; status = 'max_turns'; t0 = time.time()
    for turn in range(MAX_TURNS):
        txt, meta = chat(msgs); p = parse(txt); st = dict(turn=turn, raw=txt, **meta); steps.append(st); msgs.append(dict(role='assistant', content=txt))
        if p is None:
            bad += 1; st['parsed'] = None
            if bad > MAX_BAD: status = 'parse_fail'; break
            msgs.append(dict(role='user', content='That was not one valid JSON object of the two forms. Reply with exactly one JSON object: a tool call or the final.')); continue
        if p[0] == 'final':
            f = p[1]; final = dict(targets_raw=f.get('targets', []), targets=reg.expand(f.get('targets', [])), answer=f.get('answer'), reject=bool(f.get('reject')), reason=f.get('reason'))
            st['parsed'] = 'final'; status = 'final'; break
        calls += 1; st['parsed'] = 'tool'; st['tool'] = p[1]; st['args'] = p[2]
        res = reg.call(p[1], p[2]) if calls <= MAX_CALLS else dict(error='no tool calls left; reply with the final object now')
        st['result'] = res; msgs.append(dict(role='user', content='TOOL RESULT: ' + json.dumps(res)))
    return dict(status=status, n_tool_calls=min(calls, MAX_CALLS), final=final, steps=steps, seconds=round(time.time() - t0, 1))


SYS = {arm: system_prompt(R, arm, MAX_CALLS) for arm in ('twin', 'enumerated')}
for arm in a.arms.split(','):
    path = f'{a.out}_{arm}.jsonl'; done = set()
    if os.path.exists(path): done = {json.loads(l)['id'] for l in open(path) if l.strip()}
    with open(path, 'a') as f:
        for it in bank:
            if it['id'] in done: continue
            rec = dict(id=it['id'], category=it['category'], instruction=it['instruction'], arm=arm, model=a.model, seed=a.seed, options=OPT, **ground(arm, it['instruction']))
            f.write(json.dumps(rec) + '\n'); f.flush()
            fi = rec['final'] or {}; print(f"[h1v1] {arm} {it['id']} {rec['status']} calls={rec['n_tool_calls']} reject={fi.get('reject')} n_targets={len(fi.get('targets', []))} answer={fi.get('answer')} {rec['seconds']}s", flush=True)
print('[h1v1] done', flush=True)
