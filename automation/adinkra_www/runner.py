#!/usr/bin/env python3
"""Adinkra x WWW eval runner (plans/adinkra_www_eval.md).

Sequential queries against the :8003 panel server; one JSONL line per query.
Resumable: a (id, lang, site, tag) already present in the output is skipped,
so a killed pass continues where it stopped. Retry once on failure; second
failure is logged as a defect row and the pass moves on (never stalls).

  python3 runner.py --questions questions.json --site citrus --tag passA \
      --out /path/run.jsonl [--subset B2,T1] [--langs en,af] [--followups]
"""
import argparse, json, time, sys, urllib.request, datetime

LANGS = ["en", "af", "xh", "zu", "sw"]
URL = "http://localhost:8003/query"
TIMEOUT = 300


def ask(query, panel, lang):
    body = json.dumps({"query": query, "panel": panel, "lang": lang}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
    return r, round(time.time() - t0, 2)


def done_keys(path):
    keys = set()
    try:
        for ln in open(path):
            try:
                d = json.loads(ln)
                keys.add((d["id"], d["lang"], d["site"], d["tag"]))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    return keys


def emit(out, row):
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()


def run_item(out, done, *, qid, panel, site, tag, lang, text, intent="", expect="",
             followup_to=""):
    if (qid, lang, site, tag) in done:
        return "skip"
    row = dict(id=qid, panel=panel, site=site, tag=tag, lang=lang, intent=intent,
               expect=expect, followup_to=followup_to, question=text,
               ts=datetime.datetime.now().isoformat(timespec="seconds"))
    for attempt in (1, 2):
        try:
            resp, secs = ask(text, panel, lang)
            cmd = resp.get("command") or {}
            row.update(secs=secs, attempt=attempt, ok=True, response=resp,
                       action=cmd.get("action"), answer=cmd.get("target"),
                       empty=not (cmd.get("target") or "").strip())
            emit(out, row)
            print(f"[{tag}] {site}/{qid}/{lang} {secs}s action={row['action']}",
                  flush=True)
            return "ok"
        except Exception as e:
            if attempt == 2:
                row.update(ok=False, error=str(e)[:200], attempt=attempt)
                emit(out, row)
                print(f"[{tag}] {site}/{qid}/{lang} FAILED twice: {e}", flush=True)
                return "fail"
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--site", required=True, choices=["citrus", "klap"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default="")
    ap.add_argument("--langs", default="")
    ap.add_argument("--followups", action="store_true")
    a = ap.parse_args()

    bank = json.load(open(a.questions))
    done = done_keys(a.out)
    out = open(a.out, "a")
    subset = set(a.subset.split(",")) if a.subset else None
    langs = a.langs.split(",") if a.langs else LANGS
    n = dict(ok=0, fail=0, skip=0)

    if a.followups:
        # Pass C: replay the parent question, then send the follow-up with the
        # exchange inlined (the server is stateless single-turn).
        by_id = {q["id"]: q for q in bank["questions"]}
        for f in bank["pass_c_followups"]:
            if a.site not in f["sites"]:
                continue
            parent = by_id[f["after"]]
            for lang in f["langs"]:
                ptext = parent.get(lang) or parent["en"]
                try:
                    presp, _ = ask(ptext, parent["panel"], lang)
                    ans = (presp.get("command") or {}).get("target") or ""
                except Exception as e:
                    print(f"[{a.tag}] parent {f['after']}/{lang} failed: {e}",
                          flush=True)
                    continue
                ftext = f.get(lang) or f["en"]
                composed = (f'Earlier I asked: "{ptext}" and you answered: '
                            f'"{ans}". Now: {ftext}')
                st = run_item(out, done, qid=f"{f['after']}+F", panel=parent["panel"],
                              site=a.site, tag=a.tag, lang=lang, text=composed,
                              intent="followup", expect="", followup_to=f["after"])
                n[st] += 1
    else:
        for q in bank["questions"]:
            if a.site not in q["sites"]:
                continue
            if subset and q["id"] not in subset:
                continue
            qlangs = q.get("langs_override") or langs
            for lang in qlangs:
                text = q.get(lang)
                if not text:
                    continue
                st = run_item(out, done, qid=q["id"], panel=q["panel"], site=a.site,
                              tag=a.tag, lang=lang, text=text,
                              intent=q.get("intent", ""), expect=q.get("expect", ""))
                n[st] += 1

    print(f"[{a.tag}] {a.site} done: {n}", flush=True)
    sys.exit(1 if (n["fail"] and not n["ok"]) else 0)


if __name__ == "__main__":
    main()
