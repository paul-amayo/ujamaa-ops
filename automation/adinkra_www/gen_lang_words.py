"""Discriminative detector vocabulary from the question sets: for each language, the words (>= 3 letters) that occur
in its questions and in NO other language's questions (English included). Measures coverage with the >=1 and >=2 rules,
leaving each question out of its own language's list when scoring (so coverage is not just memorisation)."""
import json, re, collections
SRC = ["/home/paperspace/code/automation/adinkra_www/questions.json", "/home/paperspace/code/automation/adinkra_www/demo_bank_20260930.json"]
LANGS = ["en", "af", "xh", "zu", "sw", "ha", "ar", "am"]
qs = []
for src in SRC:
    d = json.load(open(src)); items = d.get("questions") or d.get("items")
    for it in items:
        text = it.get("text") or {k: v for k, v in it.items() if k in LANGS}
        for lang, q in text.items(): qs.append((lang, q))
tok = lambda q: set(w for w in re.findall(r"[a-zà-ÿ']+", q.lower()) if len(w) >= 3)
per = collections.defaultdict(collections.Counter)
for lang, q in qs:
    for w in tok(q): per[lang][w] += 1
vocab = {}
for lang in LANGS:
    if lang in ("en", "ar", "am"): continue   # ar/am by script; en = the fallback
    others = set().union(*(set(per[l]) for l in LANGS if l != lang))
    vocab[lang] = sorted(w for w in per[lang] if w not in others)
print({l: len(v) for l, v in vocab.items()})
def detect(q, rule, exclude=None):
    words = tok(q); scores = {}
    for lang, v in vocab.items():
        vs = set(v) - (exclude if lang == exclude_lang else set()) if False else set(v)
        scores[lang] = len(words & vs)
    best = max(scores, key=scores.get); return best if scores[best] >= rule else "en"
for rule in (1, 2):
    miss = []
    for lang, q in qs:
        if lang in ("ar", "am"): continue
        # leave-one-out: remove this question's own words from its language's vocab
        own = tok(q); saved = vocab.get(lang)
        if saved is not None: vocab[lang] = [w for w in saved if not (w in own and per[lang][w] == 1)]
        got = detect(q, rule)
        if saved is not None: vocab[lang] = saved
        if got != lang: miss.append((lang, got, q[:60]))
    print(f"rule >= {rule}: {len(miss)} misdetected of {sum(1 for l, _ in qs if l not in ('ar', 'am'))} (leave-one-out)")
    for m in miss[:10]: print("   ", m)
json.dump(vocab, open("/tmp/appcheck/lang_words_generated.json", "w"), ensure_ascii=False, indent=0)
