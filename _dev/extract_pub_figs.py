#!/usr/bin/env python3
"""Extract ONE representative figure (prefer Figure 1) per clinical-chapter
publication from the PMC article page CDN, into img/figures/clinical/<citekey>.jpg.
Best-effort: non-OA papers (no PMC figures) are recorded as null."""
import json, urllib.request, re, os, time, io

EMAIL = "l.martin@brainmodes.com"; TOOL = "tvbo-workshop"
UA = f"Mozilla/5.0 (compatible; {TOOL}; {EMAIL})"
V = {e["citekey"]: e for e in json.load(open("/tmp/ch4_verified.json"))}
FINAL = json.load(open("/tmp/ch4_final.json"))
keys = sorted({k for d in FINAL.values() for k in d.get("dyn_dedup", [])} | {"Hofsaehs2026"})
OUT = "img/figures/clinical"; os.makedirs(OUT, exist_ok=True)
res = {}

def get(url, raw=False, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    d = urllib.request.urlopen(req, timeout=timeout).read()
    return d if raw else d.decode("utf-8", "replace")

def pmcid_for(pmid):
    try:
        j = json.loads(get(f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
                            f"?ids={pmid}&format=json&tool={TOOL}&email={EMAIL}"))
        for rc in j.get("records", []):
            if rc.get("pmcid"):
                return rc["pmcid"]
    except Exception as e:
        print("  idconv err", pmid, repr(e)[:60])
    return None

def fig_num(fname):                       # figure number from a blob filename, or None
    m = re.search(r'(?:fig(?:ure)?[._ -]?|[._-]g)0*(\d+)', fname, re.I)
    return int(m.group(1)) if m else None

def best_figure_url(pmcid):
    html = get(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")
    blobs = re.findall(r'https://cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/[^\s"\'>]+?\.(?:jpg|jpeg|png|gif)', html)
    cands = []
    seen = set()
    for u in blobs:
        if u in seen:
            continue
        seen.add(u)
        fn = u.rsplit("/", 1)[-1].lower()
        if re.search(r'[._-]t0*\d+\.|[._-]e0*\d+\.|equation|table', fn):   # skip tables/equations
            continue
        n = fig_num(fn)
        if n:
            cands.append((n, u))
    if not cands:                          # fallback: first non-table blob
        for u in blobs:
            fn = u.rsplit("/", 1)[-1].lower()
            if not re.search(r'[._-]t0*\d+\.|equation|table', fn):
                cands.append((999, u)); break
    cands.sort()
    return cands[0][1] if cands else None

def save(url, citekey):
    from PIL import Image
    raw = get(url, raw=True, timeout=120)
    im = Image.open(io.BytesIO(raw))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size
    if w > 1000:
        im = im.resize((1000, int(h * 1000 / w)))
    p = f"{OUT}/{citekey}.jpg"
    im.convert("RGB").save(p, quality=86)
    return p

for k in keys:
    pmid = V.get(k, {}).get("pmid")
    if not pmid:
        res[k] = None; print(k, "no pmid", flush=True); continue
    pmcid = pmcid_for(pmid); time.sleep(0.35)
    if not pmcid:
        res[k] = None; print(k, "no pmcid (not OA)", flush=True); continue
    try:
        u = best_figure_url(pmcid)
        if not u:
            res[k] = None; print(k, pmcid, "no figure blob", flush=True)
        else:
            res[k] = save(u, k); print(k, pmcid, "->", res[k], flush=True)
    except Exception as e:
        res[k] = None; print(k, pmcid, "ERR", repr(e)[:70], flush=True)
    json.dump(res, open("/tmp/ch4_figs.json", "w"), indent=1)
    time.sleep(0.35)

json.dump(res, open("/tmp/ch4_figs.json", "w"), indent=1)
print("DONE", sum(1 for v in res.values() if v), "/", len(keys), "figures", flush=True)
