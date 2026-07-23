#!/usr/bin/env python3
"""Regenerate the per-disease slides of _04-clinical.qmd as figure cards
(one publication figure + text per study) from _dev/ch4_cards.json + the
extracted figures in img/figures/clinical/. Re-runnable."""
import json, os, re
from math import ceil

QMD = "slides/_04-clinical.qmd"
CARDS = json.load(open("_dev/ch4_cards.json"))
FIGDIR = "img/figures/clinical"

def _year(ck):
    m = re.search(r"(\d{4})", ck)
    return int(m.group(1)) if m else 0
try:
    NOTES = json.load(open("_dev/ch4_notes.json"))   # {disease title: abstract-derived speaker notes}
except Exception:
    NOTES = {}

CSS = """<style>
/* The disease slide is a flex column: heading + subline take their natural height,
   the .cards grid takes ALL the rest. Dense slides (.fill) split that leftover space
   into equal rows (grid-auto-rows:1fr) so they fill top-to-bottom with no whitespace
   and no overflow; sparse slides (.center) use a capped row height and sit centred.
   --cols fills the width; --cardw caps card width on sparse slides; object-fit:contain
   resizes each figure to fit without cropping; overflow:hidden is a hard stop. */
.reveal section.cards-slide { overflow:hidden; }
.cards { width:100%; margin:0.3rem auto 0; overflow:hidden; gap:0.5rem; }
/* dense: grid with a FIXED row height in logical px of the 1244x750 slide, so the
   layout is identical at every window size (Reveal scales the whole slide uniformly) */
.cards.fill { display:grid; justify-content:center; grid-auto-rows:var(--rowh,175px);
              grid-template-columns:repeat(var(--cols,4), minmax(0, 1fr)); }
/* sparse: a few capped-width cards of fixed height, centred horizontally */
.cards.center { display:flex; flex-wrap:wrap; justify-content:center; align-content:flex-start; }
.cards.center .card { flex:0 0 var(--cardw,340px); max-width:var(--cardw,340px); height:var(--rowh,340px); }
.cards .card { display:flex; flex-direction:column; min-height:0; overflow:hidden;
         border:1px solid #d4dde4; border-radius:9px; background:#fff; box-shadow:0 1px 5px rgba(40,60,90,.14); }
.cards .card > p:has(img), .cards .card .cardimg { margin:0; flex:1 1 auto; min-height:0; display:flex; }
.cards .card img { width:100%; height:100%; object-fit:contain;
         background:#f4f7f9; border-bottom:1px solid #e6edf2; padding:0.15rem; }
.cards .card .noimg { flex:1 1 auto; min-height:0; display:flex; align-items:center; justify-content:center;
         text-align:center; padding:0.3rem; background:#dbe8df; color:#36684b; font-style:italic; font-size:0.62rem; }
.cards .card .cardbody { flex:0 0 auto; padding:0.3rem 0.5rem 0.4rem; font-size:0.5rem; line-height:1.2; }
.cards .card .cardbody p { margin:0.08rem 0 0; }
.cards .card .cardbody p:last-child { display:-webkit-box; -webkit-line-clamp:3;
         -webkit-box-orient:vertical; overflow:hidden; }   /* clamp blurb so cards stay uniform */
.cards .card .cardbody strong { color:#23364a; }
.cards .card .cardbody .mdl { color:#5a7d80; font-style:italic; }
</style>"""

def grid(n):
    """cols, card width, fill/center mode, and fixed row height (logical px, 750px slide)."""
    cols = {1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3, 10: 5, 11: 4, 12: 4}.get(n, 4)
    rows = ceil(n / cols)
    mode = "center" if n <= 3 else "fill"                  # 1-3 cards centred; 4+ a filled grid
    cardw = "340px" if mode == "center" else "1fr"
    rowh = 340 if mode == "center" else {1: 430, 2: 265, 3: 175}.get(rows, 135)
    return cols, cardw, mode, rowh

def md_card(s):
    k, name, model, blurb = s["citekey"], s["name"], s["model"], s["blurb"]
    out = [":::: {.card}"]
    fig = f"{FIGDIR}/{k}.jpg"
    if os.path.exists(fig):
        out.append(f"![]({fig}){{.cardimg}}")
    else:
        out += ["::: {.noimg}", (model or name), ":::"]
    head = f"**{name}**" + (f" · [{model}]{{.mdl}}" if model else "")
    out += ["", "::: {.cardbody}", head, "", f"{blurb} [@{k}]", ":::", "::::"]
    return "\n".join(out)

def md_disease(d):
    studies = sorted(d["studies"], key=lambda s: (_year(s["citekey"]), s["citekey"]))  # oldest first (top-left)
    cols, cardw, mode, rowh = grid(len(studies))
    block = [f"## {d['title']} {{.smaller .cards-slide}}", "", f"**{d['icd']}**", "",
             f'::::: {{.cards .{mode} style="--cols:{cols}; --cardw:{cardw}; --rowh:{rowh}px"}}']
    block += [md_card(s) for s in studies]
    block.append(":::::")
    note = NOTES.get(d["title"])
    if note:
        block += ["", "::: {.notes}", note.strip(), ":::"]
    return "\n".join(block)

import re
src = open(QMD).read()
first = src.index("## Epilepsy")
take = src.index("<!-- TAKEAWAYS -->")     # sentinel before the closing takeaways slide
head, tail = src[:first], src[take:]
head = re.sub(r"<style>.*?</style>\n*", "", head, flags=re.S)   # drop any prior card CSS
nl = head.index("\n")                                          # re-inject fresh CSS after chapter H1
head = head[:nl + 1] + "\n" + CSS + "\n" + head[nl + 1:]
mid = "\n\n".join(md_disease(d) for d in CARDS.values())
open(QMD, "w").write(head + mid + "\n\n" + tail)

n_fig = sum(os.path.exists(f"{FIGDIR}/{s['citekey']}.jpg")
            for d in CARDS.values() for s in d["studies"])
n_tot = sum(len(d["studies"]) for d in CARDS.values())
print(f"regenerated {len(CARDS)} disease card-slides; {n_fig}/{n_tot} cards have a figure")
