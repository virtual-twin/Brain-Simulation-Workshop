#!/usr/bin/env python3
"""Generate the TVB-O clinical-applications ontology addon (Turtle).

Connects published TVB / whole-brain modelling STUDIES to CLINICAL DOMAINS
(diseases) cross-referenced to ICD-11 codes and MeSH descriptors, plus the
neural-mass model each study uses where known. Layered addon, like
ontology/tvb-o-axioms.ttl. Source data = the PubMed-verified classification.
"""
import json, re, datetime

V = {e["citekey"]: e for e in json.load(open("/tmp/ch4_verified.json"))}
META = json.load(open("/tmp/ch4_meta.json"))
FINAL = json.load(open("/tmp/ch4_final.json"))
try:
    FT = {e["citekey"]: e for e in json.load(open("/tmp/ch4_fulltext.json"))}  # optional, from full-text wf
except Exception:
    FT = {}

# disease -> (label, icd11 chapter num, icd11 code, MeSH id, MeSH label, default neural-mass model)
DOMAINS = {
 "Epilepsy":      ("Epilepsy","08","8A6Z","D004827","Epilepsy","Epileptor"),
 "Alzheimer":     ("Alzheimer disease & dementia","08","8A20","D000544","Alzheimer Disease",None),
 "Schizophrenia": ("Schizophrenia","06","6A20","D012559","Schizophrenia",None),
 "Parkinson":     ("Parkinson disease","08","8A00.0","D010300","Parkinson Disease",None),
 "Stroke":        ("Stroke","08","8B20","D020521","Stroke",None),
 "Tumour":        ("Brain tumor / glioma","02","2A00.0","D005910","Glioma","ReducedWongWang"),
 "MS":            ("Multiple sclerosis","08","8A40","D009103","Multiple Sclerosis",None),
 "TBI":           ("Traumatic brain injury","22","NA07","D000070642","Brain Injuries, Traumatic",None),
 "DoC":           ("Disorders of consciousness","08","8E2Z","D003244","Consciousness Disorders",None),
 "Depression":    ("Major depressive disorder","06","6A70","D003865","Depressive Disorder, Major",None),
 "OCD":           ("Obsessive-compulsive disorder","06","6B20","D009771","Obsessive-Compulsive Disorder",None),
}
CHAPTERS = {"08":"Diseases of the nervous system","02":"Neoplasms",
            "06":"Mental, behavioural or neurodevelopmental disorders","22":"Injury, poisoning or external causes"}
NMM = {  # node dynamical-model individuals: id -> label  (full-text-confirmed)
 "Epileptor":"Epileptor",
 "ReducedWongWang":"Reduced Wong-Wang / dynamic mean-field (Deco)",
 "JansenRit":"Jansen-Rit",
 "Wendling":"Wendling (Jansen-Rit-derived)",
 "StefanescuJirsa3D":"Stefánescu-Jirsa 3D",
 "DodyNextGen":"next-generation NMM (Montbrió/Dody, aQIF-derived)",
 "Hopf":"Hopf / Stuart-Landau oscillator",
 "SpikingEI":"Spiking E/I network (LIF)",
 "BasalGanglia":"Basal-ganglia circuit model",
 "WilsonCowan":"Wilson-Cowan",
 "DynamicCausalModel":"DCM canonical microcircuit",
}
NMM_PATS = [
 ("DynamicCausalModel", r"causal|canonical microcircuit|\bdcm\b"),
 ("Epileptor",          r"epileptor"),
 ("StefanescuJirsa3D",  r"stefa|sj3d"),
 ("DodyNextGen",        r"dody|next.?generation|aqif|montbri"),
 ("Wendling",           r"wendling"),
 ("Hopf",               r"hopf|stuart.?landau"),
 ("SpikingEI",          r"integrate.?and.?fire|\blif\b|spiking"),
 ("WilsonCowan",        r"wilson.?cowan"),
 ("JansenRit",          r"jansen"),
 ("ReducedWongWang",    r"wong.?wang|mean.?field|\bdmf\b"),
 ("BasalGanglia",       r"basal.?ganglia"),
]

def esc(s): return (s or "").replace("\\","\\\\").replace('"','\\"').replace("\n"," ").strip()
def model_for(citekey, disease):
    """neural-mass model id for a study, from full-text where available."""
    name = (FT.get(citekey, {}).get("neural_mass_model") or "").lower()
    if name:
        for mid, pat in NMM_PATS:
            if re.search(pat, name):
                return mid
        return None  # named but unmapped -> don't guess
    return DOMAINS[disease][5]  # disease default only for the confident ones

# assemble study -> disease (38 dynamical apps + Hofsaehs)
study2dis = {}
for dkey, v in FINAL.items():
    for k in v.get("dyn_dedup", []):
        study2dis[k] = dkey
study2dis.setdefault("Hofsaehs2026","Depression")

L = []
W = L.append
W("# ─────────────────────────────────────────────────────────────────────────")
W("# TVB-O — Clinical Applications addon")
W(f"# Generated {datetime.date(2026,6,15).isoformat()} from a PubMed-verified survey of")
W("# whole-brain / TVB dynamical-model applications to disease. Layer on top of")
W("# tvb-o.owl (like tvb-o-axioms.ttl). Connects studies -> clinical domains")
W("# (ICD-11 + MeSH) and -> the neural-mass model used.")
W("# ─────────────────────────────────────────────────────────────────────────\n")
W("@prefix tvbo:    <https://w3id.org/tvbo/> .")
W("@prefix tvbc:    <https://w3id.org/tvbo/clinical/> .")
W("@prefix owl:     <http://www.w3.org/2002/07/owl#> .")
W("@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .")
W("@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .")
W("@prefix skos:    <http://www.w3.org/2004/02/skos/core#> .")
W("@prefix dcterms: <http://purl.org/dc/terms/> .")
W("@prefix bibo:    <http://purl.org/ontology/bibo/> .")
W("@prefix fabio:   <http://purl.org/spar/fabio/> .")
W("@prefix MESH:    <http://purl.bioontology.org/ontology/MESH/> .")
W("@prefix icd11:   <http://id.who.int/icd/release/11/mms/> .\n")

# ---- vocabulary (TBox) ----
W("### Vocabulary -----------------------------------------------------------")
W("tvbc:ClinicalDomain a owl:Class ; rdfs:subClassOf skos:Concept ;")
W('  rdfs:label "Clinical domain" ; rdfs:comment "A disease/condition area a model is applied to, cross-referenced to ICD-11 and MeSH." .')
W("tvbc:ClinicalApplicationStudy a owl:Class ; rdfs:subClassOf bibo:AcademicArticle ;")
W('  rdfs:label "Clinical application study" ; rdfs:comment "A publication applying a whole-brain / neural-mass model to patient data." .')
W("tvbc:NodeDynamicsModel a owl:Class ; rdfs:subClassOf tvbo:Dynamics ;")
W('  rdfs:label "Node dynamical model" ; rdfs:comment "Local node / mesoscale dynamics used in a whole-brain or microcircuit model: neural-mass and mean-field models (most align with tvbo:Dynamics), phenomenological oscillators (Hopf/Stuart-Landau), DCM convolution models, and spiking E/I networks." .')
W("tvbc:appliesToClinicalDomain a owl:ObjectProperty ; rdfs:domain tvbc:ClinicalApplicationStudy ;")
W('  rdfs:range tvbc:ClinicalDomain ; rdfs:label "applies to clinical domain" .')
W("tvbc:hasClinicalApplication a owl:ObjectProperty ; owl:inverseOf tvbc:appliesToClinicalDomain ;")
W('  rdfs:label "has clinical application" .')
W("tvbc:usesNodeDynamics a owl:ObjectProperty ; rdfs:domain tvbc:ClinicalApplicationStudy ;")
W('  rdfs:range tvbc:NodeDynamicsModel ; rdfs:label "uses node dynamical model" .')
W("tvbc:studyKind a owl:DatatypeProperty ; rdfs:range xsd:string ; rdfs:label \"study kind\" .")
W("tvbc:icd11Code a owl:AnnotationProperty ; rdfs:label \"ICD-11 code\" .\n")

# ---- ICD-11 chapters ----
W("### ICD-11 chapters ------------------------------------------------------")
for num, name in CHAPTERS.items():
    W(f'tvbc:icd11-{num} a skos:Concept ; skos:notation "{num}" ; skos:prefLabel "{esc(name)}"@en .')
W("")

# ---- node dynamical models ----
W("### Node dynamical models ------------------------------------------------")
used_models = {model_for(k, d) for k, d in study2dis.items()}
for mid, mlabel in NMM.items():
    if mid in used_models:
        W(f'tvbc:model-{mid} a tvbc:NodeDynamicsModel ; skos:prefLabel "{esc(mlabel)}"@en .')
W("")

# ---- clinical domains ----
W("### Clinical domains (disease -> ICD-11 + MeSH) --------------------------")
for dkey,(label,chap,code,mesh_id,mesh_label,_) in DOMAINS.items():
    n = sum(1 for v in study2dis.values() if v==dkey)
    W(f"tvbc:domain-{dkey} a tvbc:ClinicalDomain ;")
    W(f'  skos:prefLabel "{esc(label)}"@en ;')
    W(f'  tvbc:icd11Code "{code}" ; skos:notation "{code}" ;')
    W(f'  skos:exactMatch MESH:{mesh_id} ;   # {esc(mesh_label)}')
    W(f"  skos:broader tvbc:icd11-{chap} ;")
    W(f'  skos:scopeNote "{n} verified dynamical-model application(s) in TVB-O survey 2026" .')
W("")

# ---- studies ----
W("### Clinical application studies -----------------------------------------")
for k in sorted(study2dis):
    d = study2dis[k]; m = META.get(k,{}); v = V.get(k,{})
    W(f"tvbc:study-{k} a tvbc:ClinicalApplicationStudy ;")
    if m.get("title"): W(f'  dcterms:title "{esc(m["title"])}" ;')
    if m.get("year"):  W(f'  dcterms:date "{m["year"]}"^^xsd:gYear ;')
    if m.get("doi"):   W(f'  bibo:doi "{esc(m["doi"])}" ;')
    if v.get("pmid"):  W(f'  fabio:hasPubMedId "{v["pmid"]}" ;')
    W(f'  tvbc:studyKind "{esc(v.get("study_kind","patient-application"))}" ;')
    mid = model_for(k,d)
    if mid: W(f"  tvbc:usesNodeDynamics tvbc:model-{mid} ;")
    W(f"  tvbc:appliesToClinicalDomain tvbc:domain-{d} .")
W("")

ttl = "\n".join(L) + "\n"
out_repo = "_dev/tvb-o-clinical.ttl"
open(out_repo,"w").write(ttl)
print("wrote", out_repo, "—", len(ttl), "bytes,", len(study2dis), "studies,", len(DOMAINS), "domains")
