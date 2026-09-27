"""Phase 1.2 / 1.3: catalogue of generator operations on true matches, and of decoys (train, US + India).

Name relation of a pair (S1 name a, record name b) = surface flags (case, fake accents, leetspeak, web handle, alias
phrase, native script, honorific, filler, legal-form change) + one primary relation of the canonical token lists:
canon_equal(_reordered), glued, acronym, truncation(_single / _typo), extra_word, typo, word_swap, invented,
multi_change, name_empty.
Address relation (a, b): surface flags (case, component reorder, number prefix, leading zero, null marker, PO box,
city-type suffix, admin-label change, abbreviation, native script) + primary relation of the canonical token sets:
canon_equal, empty, or num:<same|missing|added|truncated|typo|changed> x words:<same|missing|added|typo|changed>.

Step 3 (decoys): all (S1, record) pairs that share the canonical address key (or the canonical name key), labelled
match / owned-by-another-S1 / unowned, broken down by the name (or address) relation.
"""
import random
import re
import sys
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

import normalize as N

D = "C:/ML/Challenges/amazon/work/data/"
ACC = re.compile(r"[àáâãäåāçćčèéêëēėìíîïīñńòóôõöøōùúûüūýÿœæ]")
LEET = re.compile(r"\b(?=\w*[a-z]\w*[a-z])(?=\w*\d)\w+\b")
WEB = re.compile(r"@\w|#\w|www\.|\.(com|net|org|in|fr)\b")
ALIASRE = re.compile(r"\b(dba|d/b/a|aka|a\.k\.a|fka|f/k/a|formerly|known as|doing business as|t/a|trading as|nee)\b")
INDIC = re.compile(r"[\u0900-\u0d7f]")
HON = {"mr", "mrs", "ms", "smt", "shri", "sri", "shree", "dr", "m/s", "messrs"}
NUMPRE = re.compile(r"(n°|nº|\bno\.?\s*\d|#\s*\d|\bh\.?\s*no\b|\bd\.?\s*no\b|\(\d+\))")
LZ = re.compile(r"(^|[\s,#])0\d")
NULLRE = re.compile(r"(\bnull\b|\bn/a\b|<null>)")
POBOX = re.compile(r"\b(p\.?\s?o\.?\s*box|pmb)\b")
CITYSUF = {t for t in N.ADDR_DROP if t in {"township", "city", "cdp"} or "ship" in t or "city" in t or t in
           {"ity", "ciy", "citty", "ciyt", "ciity", "icty", "ccity", "ctiy", "cpd", "dcp", "ccdp", "cddp"}}


def words(s):
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]


def legal_of(s):
    toks = re.split(r"[^a-z0-9]+", re.sub(r"(?<=\b[a-z])\.", "", s.lower()))  # s.a.s. -> sas
    return [t for t in toks if t in N.LEGAL]


def name_flags(a, b):
    f = []
    if a == b:
        return ["exact"]
    if a.lower() == b.lower():
        f.append("case")
    if set(ACC.findall(b.lower())) - set(ACC.findall(a.lower())):
        f.append("fake_accent")
    if LEET.search(b.lower()) and not LEET.search(a.lower()):
        f.append("leetspeak")
    if WEB.search(b.lower()):
        f.append("web_handle")
    if ALIASRE.search(b.lower()):
        f.append("alias_phrase")
    if INDIC.search(b):
        f.append("native_script")
    wa, wb = set(words(a)), set(words(b))
    if (wb & HON) - wa:
        f.append("honorific")
    if (wb & N.FILLER) - wa:
        f.append("filler_added")
    la, lb = legal_of(a), legal_of(b)
    if la and not lb:
        f.append("legal_dropped")
    elif lb and not la:
        f.append("legal_added")
    elif la and lb and set(la) != set(lb):
        f.append("legal_changed")
    elif la and lb:
        ta, tb = words(a), words(b)
        pa = ta.index(la[0]) == 0 if la[0] in ta else None
        pb = tb.index(lb[0]) == 0 if lb[0] in tb else None
        if pa is not None and pb is not None and pa != pb:
            f.append("legal_moved")
    if not f:
        f.append("format_only")
    return f


def name_rel(A, B, cnsA, cnsB):
    if not B:
        return "name_empty"
    if sorted(A) == sorted(B):
        return "canon_equal" if A == B else "canon_equal_reordered"
    if cnsA and (cnsA == cnsB or (len(B) == 1 and (B[0] == cnsA or cnsA in B[0]))):
        return "glued"
    if len(B) == 1 and 2 <= len(B[0]) <= 5 and len(A) >= 2 and B[0] == "".join(t[0] for t in A):
        return "acronym"
    used, nE, nT, nU = set(), 0, 0, 0
    for tb in B:
        if tb in A and A.index(tb) not in used:
            used.add(A.index(tb)); nE += 1; continue
        best, bi = 0, -1
        for i, ta in enumerate(A):
            if i in used:
                continue
            r = fuzz.ratio(ta, tb)
            if r > best:
                best, bi = r, i
        if best >= 75:
            used.add(bi); nT += 1
        else:
            nU += 1
    nM = len(A) - len(used)
    typo = "_typo" if nT else ""
    if nU == 0 and nM == 0:
        return "typo"
    if nU == 0:
        if len(B) == 1:
            return "truncation_single" + typo
        return "truncation" + typo
    if nM == 0:
        return "extra_word" + typo
    if nE + nT == 0:
        return "invented"
    if nU == 1 and nM == 1:
        return "word_swap" + typo
    return "multi_change"


def addr_flags(a, b, admin):
    f = []
    if a == b:
        return ["exact"]
    al, bl = a.lower(), b.lower()
    if al == bl:
        f.append("case")
    ca = [c.strip() for c in al.split(",") if c.strip()]
    cb = [c.strip() for c in bl.split(",") if c.strip()]
    if sorted(ca) == sorted(cb) and ca != cb:
        f.append("component_reorder")
    if NUMPRE.search(bl) and not NUMPRE.search(al):
        f.append("number_prefix")
    if LZ.search(bl) and not LZ.search(al):
        f.append("leading_zero")
    if NULLRE.search(bl):
        f.append("null_marker")
    if POBOX.search(bl) and not POBOX.search(al):
        f.append("po_box")
    if (set(words(b)) & CITYSUF) - set(words(a)):
        f.append("city_suffix")
    adm_a = {c for c in ca if c in admin}
    adm_b = {c for c in cb if c in admin}
    if INDIC.search(b):
        f.append("native_script")
    if adm_a != adm_b:
        f.append("admin_missing" if adm_a and not adm_b and not INDIC.search(b) else "admin_changed")
    wa, wb = words(a), words(b)
    if any(N.ADDR_ABBR.get(x, x) == N.ADDR_ABBR.get(y, y) and x != y for x in set(wa) - set(wb) for y in set(wb) - set(wa)):
        f.append("abbreviation")
    if not f:
        f.append("other_format")
    return f


def _rel(xa, xb, numeric):
    if xa == xb:
        return "same"
    if xb < xa:
        return "missing"
    if xb > xa:
        return "added"
    da, db = sorted(xa - xb), sorted(xb - xa)
    if numeric:
        if any(y in x or x.startswith(y) for x in da for y in db):
            return "truncated"
        if any(Levenshtein.distance(x, y) <= 1 for x in da for y in db):
            return "typo"
        return "changed"
    if all(max(fuzz.ratio(x, y) for x in da) >= 75 for y in db) if da else False:
        return "typo"
    return "changed"


def addr_rel(A, B, aempty):
    if aempty:
        return "empty"
    if A == B:
        return "canon_equal"
    na, nb = {t for t in A if any(ch.isdigit() for ch in t)}, {t for t in B if any(ch.isdigit() for ch in t)}
    wa, wb = set(A) - na, set(B) - nb
    return f"num:{_rel(na, nb, True)}|words:{_rel(wa, wb, False)}"


def load_side(c, side, cols):
    raw = (pl.read_parquet(f"{D}train_source1.parquet") if side == "s1" else
           pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet") for k in (2, 3)])).with_row_index("idx")
    can = pl.read_parquet(f"{D}p1_canon_train_{c}_{side}.parquet")
    return can.join(raw.select("idx", "business_name", "business_address"), on="idx").select(
        pl.col("idx"), *[pl.col(k) for k in cols], "business_name", "business_address")


def pairs_frame(c, P, s1, o):
    return (P.join(s1.rename({k: f"{k}_1" for k in s1.columns if k != "idx"}).rename({"idx": "rid"}), on="rid")
            .join(o.rename({k: f"{k}_2" for k in o.columns if k != "idx"}).rename({"idx": "oid"}), on="oid"))


def show(title, counter, examples, total, k=5, top=40):
    print(f"\n--- {title} (n={total:,}) ---")
    for key, n in counter.most_common(top):
        print(f"{key:<34s} {n:>8,} {100 * n / total:6.2f}%")
        for e in examples[key][:k]:
            print(f"      {e}")


def catalogue(c, n_sample=40000, seed=0):
    cols = ["cn", "cns", "cnk", "cak", "aempty", "entity_id"]
    s1, o = load_side(c, "s1", cols), load_side(c, "o", cols)
    admin = N.admin_for(c)
    g = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).sample(n_sample, seed=seed)
    x = pairs_frame(c, g.select("rid", "oid"), s1, o)
    nf, nr, af, ar = Counter(), Counter(), Counter(), Counter()
    ex = defaultdict(list)
    for r in x.iter_rows(named=True):
        a, b = r["business_name_1"], r["business_name_2"]
        A, B = list(r["cn_1"]), list(r["cn_2"])
        fl = name_flags(a, b)
        rel = name_rel(A, B, r["cns_1"], r["cns_2"])
        for q in fl:
            nf[q] += 1
            if len(ex["n:" + q]) < 5 and random.random() < 0.3:
                ex["n:" + q].append(f"{a!r} -> {b!r}")
        nr[rel] += 1
        if len(ex["r:" + rel]) < 5 and random.random() < 0.3:
            ex["r:" + rel].append(f"{a!r} -> {b!r}   [{' '.join(A)}] vs [{' '.join(B)}]")
        aa, bb = r["business_address_1"], r["business_address_2"]
        for q in addr_flags(aa, bb, admin):
            af[q] += 1
            if len(ex["af:" + q]) < 5 and random.random() < 0.3:
                ex["af:" + q].append(f"{aa!r} -> {bb!r}")
        ra = addr_rel(r["cak_1"].split(" ") if r["cak_1"] else [], r["cak_2"].split(" ") if r["cak_2"] else [],
                      r["aempty_2"])
        ra = ra if ra in ("empty", "canon_equal") else ra
        ar[ra] += 1
        if len(ex["ar:" + ra]) < 5 and random.random() < 0.3:
            ex["ar:" + ra].append(f"{aa!r} -> {bb!r}")
    tot = x.height
    print(f"\n==================== {c}: {tot:,} random true pairs ====================")
    show("NAME surface operations (a pair can have several)", nf, {k[2:]: v for k, v in ex.items() if k.startswith("n:")}, tot)
    show("NAME canonical relation (one per pair)", nr, {k[2:]: v for k, v in ex.items() if k.startswith("r:")}, tot)
    show("ADDRESS surface operations", af, {k[3:]: v for k, v in ex.items() if k.startswith("af:")}, tot)
    show("ADDRESS canonical relation", ar, {k[3:]: v for k, v in ex.items() if k.startswith("ar:")}, tot, k=3)
    return s1, o


def decoys(c, s1, o, max_s1=30, max_o=300, n_sample=150000, seed=0):
    """Pairs sharing the canonical address key (and, separately, the name key): who owns the record?"""
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet").filter(pl.col("country") == c).select("rid", "oid")
    own = gt.rename({"rid": "owner"})
    for key, other in (("cak", "name"), ("cnk", "address")):
        k1 = s1.filter(pl.col(key) != "").select("idx", key)
        k2 = o.filter(pl.col(key) != "").select("idx", key)
        sz = k1.group_by(key).agg(n1=pl.len()).join(k2.group_by(key).agg(n2=pl.len()), on=key).filter(
            (pl.col("n1") <= max_s1) & (pl.col("n2") <= max_o))
        P = (k1.join(sz.select(key), on=key).rename({"idx": "rid"})
             .join(k2.rename({"idx": "oid"}), on=key).select("rid", "oid"))
        P = P.join(own, on="oid", how="left").with_columns(
            status=pl.when(pl.col("owner") == pl.col("rid")).then(pl.lit("match"))
            .when(pl.col("owner").is_not_null()).then(pl.lit("other_s1")).otherwise(pl.lit("unowned")))
        tot_true = gt.height
        cover = P.filter(pl.col("status") == "match").height / tot_true
        print(f"\n==================== {c}: decoys sharing the canonical {'ADDRESS' if key == 'cak' else 'NAME'} key "
              f"====================\n{P.height:,} pairs (keys with <= {max_s1} S1 and <= {max_o} records); "
              f"they contain {cover:.3f} of all true pairs; status: "
              + ", ".join(f"{s} {n:,}" for s, n in P.group_by("status").len().sort("status").iter_rows()))
        Ps = P.sample(min(n_sample, P.height), seed=seed)
        x = pairs_frame(c, Ps, s1, o)
        stat, ex = defaultdict(Counter), defaultdict(list)
        for r in x.iter_rows(named=True):
            if key == "cak":
                rel = name_rel(list(r["cn_1"]), list(r["cn_2"]), r["cns_1"], r["cns_2"])
                fl = [q for q in name_flags(r["business_name_1"], r["business_name_2"]) if q.startswith("legal")]
                rel = rel + ("+" + fl[0] if fl and rel.startswith("canon") else "")
                txt = f"{r['business_name_1']!r} -> {r['business_name_2']!r}"
            else:
                rel = addr_rel(r["cak_1"].split(" ") if r["cak_1"] else [], r["cak_2"].split(" ") if r["cak_2"] else [],
                               r["aempty_2"])
                txt = f"{r['business_name_1']!r} | {r['business_address_1']!r} -> {r['business_name_2']!r} | {r['business_address_2']!r}"
            stat[rel][r["status"]] += 1
            if len(ex[(rel, r["status"])]) < 3 and random.random() < 0.2:
                ex[(rel, r["status"])].append(txt)
        rows = sorted(stat.items(), key=lambda kv: -sum(kv[1].values()))
        print(f"{other + ' relation':<40s} {'pairs':>8s} {'match':>7s} {'other_S1':>8s} {'unowned':>8s}")
        for rel, cnt in rows[:28]:
            n = sum(cnt.values())
            print(f"{rel:<40s} {n:>8,} {cnt['match'] / n:7.3f} {cnt['other_s1'] / n:8.3f} {cnt['unowned'] / n:8.3f}")
            for st in ("match", "other_s1", "unowned"):
                for e in ex[(rel, st)][:2]:
                    print(f"      [{st}] {e}")


if __name__ == "__main__":
    random.seed(0)
    for c in sys.argv[1:] or ["US", "India"]:
        s1, o = catalogue(c)
        decoys(c, s1, o)
