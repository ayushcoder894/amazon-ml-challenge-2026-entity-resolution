"""Text normalisation: accent stripping, Indic -> Latin transliteration, abbreviation canonicalisation."""
import re
import unicodedata

import polars as pl

# ---------------------------------------------------------------- Indic transliteration
# Brahmic scripts share one layout: each script block is 0x80 wide and the offset of a
# letter inside its block is (almost) the same across scripts. Fold everything onto the
# Devanagari offsets, then map offsets -> Latin with a small ITRANS-like table.
_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g",
    0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOW = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo", 0x0B: "ri", 0x0E: "e",
        0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x72: "a"}
_MAT = {0x3E: "a", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri", 0x45: "e", 0x46: "e",
        0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x57: "au"}
_MISC = {0x01: "n", 0x02: "n", 0x03: "h", 0x3C: "", 0x4D: "", 0x70: "n", 0x71: ""}
# digits 0x66-0x6F handled separately


def _off(ch):
    c = ord(ch)
    for b in _BLOCKS:
        if b <= c < b + 0x80:
            return c - b
    return None


def translit(s: str) -> str:
    """Rough phonetic Latin rendering of any Brahmic-script text; non-Indic chars pass through."""
    out = []
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        o = _off(ch)
        if o is None:
            out.append(ch)
            i += 1
            continue
        if o in _CONS:
            out.append(_CONS[o])
            nxt = _off(s[i + 1]) if i + 1 < n else None
            if nxt is not None and nxt in _MAT:
                out.append(_MAT[nxt])
                i += 2
                continue
            if nxt == 0x4D:  # virama: no inherent vowel
                i += 2
                continue
            if nxt == 0x3C:  # nukta
                i += 1
            # inherent schwa, dropped at word end (Hindi-style schwa deletion)
            nn = s[i + 1] if i + 1 < n else " "
            if _off(nn) is not None or nn.isalpha():
                out.append("a")
            i += 1
            continue
        if o in _VOW:
            out.append(_VOW[o])
        elif o in _MAT:
            out.append(_MAT[o])
        elif o in _MISC:
            out.append(_MISC[o])
        elif 0x66 <= o <= 0x6F:
            out.append(str(o - 0x66))
        i += 1
    return "".join(out)


_INDIC_RE = re.compile(r"[ऀ-ൿ]")


def has_indic(s: str) -> bool:
    return bool(_INDIC_RE.search(s))


# ---------------------------------------------------------------- canonical token maps
ABBR = {
    # street types (US / India / France) -> canonical short form
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "drive": "dr", "lane": "ln", "court": "ct",
    "place": "pl", "plaza": "plz", "circle": "cir", "parkway": "pkwy", "highway": "hwy", "hiway": "hwy",
    "terrace": "ter", "trail": "trl", "square": "sq", "way": "wy", "suite": "ste", "apartment": "apt",
    "building": "bldg", "floor": "flr", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw", "mount": "mt",
    "fort": "ft", "point": "pt", "center": "ctr", "centre": "ctr", "number": "no", "nagar": "ngr",
    "rue": "r", "chemin": "ch", "impasse": "imp", "allee": "all", "route": "rte", "quai": "qu",
    "passage": "pas", "chaussee": "chs", "faubourg": "fg", "residence": "res", "cours": "crs",
    "sector": "sec", "phase": "ph", "block": "blk", "colony": "col", "opposite": "opp", "opp": "opp",
    "near": "nr", "behind": "bh", "first": "1", "second": "2", "third": "3", "fourth": "4",
    "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "general": "gen", "marechal": "mal", "docteur": "dr", "doctor": "dr", "president": "pdt",
    # name / legal forms
    "private": "pvt", "limited": "ltd", "corporation": "corp", "incorporated": "inc", "company": "co",
    "and": "&", "et": "&", "brothers": "bros", "international": "intl", "associates": "assoc",
    "association": "assn", "services": "service", "technologies": "technology", "tech": "technology",
    "solutions": "solution", "industries": "industry", "enterprises": "enterprise",
    "manufacturing": "mfg", "department": "dept", "university": "univ", "institute": "inst",
    "hospital": "hosp", "shri": "sri", "shree": "sri", "sree": "sri",
    "systems": "sys", "system": "sys", "che": "ch", "chem": "ch", "sainte": "ste", "boul": "blvd", "fbg": "fg",
}

# legal forms / honorifics / filler that the noise generator adds or drops freely
NAME_STOP = {
    "pvt", "ltd", "llc", "l", "c", "inc", "corp", "co", "llp", "plc", "lp", "pc", "pllc", "the", "&", "of",
    "sa", "sas", "sasu", "sarl", "sci", "eurl", "snc", "scp", "selarl", "gie", "sca", "de", "du", "des", "la",
    "le", "les", "d", "sri", "smt", "mr", "mrs", "ms", "m", "dr", "public", "group", "holdings", "holding",
    "service", "ctr", "india", "france", "usa", "us", "dba", "aka", "a", "k", "t", "fka", "com", "www",
    "downtown", "partners", "trust", "and", "enterprise", "ventures", "venture",
    # generator noise found by S2/S3-vs-S1 token over-representation (US / France)
    "southside", "northside", "eastgate", "westgate", "midtown", "greater", "lakeside", "riverside", "central",
    "formerly", "known", "doing", "nee", "lnc", "as",
    "participations", "distribution", "developpement", "groupe", "associes", "etablissements", "ets", "societe",
    # single letters (dotted legal forms S.A.R.L., E.U.R.L., L.L.C. split into letters)
} | set("abcdefghijklmnopqrstuvwxyz")

# state names (US + India + French regions) -> codes, and native-script Indian state names get
# transliterated first so they are handled by the same map via fuzzy prefix later.
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def ascii_fold_expr(col: str) -> pl.Expr:
    """Vectorised lowercase + common accent folding (fast path, no Python per-row)."""
    e = pl.col(col).str.to_lowercase()
    for a, b in [("[àáâãäåā]", "a"), ("[çćč]", "c"), ("[èéêëēė]", "e"), ("[ìíîïī]", "i"), ("[ñń]", "n"),
                 ("[òóôõöøō]", "o"), ("[ùúûüū]", "u"), ("[ýÿ]", "y"), ("œ", "oe"), ("æ", "ae"), ("ß", "ss"),
                 ("[’‘`´]", "'"), ("°", " ")]:
        e = e.str.replace_all(a, b)
    return e


_TDICT = None
_INDIC_WORD = re.compile(r"[ऀ-ൿ]+")
DATA_DIR = "C:/ML/Challenges/amazon/work/data/"


def _tdict():
    """Native-script -> Latin word dictionaries learned from training matches (translit_dict.py)."""
    global _TDICT
    if _TDICT is None:
        import json
        import os
        _TDICT = {}
        for f in ("translit_addr_dict.json", "translit_dict.json"):  # name dictionary wins on conflicts
            p = os.path.join(DATA_DIR, f)
            if os.path.exists(p):
                _TDICT.update(json.load(open(p, encoding="utf8")))
    return _TDICT


def translit_smart(s: str) -> str:
    """Dictionary lookup per native-script word, rule-based transliteration as fallback."""
    d = _tdict()
    return _INDIC_WORD.sub(lambda m: d.get(m.group()) or translit(m.group()), s)


def fold_series(s: pl.Series) -> pl.Series:
    """Lowercase + accents + transliterate Indic (Python path only for rows containing Indic)."""
    df = pl.DataFrame({"x": s}).with_columns(x=ascii_fold_expr("x"))
    mask = df["x"].str.contains(r"[ऀ-ൿ]")
    if mask.any():
        idx = mask.arg_true()
        vals = [translit_smart(v) for v in df["x"].gather(idx).to_list()]
        out = df["x"].scatter(idx, pl.Series(vals, dtype=pl.Utf8))
    else:
        out = df["x"]
    return out


_OCR = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g", "2": "z", "6": "b"}


def addr_tokens_expr(col: str) -> pl.Expr:
    """Address tokens: ordinals stripped (26th -> 26), digits split from letters, leading zeros dropped."""
    return (pl.col(col)
            .str.replace_all(r"(\d)(st|nd|rd|th)\b", "$1")
            .str.replace_all(r"([0-9])([a-z])", "$1 $2")
            .str.replace_all(r"([a-z])([0-9])", "$1 $2")
            .str.replace_all(r"[^a-z0-9]+", " ")
            .str.replace_all(r"\b0+(\d)", "$1")
            .str.strip_chars()
            .str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "").replace(ABBR)))


def name_tokens_expr(col: str) -> pl.Expr:
    """Name tokens: OCR digit->letter repair inside words (Visi0n -> vision), '&'/and unified."""
    e = (pl.col(col)
         .str.replace_all(r"'s\b", "s")
         .str.replace_all(r"\.com\b", " com")
         .str.replace_all(r"[^a-z0-9&]+", " "))
    # repair digits sandwiched in / attached to letters (up to two passes)
    for d, l in _OCR.items():
        e = e.str.replace_all(rf"([a-z]){d}", f"${{1}}{l}").str.replace_all(rf"\b{d}([a-z]{{2,}})", f"{l}$1")
    return (e.str.replace_all("&", " & ").str.strip_chars().str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "").replace(ABBR)))


def skeleton_expr(e: pl.Expr) -> pl.Expr:
    """Crude phonetic consonant skeleton of a single token (transliteration / typo tolerant).
    eastern|eestarn -> strn, prime|praim -> prm, hospitality|hasapitaliti -> hspttlt."""
    for a, b in [("ph", "f"), ("ck", "k"), ("ch", "k"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"),
                 ("kh", "k"), ("gh", "g"), ("q", "k"), ("c", "k"), ("x", "ks"), ("z", "s"), ("w", "v"),
                 ("j", "g"), ("[aeiouy]", "")] + [(c + c + "+", c) for c in "bdfghklmnprstv"]:
        e = e.str.replace_all(a, b)
    return e
