"""Canonical name / address forms for the generator-driven matcher (vectorised polars; works per country).

Country handling: `country` is an open string label. Only the list of administrative-area labels (state / region /
department names and codes) is keyed by country; unknown countries fall back to labels learned from the data
(`learn_admin`): digit-free address components that are very frequent among that country's Source-1 addresses.
Everything else (legal forms, fillers, abbreviations) is one generic map shared by all countries.

canon_name  : lower, accent fold, strip web wrappers (@ # www .com), reverse leetspeak inside words, drop legal forms,
              honorifics, alias markers and learned filler words, drop single letters -> token list (order kept)
canon_addr  : lower, accent fold, drop non-Latin script, split into comma components, drop admin / null / PO-box /
              unit components, strip number prefixes (N°, No., #, H.No) and leading zeros and ordinal suffixes,
              expand street-type abbreviations, drop single letters -> sorted unique token list
"""
import polars as pl

FOLD = [("[àáâãäåāą]", "a"), ("[çćč]", "c"), ("[èéêëēėę]", "e"), ("[ìíîïī]", "i"), ("[ñń]", "n"), ("[òóôõöøō]", "o"),
        ("[ùúûüū]", "u"), ("[ýÿ]", "y"), ("œ", "oe"), ("æ", "ae"), ("ß", "ss"), ("[ºª°]", " ")]

LEGAL = {"llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private", "plc",
         "llp", "lp", "pc", "pllc", "lnc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "scp", "selarl",
         "sca", "gie", "cie", "gmbh", "ag", "bv", "nv", "pty", "opc"}
HONORIFIC = {"mr", "mrs", "ms", "smt", "shri", "sri", "shree", "dr", "m", "s", "messrs"}
ALIAS = {"dba", "aka", "fka", "nee", "formerly", "known", "as", "doing", "business", "trading", "ta"}
# learned filler words (p1_filler.py: share in S2/S3 names >> share in S1 names; see p1_filler_name.txt)
FILLER = {"com", "www", "net", "org", "in", "fr", "midtown", "northside", "southside", "eastgate", "westgate", "greater",
          "riverside", "lakeside", "downtown", "authority", "sys", "id", "the", "and", "et", "of", "participations",
          "holding", "holdings", "distribution", "international", "trading", "associes", "developpement", "groupe",
          "group", "france", "india", "lndia", "fils", "freres", "frs", "services", "service", "center", "centre"}
NAME_DROP = LEGAL | HONORIFIC | ALIAS | FILLER

ADDR_ABBR = {
    "street": "st", "str": "st", "stret": "st", "strete": "st", "saint": "st", "sainte": "ste", "road": "rd",
    "raod": "rd", "avenue": "av", "ave": "av", "avn": "av", "aveue": "av", "aenue": "av", "aveneu": "av", "aveune": "av",
    "avnue": "av", "aevnue": "av", "avneue": "av", "avene": "av", "boulevard": "bd", "blvd": "bd", "boul": "bd",
    "drive": "dr", "lane": "ln", "court": "ct", "place": "pl", "plaza": "plz", "circle": "cir", "parkway": "pkwy",
    "highway": "hwy", "terrace": "ter", "trail": "trl", "square": "sq", "way": "wy", "mount": "mt", "fort": "ft",
    "point": "pt", "north": "n", "south": "s", "east": "e", "west": "w", "rue": "r", "chemin": "ch", "chem": "ch",
    "impasse": "imp", "allee": "all", "alee": "all", "route": "rte", "quai": "qu", "cours": "crs", "passage": "pas",
    "pass": "pas", "psg": "pas", "faubourg": "fg", "fbg": "fg", "residence": "res", "nagar": "ngr", "sector": "sec",
    "block": "blk", "blokc": "blk", "colony": "col", "marg": "mg", "general": "gen", "marechal": "mal",
    "docteur": "dr", "doctor": "dr", "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
}
# tokens that carry no location identity (unit designators, null markers, PO boxes, city-type suffixes)
ADDR_DROP = {"null", "na", "none", "nil", "n", "no", "num", "numero", "unit", "apt", "apartment", "suite", "ste",
             "flat", "floor", "fl", "folor", "flta", "room", "rm", "bldg", "building", "pmb", "po", "box", "bp", "cdp",
             "township", "city", "ity", "ciy", "citty", "ciyt", "ciity", "icty", "ccity", "ctiy", "ownship", "towship",
             "townshp", "tonship", "ttownship", "townshiip", "townsip", "twnship", "townhip", "cpd", "dcp", "ccdp",
             "cddp", "town", "village", "h", "hno", "d", "dno", "plot", "plto", "shop", "sohp", "door", "dor", "doro",
             "bis", "ter", "region", "divreportingcircle"}

_US = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
       "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
       "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
       "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
       "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
       "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
       "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
       "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
       "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
       "district of columbia": "dc", "puerto rico": "pr"}
_IN = {"andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
       "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
       "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
       "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "or", "punjab": "pb",
       "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
       "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "ua", "west bengal": "wb", "delhi": "dl",
       "new delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "jammu & kashmir": "jk", "ladakh": "la",
       "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "dadra and nagar haveli": "dn",
       "daman and diu": "dd", "lakshadweep": "ld", "andaman and nicobar islands": "an", "ct": "cg", "ts": "tg",
       "ori": "od", "tl": "tg"}
_FR = {"hauts-de-france": "hdf", "hauts de france": "hdf", "nouvelle-aquitaine": "naq", "nouvelle aquitaine": "naq",
       "pays de la loire": "pdl", "nord": "59", "pas-de-calais": "62", "pas de calais": "62", "gironde": "33",
       "loire-atlantique": "44", "loire atlantique": "44"}
ADMIN = {"US": set(_US) | set(_US.values()), "India": set(_IN) | set(_IN.values()), "France": set(_FR)}


def fold(e: pl.Expr) -> pl.Expr:
    e = e.str.to_lowercase()
    for a, b in FOLD:
        e = e.str.replace_all(a, b)
    return e


_LEET_D = ["0", "1", "3", "4", "5", "6", "7", "8", "@"]
_LEET_L = ["o", "l", "e", "a", "s", "g", "t", "b", "a"]


INDIC = "[ऀ-ൿ]"


def translit_col(col: str) -> pl.Expr:
    """Native-script (Indic) words -> Latin: dictionary learned from aligned training matches, rule-based fallback
    (norm.translit_smart); only applied to strings that contain such characters."""
    import norm
    return pl.when(pl.col(col).str.contains(INDIC)).then(
        pl.col(col).map_elements(norm.translit_smart, return_dtype=pl.String)).otherwise(pl.col(col))


def canon_name(col: str) -> pl.Expr:
    """Token list (order kept) of the identity-bearing name words."""
    s = fold(translit_col(col))
    s = (s.str.replace_all(r"www\.", " ").str.replace_all(r"\.(com|net|org|co\.in|in|fr)\b", " ")
         .str.replace_all(r"\bt/a\b", " ").str.replace_all(r"[^a-z0-9@ ]+", " "))
    t = s.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    # leetspeak: a digit inside an otherwise alphabetic word ("mais0n", "5ervices", "hea1th")
    t = t.list.eval(pl.when(pl.element().str.contains(r"[a-z]") & pl.element().str.contains(r"[0-9@]")
                            & (pl.element().str.count_matches(r"[a-z]") >= 2))
                    .then(pl.element().str.replace_many(_LEET_D, _LEET_L)).otherwise(pl.element()))
    t = t.list.eval(pl.element().str.replace_all("@", ""))
    return t.list.eval(pl.element().filter((pl.element().str.len_chars() > 1) & ~pl.element().is_in(list(NAME_DROP))))


def canon_addr(col: str, admin: set) -> pl.Expr:
    """Sorted unique token list of the location-bearing address words (admin labels dropped as whole components)."""
    s = (fold(pl.col(col)).str.replace_all(r"[^\x00-\x7f]+", " ")
         .str.replace_all(r"\b(p\.?\s?o\.?\s*box|pmb|b\.?p\.?)\s*[#:]?\s*[a-z0-9-]*\d[a-z0-9-]*", " "))
    comps = s.str.split(",").list.eval(pl.element().str.replace_all(r"[^a-z0-9 ]+", " ").str.replace_all(r"\s+", " ")
                                       .str.strip_chars())
    comps = comps.list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(list(admin))))
    t = comps.list.join(" ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    t = t.list.eval(pl.element().str.replace(r"^(\d+)(st|nd|rd|th|er|e|eme|re)$", "$1").str.replace(r"^0+(\d)", "$1")
                    .replace(ADDR_ABBR))
    keep = ((pl.element().str.len_chars() > 1) | pl.element().str.contains(r"^\d$")) & ~pl.element().is_in(list(ADDR_DROP))
    return t.list.eval(pl.element().filter(keep)).list.unique().list.sort()


def learn_admin(s1_addr: pl.Series, min_share=0.02) -> set:
    """Fallback for an unknown country: digit-free comma components found in >= min_share of S1 addresses."""
    c = (pl.DataFrame({"a": s1_addr}).select(fold(pl.col("a")).str.split(",").alias("c")).explode("c")
         .select(pl.col("c").str.replace_all(r"[^a-z0-9 ]+", " ").str.replace_all(r"\s+", " ").str.strip_chars()))
    vc = c.filter(~pl.col("c").str.contains(r"\d") & (pl.col("c") != "")).group_by("c").len()
    return set(vc.filter(pl.col("len") >= min_share * len(s1_addr))["c"].to_list())


def admin_for(country: str, s1_addr: pl.Series = None) -> set:
    if country in ADMIN:
        return ADMIN[country]
    return learn_admin(s1_addr) if s1_addr is not None else set()


def add_canon(df: pl.DataFrame, country: str, admin: set, name="business_name", addr="business_address") -> pl.DataFrame:
    """Adds cn (name tokens, order kept), cnk (sorted name key), cns (no-space name), ca (address tokens),
    cak (address key), aempty."""
    return df.with_columns(cn=canon_name(name), ca=canon_addr(addr, admin),
                           aempty=pl.col(addr).str.strip_chars() == "").with_columns(
        cnk=pl.col("cn").list.sort().list.join(" "), cns=pl.col("cn").list.join(""),
        cak=pl.col("ca").list.join(" "))
