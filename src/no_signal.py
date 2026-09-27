"""Name-only namesake signals on India TRAIN (harder split): record script vs owner state; record legal form vs S1 legal form."""
import re
import polars as pl
import decide

D = "C:/ML/Challenges/amazon/work/data/"
DN = "D:/amazon_er/dense/"
SCRIPTS = {"deva": "\u0900-\u097F", "beng": "\u0980-\u09FF", "guru": "\u0A00-\u0A7F", "gujr": "\u0A80-\u0AFF", "orya": "\u0B00-\u0B7F",
           "taml": "\u0B80-\u0BFF", "telu": "\u0C00-\u0C7F", "knda": "\u0C80-\u0CFF", "mlym": "\u0D00-\u0D7F"}
STATES = ["andaman and nicobar", "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chandigarh", "chhattisgarh", "dadra", "daman", "delhi",
          "goa", "gujarat", "haryana", "himachal pradesh", "jammu", "kashmir", "jharkhand", "karnataka", "kerala", "ladakh", "lakshadweep",
          "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "puducherry", "pondicherry", "punjab",
          "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand", "uttaranchal", "west bengal", "chattisgarh"]
ALIAS = {"orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand", "chattisgarh": "chhattisgarh", "kashmir": "jammu"}
STRE = re.compile(r"\b(" + "|".join(sorted(STATES, key=len, reverse=True)) + r")\b")


def script_expr(col):
    e = pl.lit("latin")
    for k, r in reversed(list(SCRIPTS.items())):
        e = pl.when(pl.col(col).str.contains(f"[{r}]")).then(pl.lit(k)).otherwise(e)
    return e


def state_of(a):
    m = STRE.findall((a or "").lower())
    return ALIAS.get(m[-1], m[-1]) if m else "?"


def legal(n):
    n = (n or "").lower()
    if re.search(r"\bllp\b|एलएलपी", n): return "llp"
    if re.search(r"\b(pvt|private)\b|प्राइवेट|प्रा\.|ప్రైవేట్|பிரைவேட்|ಪ್ರೈವೇಟ್|പ്രൈവറ്റ്|প্রাইভেট|પ્રાઇવેટ|પ્રા\.|ପ୍ରାଇଭେଟ୍", n): return "pvt"
    if re.search(r"\b(ltd|limited)\b|लिमिटेड|లిమిటెడ్|லிமிடெட்|ಲಿಮಿಟೆಡ್|ലിമിറ്റഡ്|লিমিটেড|લિમિટેડ|ଲିମିଟେଡ୍", n): return "ltd"
    return "none"


if __name__ == "__main__":
    c = "India"
    keep = pl.read_parquet(f"{D}keep_v11.parquet")
    s1c = pl.read_parquet(f"{D}train_s1_norm.parquet", columns=["rid", "country"]).filter(pl.col("country") == c).select("rid")
    raw1 = pl.read_parquet(f"{D}train_source1.parquet", columns=["business_name", "business_address"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).join(s1c, on="rid", how="semi")
    raw1 = raw1.with_columns(state=pl.Series([state_of(a) for a in raw1["business_address"].to_list()]), leg1=pl.Series([legal(n) for n in raw1["business_name"].to_list()]))
    print("S1 state found:", f"{(raw1['state'] != '?').mean():.1%}")
    oc = pl.read_parquet(f"{D}p1_canon_train_{c}_o.parquet", columns=["idx", "cns", "aempty"]).rename({"idx": "oid"})
    raw2 = pl.concat([pl.read_parquet(f"{D}train_source{k}.parquet", columns=["business_name"]) for k in (2, 3)]).with_row_index("oid").with_columns(pl.col("oid").cast(pl.UInt32)).join(oc, on="oid", how="semi")
    raw2 = raw2.with_columns(script=script_expr("business_name"))
    raw2 = raw2.with_columns(leg2=pl.Series([legal(n) for n in raw2["business_name"].to_list()]))
    gt = pl.read_parquet(f"{D}train_gt_rid.parquet", columns=["rid", "oid"]).join(s1c, on="rid", how="semi")
    T = gt.join(raw1.select("rid", "state", "leg1"), on="rid").join(raw2.select("oid", "script", "leg2"), on="oid")
    tab = T.group_by("state", "script").len().pivot(on="script", index="state", values="len").fill_null(0)
    print(tab.sort("state"))
    lt = T.group_by("leg1", "leg2").len().pivot(on="leg2", index="leg1", values="len").fill_null(0)
    print(lt)
    raw1.select("rid", "state", "leg1").write_parquet("D:/amazon_er/france_night/data/no_s1_train_India.parquet")
    raw2.select("oid", "script", "leg2").write_parquet("D:/amazon_er/france_night/data/no_o_train_India.parquet")
    no = raw2.join(oc.filter(pl.col("aempty")).select("oid"), on="oid", how="semi")
    print("name-only records by script:", no.group_by("script").len().sort("len", descending=True))
