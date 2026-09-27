"""LLM group judgment (in-context clustering) with Qwen2.5-7B-Instruct, zero training: for each reference business the model
sees ALL its candidate records at once and lists the ones that are the same business. Train groups first (evaluated
locally against labels), then test groups. Writes out/grp_{train,test}_ans.parquet (rid, answer) incrementally.
  python group_llm.py [--limit_test N]"""
import argparse
import json
import os
import time

import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = ("You match business records. A reference business is listed with candidate records from other databases. "
       "Records of the SAME business may have typos, abbreviations, missing or extra words, different casing, a legal "
       "suffix added or dropped, a web-style name, or a shortened address. A record at a nearby but DIFFERENT street "
       "number is usually a DIFFERENT business (a neighbour or sister company) - but not always: the same business can "
       "appear with a changed number, especially when several of its records agree with each other. Records with no "
       "address can only be judged by name. Answer ONLY with the numbers of the records that are the same business as "
       "the reference, comma-separated, or 'none'.")


def prompt(tok, ex, ref, recs):
    msgs = [{"role": "system", "content": SYS}]
    for e in ex:
        msgs += [{"role": "user", "content": f"Reference: {e['ref']}\nRecords:\n{e['recs']}"}, {"role": "assistant", "content": e["ans"]}]
    msgs.append({"role": "user", "content": f"Reference: {ref}\nRecords:\n{recs}"})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def groups(df):
    df = df.sort(["rid", "k"])
    out = []
    for (rid,), g in df.group_by(["rid"], maintain_order=True):
        r0 = g.row(0, named=True)
        recs = "\n".join(f"[{i + 1}] {r['rn']} | {r['ra'] or '(no address)'}" for i, r in enumerate(g.iter_rows(named=True)))
        out.append((int(rid), f"{r0['en']} | {r0['ea']}", recs))
    return out


def run(split, tok, model, ex, a, limit=0):
    path = f"{a.out}/grp_{split}_ans.parquet"
    df = pl.read_parquet(f"{a.data}/grp_{split}.parquet")
    G = groups(df)
    skip = {e["rid"] for e in ex}
    G = [g for g in G if g[0] not in skip]
    if limit:
        G = G[:limit]
    done = pl.read_parquet(path) if os.path.exists(path) else pl.DataFrame({"rid": [], "answer": []}, schema={"rid": pl.Int64, "answer": pl.String})
    seen = set(done["rid"].to_list())
    G = [g for g in G if g[0] not in seen]
    P = [prompt(tok, ex, g[1], g[2]) for g in G]
    order = sorted(range(len(P)), key=lambda i: len(P[i]))
    rows, t0 = [], time.time()
    print(time.strftime("%H:%M:%S"), f"[{split}] {len(P):,} groups to judge", flush=True)
    for b in range(0, len(order), a.bs):
        ids = order[b:b + a.bs]
        enc = tok([P[i] for i in ids], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [(G[i][0], t.strip()) for i, t in zip(ids, txt)]
        if (b // a.bs) % 20 == 0 or b + a.bs >= len(order):
            el = time.time() - t0
            print(time.strftime("%H:%M:%S"), f"[{split}] {min(b + a.bs, len(order)):,}/{len(order):,} ({(b + a.bs) / el:.1f}/s, ETA {(len(order) - b - a.bs) / max((b + a.bs) / el, 1e-6) / 60:.0f} min)", flush=True)
            pl.concat([done, pl.DataFrame(rows, schema={"rid": pl.Int64, "answer": pl.String}, orient="row")]).write_parquet(path)
    pl.concat([done, pl.DataFrame(rows, schema={"rid": pl.Int64, "answer": pl.String}, orient="row")]).write_parquet(path)
    print(time.strftime("%H:%M:%S"), f"[{split}] DONE -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=f"{HERE}/out")
    ap.add_argument("--data", default=f"{HERE}/data")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--limit_test", type=int, default=0)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    ex = json.load(open(f"{a.data}/grp_fewshot.json", encoding="utf-8"))
    run("train", tok, model, ex, a)
    run("test", tok, model, ex, a, a.limit_test)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
