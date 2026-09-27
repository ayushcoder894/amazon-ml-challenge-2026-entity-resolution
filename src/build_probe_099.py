"""Build controlled submissions from probeA2 without loading the 100 MB TSV into RAM.

safe_remove: apply only the addressed-record removals proposed by the 50/50 set model.
high_upside: safe_remove plus the France probeB additions (experimental, not recommended
unless the user explicitly chooses the high-variance final slot).
"""
import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

D=Path('C:/ML/Challenges/amazon/work/data')
OUT=Path('C:/ML/Challenges/amazon/work/output')
N=Path('D:/amazon_er/france_night')
R=N/'probe_099'
DEST=Path('D:/amazon_er/final_probes')
RES=Path('C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource')


def pair_changes(mode):
    rem=[]; add=[]
    for c in ['India','US']:
        rem.append(pl.read_parquet(R/f'set_addressed_{c}_w0.5_remove.parquet').with_columns(country=pl.lit(c)))
    if mode=='high_upside':
        add.append(pl.read_parquet(R/'probeA2_to_probeB_pairs.parquet').filter((pl.col('country')=='France')&(pl.col('kind')=='add')).select('rid','record_id','country'))
    return pl.concat(rem), (pl.concat(add) if add else None)


def keyed(mode):
    rm,ad=pair_changes(mode)
    s=pl.read_parquet(D/'test_s1_norm.parquet',columns=['rid','entity_id','country']).select('rid',source1_entity_id='entity_id','country')
    tabs=[]
    for c in ['India','US']:
        o=pl.read_parquet(D/f'p1_canon_test_{c}_o.parquet',columns=['idx','entity_id']).rename({'idx':'oid','entity_id':'record_id'})
        tabs.append(rm.filter(pl.col('country')==c).join(o,on='oid').select('rid','record_id','country'))
    rm=pl.concat(tabs).join(s,on=['rid','country']).select('source1_entity_id','record_id')
    if ad is not None:
        ad=ad.join(s,on=['rid','country']).select('source1_entity_id','record_id')
    return rm,ad


def main(mode):
    DEST.mkdir(parents=True,exist_ok=True)
    rm,ad=keyed(mode)
    changes=defaultdict(lambda:[set(),set()])
    for sid,rid in rm.iter_rows(): changes[sid][0].add(rid)
    if ad is not None:
        for sid,rid in ad.iter_rows(): changes[sid][1].add(rid)
    src=OUT/'matching_results_probeA2.tsv'
    dst=DEST/f'matching_results_probeC_{mode}.tsv'
    rows=removed=added=0
    with src.open(encoding='utf8',newline='') as fi,dst.open('w',encoding='utf8',newline='') as fo:
        ri=csv.reader(fi,delimiter='\t'); wo=csv.writer(fo,delimiter='\t',lineterminator='\n')
        wo.writerow(next(ri))
        for sid,val in ri:
            old=set(val.split(',')) if val else set(); rr,aa=changes.get(sid,(set(),set()))
            cur=(old-rr)|aa; removed+=len(old-cur); added+=len(cur-old); rows+=1
            wo.writerow([sid,','.join(sorted(cur))])
    if rows!=1732544: raise ValueError(f'row count {rows}')
    cand=OUT/'candidate_pairs_probeA2.tsv'
    cmd=[sys.executable,str(RES/'utils/validate_submission.py'),'--matching',str(dst),'--candidate',str(cand),'--test-dir',str(RES/'dataset/test'),'--check-ids']
    print(f'{mode}: rows {rows:,}; removed {removed:,}; added {added:,}; validating...',flush=True)
    subprocess.run(cmd,check=True)
    print(dst,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['safe_remove','high_upside']);a=ap.parse_args();main(a.mode)
