"""Bounded continuation diagnostics and a separate, shift-dependent US submission.
Always invoke via guard.sh with LIMIT=58 and PYTHONUTF8=1. Outputs live on D:.
"""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import polars as pl

import decide
import evaluate as ev
from nearby_number import pattern, sib_agree
from france_rules import categorise

D=Path('C:/ML/Challenges/amazon/work/data')
OUT=Path('C:/ML/Challenges/amazon/work/output')
N=Path('D:/amazon_er/france_night')
DEST=N/'review_20260927'
DEST.mkdir(exist_ok=True)
RD=dict(separator='\t', quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def ids(c):
    s=pl.read_parquet(D/f'p1_canon_test_{c}_s1.parquet',columns=['idx','entity_id']).rename({'idx':'rid','entity_id':'source1_entity_id'})
    o=pl.read_parquet(D/f'p1_canon_test_{c}_o.parquet',columns=['idx','entity_id']).rename({'idx':'oid','entity_id':'record_id'})
    return s,o


def submission(c, version, candidate=False):
    s,o=ids(c)
    col='candidate_entity_ids' if candidate else 'matched_entity_ids'
    path=OUT/f"{'candidate_pairs' if candidate else 'matching_results'}_{version}.tsv"
    x=(pl.read_csv(path,**RD).join(s,on='source1_entity_id',how='inner').filter(pl.col(col)!='')
       .with_columns(record_id=pl.col(col).str.split(',')).explode('record_id')
       .join(o,on='record_id').select('rid','oid'))
    assert x.height==x.unique().height
    return x


def dump_json(name,obj):
    (DEST/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps(obj,indent=2),flush=True)


def f(tp,k,n):
    return 1.0 if k==n==0 else 0.0 if tp==0 else 1.25*tp/(k+.25*n)


def scenario(t, precision):
    gain=0.0
    for k,m,count in t.group_by('k','m').len().iter_rows():
        for true in range(m+1):
            prob=math.comb(m,true)*precision**true*(1-precision)**(m-true)
            nt=k-m+true
            gain+=count*prob*(f(k-m,k-m,nt)-f(nt,k,nt))
    return gain/663106


def us():
    keep=pl.read_parquet(D/'keep_v11.parquet')
    s1=(pl.read_parquet(D/'train_s1_norm.parquet',columns=['rid','country']).filter(pl.col('country')=='US')
        .join(keep,on='rid',how='semi').with_columns(fold=(pl.col('rid').hash(seed=11)%2)))
    gt=pl.read_parquet(D/'train_gt_rid.parquet',columns=['rid','oid']).join(s1.select('rid'),on='rid',how='semi')
    u=(pl.scan_parquet('D:/amazon_er/dense/train_joint9.parquet').filter(pl.col('country')=='US')
       .select('rid','oid',p2=pl.col('pj')).collect())
    p=decide.ef_predict(u,'p2',0.,1.,.05)
    pat,o=pattern('train','US')
    a=sib_agree(pat.join(p,on=['rid','oid'],how='semi'),p,o,'sib').filter((pl.col('sib')=='odd one out') & pl.col('diff').is_between(1,50))
    new=p.join(a.select('rid','oid'),on=['rid','oid'],how='anti')
    stats={'rule':'US only; same canonical name and nonnumeric address words; non-truncated numeric difference 1..50; every informative other predicted sibling supports S1 number and none supports record number',
           'validation':[], 'caution':'Shift-dependent variant. Test correctness is not known. No hand-made labels were used for local validation.'}
    for fold in [0,1,None]:
        ss=s1 if fold is None else s1.filter(pl.col('fold')==fold)
        g=gt.join(ss.select('rid'),on='rid',how='semi')
        before=ev.macro_f05(p.join(ss.select('rid'),on='rid',how='semi'),g,ss['rid'])
        after=ev.macro_f05(new.join(ss.select('rid'),on='rid',how='semi'),g,ss['rid'])
        ar=a.join(ss.select('rid'),on='rid',how='semi')
        stats['validation'].append(dict(fold=fold,baseline=before,variant=after,delta=after-before,removed=ar.height,
                                          true_removed=ar.join(g,on=['rid','oid']).height))
    del pat,o,u,p,new,a,gt
    gc.collect()
    p=submission('US','v19')
    pat,o=pattern('test','US')
    accepted=sib_agree(pat.join(p,on=['rid','oid'],how='semi'),p,o,'sib')
    a=accepted.filter((pl.col('sib')=='odd one out') & pl.col('diff').is_between(1,50))
    a.write_parquet(DEST/'us_near_targeted_removed.parquet')
    s,oi=ids('US')
    removed=a.select('rid','oid').join(s,on='rid').join(oi,on='oid')
    forbidden=accepted.filter(pl.col('sib')=='siblings share rec').select('rid','oid')
    assert a.join(forbidden,on=['rid','oid']).is_empty()
    tab=a.group_by('rid').agg(m=pl.len()).join(p.group_by('rid').agg(k=pl.len()),on='rid')
    stats['test']=dict(removed=a.height,entities=a['rid'].n_unique(),siblings_share_record_removed=0,
                       bins=a.group_by('dbin').len().to_dicts())
    stats['scenario_assuming_other_predictions_correct']=[dict(precision=q,US_delta=scenario(tab,q),LB_delta=scenario(tab,q)*663106/1732544) for q in [.4,.6,.73,.9,.979]]
    by_id={r:set(v) for r,v in removed.group_by('source1_entity_id').agg(pl.col('record_id')).iter_rows()}
    mp=DEST/'matching_results_v19_USnear_targeted.tsv'
    cp=DEST/'candidate_pairs_v19_USnear_targeted.tsv'
    changes=0
    with (OUT/'matching_results_v19.tsv').open(encoding='utf-8',newline='') as fi, mp.open('w',encoding='utf-8',newline='') as fo:
        rd=csv.reader(fi,delimiter='\t'); wr=csv.writer(fo,delimiter='\t',lineterminator='\n')
        wr.writerow(next(rd))
        for r,vals in rd:
            if r in by_id:
                old=vals.split(',') if vals else []
                nn=[v for v in old if v not in by_id[r]]
                changes+=len(old)-len(nn)
                vals=','.join(nn)
            wr.writerow([r,vals])
    assert changes==a.height
    shutil.copyfile(OUT/'candidate_pairs_v19.tsv',cp)
    validator=Path('C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py')
    subprocess.run([sys.executable,str(validator),'--matching',str(mp),'--candidate',str(cp),'--test-dir',str(validator.parent.parent/'dataset/test')],check=True)
    stats['outputs']=[str(mp),str(cp)]
    dump_json('us_near_targeted_report.json',stats)


def deltas():
    rows=[]
    for c in ['India','US']:
        old=submission(c,'v13_ce'); new=submission(c,'v19')
        for kind,x in [('added',new.join(old,on=['rid','oid'],how='anti')),('removed',old.join(new,on=['rid','oid'],how='anti'))]:
            print(c,kind,x.height,flush=True)
            x.write_parquet(DEST/f'{c}_v13ce_v19_{kind}.parquet')
            cat=categorise(x.with_columns(p=pl.lit(0.0)),'test',c)
            grouped=cat.group_by('name','addr','sh').len().sort('len',descending=True)
            grouped.write_csv(DEST/f'{c}_v13ce_v19_{kind}_categories.csv')
            rows.append(dict(country=c,kind=kind,pairs=x.height,entities=x['rid'].n_unique(),top_categories=grouped.head(12).to_dicts()))
        del old,new
        gc.collect()
    dump_json('v13ce_v19_changes.json',rows)


def france():
    from france_cats import canon
    from france_v15 import raw_form_match
    s,o,_=canon('test','France')
    u=pl.read_parquet(N/'data/france_union.parquet').select('rid','oid','p','pb')
    pred=pl.read_parquet(N/'data/france_v19_pairs.parquet').select('rid','oid')
    cand=submission('France','v19',candidate=True)
    # Direct exact-address enumeration is independent of the candidate generator.
    addr=(s.filter(pl.col('cak')!='').select(rid='idx',key='cak').join(
          o.filter(~pl.col('aempty') & (pl.col('cak')!='')).select(oid='idx',key='cak'),on='key').select('rid','oid'))
    missing=addr.join(cand,on=['rid','oid'],how='anti')
    mc=categorise(missing.with_columns(p=pl.lit(0.)),'test','France')
    raw1=pl.read_parquet(D/'test_source1.parquet',columns=['business_name','business_address']).with_row_index('rid')
    raw2=pl.concat([pl.read_parquet(D/f'test_source{k}.parquet',columns=['business_name','business_address']) for k in [2,3]]).with_row_index('oid')
    def raw(x):
        return x.join(raw1.rename({'business_name':'entity_name','business_address':'entity_address'}),on='rid').join(
            raw2.rename({'business_name':'record_name','business_address':'record_address'}),on='oid')
    mc=raw(mc)
    mc=mc.with_columns(raw_form=pl.Series([raw_form_match(a,b) for a,b in mc.select('entity_name','record_name').iter_rows()],dtype=pl.String))
    strong=mc.filter(pl.col('name').is_in(['canon_equal','canon_equal_reordered','acronym','truncation_single_prefix','truncation','extra_word','invented_pseudo_word']) | pl.col('raw_form').is_not_null())
    strong.write_parquet(DEST/'france_strong_exact_address_not_candidates.parquet')
    strong.head(200).write_csv(DEST/'france_strong_exact_address_not_candidates_examples.csv')
    # One changed canonical token in each direction, then use the established categorizer.
    swaps=(u.join(s.select(rid='idx',a='cn'),on='rid').join(o.select(oid='idx',b='cn'),on='oid')
           .filter((pl.col('a').list.set_difference('b').list.len()==1) & (pl.col('b').list.set_difference('a').list.len()==1))
           .select('rid','oid','p','pb'))
    cats=categorise(swaps.select('rid','oid','p'),'test','France').filter(pl.col('name').str.starts_with('word_swap_real_word'))
    cats=(cats.join(swaps.select('rid','oid','pb'),on=['rid','oid']).join(pred.with_columns(accepted=pl.lit(True)),on=['rid','oid'],how='left')
          .join(pred.select('oid').unique().with_columns(record_taken=pl.lit(True)),on='oid',how='left')
          .with_columns(pl.col('accepted','record_taken').fill_null(False)))
    sc=pl.read_parquet('D:/amazon_er/gpu3_results/ce3_e5l_france_test.parquet').rename({'s':'e5l'})
    cats=cats.join(sc,on=['rid','oid'],how='left').join(pl.read_parquet('D:/amazon_er/gpu3_results/ce3_bgem3_france_test.parquet').rename({'s':'bgem3'}),on=['rid','oid'],how='left')
    tab=cats.group_by('name','addr','sh').agg(n=pl.len(),accepted=pl.col('accepted').sum(),taken=pl.col('record_taken').sum(),
             p_mean=pl.col('p').mean(),pb_mean=pl.col('pb').mean(),e5l=pl.col('e5l').mean(),bgem3=pl.col('bgem3').mean()).sort('n',descending=True)
    tab.write_csv(DEST/'france_word_swap_breakdown.csv')
    raw(cats.sort('pb',descending=True).head(200)).write_csv(DEST/'france_word_swap_examples.csv')
    dump_json('france_diagnostics.json',dict(exact_address_pairs=addr.height,not_in_submitted_candidates=missing.height,
         strong_missing=strong.height,strong_missing_by_category=strong.group_by('name','sh').len().sort('len',descending=True).to_dicts(),
         scored_union_pairs_not_in_candidate_file=u.join(cand,on=['rid','oid'],how='anti').height,
         submitted_candidates_not_in_union=cand.join(u.select('rid','oid'),on=['rid','oid'],how='anti').height,
         word_swap_categories=tab.head(25).to_dicts(),
         caveat='Strong-pattern pairs are hypotheses, not labeled matches. No new France labels or predictions were created. One-token swaps are canonically defined; counts need not equal a prior hand-defined category-word subset.'))


def gaps():
    x=pl.read_parquet(DEST/'france_strong_exact_address_not_candidates.parquet')
    u=pl.read_parquet(N/'data/france_union.parquet',columns=['rid','oid','p','pb'])
    pr=pl.read_parquet(N/'data/france_v19_pairs.parquet').select('rid','oid')
    x=(x.drop('p').join(u.with_columns(scored=pl.lit(True)),on=['rid','oid'],how='left')
       .join(pr.select('oid').unique().with_columns(taken=pl.lit(True)),on='oid',how='left')
       .with_columns(pl.col('scored','taken').fill_null(False)))
    x.write_parquet(DEST/'france_strong_gap_status.parquet')
    unique=x.filter(~pl.col('sh') & ~pl.col('scored'))
    unique.write_csv(DEST/'france_unique_address_unscored_examples.csv')
    dump_json('france_gap_status.json',dict(counts=x.group_by('sh','scored','taken').len().sort('len',descending=True).to_dicts(),
        unique_address_never_scored=unique.height,unique_unscored_categories=unique.group_by('name').len().sort('len',descending=True).to_dicts()))


def trace():
    x=pl.read_parquet(DEST/'france_strong_gap_status.parquet').filter(~pl.col('sh') & ~pl.col('scored')).select('rid','oid','name','taken')
    raw=(pl.scan_parquet('D:/amazon_er/data/test_cand.parquet').select('rid','oid','r1','r2')
         .join(x.select('rid','oid').lazy(),on=['rid','oid'],how='semi').collect())
    den=(pl.scan_parquet('D:/amazon_er/dense/dense_test_new.parquet').select('rid','oid','fwd_rank','rev_rank')
         .join(x.select('rid','oid').lazy(),on=['rid','oid'],how='semi').collect())
    x=x.join(raw.with_columns(in_raw=pl.lit(True)),on=['rid','oid'],how='left').join(den.with_columns(in_dense=pl.lit(True)),on=['rid','oid'],how='left')
    x=x.with_columns(pl.col('in_raw','in_dense').fill_null(False))
    x.write_parquet(DEST/'france_unscored_stage_trace.parquet')
    dump_json('france_unscored_stage_trace.json',dict(pairs=x.height,stages=x.group_by('in_raw','in_dense','taken').len().to_dicts(),
        note='Raw blocker cache is test_cand.parquet. Absence there means never reached that stored pool, not proof of which individual key or rank filter failed.',
        raw_present_examples=x.filter(pl.col('in_raw')).head(20).to_dicts(),dense_present_examples=x.filter(pl.col('in_dense')).head(20).to_dicts()))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['us','deltas','france','gaps','trace']); args=ap.parse_args()
    {'us':us,'deltas':deltas,'france':france,'gaps':gaps,'trace':trace}[args.mode]()
