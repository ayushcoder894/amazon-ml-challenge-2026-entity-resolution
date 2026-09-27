"""Audit and prepare bounded probes from the public 0.987 probeA2 baseline.
Invoke heavy modes through guard.sh. All new large outputs are on D:.
"""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
from itertools import zip_longest

import polars as pl

D = Path('C:/ML/Challenges/amazon/work/data')
OUT = Path('C:/ML/Challenges/amazon/work/output')
N = Path('D:/amazon_er/france_night')
R = N / 'probe_099'
R.mkdir(exist_ok=True)
TOTAL = 1732544


def score(tp, npred, nt):
    return 1.0 if npred == nt == 0 else 1.25 * tp / (npred + .25 * nt)


def audit():
    s = pl.read_parquet(D/'test_s1_norm.parquet', columns=['rid','entity_id','country'])
    lookup = {eid:(rid,c) for rid,eid,c in s.iter_rows()}
    comparisons = [('v19','probeA2'), ('probeA2','probeA3'), ('probeA2','probeA19'),
                   ('probeA2','probeB19'), ('probeA2','probeB'), ('probeA2','v20'), ('probeA2','v21')]
    reports=[]
    for base,new in comparisons:
        rows=[]; ent=[]; n=0
        with (OUT/f'matching_results_{base}.tsv').open(encoding='utf8',newline='') as fb, (OUT/f'matching_results_{new}.tsv').open(encoding='utf8',newline='') as fn:
            rb,rn=csv.reader(fb,delimiter='\t'),csv.reader(fn,delimiter='\t')
            assert next(rb)==next(rn)
            for a,b in zip_longest(rb,rn):
                assert a is not None and b is not None and a[0]==b[0], (base,new,n)
                n+=1
                old=set(a[1].split(',')) if a[1] else set()
                cur=set(b[1].split(',')) if b[1] else set()
                add,rem=cur-old,old-cur
                if not add and not rem:
                    continue
                rid,c=lookup[a[0]]
                ent.append((rid,c,len(old),len(cur),len(add),len(rem)))
                rows.extend((rid,c,rid2,'add') for rid2 in add)
                rows.extend((rid,c,rid2,'remove') for rid2 in rem)
        assert n==TOTAL
        x=pl.DataFrame(rows,schema={'rid':pl.UInt32,'country':pl.String,'record_id':pl.String,'kind':pl.String},orient='row')
        e=pl.DataFrame(ent,schema={'rid':pl.UInt32,'country':pl.String,'before':pl.UInt32,'after':pl.UInt32,'added':pl.UInt32,'removed':pl.UInt32},orient='row')
        x.write_parquet(R/f'{base}_to_{new}_pairs.parquet')
        e.write_parquet(R/f'{base}_to_{new}_entities.parquet')
        report={'base':base,'new':new,'changes':x.group_by('country','kind').len().to_dicts(),
                'entities':e.group_by('country').len().to_dicts()}
        if e.height and e['added'].sum()==0:
            table=e.group_by('before','removed').len()
            estimates=[]
            for p in [0.,.2,.4,.6,.8,1.]:
                gain=0
                for k,m,count in table.iter_rows():
                    left=k-m
                    for true in range(m+1):
                        prob=math.comb(m,true)*p**true*(1-p)**(m-true)
                        nt=left+true
                        gain+=count*prob*(score(left,left,nt)-score(nt,k,nt))
                estimates.append({'removed_true_fraction':p,'LB_delta':gain/TOTAL})
            report['removal_scenarios_assuming_kept_correct_no_other_FN']=estimates
        reports.append(report)
        print(json.dumps(report),flush=True)
    (R/'comparison_report.json').write_text(json.dumps(reports,indent=2),encoding='utf8')


def details():
    pl.Config.set_tbl_rows(45)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_fmt_str_lengths(65)
    for name in ['probeB_adds','pool_France_blend','france_union']:
        print(name, pl.read_parquet_schema(N/'data'/f'{name}.parquet'), flush=True)
    for c in ['US','India','France']:
        o=pl.read_parquet(D/f'p1_canon_test_{c}_o.parquet',columns=['idx','entity_id']).rename({'idx':'oid','entity_id':'record_id'})
        for new in ['probeA2','probeA3','probeB']:
            base='v19' if new=='probeA2' else 'probeA2'
            x=pl.read_parquet(R/f'{base}_to_{new}_pairs.parquet').filter(pl.col('country')==c).join(o,on='record_id')
            if not x.height: continue
            print('\nCHANGES',base,new,c,x.height,flush=True)
            if c!='France':
                shift=pl.read_parquet(N/'data'/f'shift3_test_{c}.parquet')
                cols=['rid','oid','p','ws','sib','dbin','pb']
                z=x.join(shift.select(cols),on=['rid','oid'],how='left')
                print(z.group_by('kind','ws','sib','dbin','pb').agg(n=pl.len(),p=pl.col('p').mean()).sort('n',descending=True))
            else:
                pool=pl.read_parquet(N/'data'/'pool_France_blend.parquet')
                z=x.join(pool,on=['rid','oid'],how='left')
                cols=[v for v in ['kind','name','addr','clus'] if v in z.columns]
                print(z.group_by(cols).agg(n=pl.len(),p=pl.col('p').mean()).sort('n',descending=True))
            z.write_parquet(R/f'{base}_to_{new}_{c}_details.parquet')
            del z,x
        del o
        gc.collect()


def residuals():
    """Compare addressed uncertain accepted pairs using real OOF labels and test text."""
    import decide
    from france_rules import categorise
    for c in ['US','India']:
        s=pl.read_parquet(D/'train_s1_norm.parquet',columns=['rid','country']).filter(pl.col('country')==c).join(pl.read_parquet(D/'keep_v11.parquet'),on='rid',how='semi')
        u=pl.scan_parquet('D:/amazon_er/dense/train_joint9.parquet').filter(pl.col('country')==c).select('rid','oid','y',p='pj').collect().join(s,on='rid',how='semi')
        pred=decide.ef_predict(u.select('rid','oid',p2='p'),'p2',0.,1.,.05)
        a=u.join(pred,on=['rid','oid'],how='semi').filter(pl.col('p')<.97)
        del u
        z=categorise(a,'train',c).filter(pl.col('addr')!='empty')
        z=z.with_columns(pb=pl.when(pl.col('p')>=.9).then(pl.lit('high')).otherwise(pl.lit('low')))
        z.write_parquet(R/f'uncertain_train_{c}.parquet')
        ta=z.group_by('name','addr','sh','pb').agg(ntr=pl.len(),true_tr=pl.col('y').sum(),precision=pl.col('y').mean())
        ot=pl.read_parquet(D/f'p1_canon_test_{c}_o.parquet',columns=['idx','cak','cnk']).rename({'idx':'oid'})
        te=pl.read_parquet(N/'data'/f'shift_all_test_{c}.parquet').filter((pl.col('pb')!='5 p>=.97') & ~pl.col('key').str.contains('name-only'))
        p=pl.scan_parquet('D:/amazon_er/dense/test_union_p.parquet').select('rid','oid','p').collect()
        te=te.select('rid','oid').join(p,on=['rid','oid'])
        del p
        zt=categorise(te,'test',c).with_columns(pb=pl.when(pl.col('p')>=.9).then(pl.lit('high')).otherwise(pl.lit('low')))
        zt.write_parquet(R/f'uncertain_test_{c}.parquet')
        tb=zt.group_by('name','addr','sh','pb').agg(nte=pl.len())
        nte=pl.read_parquet(D/f'p1_canon_test_{c}_s1.parquet',columns=['idx']).height
        both=ta.join(tb,on=['name','addr','sh','pb'],how='full',coalesce=True).fill_null(0).with_columns(expected_true=pl.col('true_tr')*nte/s.height).with_columns(excess=pl.col('nte')-pl.col('expected_true'))
        both.write_parquet(R/f'uncertain_cells_{c}.parquet')
        print(c, 'n_train_entities',s.height,'n_test_entities',nte,flush=True)
        print(json.dumps(both.filter(pl.col('nte')>=100).sort('excess',descending=True).head(24).to_dicts()),flush=True)
        del s,pred,a,z,zt,ta,tb,te,ot,both
        gc.collect()


def scenarios():
    reports=[]
    for new,c,operation in [('probeA3','US','remove'),('probeA3','India','remove'),('probeB','France','add')]:
        e=pl.read_parquet(R/f'probeA2_to_{new}_entities.parquet').filter(pl.col('country')==c)
        table=e.group_by('before','added' if operation=='add' else 'removed').len()
        ans={'variant':new,'country':c,'operation':operation,'pairs':int(e['added' if operation=='add' else 'removed'].sum()),'entities':e.height,'initially_empty':e.filter(pl.col('before')==0).height,'scenarios':[]}
        for p in [0.,.2,.4,.6,.7,.8,.9,1.]:
            gain=0.
            for k,m,n in table.iter_rows():
                for t in range(m+1):
                    prob=math.comb(m,t)*p**t*(1-p)**(m-t)
                    if operation=='add':
                        nt=k+t
                        d=score(k+t,k+m,nt)-score(k,k,nt)
                    else:
                        left=k-m;nt=left+t
                        d=score(left,left,nt)-score(nt,k,nt)
                    gain+=n*prob*d
            ans['scenarios'].append({'changed_true_fraction':p,'full_test_macro_delta':gain/TOTAL})
        reports.append(ans)
        print(json.dumps(ans),flush=True)
    (R/'scenario_report.json').write_text(json.dumps(reports,indent=2),encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['audit','details','residuals','scenarios']); a=ap.parse_args()
    {'audit':audit,'details':details,'residuals':residuals,'scenarios':scenarios}[a.mode]()
