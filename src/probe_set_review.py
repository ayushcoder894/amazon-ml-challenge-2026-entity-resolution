"""Review saved set-model scores; keep all name-only decisions and A2 vetoes.
No retraining, no inferred labels. Run every mode through guard.sh.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

import decide
from q7_check import per_entity

D=Path('C:/ML/Challenges/amazon/work/data')
N=Path('D:/amazon_er/france_night')
R=N/'probe_099'
DN=Path('D:/amazon_er/dense')


def modified(base, proposal, addressed):
    return pl.concat([base.join(addressed,on='oid',how='anti'),
                      proposal.join(addressed,on='oid',how='semi')]).unique()


def main(mode):
    rows=[]
    for c in ['India','US']:
        split='train' if mode=='validate' else 'test'
        s=pl.read_parquet(D/f'{split}_s1_norm.parquet',columns=['rid','country']).filter(pl.col('country')==c)
        if split=='train':
            s=s.join(pl.read_parquet(D/'keep_v11.parquet'),on='rid',how='semi')
            u=pl.scan_parquet(DN/'train_joint9.parquet').filter(pl.col('country')==c).select('rid','oid',p='pj').collect().join(s,on='rid',how='semi')
            ps=pl.read_parquet(DN/'jalt_set2_oof.parquet').rename({'p':'ps'}).join(s,on='rid',how='semi')
            base=decide.ef_predict(u,'p',0.,1.,.05)
            gt=pl.read_parquet(D/'train_gt_rid.parquet',columns=['rid','oid']).join(s,on='rid',how='semi')
            be=per_entity(base,gt,s).select('rid',fbase='f')
            deny=None
        else:
            u=pl.scan_parquet(DN/'test_union_p.parquet').select('rid','oid','p').collect().join(s,on='rid',how='semi')
            ps=pl.read_parquet(DN/'test_union_p_set2.parquet').rename({'p':'ps'}).join(s,on='rid',how='semi')
            base=pl.read_parquet(N/'data'/f'shift_all_test_{c}.parquet',columns=['rid','oid'])
            o=pl.read_parquet(D/f'p1_canon_test_{c}_o.parquet',columns=['idx','entity_id']).rename({'idx':'oid','entity_id':'record_id'})
            deny=pl.read_parquet(R/'v19_to_probeA2_pairs.parquet').filter((pl.col('country')==c)&(pl.col('kind')=='remove')).join(o,on='record_id').select('rid','oid')
        addr=pl.read_parquet(D/f'p1_canon_{split}_{c}_o.parquet',columns=['idx','aempty']).filter(~pl.col('aempty')).select(oid='idx')
        assert ps.select(pl.struct('rid','oid').n_unique()).item()==ps.height
        u=u.join(ps,on=['rid','oid'],how='left').with_columns(pl.col('ps').fill_null(pl.col('p')))
        del ps
        print(c,'score coverage rows',u.height,'baseline accepted',base.height,flush=True)
        for weight in ([.5,1.] if mode=='validate' else [.5,1.]):
            x=u.select('rid','oid',q=(1-weight)*pl.col('p')+weight*pl.col('ps'))
            p=modified(base,decide.ef_predict(x,'q',0.,1.,.05),addr)
            if deny is not None: p=p.join(deny,on=['rid','oid'],how='anti')
            rem=base.join(p,on=['rid','oid'],how='anti')
            add=p.join(base,on=['rid','oid'],how='anti')
            row={'country':c,'weight':weight,'removed':rem.height,'added':add.height}
            if split=='train':
                variants={'full':p, 'remove_only':base.join(rem,on=['rid','oid'],how='anti'),
                          'add_only':pl.concat([base,add]).unique()}
                vals={}
                for vn,vp in variants.items():
                    pe=per_entity(vp,gt,s).select('rid',fnew='f').join(be,on='rid').with_columns(delta=pl.col('fnew')-pl.col('fbase'),fold=(pl.col('rid').hash(seed=11)%2))
                    a=pe['delta'].to_numpy(); se=float(a.std(ddof=1)/np.sqrt(len(a)))
                    vals[vn]=dict(score=float(pe['fnew'].mean()),delta=float(a.mean()),ci95=[float(a.mean()-1.96*se),float(a.mean()+1.96*se)],folds=pe.group_by('fold').agg(delta=pl.col('delta').mean()).sort('fold').to_dicts())
                    if vn=='full': pe.write_parquet(R/f'set_addressed_{c}_w{weight}_validation.parquet')
                row.update(base=float(be['fbase'].mean()),variants=vals)
            else:
                rem.write_parquet(R/f'set_addressed_{c}_w{weight}_remove.parquet')
                add.write_parquet(R/f'set_addressed_{c}_w{weight}_add.parquet')
            print(json.dumps(row),flush=True)
            rows.append(row)
            del p,x,rem,add
            gc.collect()
        del u,base,addr,s
        gc.collect()
    (R/f'set_addressed_{mode}.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['validate','test']);a=ap.parse_args();main(a.mode)
