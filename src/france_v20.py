"""Build Qwen + v18 protection + rule S only after the real two-country check passes.
Run under guard.sh, then optionally assemble/validate with --assemble. Never uploads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import polars as pl

from q7_check import build, protect
from france_v19 import rule_s

N=Path('D:/amazon_er/france_night')


def main(qfile, report, assemble=False, ws_override=None, alpha_override=None, name='v20'):
    check=json.loads(Path(report).read_text(encoding='utf-8'))
    if check.get('synthetic') or not check.get('complete') or not check.get('eligible_for_v20'):
        raise ValueError('Real, complete, positive held-out results in BOTH proxy countries are required')
    if any(s in Path(qfile).name.lower() for s in ['fake','smoke']):
        raise ValueError('Synthetic France scores are not allowed')
    q=pl.read_parquet(qfile).select(pl.col('rid','oid').cast(pl.UInt32),'s')
    from france_v15 import mix_q7
    mix_q7(q.head(0),q,.5)
    band=pl.read_parquet('C:/ML/Challenges/amazon/work/gpu5/data/band_france.parquet').filter(pl.col('p')>=.2).select('rid','oid')
    if q.height!=band.height or q.join(band,on=['rid','oid'],how='anti').height:
        raise ValueError(f'France score coverage mismatch: {q.height:,} scores vs {band.height:,} expected p>=0.2 pairs')
    os.environ['Q7_FILE']=str(Path(qfile).resolve())
    ws,alpha=str(check['selected']['ws']),check['selected']['alpha']
    if ws_override is not None:  # proxy-reviewed alternative (both halves, both countries positive in the report)
        ws,alpha=str(ws_override),alpha_override
    A,_,_=build('France','0','avg')
    B,U,split=build('France',ws,'q7',alpha)
    # France-only extra protections from v18; use the same ranking and ownership logic.
    from q7_check import PROT
    cats=PROT+[('truncation_single_prefix','canon_equal'),('truncation','canon_equal')]
    C=protect(A,B,U,split,'France',cats=cats)
    S=rule_s('France','test',C)
    P=pl.concat([C,S]).unique()
    assert P['oid'].n_unique()==P.height
    old=pl.read_parquet(N/'data/france_v19_pairs.parquet').select('rid','oid')
    P.write_parquet(N/f'data/france_{name}_pairs.parquet')
    meta=dict(qfile=str(Path(qfile).resolve()),q_sha256=hashlib.sha256(Path(qfile).read_bytes()).hexdigest(),
              validation_report=str(Path(report).resolve()),ws=float(ws),alpha=alpha,pairs=P.height,
              added_vs_v19=P.join(old,on=['rid','oid'],how='anti').height,
              removed_vs_v19=old.join(P,on=['rid','oid'],how='anti').height,rule_s_additions=S.height)
    (N/f'review_20260927/france_{name}_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)
    if assemble:
        env=dict(os.environ,FR_PAIRS=f'france_{name}_pairs',SUBMISSION_OUT=str(N/'review_20260927'),PYTHONUTF8='1')
        subprocess.run([sys.executable,str(Path(__file__).with_name('build_v15.py')),'v13_e9',name],env=env,check=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('qfile'); ap.add_argument('report'); ap.add_argument('--assemble',action='store_true'); ap.add_argument('--ws',type=float); ap.add_argument('--alpha',type=float); ap.add_argument('--name',default='v20')
    args=ap.parse_args(); main(args.qfile,args.report,args.assemble,args.ws,args.alpha,args.name)
