"""Verify baseline parity and both review artifacts; never submits to the leaderboard."""
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import subprocess
import sys

import polars as pl

R=Path('D:/amazon_er/france_night/review_20260927')
OUT=Path('C:/ML/Challenges/amazon/work/output')
RES=Path('C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    a=OUT/'matching_results_v19.tsv'; b=R/'matching_results_v19_audited.tsv'
    n=0
    with a.open(encoding='utf-8',newline='') as fa,b.open(encoding='utf-8',newline='') as fb:
        aa=csv.reader(fa,delimiter='\t'); bb=csv.reader(fb,delimiter='\t')
        assert next(aa)==next(bb)
        for old,new in zip_longest(aa,bb):
            assert old is not None and new is not None and old[0]==new[0]
            assert set(old[1].split(','))==set(new[1].split(',')),old[0]
            n+=1
    assert n==1732544
    source=R/'candidate_pairs_v19_audited.tsv'; target=R/'candidate_pairs_v19_USnear_targeted.tsv'
    shutil.copyfile(source,target)
    assert digest(source)==digest(target)
    for name in ['v19_audited','v19_USnear_targeted']:
        subprocess.run([sys.executable,str(RES/'utils/validate_submission.py'),'--matching',str(R/f'matching_results_{name}.tsv'),
                        '--candidate',str(R/f'candidate_pairs_{name}.tsv'),'--test-dir',str(RES/'dataset/test'),'--check-ids'],check=True)
    manifest=dict(v19_prediction_rows_unchanged=n,candidate_pool_pairs=18523803,candidate_mean=18523803/n,
                  files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in R.glob('*.tsv')},
                  leaderboard_submissions_used=0,real_Qwen_outputs_available=False)
    (R/'verified_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
