"""Build a France-blanked diagnostic from the exact v19 leaderboard baseline.

Use only as a country-score probe. Matching predictions outside France are byte-for-byte
unchanged; candidate pairs remain the full audited v19 scored pool.
"""
import csv
from pathlib import Path

import polars as pl


ROOT = Path('D:/amazon_er/france_night/review_20260927')
BASE = Path('C:/ML/Challenges/amazon/work/output/matching_results_v19.tsv')
OUT = ROOT / 'matching_results_v19_blankFR.tsv'
S1 = Path('C:/ML/Challenges/amazon/work/data/test_s1_norm.parquet')


def main():
    fr_ids = set(pl.read_parquet(S1, columns=['entity_id', 'country'])
                 .filter(pl.col('country') == 'France')['entity_id'].to_list())
    if len(fr_ids) != 259452:
        raise ValueError(f'Unexpected France entity count: {len(fr_ids)}')
    n = blanked = 0
    with BASE.open('r', encoding='utf-8', newline='') as src, OUT.open('w', encoding='utf-8', newline='') as dst:
        reader = csv.reader(src, delimiter='\t')
        writer = csv.writer(dst, delimiter='\t', lineterminator='\n')
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if len(row) != 2:
                raise ValueError(f'Malformed row {n+2}')
            if row[0] in fr_ids:
                blanked += 1
                row[1] = ''
            writer.writerow(row)
            n += 1
    if n != 1732544 or blanked != len(fr_ids):
        raise ValueError(f'Row/count mismatch: {n}, {blanked}')
    print(f'{OUT} rows={n:,} France blanked={blanked:,}; candidate file = {ROOT / "candidate_pairs_v19_audited.tsv"}')


if __name__ == '__main__':
    main()
