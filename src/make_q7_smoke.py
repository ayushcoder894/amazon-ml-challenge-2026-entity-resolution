"""Create a small, explicitly synthetic stand-in from the existing fake scores."""
from pathlib import Path
import polars as pl

root=Path('D:/amazon_er/gpu5_results')
b=pl.read_parquet('C:/ML/Challenges/amazon/work/gpu5/data/band_loco.parquet',columns=['rid','oid','country'])
q=pl.read_parquet(root/'FAKE_q7_loco.parquet').join(b,on=['rid','oid'])
ents=pl.concat([q.filter(pl.col('country')==c).select('rid').unique().sort('rid').head(100) for c in ['India','US']])
q.join(ents,on='rid',how='semi').select('rid','oid','s').write_parquet(root/'SMOKE_q7_tiny.parquet')
print('Created explicitly synthetic 100-entity-per-country smoke file; never use for model selection')
