import polars as pl, os, sys
R = r"C:\ML\Challenges\amazon\6ab10eb3b23ba_student_resource\student_resource\dataset"
O = r"C:\ML\Challenges\amazon\work\data"
for split in ["train", "test"]:
    for f in sorted(os.listdir(os.path.join(R, split))):
        if not f.endswith(".tsv"): continue
        df = pl.read_csv(os.path.join(R, split, f), separator="\t", quote_char=None,
                         infer_schema=False, missing_utf8_is_empty_string=True)
        df.write_parquet(os.path.join(O, f.replace(".tsv", ".parquet")))
        print(f, df.shape, df.columns, flush=True)
