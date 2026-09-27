"""Build the final submission package folder (code with portable paths + README + requirements + outputs)."""
import os
import re
import shutil
import sys

W = "C:/ML/Challenges/amazon/work/"
TEAM = sys.argv[1] if len(sys.argv) > 1 else "TEAM"
PKG = f"{W}submission/{TEAM}_submission/"
CODE = f"{PKG}code/business_entity_resolution/"
SRC_FILES = ["to_parquet.py", "make_gt.py", "norm.py", "segment.py", "translit_dict.py", "prep.py", "block.py",
             "features.py", "siblings.py", "train.py", "train2.py", "train3.py", "predict3.py", "decide.py",
             "evaluate.py", "ce.py", "ce_combine.py", "final_ce.py", "ce_multi.py", "twin_check.py", "twin_feats.py", "make_drop_folds.py", "orphan.py", "france_variant.py", "loco_blend.py"]
HEADER = ('import os as _os\n'
          '_HERE = _os.path.dirname(_os.path.abspath(__file__))\n'
          'WORK = _os.environ.get("ER_WORK", _os.path.join(_HERE, "..", "work")).replace("\\\\", "/").rstrip("/") + "/"\n'
          'RESOURCE = _os.environ.get("ER_RESOURCE", _os.path.join(_HERE, "..", "student_resource")).replace("\\\\", "/")'
          '.rstrip("/") + "/"\n')


def portable(text):
    text = text.replace('"C:/ML/Challenges/amazon/6ab10eb3b23ba_student_resource/student_resource/', 'RESOURCE + "')
    text = text.replace('r"C:\\ML\\Challenges\\amazon\\6ab10eb3b23ba_student_resource\\student_resource\\dataset"',
                        'RESOURCE + "dataset"')
    text = text.replace('r"C:\\ML\\Challenges\\amazon\\work\\data"', 'WORK + "data"')
    text = text.replace('"C:/ML/Challenges/amazon/work/', 'WORK + "')
    text = re.sub(r'WORK \+ "(\w+/?)"', r'WORK + "\1"', text)
    assert "C:/ML" not in text and "C:\\ML" not in text, "hard-coded path left"
    # insert header after the module docstring
    m = re.match(r'(\s*""".*?"""\n)', text, re.S)
    return (m.group(1) + HEADER + text[m.end():]) if m else HEADER + text


def main():
    if os.path.exists(PKG):
        shutil.rmtree(PKG)
    os.makedirs(f"{CODE}src", exist_ok=True)
    os.makedirs(f"{PKG}output", exist_ok=True)
    for f in SRC_FILES:
        with open(f"{W}src/{f}", encoding="utf8") as fh:
            txt = fh.read()
        with open(f"{CODE}src/{f}", "w", encoding="utf8") as fh:
            fh.write(portable(txt))
    for f in ("README.md", "requirements.txt", "run_all.sh"):
        shutil.copy(f"{W}package_files/{f}", f"{CODE}{f}")
    shutil.copy(f"{W}package_files/Documentation_template.md", f"{PKG}Documentation_template.md")
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(f"{W}output/{f}", f"{PKG}output/{f}")
    print("package at", PKG)


if __name__ == "__main__":
    main()
