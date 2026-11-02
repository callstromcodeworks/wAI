"""Build the visualization page: viz/template.html with the data from extract.py inlined, one self-contained file
(three.js comes from cdnjs, fonts from Google Fonts).

    python model/viz/build.py                     # runs/viz/data.json -> runs/viz/index.html
    python model/viz/build.py --data D --out F"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(MODEL, "runs", "viz", "data.json"))
ap.add_argument("--out", default=os.path.join(MODEL, "runs", "viz", "index.html"))
a = ap.parse_args()
t = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
d = open(a.data, encoding="utf-8").read()
assert t.count("/*DATA*/null") == 1
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
open(a.out, "w", encoding="utf-8").write(t.replace("/*DATA*/null", d.replace("</", "<\\/")))
print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")
