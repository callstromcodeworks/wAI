"""Label-free arrival against the labeled baseline: what each specialty gained, what was forgotten, how well the
buffer caught the new material.

    python model/stream_compare.py C2_spec_instruct_chat C2s_spec_instruct_chat S1_stream S1o_stream_oracle

Runs are read from runs/forgetting (this branch) or runs/main_forgetting (main's results), first 10 stages only:
- gain: bpb just before the specialty's own stage minus just after;
- total: bpb at the start minus after stage 10 (transfer included, forgetting subtracted);
- english / code: bpb change on familiar text from the start to stage 10 (forgetting is positive);
- for stream runs, the buffer each day: windows kept, precision (share that was the specialty), recall (share of
  the specialty's windows kept), familiar text kept by mistake."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    for d in ("forgetting", "main_forgetting"):
        p = os.path.join(HERE, "runs", d, f"continual_{name}.json")
        if os.path.exists(p):
            return json.load(open(p))
    raise SystemExit(f"no continual_{name}.json in runs/forgetting or runs/main_forgetting")


def main():
    names = sys.argv[1:]
    runs = [load(n) for n in names]
    doms = runs[0]["domains"].split(",")[:10]
    for r in runs:
        assert r["domains"].split(",")[:10] == doms, "runs must share the first 10 specialties, in order"
    cols = [n[:18] for n in names]
    print(f"{'specialty':11}" + "".join(f" | {c:>18}" for c in cols) + "   (gain in own stage / total at stage 10)")
    tot = {n: [] for n in names}
    for s, d in enumerate(doms, 1):
        cells = []
        for n, r in zip(names, runs):
            c = r["curve"]
            g, t = c[s - 1][d] - c[s][d], c[0][d] - c[10][d]
            tot[n].append(t)
            cells.append(f"{g:.4f} / {t:.4f}")
        print(f"{d:11}" + "".join(f" | {x:>18}" for x in cells))
    print(f"{'mean total':11}" + "".join(f" | {np.mean(tot[n]):18.4f}" for n in names))
    for key, label in (("fineweb-edu", "english"), ("code", "code")):
        print(f"{label:11}" + "".join(f" | {r['curve'][10][key] - r['curve'][0][key]:+18.4f}" for r in runs))
    for n, r in zip(names, runs):
        st = r.get("stream_stats") or []
        if not st:
            continue
        print(f"\n{n} buffer by day ({r.get('surprise')}, q {r.get('surprise_q')}"
              f"{', oracle' if r.get('stream_oracle') else ''}):")
        for i, (d, x) in enumerate(zip(doms, st), 1):
            print(f"  day {i:2} {d:11} kept {x['kept']:5}/{x['read']:5} | precision {x['precision']:.3f} "
                  f"recall {x['recall']:.3f} familiar kept {x['false_keep']:.3f}")
        print(f"  mean precision {np.mean([x['precision'] for x in st]):.3f}, recall "
              f"{np.mean([x['recall'] for x in st]):.3f}")


if __name__ == "__main__":
    main()
