"""Does a specialty learn less when it comes later? Compares continual runs of the same specialties in different orders.

    python model/order_effect.py C2_spec_instruct_chat C2s_spec_instruct_chat R1_spec_reverse

Per specialty, from runs/forgetting/continual_<name>.json (first 10 stages):
- gain: bpb just before its own stage minus just after (what that stage learned);
- total: bpb at the start minus after stage 10 (what the run as a whole gained on it, transfer from other stages
  included, forgetting subtracted).
The last run is compared with the mean of the others (same settings, another order): for each specialty the change in
position and the ratio of gains. Plasticity that falls with accumulated knowledge shows as ratios below 1 for
specialties that moved later and above 1 for those that moved earlier (a negative correlation). The first two runs'
difference gives the noise."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    r = json.load(open(os.path.join(HERE, "runs", "forgetting", f"continual_{name}.json")))
    c = r["curve"]
    return {d: dict(pos=s, before=c[s - 1][d], gain=c[s - 1][d] - c[s][d], total=c[0][d] - c[10][d])
            for s, d in enumerate(r["domains"].split(",")[:10], 1)}


def main():
    names = sys.argv[1:]
    runs = [load(n) for n in names]
    doms = list(runs[0])
    assert all(set(r) == set(doms) for r in runs), "runs must cover the same 10 specialties"
    print(f"{'specialty':11}" + "".join(f" | {n[:14]:>14}: pos  gain  total" for n in names))
    for d in doms:
        print(f"{d:11}" + "".join(f" | {'':>14}  {r[d]['pos']:>3} {r[d]['gain']:.4f} {r[d]['total']:.4f}" for r in runs))
    for n, r in zip(names, runs):
        print(f"{n}: mean gain {np.mean([r[d]['gain'] for d in doms]):.4f}, mean total {np.mean([r[d]['total'] for d in doms]):.4f}")
    if len(runs) >= 3:
        a, b = runs[0], runs[1]
        noise = np.std([b[d]["gain"] - a[d]["gain"] for d in doms], ddof=1)
        print(f"noise ({names[1]} - {names[0]}): sd of gain differences {noise:.4f}")
    if len(runs) >= 2:
        ref, last = runs[:-1], runs[-1]
        print(f"\n{names[-1]} vs mean of {', '.join(names[:-1])}:")
        print(f"{'specialty':11} moved   gain ref -> new  ratio   total ref -> new  ratio")
        moved, lg, lt = [], [], []
        for d in doms:
            g0, t0 = np.mean([r[d]["gain"] for r in ref]), np.mean([r[d]["total"] for r in ref])
            dp = last[d]["pos"] - np.mean([r[d]["pos"] for r in ref])
            print(f"{d:11} {dp:+5.0f}   {g0:.4f} -> {last[d]['gain']:.4f}  {last[d]['gain'] / g0:5.2f}   "
                  f"{t0:.4f} -> {last[d]['total']:.4f}  {last[d]['total'] / t0:5.2f}")
            moved.append(dp)
            lg.append(np.log(last[d]["gain"] / g0))
            lt.append(np.log(last[d]["total"] / t0))
        if np.std(moved) == 0:
            return  # same order: nothing to correlate
        for label, y in (("gain", lg), ("total", lt)):
            r = np.corrcoef(moved, y)[0, 1]
            slope = np.polyfit(moved, y, 1)[0]
            print(f"{label}: correlation of log ratio with position change {r:+.2f}; "
                  f"{(np.exp(slope * 9) - 1) * 100:+.0f}% per 9 positions later")


if __name__ == "__main__":
    main()
