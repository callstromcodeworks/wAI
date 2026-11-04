"""Recurring topics (wai_struct STRUCT_PLAN 4b): what each run kept, how far topics drifted while away, and how fast
they came back.

    python model/recur_report.py R2_grow R2_practice [--ref S1_stream]

Reads runs/forgetting/continual_<name>.json from forgetting.py --repeats N (with --track_every for savings).
- kept: each specialty's gain (bpb) at the end against the start, beside --ref (S1: the same text in one session);
- after each appearance: the gain right after that day's night;
- drift: how much worse the topic got between the night after one appearance and the day before the next (frozen
  parts don't decay: this is interference from what was learned in between);
- savings (Ebbinghaus): on a return, the wake steps needed to get back to the level the previous appearance's wake
  ended at, against the wake steps the topic first needed to reach that level (cumulative over appearances). 100%:
  nothing was lost; 0%: relearned as slowly as the first time. From the --track_every scores (held-out windows);
- matching (--practice): how often a day was matched to its own topic, and the parts grown."""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return json.load(open(os.path.join(HERE, "runs", "forgetting", f"continual_{name}.json")))


def first_reach(steps, bpb, level, offset=0):
    """Steps (from offset) until bpb first falls to level, interpolated between scores; None if it never does."""
    for i, (s, b) in enumerate(zip(steps, bpb)):
        if b <= level:
            if i == 0:
                return offset + s
            s0, b0 = steps[i - 1], bpb[i - 1]
            return offset + s0 + (s - s0) * (b0 - level) / max(b0 - b, 1e-12)
    return None


def report(r, ref):
    names = list(dict.fromkeys(r["day_names"]))
    curve, start = r["curve"], r["curve"][0]
    end = curve[-1]
    print(f"\n=== {r['name']}: {r['repeats']} appearances per specialty, {len(r['day_names'])} days, "
          f"practice {'oracle' if r.get('practice_oracle') else r.get('practice', 0)}, "
          f"{r['parts_at'][-1] if r.get('parts_at') else '?'} grown parts at the end")
    head = f"{'specialty':11} | kept   " + (f"| {ref['name'][:10]:10} " if ref else "") + "| after each appearance"
    print(head)
    kept = {}
    for n in names:
        days = [d + 1 for d, x in enumerate(r["day_names"]) if x == n]
        kept[n] = start[n] - end[n]
        after = " ".join(f"{start[n] - curve[d][n]:+.4f}" for d in days)
        refk = f"| {ref['curve'][0][n] - ref['curve'][-1][n]:+.4f}    " if ref else ""
        print(f"{n:11} | {kept[n]:+.4f} {refk}| {after}")
    mean = sum(kept[n] for n in names) / len(names)
    refm = (sum(ref["curve"][0][n] - ref["curve"][-1][n] for n in names) / len(names)) if ref else None
    print(f"{'mean':11} | {mean:+.4f} " + (f"| {refm:+.4f}    " if ref else "") +
          f"| English {end['fineweb-edu'] - start['fineweb-edu']:+.4f}, code {end['code'] - start['code']:+.4f} (drift)")

    print("drift while away (bpb worse at the day before a return than after the previous appearance's night):")
    for n in names:
        days = [d + 1 for d, x in enumerate(r["day_names"]) if x == n]
        drift = [curve[b - 1][n] - curve[a][n] for a, b in zip(days, days[1:])]
        if drift:
            print(f"  {n:11} " + " ".join(f"{x:+.4f}" for x in drift))

    tracks = r.get("tracks") or []
    if tracks:
        print("savings on return (wake steps to win back the previous level / steps it first took; 1 - ratio):")
        for n in names:
            ts = [t for t in tracks if t["name"] == n]
            out, done = [], 0  # done: wake steps the topic had before this appearance
            for i in range(1, len(ts)):
                prev, cur = ts[i - 1], ts[i]
                done += prev["step"][-1]
                level = prev["bpb"][-1]
                orig, off = None, 0
                for t in ts[:i]:
                    orig = first_reach(t["step"], t["bpb"], level, off)
                    if orig is not None:
                        break
                    off += t["step"][-1]
                back = first_reach(cur["step"], cur["bpb"], level)
                if orig is None or orig <= 0:
                    out.append("n/a")
                elif back is None:
                    out.append(f"not regained in {cur['step'][-1]} steps")
                else:
                    out.append(f"{back:.0f}/{orig:.0f} steps = {1 - back / orig:.0%}")
            if out:
                print(f"  {n:11} " + " | ".join(out))

    if r.get("matches"):
        m = r["matches"]
        ok = sum(x["correct"] for x in m)
        print(f"matching: {ok}/{len(m)} days right (returns matched to their own topic, first appearances to none)")
        for x in m:
            if not x["correct"]:
                print(f"  wrong: day {x['day']} ({x['name']}) -> "
                      f"{'day ' + str(x['matched']) if x['matched'] else 'none'} | votes {x['share']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default="S1_stream", help="one-session run to compare kept gains with ('' for none)")
    a = ap.parse_args()
    ref = load(a.ref) if a.ref else None
    for n in a.runs:
        report(load(n), ref)


if __name__ == "__main__":
    main()
