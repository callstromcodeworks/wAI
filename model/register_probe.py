"""Does the model track register? Formal vs casual contexts, and whether its word choice follows.

    python model/register_probe.py runs/sweep_b32k_sparseB32c2_t1b [more model dirs ...]

Each item: a formal and a casual context ending where either of two same-meaning words fits (formal word, casual
word). register shift = [log p(formal) - log p(casual)] in the formal context minus the same in the casual
context: > 0 means word choice follows the register, 0 means it ignores it. Words may span several tokens (the
continuation's total log-probability is used)."""
import argparse
import math
import os

import torch

from forgetting import load_model
from tokenizer import VOCABS, Tokenizer

# (formal word, casual word, formal context, casual context): either word fits the slot in both
ITEMS = [
    (" assist", " help",
     "Dear Professor Williams,\n\nI am writing to request an extension for the final assignment. I would be most grateful if you could",
     "hey so my laptop totally died last night lol and the essay is due tmrw, can u"),
    (" purchase", " buy",
     "The committee has reviewed the quarterly budget and approved the proposal. The department may now proceed to",
     "omg the new shoes finally dropped and theyre on sale, im so gonna"),
    (" children", " kids",
     "The study examined educational outcomes among primary school pupils. Researchers found that the",
     "lol my sister came over with her whole crew today and the"),
    (" therefore", " so",
     "The sample size was insufficient to establish statistical significance. The results should",
     "honestly i was super tired and kinda grumpy all day, it was"),
    (" approximately", " about",
     "According to the official census report, the population of the region has increased to",
     "dude the line at the coffee place was insane, i waited"),
    (" obtain", " get",
     "Applicants who wish to renew their licence must complete the official form in order to",
     "ugh i need to go to the store real quick to"),
    (" commence", " start",
     "The ceremony is scheduled for nine o'clock. Guests are kindly requested to be seated before the proceedings",
     "ok everyone grab a snack, the movie's gonna"),
    (" inquire", " ask",
     "Should you require further information regarding your application, please do not hesitate to",
     "if ur confused about the homework just"),
    (" sufficient", " enough",
     "The auditors concluded that the documentation provided was not",
     "nah we cant all fit in my car, theres not"),
    (" However", " But",
     "The proposed method demonstrates strong performance on the benchmark datasets.",
     "i really wanted to go to the party tonight lol."),
    (" Good morning", " Hey",
     "Minutes of the Board Meeting, 14 March.\n\nThe Chair opened the meeting at 9:00.\n\nChair:",
     "*group chat*\n\njess: anyone up??\n\nmike:"),
    (" residence", " place",
     "The defendant stated under oath that on the night in question he remained at his",
     "yo wanna come chill at my"),
]


@torch.no_grad()
def logprob(model, tok, context, word):
    ctx = [int(tok.eot)] + [int(t) for t in tok.encode(context)]
    w = [int(t) for t in tok.encode(word)]
    ids = torch.tensor([ctx + w], device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        h = model(ids[:, :-1])
    logp = torch.log_softmax(model.logits(h[0, len(ctx) - 1:]).float(), -1)
    return logp[torch.arange(len(w)), torch.tensor(w, device="cuda")].sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    for path in a.ckpts:
        model, ck = load_model(path)
        model.eval()
        tok = Tokenizer(VOCABS[ck["args"]["vocab"]])
        shifts, right = [], 0
        print(f"== {os.path.basename(path.rstrip('/'))}")
        for f, c, fctx, cctx in ITEMS:
            in_formal = logprob(model, tok, fctx, f) - logprob(model, tok, fctx, c)   # > 0: prefers formal word
            in_casual = logprob(model, tok, cctx, f) - logprob(model, tok, cctx, c)
            shifts.append(in_formal - in_casual)
            right += (in_formal > 0) + (in_casual < 0)
            print(f"  {f.strip():>13s} / {c.strip():<6s} | formal ctx {in_formal:+6.2f} | casual ctx {in_casual:+6.2f} | "
                  f"shift {in_formal - in_casual:+6.2f}")
        mean = sum(shifts) / len(shifts)
        print(f"  mean register shift {mean:+.2f} nats (x{math.exp(mean):.1f} odds), {sum(s > 0 for s in shifts)}/{len(shifts)} "
              f"items shift the right way; matching word preferred in {right}/{2 * len(ITEMS)} contexts", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
