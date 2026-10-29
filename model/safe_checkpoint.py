"""Checkpoints that can't run code when loaded: safetensors weights plus a JSON config.

    python model/safe_checkpoint.py export runs/sweep_m32k_concept/ckpt.pt   # -> model.safetensors + model.json
    model, meta = safe_checkpoint.load_model("runs/sweep_m32k_concept")       # at home

A ckpt.pt is a Python pickle: torch.load(..., weights_only=False) runs whatever code it contains, so a
checkpoint from a machine you don't control (a rented GPU) could execute code on yours. Export on the
machine that trained it, bring back only model.safetensors and model.json, and load those: safetensors
holds raw tensors and nothing else, and the JSON is parsed as data. The optimizer state is left out,
so exported runs are for evaluation and generation, not for resuming training.
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def export(ckpt_path, out_dir=None):
    """Write model.safetensors and model.json next to ckpt_path (run this where the checkpoint was made)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out_dir = out_dir or os.path.dirname(os.path.abspath(ckpt_path))
    state = {k: v.detach().contiguous().clone() for k, v in ck["model"].items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    meta = {"config": ck["config"], "args": ck.get("args", {}), "step": ck.get("step"),
            "tokens_seen": ck.get("tokens_seen")}
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return out_dir


def load_model(path, device="cuda", **overrides):
    """(model, meta) from a run directory (or its model.safetensors), without unpickling anything.
    overrides change config fields that don't affect the weights (e.g. seq_len for longer RoPE tables)."""
    run = path if os.path.isdir(path) else os.path.dirname(path)
    with open(os.path.join(run, "model.json")) as f:
        meta = json.load(f)
    from brain import BrainConfig, BrainGPT
    from gpt import GPT, GPTConfig
    cfg_dict = {**meta["config"], **overrides}
    is_brain = cfg_dict.get("arch") == "brain"
    cfg = (BrainConfig if is_brain else GPTConfig)(**cfg_dict)
    with torch.device(device):
        model = BrainGPT(cfg) if is_brain else GPT(cfg)
    model.load_state_dict(load_file(os.path.join(run, "model.safetensors"), device=str(device)))
    return model, meta


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "export":
        raise SystemExit(__doc__)
    for p in sys.argv[2:]:
        print("exported", export(p))
