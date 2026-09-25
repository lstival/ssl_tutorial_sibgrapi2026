"""
Modality-agnostic training-loop plumbing shared by the offline pretraining / evaluation scripts
of both tracks (src/remote_sensing/pretraining and src/time_series/pretraining).

The notebooks never import this module: tutorial_rs.py / tutorial_ts.py stay single-file so a
Colab runtime can download just one of them. Each track's `pretrain_utils*.py` re-exports these
helpers, so the scripts keep importing from their own track.
"""

import json
import math
import os
import time

import torch


def enable_fast_cuda(device):
    """TF32 matmuls + cuDNN autotuning: free speedups for fixed-shape transformers on Ampere+."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def warmup_cosine_lambda(total_steps, warmup_steps, floor=1.0 / 50):
    """
    `LambdaLR` multiplier: linear warmup over `warmup_steps`, then cosine decay to `floor` * lr
    at `total_steps`. The lr/50 floor matches the `eta_min=lr/50` the cosine-only runs use.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def train_log_path(out_path):
    """
    JSONL log path derived from the checkpoint path. RunningLogger appends, so a fixed per-folder
    name would interleave a new run's steps (restarting at 1) into an older run's history.
    """
    return os.path.splitext(out_path)[0] + "_train_log.jsonl"


def save_encoder_checkpoint(encoder, out_path, extra=None):
    """
    Save only the encoder's state dict (no pretraining-only head/decoder) -- the format the
    notebooks' `try_load_checkpoint` loads into a bare encoder -- plus an optional JSON sidecar.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(encoder.state_dict(), out_path)
    if extra is not None:
        meta_path = os.path.splitext(out_path)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2)
    print(f"Saved encoder checkpoint to {out_path}")


def save_train_state(path, *, step, model, optimizer, scheduler, extra=None):
    """
    Save a *complete* training checkpoint (model + optimizer + scheduler + step), separate from
    the encoder-only checkpoint the notebooks consume, so `--resume` continues exactly where the
    run left off. Written atomically: an interrupt mid-write leaves the previous file intact.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if extra is not None:
        state["extra"] = extra
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_train_state(path, model, optimizer, scheduler, device):
    """
    Restore a checkpoint written by `save_train_state`.

    Returns (step, extra): the loop continues from step+1, and `extra` is whatever the caller
    stashed (e.g. DINO's teacher and center). Returns (0, {}) if no checkpoint exists.
    """
    if not os.path.isfile(path):
        return 0, {}
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"Resumed training state from {path} at step {state['step']}")
    return state["step"], state.get("extra", {})


class RunningLogger:
    """Minimal step/metric logger: prints every `print_every` steps and appends a JSONL history."""

    def __init__(self, log_path, print_every=50):
        self.log_path = log_path
        self.print_every = print_every
        self.start_time = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._f = open(log_path, "a", encoding="utf-8")

    def log(self, step, **metrics):
        record = {"step": step, "elapsed_sec": round(time.time() - self.start_time, 1), **metrics}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        if step % self.print_every == 0:
            metric_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in metrics.items())
            print(f"[step {step:6d} | {record['elapsed_sec']:8.1f}s] {metric_str}")

    def close(self):
        self._f.close()
