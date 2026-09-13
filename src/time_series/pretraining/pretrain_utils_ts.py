"""
Small shared helpers for the time-series pretraining scripts in this folder
(train_contrastive_ts.py, train_mae_ts.py, train_dino_ts.py). The counterpart of
`pretraining/pretrain_utils.py` for the time series part.

Kept separate from src/time_series/tutorial_ts.py: this module is training-loop plumbing
(checkpoint saving, logging), not something the notebooks import. As in the remote sensing part,
`src/time_series/tutorial_ts.py` remains the single source of truth for the model architecture --
these scripts import `build_ts_encoder` / `TransformerBlock` from it directly.
"""

import json
import os
import sys
import time

import torch

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)


def save_encoder_checkpoint(encoder, out_path, extra=None):
    """
    Save only the encoder's state dict (not any pretraining-only head/decoder), matching what
    src/time_series/tutorial_ts.py's `try_load_checkpoint` loads into a bare `build_ts_encoder()`
    in Notebooks 1-4.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(encoder.state_dict(), out_path)
    if extra is not None:
        meta_path = os.path.splitext(out_path)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2)
    print(f"Saved encoder checkpoint to {out_path}")


class RunningLogger:
    """Minimal step/loss logger: prints periodically and appends a JSONL history file."""

    def __init__(self, log_path, print_every=100):
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
