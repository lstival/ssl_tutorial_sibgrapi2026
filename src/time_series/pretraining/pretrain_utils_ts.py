"""
Shared helpers for the time-series pretraining scripts in this folder (train_contrastive_ts.py,
train_mae_ts.py, train_dino_ts.py): puts src/time_series (tutorial_ts.py, the single source of
truth for the model architecture) and src/common on sys.path, and re-exports the
modality-agnostic training plumbing from src/common/train_utils.py.
"""

import os
import sys

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _path in (SRC_DIR, os.path.join(SRC_DIR, "..", "common")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train_utils import (  # noqa: E402,F401 -- re-exported for the scripts in this folder
    RunningLogger,
    enable_fast_cuda,
    save_encoder_checkpoint,
    train_log_path,
    warmup_cosine_lambda,
)

# The time-series runs are ~10x more steps/s than the ViT ones; print a little less often.
LOG_PRINT_EVERY = 100
