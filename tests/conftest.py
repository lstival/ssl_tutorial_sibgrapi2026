"""
Put the repo's script folders on sys.path, mirroring how the notebooks and pretraining scripts
import each other (the repo is intentionally not a pip-installable package: Colab downloads
tutorial_rs.py / tutorial_ts.py as single files).
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO_ROOT, "src")

for path in (
    os.path.join(SRC, "common"),
    os.path.join(SRC, "remote_sensing"),
    os.path.join(SRC, "remote_sensing", "pretraining"),
    os.path.join(SRC, "time_series"),
    os.path.join(SRC, "time_series", "pretraining"),
):
    if path not in sys.path:
        sys.path.insert(0, path)
