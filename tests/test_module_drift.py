"""
Drift guard for the code that tutorial_rs.py and tutorial_ts.py deliberately duplicate.

Both modules must stay single-file (Colab downloads exactly one of them), so helpers such as the
checkpoint downloader cannot be shared by import. This test makes the copies fail loudly when
one side is edited without the other. Docstrings are ignored; code must match exactly.
"""

import ast
import os

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
SHARED = [
    "ContrastiveTransformations",
    "TransformerBlock",
    "_looks_like_lfs_pointer",
    "check_colab_gpu",
    "download_files",
    "get_device",
    "plot_curve",
    "resolve_local_checkpoint",
    "setup_plotting",
    "try_load_checkpoint",
]


def _definitions(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            body = getattr(sub, "body", None)
            if (isinstance(sub, (ast.FunctionDef, ast.ClassDef)) and body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                sub.body = body[1:] or [ast.Pass()]
        out[node.name] = ast.dump(node)
    return out


RS = _definitions(os.path.join(SRC, "remote_sensing", "tutorial_rs.py"))
TS = _definitions(os.path.join(SRC, "time_series", "tutorial_ts.py"))


@pytest.mark.parametrize("name", SHARED)
def test_shared_definition_is_identical(name):
    assert RS[name] == TS[name], f"{name} differs between tutorial_rs.py and tutorial_ts.py"
