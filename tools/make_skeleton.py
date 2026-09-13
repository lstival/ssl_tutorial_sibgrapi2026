"""
Generates the participant ("skeleton") version of each mechanism notebook from the filled
(reference) version.

Each filled notebook marks the mechanism-defining lines with sentinel comments:

    # --- SOLUTION START (<hint>) ---
    ... reference implementation ...
    # --- SOLUTION END ---

This script replaces every such block with:

    # TODO: <hint>
    raise NotImplementedError("Fill in <hint> -- see the notebook markdown above.")

so the filled notebook is the single source of truth and skeletons never drift from it.

Usage:
    python tools/make_skeleton.py                                       # all notebooks
    python tools/make_skeleton.py notebooks/time_series/01_contrastive_simclr.ipynb
"""

import glob
import json
import os
import re
import sys

SOLUTION_START_RE = re.compile(r"^(\s*)# --- SOLUTION START \((.+?)\) ---\s*$")
SOLUTION_END_RE = re.compile(r"^\s*# --- SOLUTION END ---\s*$")

SKELETON_SUFFIX = "_skeleton"


def strip_solutions(source_lines):
    """
    Replace every SOLUTION START/END block in a list of source lines with a
    `# TODO: <hint>` + `raise NotImplementedError(...)` stub, preserving indentation.
    Returns (new_lines, num_blocks_replaced).
    """
    new_lines = []
    in_block = False
    hint = None
    indent = ""
    num_blocks = 0

    for line in source_lines:
        start_match = SOLUTION_START_RE.match(line)
        if start_match and not in_block:
            in_block = True
            indent, hint = start_match.group(1), start_match.group(2)
            new_lines.append(f"{indent}# TODO: {hint}\n")
            new_lines.append(
                f'{indent}raise NotImplementedError("Fill in {hint} -- see the notebook markdown above.")\n'
            )
            num_blocks += 1
            continue

        if in_block:
            if SOLUTION_END_RE.match(line):
                in_block = False
            continue  # drop every line inside the solution block

        new_lines.append(line)

    return new_lines, num_blocks


def make_skeleton(nb_path, out_path):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    total_blocks = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_source, n = strip_solutions(cell["source"])
        if n:
            cell["source"] = new_source
            total_blocks += n

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    return total_blocks


def skeleton_path_for(nb_path):
    base, ext = os.path.splitext(nb_path)
    return f"{base}{SKELETON_SUFFIX}{ext}"


def main():
    # Both notebook sets share the same SOLUTION-sentinel convention.
    notebook_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "notebooks", "remote_sensing"),
        os.path.join(os.path.dirname(__file__), "..", "notebooks", "time_series"),
    ]

    if len(sys.argv) > 1:
        nb_paths = sys.argv[1:]
    else:
        nb_paths = sorted(
            p
            for d in notebook_dirs
            for p in glob.glob(os.path.join(d, "*.ipynb"))
            if not p.endswith(f"{SKELETON_SUFFIX}.ipynb")
        )

    if not nb_paths:
        print(f"No notebooks found in {notebook_dirs}")
        return

    for nb_path in nb_paths:
        out_path = skeleton_path_for(nb_path)
        n_blocks = make_skeleton(nb_path, out_path)
        rel_in = os.path.relpath(nb_path)
        rel_out = os.path.relpath(out_path)
        if n_blocks:
            print(f"{rel_in} -> {rel_out} ({n_blocks} blank(s))")
        else:
            print(f"{rel_in} -> {rel_out} (no SOLUTION blocks found; skeleton == filled)")


if __name__ == "__main__":
    main()
