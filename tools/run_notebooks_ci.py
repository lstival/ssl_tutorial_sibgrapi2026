"""
Execute every teaching notebook headless and report which ones broke.

This is the automated stand-in for "does this actually run for a participant", both in CI
(CPU, GitHub Actions) and when run by hand on a GPU box or inside Colab (see
notebooks/colab_smoke_test.ipynb, which calls this same script). It does not replace a real
Colab GPU run -- CPU execution exercises the same code paths, including the "no pretrained
checkpoint -> short live-training fallback" branch every notebook has, but it cannot catch
GPU-only bugs (dtype/device mismatches that only surface on cuda, OOM at Colab's GPU memory
size, etc). Run the Colab smoke test before the actual tutorial session for that.

    python tools/run_notebooks_ci.py                  # every notebook, both tracks
    python tools/run_notebooks_ci.py --track remote_sensing
    python tools/run_notebooks_ci.py --track time_series --only 00,01
    python tools/run_notebooks_ci.py --timeout 1800    # per-notebook cell timeout, seconds

Exit status is 0 only if every executed notebook completed without error.
"""

import argparse
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS = ["remote_sensing", "time_series"]
STAGES = ["00_setup_and_data", "01_contrastive_simclr", "02_masking_mae",
          "03_distillation_dino", "04_comparative_evaluation"]


def discover(tracks, only):
    """Notebooks for the requested tracks/stages, in teaching order."""
    jobs = []
    for track in tracks:
        nb_dir = os.path.join(REPO_ROOT, "notebooks", track)
        for stage in STAGES:
            if only and stage[:2] not in only:
                continue
            path = os.path.join(nb_dir, f"{stage}.ipynb")
            if os.path.isfile(path):
                jobs.append((track, stage, path))
    return jobs


def run_one(path, out_dir, timeout):
    import papermill as pm

    out_path = os.path.join(out_dir, os.path.basename(path))
    start = time.time()
    try:
        pm.execute_notebook(
            path,
            out_path,
            cwd=os.path.dirname(path),
            kernel_name="python3",
            execution_timeout=timeout,
            progress_bar=False,
        )
        return True, time.time() - start, None
    except Exception as e:  # noqa: BLE001 - report every failure mode, not just papermill's
        detail = str(e)
        # papermill wraps the participant-facing traceback inside the exception message;
        # keep only the last chunk so the report stays readable.
        tail = "\n".join(detail.splitlines()[-25:])
        return False, time.time() - start, tail


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", action="append", choices=TRACKS,
                    help="restrict to one track (repeatable); default: both")
    p.add_argument("--only", default=None,
                    help="comma-separated stage prefixes to run, e.g. 00,01,04")
    p.add_argument("--timeout", type=int, default=1800,
                    help="per-cell timeout in seconds (default 1800)")
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "artifacts", "ci_run"),
                    help="where executed notebooks are written")
    p.add_argument("--summary-file", default=None,
                    help="append a Markdown report here (e.g. $GITHUB_STEP_SUMMARY)")
    args = p.parse_args()

    try:
        import papermill  # noqa: F401
    except ImportError:
        print("papermill is required: pip install papermill ipykernel", file=sys.stderr)
        return 2

    tracks = args.track or TRACKS
    only = set(args.only.split(",")) if args.only else None
    jobs = discover(tracks, only)
    if not jobs:
        print("No notebooks matched the given --track/--only filters.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    print(f"Running {len(jobs)} notebook(s) headless (timeout={args.timeout}s/cell)...\n")
    for track, stage, path in jobs:
        label = f"{track}/{stage}"
        print(f"-> {label} ...", flush=True)
        ok, elapsed, error = run_one(path, args.out_dir, args.timeout)
        status = "PASS" if ok else "FAIL"
        print(f"   {status}  ({elapsed:.0f}s)")
        if not ok:
            print(f"   {error}\n")
        results.append((label, ok, elapsed, error))

    n_ok = sum(1 for _, ok, _, _ in results if ok)
    n = len(results)
    print(f"\n{n_ok}/{n} notebooks passed.")

    lines = ["# Notebook CI report", "", "| Notebook | Status | Time |", "|---|---|---|"]
    for label, ok, elapsed, _ in results:
        lines.append(f"| {label} | {'PASS' if ok else 'FAIL'} | {elapsed:.0f}s |")
    failed = [(label, error) for label, ok, _, error in results if not ok]
    if failed:
        lines.append("")
        lines.append("## What broke")
        for label, error in failed:
            lines.append(f"\n### {label}\n```\n{error}\n```")
    report = "\n".join(lines)

    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    else:
        print("\n" + report)

    return 0 if n_ok == n else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
