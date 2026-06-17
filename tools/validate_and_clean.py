"""Vetted utility: run validation and optionally remove the output folder.

This script invokes the training module's validation step using the same
Python interpreter and can optionally remove the model directory afterwards.

Usage:
    python tools/validate_and_clean.py --input data/raw --model-dir scratch_models [--clean]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation and optionally clean the model directory")
    parser.add_argument("--input", default="data/raw", help="Input folder (raw or processed) to validate")
    parser.add_argument("--model-dir", default="scratch_models", help="Directory to write validation outputs")
    parser.add_argument("--clean", action="store_true", help="Remove the model directory after running validation")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "src.train", "--input", args.input, "--model-dir", args.model_dir, "--validate-only"]
    print("Running:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            print(f"Validation command exited with code {res.returncode}")
    except Exception as e:
        print("Failed to run validation:", e)
        return 1

    if args.clean:
        p = Path(args.model_dir)
        if p.exists():
            try:
                shutil.rmtree(p)
                print(f"Removed directory: {p}")
            except Exception as e:
                print(f"Failed to remove {p}: {e}")
                return 1
        else:
            print(f"No directory to remove: {p}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
