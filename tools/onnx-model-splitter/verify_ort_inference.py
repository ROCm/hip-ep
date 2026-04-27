#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
r"""
Verify extracted ONNX models by running ORT inference with dummy inputs.
Checks that models load successfully and produce valid outputs.

Each model is tested in a subprocess so that native ORT crashes
(e.g. access violations) don't kill the whole verification run.

Usage:
  python verify_ort_inference.py <output_dir> [--mode single_op|single_layer|full_model]

Examples:
  python verify_ort_inference.py D:\liuc\0-modelzoo\llm\model\space
  python verify_ort_inference.py D:\liuc\0-modelzoo\llm\model\space --mode single_op
"""

import os
import sys
import argparse
import subprocess


def _run_in_subprocess(fpath, verbose=False, timeout=300):
    """Run ORT inference for a single model in an isolated subprocess."""
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_verify_one_model.py"
    )
    cmd = [sys.executable, script, fpath]
    if verbose:
        cmd.append("--verbose")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (exceeded {}s)".format(timeout)

    stdout = result.stdout.strip()
    if result.returncode != 0:
        if result.returncode < 0 or result.returncode > 100:
            return False, "ORT crashed (exit code {})".format(result.returncode)
        return False, stdout if stdout else "exit code {}".format(result.returncode)
    return True, stdout


def _collect_onnx_files_single_op(d):
    """Yield (display_label, full_path) for each onnx in single_op subfolders."""
    if not os.path.isdir(d):
        return
    for folder in sorted(os.listdir(d)):
        folder_path = os.path.join(d, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in sorted(os.listdir(folder_path)):
            if f.endswith(".onnx"):
                yield f"{folder}/{f}", os.path.join(folder_path, f)


def _collect_onnx_files_flat(d):
    """Yield (display_label, full_path) for each onnx directly in dir."""
    if not os.path.isdir(d):
        return
    for f in sorted(os.listdir(d)):
        if f.endswith(".onnx"):
            yield f, os.path.join(d, f)


def verify_dir(d, collector, verbose=False):
    if not os.path.isdir(d):
        print("  [NOT FOUND]")
        return True

    files = list(collector(d))
    if not files:
        print("  (no .onnx files found)")
        return True

    all_ok = True
    total = len(files)
    passed = 0

    for label, fpath in files:
        print(f"    testing {label} ...", end="", flush=True)
        ok, msg = _run_in_subprocess(fpath, verbose=verbose)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            all_ok = False
        print(f"\r    [{status}] {label:50s} {msg}")

    print(f"  >> {passed}/{total} passed")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify extracted ONNX models via ORT inference"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Base output directory containing single_op/, single_layer/, full_model/",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["single_op", "single_layer", "full_model"],
        help="Only verify a specific mode (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed input/output info for each model",
    )
    args = parser.parse_args()

    base = args.output_dir
    if not os.path.isdir(base):
        print(f"ERROR: {base} is not a directory")
        sys.exit(1)

    modes = [args.mode] if args.mode else ["single_op", "single_layer", "full_model"]
    overall_ok = True

    for mode in modes:
        print("=" * 70)
        print(mode)
        print("=" * 70)
        d = os.path.join(base, mode)
        collector = (
            _collect_onnx_files_single_op
            if mode == "single_op"
            else _collect_onnx_files_flat
        )
        ok = verify_dir(d, collector, verbose=args.verbose)
        if not ok:
            overall_ok = False
        print()

    print("=" * 70)
    if overall_ok:
        print("RESULT: ALL ORT INFERENCE CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 70)

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
