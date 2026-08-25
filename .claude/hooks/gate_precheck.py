#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Mechanical gate checks for the hybrid NPU + GPU work.

Run by npu-gate-verifier before it applies judgement. This covers only what can be
decided by inspection, so that expensive review attention goes to what cannot:
whether the work could pass while being wrong.

Checks, in order of how often they matter:

  1. A test claiming NPU coverage sets strict mode, asserts dispatch, and asserts the
     boundary-copy counters are zero. All three, or it can pass by falling back or by
     copying.
  2. The copy escape hatch is not left enabled anywhere outside documentation.
  3. Library changes are accompanied by tests.
  4. Library changes are accompanied by documentation updates, per repository policy.

Exit status is 1 if any blocking finding exists, 0 otherwise. Advisory findings never
set a non-zero status; they are for the reviewer to weigh.

Usage:
    python .claude/hooks/gate_precheck.py [--base <git-ref>]
"""

import argparse
import subprocess
import sys
from pathlib import Path

STRICT_FLAG = "HIPDNN_EP_NPU_STRICT"
COPY_ESCAPE_HATCH = "HIPDNN_EP_NPU_ALLOW_COPY"

NPU_TEST_MARKERS = ("npu", "prefill_backend")
DISPATCH_MARKERS = ("dispatch", "used_npu", "ran_on_npu", "backend")
COUNTER_MARKERS = ("copy_count", "boundary_copy", "copies", "copy_counter")


def run(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def changed_files(base: str) -> list[str]:
    out = run(["git", "diff", "--name-only", f"{base}...HEAD"])
    staged = run(["git", "diff", "--name-only", "--cached"])
    unstaged = run(["git", "diff", "--name-only"])
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"])
    seen = {}
    for blob in (out, staged, unstaged, untracked):
        for line in blob.splitlines():
            line = line.strip()
            if line:
                seen[line] = None
    return list(seen)


def read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args()

    files = changed_files(args.base)
    if not files:
        print("No changed files detected. Nothing to check.")
        return 0

    print(f"Changed files ({len(files)}):")
    for path in sorted(files):
        print(f"  {path}")
    print()

    blocking: list[str] = []
    advisory: list[str] = []

    touched_lib = [f for f in files if f.startswith(("lib/", "include/", "morphizen/"))]
    touched_tests = [f for f in files if f.startswith("test/")]
    touched_docs = [f for f in files if f.startswith("docs/") or f == "CLAUDE.md"]

    # 1. Tests claiming NPU coverage must be unable to pass by fallback or by copying.
    for path in touched_tests:
        body = read(path)
        lowered = body.lower()
        if not any(marker in lowered for marker in NPU_TEST_MARKERS):
            continue
        if STRICT_FLAG not in body:
            blocking.append(
                f"{path}: appears to cover the NPU path but does not set {STRICT_FLAG}. "
                "Without strict mode it passes on silent fallback."
            )
        if not any(marker in lowered for marker in DISPATCH_MARKERS):
            blocking.append(
                f"{path}: no visible dispatch assertion. Tests must assert the NPU ran, "
                "not infer it from correct output."
            )
        if not any(marker in lowered for marker in COUNTER_MARKERS):
            blocking.append(
                f"{path}: no visible boundary-copy counter assertion. A copy produces "
                "correct numbers, so numeric comparison cannot detect it."
            )

    # 2. The copy escape hatch must not be enabled in committed code. The agent tooling
    # under .claude/ is exempt because this file names the hatch in order to search for
    # it, so scanning it reports itself and every run gains a finding that is not real.
    for path in files:
        if path.startswith((".claude/", "docs/")) or path.endswith(".md"):
            continue
        if COPY_ESCAPE_HATCH in read(path):
            blocking.append(
                f"{path}: references {COPY_ESCAPE_HATCH}. This is a bring-up-only escape "
                "hatch; no phase may complete while it is active."
            )

    # 3 and 4. Repository policy: tests and documentation travel with the change.
    if touched_lib and not touched_tests:
        advisory.append(
            "Library code changed but no test changed. Confirm the behaviour is covered "
            "by an existing test, or explain why it cannot be."
        )
    if touched_lib and not touched_docs:
        advisory.append(
            "Library code changed but no documentation changed. Repository policy makes "
            "documentation part of the change; confirm nothing documented was affected."
        )

    if blocking:
        print("BLOCKING:")
        for item in blocking:
            print(f"  - {item}")
        print()
    if advisory:
        print("ADVISORY (judgement required):")
        for item in advisory:
            print(f"  - {item}")
        print()
    if not blocking and not advisory:
        print("No mechanical findings.")

    print(
        "Not checked here, and not checkable locally: registration behaviour, NPU "
        "numerics, the phase transition, and every performance claim. Those require the "
        "remote gfx1151 host."
    )
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
