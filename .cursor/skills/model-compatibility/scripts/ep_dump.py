#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Dump the graph the EP actually compiles, as MLIR text.

Runs hip-onnx-runner with an init-only MorphiZen config: creating the session
drives EP initialization and the pass.init dump, and --no-run skips inference
afterwards. MORPHIZEN_SAVE_MLIR_AS_TEXT selects text over bytecode, and
without it the dump is unreadable to everything downstream.

Weights are not inlined -- onnx.Constant carries location/offset/size
pointing at the external data file -- so the dump stays small even for
multi-GB models.

Used both as a command, for the whole-graph dump, and as a module, by
single_op_probe.py for its one-node models. Having one implementation is the
point: the env var above was set in only one of two copies, and the copy
that forgot it produced bytecode that crashed the reader.

Usage:
  python ep_dump.py <model.onnx> [out_dir] --package <gpu-test-package>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DUMP_FILENAME = "ep_input.mlir"
DUMP_TIMEOUT_SEC = 600

# Exit code the orchestrator and SKILL.md document for "tell the user where
# the package is".
NO_PACKAGE_EXIT = 10

# Lines worth quoting back when no dump appears: the runner usually says why,
# and "Sequence type conversion is not implemented yet" is far more useful
# than the exit code that follows it.
_WHY = re.compile(
    r"Error in ORT API|error:|Exception|not implemented|failed", re.IGNORECASE
)

REQUIRED_OPTIONS = ("--no-run", "--allow-cpu-fallback", "--provider-options")


@dataclass
class DumpResult:
    path: Path | None
    # The root cause as the tool reported it, with no wrapping. Callers that
    # want "failed with exit code N" add it; single_op_probe puts this
    # straight into a report, where the exit code is noise.
    reason: str
    log: str
    exit_code: int = 0


def resolve_package(root: str | None) -> Path:
    """Package root from the argument, then the environment, then give up."""
    chosen = root or os.environ.get("GPU_TEST_PACKAGE_ROOT", "")
    if not chosen.strip():
        print(
            "[GPU_TEST_PACKAGE_NOT_CONFIGURED] ep_dump.py: no package root. Pass "
            "--package <path> or set GPU_TEST_PACKAGE_ROOT to the package "
            "containing bin/hip-onnx-runner."
        )
        raise SystemExit(NO_PACKAGE_EXIT)
    return Path(chosen).resolve()


def find_tool(package: Path, stem: str) -> Path:
    for name in (f"{stem}.exe", stem):
        candidate = package / "bin" / name
        if candidate.exists():
            return candidate
    raise SystemExit(f"{stem} not found under {package / 'bin'}")


def check_options(runner: Path) -> None:
    """Fail early on a package too old for the flags the dump needs.

    Such a package prints its usage text and exits, which otherwise shows up
    much later as a missing dump.
    """
    try:
        help_text = subprocess.run(
            [str(runner), "--help"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"could not run {runner}: {exc}") from exc
    blob = (help_text.stdout or "") + (help_text.stderr or "")
    missing = [o for o in REQUIRED_OPTIONS if o not in blob]
    if missing:
        raise SystemExit(
            f"{runner.name} is missing required option(s): {', '.join(missing)}. "
            f"Update the gpu-test-package at {runner.parent.parent}."
        )


def default_config() -> Path:
    return Path(__file__).resolve().parent / "morphizen_init_config.json"


def run_dump(
    runner: Path,
    config: Path,
    model: Path,
    out_dir: Path,
    filename: str = DUMP_FILENAME,
    timeout: int = DUMP_TIMEOUT_SEC,
) -> DumpResult:
    """Import one model and return the MLIR the EP was handed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename

    env = dict(os.environ)
    env["MORPHIZEN_SAVE_MLIR_AS_TEXT"] = "1"

    argv = [
        str(runner),
        "-m",
        str(model),
        "--no-run",
        "--allow-cpu-fallback",
        "--provider-options",
        f"config_file={config}",
        # Per-run, so it is passed here rather than baked into the shared
        # config; the runner merges it over the config's own options.
        # Forward slashes keep the value clear of escaping issues.
        "--provider-options",
        f"pass.init.directory={out_dir.as_posix()}",
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            # Run from the package's bin so its libraries resolve.
            cwd=str(runner.parent),
        )
    except subprocess.TimeoutExpired:
        return DumpResult(None, f"the importer timed out after {timeout}s", "", -1)

    log = (proc.stdout or "") + (proc.stderr or "")
    if target.exists() and target.stat().st_size > 0:
        return DumpResult(target, "", log, proc.returncode)

    reason = ""
    for line in log.splitlines():
        if _WHY.search(line):
            # Trim glog's timestamp and source-location preamble.
            _, _, tail = line.partition("] ")
            reason = (tail or line).strip()[:300]
    if target.exists():
        reason = reason or f"the dump is empty: {target}"
    elif not reason:
        reason = f"no dump was created at {target}"
    return DumpResult(None, reason, log, proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Where the dump lands. Defaults to the model's own directory.",
    )
    ap.add_argument("--package", default="")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--filename", default=DUMP_FILENAME)
    args = ap.parse_args()

    package = resolve_package(args.package)
    runner = find_tool(package, "hip-onnx-runner")
    check_options(runner)

    config = args.config or default_config()
    if not config.exists():
        raise SystemExit(
            f"MorphiZen config not found: {config} (default ships at "
            "scripts/morphizen_init_config.json; pass --config <file> to override)"
        )

    model = args.model.resolve()
    out_dir = (args.out_dir or model.parent).resolve()

    print(f"Model:            {model}")
    print(f"Package bin:      {runner.parent}")
    print(f"MorphiZen config: {config}")
    print(f"Expected output:  {out_dir / args.filename}")
    print(flush=True)

    result = run_dump(runner, config.resolve(), model, out_dir, args.filename)
    if result.log:
        print(result.log.rstrip())
    if result.path is None:
        # stdout, not the SystemExit message: PowerShell runs this under
        # $ErrorActionPreference = "Stop", where any native stderr becomes a
        # terminating error before the caller can read the reason.
        where = (
            f"{runner.name} exited with code {result.exit_code}: "
            if result.exit_code
            else ""
        )
        print(f"ERROR: {where}{result.reason}")
        raise SystemExit(1)

    print()
    print(f"OK: dumped EP input MLIR ({result.path.stat().st_size} bytes)")
    print(f"    {result.path}")


if __name__ == "__main__":
    main()
