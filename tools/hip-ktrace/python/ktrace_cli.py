#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Unified command line for ktrace's Python post-processing tools.

    python python/ktrace_cli.py finalize <capture-dir>
    python python/ktrace_cli.py analyze <capture>
    python python/ktrace_cli.py chrome <capture> [out.json] [options]
    python python/ktrace_cli.py export <capture> --chrome-json --chrome-pftrace
    python python/ktrace_cli.py validate --output-dir <dir> --prefix <name>

Each subcommand delegates to the existing implementation, so its detailed
``--help`` remains the source of truth.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


COMMANDS = {
    "finalize": "Convert private spools to rocprofv3 CSV/JSON",
    "analyze": "Generate the ktrace text performance report",
    "chrome": "Export a Chrome/Perfetto trace",
    "perfetto": "Export a native Perfetto TrackEvent protobuf",
    "export": "Export JSON and/or pftrace from one event stream",
    "validate": "Validate ROCm 7.2.1 rocprofv3 wire compatibility",
}


def _print_help() -> None:
    print("usage: ktrace_cli.py <command> [options]\n")
    print("commands:")
    width = max(map(len, COMMANDS))
    for name, description in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print(
        "\nRun `python python\\ktrace_cli.py <command> --help` for command-specific options."
    )


def _finalize(argv: list[str]) -> int:
    from rocprofv3_export import finalize_directory

    parser = argparse.ArgumentParser(prog="ktrace finalize")
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        finalize_directory(args.capture_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    return 0


def _validate(argv: list[str]) -> int:
    script = Path(__file__).resolve().parent.parent / "tests" / "rocprofv3_schema.py"
    return subprocess.run([sys.executable, str(script), *argv], check=False).returncode


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"error: unknown command {command!r}\n", file=sys.stderr)
        _print_help()
        return 2

    if command == "finalize":
        return _finalize(rest)
    if command == "validate":
        return _validate(rest)
    if command == "analyze":
        from analyze import main as command_main
    elif command == "perfetto":
        from to_perfetto import main as command_main
    elif command == "export":
        from trace_export import main as command_main
    else:
        from to_chrome_trace import main as command_main
    result = command_main(rest)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
