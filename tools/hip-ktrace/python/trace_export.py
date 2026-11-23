#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Single-scan Chrome JSON and native pftrace export."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rocprofv3_reader import ensure_capture
from to_chrome_trace import main as chrome_main
from to_perfetto import PftraceSink


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="export one ktrace event stream as JSON and/or pftrace"
    )
    parser.add_argument("capture")
    parser.add_argument("--chrome-json", nargs="?", const="", default=None)
    parser.add_argument("--chrome-pftrace", nargs="?", const="", default=None)
    parser.add_argument("--start-ms", type=float)
    parser.add_argument("--span-ms", type=float)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--phase")
    flow = parser.add_mutually_exclusive_group()
    flow.add_argument("--flows", action="store_true")
    flow.add_argument("--no-flows", action="store_true")
    args = parser.parse_args(argv)

    capture = ensure_capture(args.capture)
    if args.chrome_json is None and args.chrome_pftrace is None:
        args.chrome_json = ""
    json_path = (
        None
        if args.chrome_json is None
        else Path(args.chrome_json or capture.auto_output("_ktrace_chrome.json"))
    )
    pftrace_path = (
        None
        if args.chrome_pftrace is None
        else Path(
            args.chrome_pftrace or capture.auto_output("_ktrace_perfetto.pftrace")
        )
    )

    chrome_options = [args.capture]
    if args.start_ms is not None:
        chrome_options += ["--start-ms", str(args.start_ms)]
    if args.span_ms is not None:
        chrome_options += ["--span-ms", str(args.span_ms)]
    if args.all:
        chrome_options.append("--all")
    if args.phase:
        chrome_options += ["--phase", args.phase]
    if args.flows:
        chrome_options.append("--flows")
    if args.no_flows:
        chrome_options.append("--no-flows")

    sink = None
    if pftrace_path:
        images = None if capture.is_legacy else capture.domain_path("cpu_images")
        sink = PftraceSink(pftrace_path, images)
    if json_path:
        chrome_main(
            [args.capture, str(json_path), *chrome_options[1:]], event_sink=sink
        )
    else:
        with tempfile.TemporaryDirectory(prefix="ktrace-export-") as temp:
            chrome_main(
                [
                    args.capture,
                    str(Path(temp) / "suppressed.json"),
                    *chrome_options[1:],
                ],
                event_sink=sink,
                suppress_json=True,
            )
    if sink:
        packets = sink.close()
        print(
            f"-> {pftrace_path} "
            f"({pftrace_path.stat().st_size / 1e6:.1f} MB, {packets:,} packets)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
