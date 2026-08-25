#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Block build.py invocations that rely on GPU architecture auto-detection.

build.py detects the *local* GPU's architecture. On this project the build host is
not the test host, so detection silently produces binaries for the wrong target. A
mismatch builds cleanly and fails only when a kernel launches, far from its cause.

Worse than a crash: building for gfx1150 rather than gfx1151 would mask the UMA
aliasing bug that the registered-memory rule exists to catch, turning tests green by
removing the failure mode they were written to provoke.

This guard requires the choice to be explicit. It does not second-guess an explicit
value, because building for the local GPU is legitimate when testing GPU-only paths.

Wired as a PreToolUse hook on Bash. Exit 2 blocks the call and shows stderr to the
agent; exit 0 allows it.
"""

import json
import re
import sys

# Subcommands that never compile device code, so the architecture is irrelevant.
ARCH_IRRELEVANT_FLAGS = ("--mock", "--clean", "--help", "-h")

BUILD_INVOCATION = re.compile(r"(?:^|[\s;&|/\\])build\.py(?:\s|$)")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        # Never block on a malformed payload; a broken guard must not wedge the loop.
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    if not BUILD_INVOCATION.search(command):
        return 0

    if any(flag in command for flag in ARCH_IRRELEVANT_FLAGS):
        return 0

    if "--hip_arch" in command:
        return 0

    sys.stderr.write(
        "Blocked: this build.py invocation has no --hip_arch and would fall back to\n"
        "detecting the LOCAL GPU's architecture.\n"
        "\n"
        "This build host is not the test host. Auto-detection here produces binaries\n"
        "that build cleanly and then fail at kernel launch on the remote, and building\n"
        "for gfx1150 would additionally mask the aliasing bug the registered-memory\n"
        "rule exists to catch.\n"
        "\n"
        "Pass the target explicitly:\n"
        "    python build.py --hip_arch gfx1151\n"
        "\n"
        "Use --mock instead if you meant a compiler-and-mock-runtime build, which needs\n"
        "no GPU architecture at all.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
