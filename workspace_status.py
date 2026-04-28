"""Bazel workspace status script — platform-independent replacement for workspace_status.sh.

Outputs STABLE_* key-value pairs consumed by Bazel stamped build actions.
Changes to STABLE_* keys invalidate stamped targets (e.g. gen_flexmlversion_h).

Run by Bazel before each build when --stamp is passed.
"""

import subprocess
import sys


def main():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        git_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        git_hash = "unknown"

    print(f"STABLE_GIT_HASH {git_hash}")


if __name__ == "__main__":
    main()
