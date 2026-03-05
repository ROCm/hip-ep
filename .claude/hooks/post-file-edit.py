#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

# Hook: PostToolUse for Write|Edit
# Reminds to commit and push changes after editing files

import sys
import json
import subprocess


def get_current_branch():
    """Get the current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "<branch>"


def main():
    try:
        branch = get_current_branch()

        # Warn if on main branch
        if branch == "main":
            response = {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "⚠️ WARNING: You're on 'main' branch!\n\nYou should create a feature branch BEFORE making changes:\n1. git checkout -b feature/<name>\n2. Then commit your changes\n\nSee docs/workflows/git-workflow.md section 'CRITICAL: Feature Branch FIRST'",
                }
            }
        else:
            response = {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": f'WORKFLOW REMINDER: After modifying files:\n1. git add <files>\n2. git commit -m "message"\n3. git push fork {branch}',
                }
            }

        print(json.dumps(response, indent=2))

    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
