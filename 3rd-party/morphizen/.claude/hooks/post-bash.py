#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

# Hook: PostToolUse for Bash commands
# Reminds workflow after git operations

import sys
import json
import subprocess
import re


def get_current_branch():
    """Get the current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "<branch>"


def has_upstream_tracking():
    """Check if current branch has upstream tracking configured."""
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    try:
        # Read the tool input JSON from stdin
        input_data = json.load(sys.stdin)
        command = input_data.get("command", "")

        # After git commit, remind to push
        if re.search(r"git\s+commit", command):
            branch = get_current_branch()

            # Warn if somehow committed on main (shouldn't happen due to pre-hook)
            if branch == "main":
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "⚠️ WARNING: You committed on 'main' branch!\n\nThis violates the git workflow. You should:\n1. Undo this commit: git reset --soft HEAD~1\n2. Create feature branch: git checkout -b feature/<name>\n3. Re-commit: git commit -m \"...\"\n4. Push: git push -u fork feature/<name>\n\nSee docs/workflows/git-workflow.md",
                    }
                }
            else:
                # Check if upstream tracking is configured
                if has_upstream_tracking():
                    push_cmd = f"git push fork {branch}"
                else:
                    push_cmd = f"git push -u fork {branch}"

                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": f"PUSH REMINDER: Commit completed. Now push to fork:\n{push_cmd}",
                    }
                }

            print(json.dumps(response, indent=2))
            return 0

        # After git push, provide reminders
        if re.search(r"git\s+push\s+fork\s+", command):
            branch = get_current_branch()
            messages = []

            # First push (with -u) - remind to create PR
            if re.search(r"-u|--set-upstream", command) and branch != "main":
                messages.append(
                    "💡 REMINDER: Create a draft PR immediately:\ngh pr create --draft\n\nThis makes your work visible to the team early.\nSee docs/workflows/git-workflow.md"
                )

            # Push without -u and no tracking - warn
            elif (
                not re.search(r"-u|--set-upstream", command)
                and branch != "main"
                and not has_upstream_tracking()
            ):
                messages.append(
                    f"⚠️ WARNING: Pushed without setting upstream tracking!\n\nFix it now:\ngit branch --set-upstream-to=fork/{branch}\n\nNext time use: git push -u fork {branch}"
                )

            if messages:
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "\n\n".join(messages),
                    }
                }
                print(json.dumps(response, indent=2))
                return 0

    except json.JSONDecodeError:
        pass
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
