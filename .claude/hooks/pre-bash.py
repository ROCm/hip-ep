#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

# Hook: PreToolUse for Bash commands
# Enforces git workflow rules from docs/workflows/git-workflow.md

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
        return None


def check_ai_mentions(command):
    """Check if commit message contains AI/tool mentions."""
    # Extract commit message from command
    match = re.search(r'-m\s+["\'](.+?)["\']', command, re.DOTALL)
    if match:
        message = match.group(1)
        ai_patterns = [
            r"co-authored-by:\s*claude",
            r"generated\s+with\s+claude",
            r"claude\s+code",
            r"🤖",
            r"\bai\b",
            r"\bllm\b",
            r"automation\s+tool",
        ]
        for pattern in ai_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True
    return False


def main():
    try:
        # Read the tool input JSON from stdin
        input_data = json.load(sys.stdin)

        # Extract command from tool_input (correct JSON structure)
        tool_input = input_data.get("tool_input", {})
        command = tool_input.get("command", "")

        # Rule 1: Block git push origin (origin is read-only)
        if re.search(r"git\s+push\s+origin", command):
            response = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": "WARNING: Pushing to 'origin' is NOT allowed. origin (ROCm/MorphiZen) is read-only.\n\nYou MUST push to 'fork' instead:\ngit push fork <branch>\n\nDo you still want to proceed?",
                }
            }
            print(json.dumps(response, indent=2))
            return 0

        # Rule 2: Auto-run pre-commit before git commit
        if re.search(r"git\s+commit", command):
            # Run pre-commit on staged files to auto-fix formatting
            try:
                result = subprocess.run(
                    ["pre-commit", "run"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                # If pre-commit failed (made changes or found issues), block commit
                if result.returncode != 0:
                    response = {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": f'BLOCKED: Pre-commit hooks made changes or found issues.\n\nPre-commit output:\n{result.stdout}\n\nThe files have been auto-fixed. Stage the changes and commit again:\n  git add -u\n  git commit -m "..."\n\nSee docs/pre-commit-setup.md',
                        }
                    }
                    print(json.dumps(response, indent=2))
                    return 0
            except FileNotFoundError:
                # pre-commit not installed - continue with commit
                pass
            except subprocess.TimeoutExpired:
                # pre-commit timeout - block to prevent issues
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "BLOCKED: Pre-commit hook timed out after 30 seconds.\n\nThis may indicate:\n- Large number of files to check\n- Network issues downloading tools\n\nTry running manually:\n  pre-commit run\n\nThen commit again.",
                    }
                }
                print(json.dumps(response, indent=2))
                return 0

            # Rule 3: Block commits on main branch
            branch = get_current_branch()
            if branch == "main":
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "BLOCKED: Cannot commit on 'main' branch.\n\nWorkflow violation: You must work on a feature branch.\n\n1. Sync main: git checkout main && git pull origin main\n2. Create feature branch: git checkout -b feature/<name>\n3. Then commit your changes\n\nSee docs/workflows/git-workflow.md for details.",
                    }
                }
                print(json.dumps(response, indent=2))
                return 0

            # Rule 3: Check for AI/tool mentions in commit messages
            if check_ai_mentions(command):
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "BLOCKED: Commit message contains AI/tool mentions.\n\nProhibited content:\n- No 'Co-Authored-By: Claude' or similar\n- No 'Generated with Claude Code' footers\n- No AI assistant, LLM, or automation tool mentions\n- No tool-specific references\n\nWrite commits as if authored by a human developer.\nSee docs/workflows/git-workflow.md section 'Commit and PR Content Guidelines'.",
                    }
                }
                print(json.dumps(response, indent=2))
                return 0

        # Rule 4: Warn about git add -A or git add . (prefer specific files)
        if re.search(r"git\s+add\s+(-A|\.)\s*$", command):
            response = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": "WARNING: Using 'git add -A' or 'git add .' is discouraged.\n\nThis can accidentally stage:\n- Sensitive files (.env, credentials)\n- Large binaries\n- Unintended changes\n\nPreferred: git add <specific-files>\n\nDo you still want to proceed?",
                }
            }
            print(json.dumps(response, indent=2))
            return 0

        # Rule 5: Block push on main branch
        if re.search(r"git\s+push", command):
            branch = get_current_branch()
            if branch == "main":
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "BLOCKED: Cannot push from 'main' branch.\n\nWorkflow violation: All changes must go through Pull Requests.\n\n1. Create feature branch: git checkout -b feature/<name>\n2. Make your changes\n3. Push to fork: git push fork <branch>\n4. Create PR: gh pr create --draft\n\nSee docs/workflows/git-workflow.md for details.",
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
