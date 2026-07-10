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
import os
import shlex

# Whitelist of allowed text file extensions
TEXT_EXTENSIONS = {
    # Source code
    ".cpp",
    ".hpp",
    ".h",
    ".c",
    ".cc",
    ".cxx",
    ".hxx",
    ".py",
    ".pyx",
    ".pxd",
    ".sh",
    ".bash",
    ".zsh",
    # Config
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".proto",
    ".cmake",
    ".in",
    # Documentation
    ".md",
    ".txt",
    ".rst",
    ".adoc",
    ".tex",
    # Web
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    # Data
    ".csv",
    ".tsv",
    ".log",
    # Build/Git
    ".gitignore",
    ".gitattributes",
    ".clang-format",
    ".clang-tidy",
    # MLIR
    ".mlir",
    ".td",
}


def get_current_branch():
    """Get the current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


def check_ai_mentions(parts):
    """Check if commit message contains AI/tool mentions."""
    # Find -m flag and extract message from parsed parts
    message = None
    for i, part in enumerate(parts):
        if part == "-m" and i + 1 < len(parts):
            message = parts[i + 1]
            break

    if not message:
        return False

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


def is_binary_content(filepath):
    """Check if file contains binary content (null bytes)."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)  # Read first 1KB
            return b"\x00" in chunk  # Null bytes indicate binary
    except (IOError, OSError):
        return False


def check_files_for_binaries(paths):
    """Check if any paths are binaries. Returns (is_safe, error_message)."""
    blocked_files = []

    for path in paths:
        # Check for wildcards
        if "*" in path or "?" in path:
            return False, f"Wildcards not allowed: {path}"

        # Check if directory
        if os.path.isdir(path):
            return False, f"Directories not allowed: {path}/"

        # Check if file exists
        if not os.path.isfile(path):
            # File doesn't exist yet (might be in .gitignore or deleted)
            # Let git handle this case
            continue

        # Check extension against whitelist
        ext = os.path.splitext(path)[1].lower()

        # Special case: files without extension (README, LICENSE, Makefile, etc.)
        if not ext:
            # Check if filename is known text file
            basename = os.path.basename(path)
            if basename in {
                "README",
                "LICENSE",
                "Makefile",
                "Dockerfile",
                "CMakeLists.txt",
            }:
                continue  # Allowed
            # Unknown no-extension file - check content
            if is_binary_content(path):
                blocked_files.append(path)
            continue

        # Check whitelist
        if ext in TEXT_EXTENSIONS:
            continue  # Allowed

        # Unknown extension - check content
        if is_binary_content(path):
            blocked_files.append(path)

    if blocked_files:
        files_list = "\n".join(f"  - {f}" for f in blocked_files)
        return (
            False,
            f"Binary files detected:\n{files_list}\n\nUse Git LFS for binary files.",
        )

    return True, ""


def main():
    try:
        # Read the tool input JSON from stdin
        input_data = json.load(sys.stdin)

        # Extract command from tool_input (correct JSON structure)
        tool_input = input_data.get("tool_input", {})
        command = tool_input.get("command", "")

        # Parse command with shlex for consistent handling
        try:
            parts = shlex.split(command)
        except ValueError:
            # shlex parsing failed - complex quoting, reject for safety
            response = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "BLOCKED: Cannot parse command (complex quoting).\n\nPlease simplify the command.",
                }
            }
            print(json.dumps(response, indent=2))
            return 0

        # Check if it's a git command
        if not parts or parts[0] != "git":
            return 0  # Not a git command, allow

        if len(parts) < 2:
            return 0  # Just "git" with no subcommand, allow

        git_subcommand = parts[1]

        # Rule 1: Block git push origin (origin is read-only)
        if git_subcommand == "push" and len(parts) >= 3 and parts[2] == "origin":
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
        if git_subcommand == "commit":
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

            # Rule 3.5: Check for AI/tool mentions in commit messages
            if check_ai_mentions(parts):
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "BLOCKED: Commit message contains AI/tool mentions.\n\nProhibited content:\n- No 'Co-Authored-By: Claude' or similar\n- No 'Generated with Claude Code' footers\n- No AI assistant, LLM, or automation tool mentions\n- No tool-specific references\n\nWrite commits as if authored by a human developer.\nSee docs/workflows/git-workflow.md section 'Commit and PR Content Guidelines'.",
                    }
                }
                print(json.dumps(response, indent=2))
                return 0

        # Rule 4: Block git add -A or git add . (changed from warn to block)
        if git_subcommand == "add" and len(parts) >= 3:
            # Check if last argument is -A or .
            last_arg = parts[-1]
            if last_arg == "-A" or last_arg == ".":
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "BLOCKED: Cannot use 'git add -A' or 'git add .'.\n\nThis can accidentally stage:\n- Binary files\n- Sensitive files (.env, credentials)\n- Unintended changes\n\nUse explicit file names:\n  git add file1.cpp file2.hpp\n\nOr to update already-tracked files:\n  git add -u\n\nSee docs/workflows/git-workflow.md",
                    }
                }
                print(json.dumps(response, indent=2))
                return 0

        # Rule 4.5: Check for binary files in git add (whitelist approach)
        if git_subcommand == "add":
            # Skip if it's -u (update tracked files only - safe)
            if "-u" in parts:
                return 0  # Allow -u

            # Extract paths (skip 'git' and 'add')
            paths = []
            for i, part in enumerate(parts):
                if i < 2:  # Skip 'git' and 'add'
                    continue
                if part.startswith("-"):  # Skip flags
                    continue
                paths.append(part)

            if not paths:
                return 0  # No files to check

            # Check paths for binaries
            is_safe, error_msg = check_files_for_binaries(paths)
            if not is_safe:
                response = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f'BLOCKED: {error_msg}\n\nOnly text files are allowed in the repository.\n\nFor binary files:\n1. Add file to .gitignore\n2. Use Git LFS: git lfs track "*.png"\n\nFor legitimate text files with unusual extensions:\n1. Add extension to TEXT_EXTENSIONS whitelist in .claude/hooks/pre-bash.py\n\nSee docs/workflows/git-workflow.md',
                    }
                }
                print(json.dumps(response, indent=2))
                return 0

        # Rule 5: Block push on main branch
        if git_subcommand == "push":
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
