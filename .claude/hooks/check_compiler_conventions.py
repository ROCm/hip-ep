#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check two compiler-tree conventions that no existing tool enforces.

pre-commit already handles license headers and formatting, and lintrunner handles
style. Neither covers these, and both are easy for an agent to forget:

1. Generic compiler code must not name hardware. The dialect and its passes are
   architecture-agnostic, so a GPU name is always an anecdote in the wrong place.
   Model names are treated the same way in implementation files, but are allowed in
   .td dialect definitions, where naming a model family can legitimately document why
   an attribute exists.
2. A newly added pass or rewrite must carry a Before:/After: MLIR snippet.

This inspects only the text being **added**, never the whole file. That distinction is
what makes the check usable: most existing pass files predate the snippet convention,
so whole-file checking would fire on unrelated edits until someone disabled the hook.
For the same reason the snippet rule applies only to newly created files.

Wired as a PostToolUse hook on Write/Edit/MultiEdit. Exit 2 surfaces the message to the
agent so it can fix the change immediately; exit 0 stays silent.
"""

import json
import re
import sys
from pathlib import Path

GOVERNED_PREFIXES = (
    "lib/conversion/",
    "lib/dialect/",
    "include/hip/",
    "lib/compiler/",
)

SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".td"}

HARDWARE_IN_COMMENTS = re.compile(
    r"\b(gfx\d{3,4}|strix|raphael|radeon|rdna\d?|navi\d*)\b", re.IGNORECASE
)

MODEL_IN_COMMENTS = re.compile(
    r"\b(llama|phi-?\d|qwen|mistral|deepseek|gpt-oss|whisper)\b", re.IGNORECASE
)

COMMENT_LINE = re.compile(r"^\s*(//|/\*|\*|#)|//")

DEFINES_PASS = re.compile(
    r"runOnOperation\s*\(|PassWrapper|struct\s+\w*Pass\b|def\s+\w+\s*:\s*Pass<"
)


def added_text(tool_name: str, tool_input: dict) -> str:
    """Return only the text this call introduced."""
    if tool_name == "Write":
        return tool_input.get("content") or ""
    if tool_name == "Edit":
        return tool_input.get("new_string") or ""
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits") or []
        return "\n".join((e or {}).get("new_string") or "" for e in edits)
    return ""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}

    raw_path = tool_input.get("file_path") or ""
    if not raw_path:
        return 0

    path = Path(raw_path)
    normalized = path.as_posix().lower()
    if not any(marker in normalized for marker in GOVERNED_PREFIXES):
        return 0
    if path.suffix.lower() not in SOURCE_SUFFIXES:
        return 0

    new_text = added_text(tool_name, tool_input)
    if not new_text.strip():
        return 0

    findings = []
    is_dialect_definition = path.suffix.lower() == ".td"

    for line in new_text.splitlines():
        if not COMMENT_LINE.search(line):
            continue
        stripped = line.strip()
        hardware = HARDWARE_IN_COMMENTS.search(line)
        if hardware:
            findings.append(
                f"  names hardware '{hardware.group(0)}': {stripped}\n"
                "    Compiler code is architecture-agnostic. Put reproduction details "
                "in a test, a design doc, or the commit message."
            )
        if not is_dialect_definition:
            model = MODEL_IN_COMMENTS.search(line)
            if model:
                findings.append(
                    f"  names model '{model.group(0)}': {stripped}\n"
                    "    Describe the IR pattern, not the model that happens to "
                    "trigger it."
                )

    # Only for newly created files: an existing file may predate this convention, and
    # flagging it on an unrelated edit is the noise that gets a hook switched off.
    if tool_name == "Write" and DEFINES_PASS.search(new_text):
        if not ("Before:" in new_text and "After:" in new_text):
            findings.append(
                "  defines a pass or rewrite with no 'Before:' / 'After:' MLIR "
                "snippet.\n    Required for new passes in this repository, and it must "
                "stay accurate if the transformation changes."
            )

    if not findings:
        return 0

    sys.stderr.write(
        f"Convention check on {path.as_posix()}:\n\n"
        + "\n".join(findings)
        + "\n\nFix these before reporting the task complete.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
