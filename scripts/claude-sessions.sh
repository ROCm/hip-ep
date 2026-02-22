#!/bin/bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# List Claude Code sessions for current project

PROJECT_DIR=$(pwd)
PROJECT_NAME=$(echo "$PROJECT_DIR" | sed 's|/|-|g' | sed 's|C:||' | sed 's|^-||')
SESSION_INDEX="$HOME/.claude/projects/C--Develop-m-MorphiZen-integration-morphizen-mlir-compiler/sessions-index.json"

if [[ ! -f "$SESSION_INDEX" ]]; then
    echo "No sessions found for this project"
    exit 1
fi

echo "Claude Code Sessions for: $PROJECT_DIR"
echo "=========================================="
echo ""

jq -r '.entries[] |
    "Session: \(.summary)\n" +
    "  ID: \(.sessionId)\n" +
    "  Created: \(.created)\n" +
    "  Modified: \(.modified)\n" +
    "  Messages: \(.messageCount)\n" +
    "  Branch: \(.gitBranch)\n"' \
    "$SESSION_INDEX" | head -100

echo ""
echo "Total sessions: $(jq '.entries | length' "$SESSION_INDEX")"
