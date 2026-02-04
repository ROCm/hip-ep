#!/bin/bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Finalization script for /fix-issue skill
# Handles backlog updates, issue file deletion, and final commit

set -euo pipefail

# Parse arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <issue_num> <pr_number> <backlog_update_content>"
    echo "Example: $0 042 123 'backlog update text'"
    exit 1
fi

ISSUE_NUM="$1"
PR_NUMBER="$2"
BACKLOG_UPDATE="$3"

echo "📝 Finalizing issue #${ISSUE_NUM}..."

# Find issue and plan files
ISSUE_FILE=$(ls docs/project/issues/${ISSUE_NUM}-*.md 2>/dev/null | head -1)
PLAN_FILE=$(ls docs/project/plans/${ISSUE_NUM}-*.md 2>/dev/null | head -1)

if [ -z "$ISSUE_FILE" ]; then
    echo "❌ Issue file not found: docs/project/issues/${ISSUE_NUM}-*.md"
    exit 1
fi

# Update backlog.md with provided content
echo "📋 Updating backlog.md..."
echo "$BACKLOG_UPDATE" > /tmp/backlog_update.txt

# The backlog update content is passed as a complete replacement
# Skill should generate the full updated backlog.md content
cat /tmp/backlog_update.txt > docs/project/backlog.md

git add docs/project/backlog.md

# Delete issue file
echo "🗑️  Deleting issue file: $ISSUE_FILE"
git rm "$ISSUE_FILE"

# Delete plan file if exists
if [ -n "$PLAN_FILE" ]; then
    echo "🗑️  Deleting plan file: $PLAN_FILE"
    git rm "$PLAN_FILE"
fi

# Pre-commit retry loop
echo "🔧 Running pre-commit hooks..."
while true; do
    if pre-commit run; then
        break
    fi
    echo "⚠️  Pre-commit made changes - re-staging..."
    git add -u
done

# Commit backlog updates
COMMIT_MSG="docs: complete issue #${ISSUE_NUM}"
git commit -m "$COMMIT_MSG"
echo "✅ Committed: $COMMIT_MSG"

# Push changes
CURRENT_BRANCH=$(git branch --show-current)
echo "📤 Pushing to fork..."
git push fork "$CURRENT_BRANCH"

# Output results
echo ""
echo "STATUS:FINALIZED"
echo "ISSUE_NUM:$ISSUE_NUM"
echo "PR_NUMBER:$PR_NUMBER"
