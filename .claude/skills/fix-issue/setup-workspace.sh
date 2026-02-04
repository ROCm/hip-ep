#!/bin/bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Workspace setup script for /fix-issue skill
# Handles Phase 1 (prerequisites) + Phase 2 (branch, commit, PR)

set -euo pipefail

# ============================================================================
# Phase 1: Prerequisites Check
# ============================================================================

echo "🔍 Checking prerequisites..."

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Not on main branch (currently on: $CURRENT_BRANCH)"
    echo "   Run: git checkout main && git pull origin main"
    exit 1
fi

# Check uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Uncommitted changes detected:"
    git status --short
    echo "   Commit or stash changes first"
    exit 1
fi

# Sync main branch
echo "📥 Syncing main branch..."
git pull origin main

# ============================================================================
# Phase 2: Workspace Setup
# ============================================================================

# Parse issue number from argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <issue_number>"
    echo "Example: $0 042"
    exit 1
fi

ISSUE_NUM="$1"

# Find issue file
ISSUE_FILE=$(ls docs/project/issues/${ISSUE_NUM}-*.md 2>/dev/null | head -1)
if [ -z "$ISSUE_FILE" ]; then
    echo "❌ Issue file not found: docs/project/issues/${ISSUE_NUM}-*.md"
    exit 1
fi

# Extract slug from filename
ISSUE_SLUG=$(basename "$ISSUE_FILE" .md | sed "s/^${ISSUE_NUM}-//")

echo "📝 Issue #${ISSUE_NUM}: ${ISSUE_SLUG}"

# Create feature branch
BRANCH_NAME="feature/issue-${ISSUE_NUM}-${ISSUE_SLUG}"
echo "🌿 Creating branch: $BRANCH_NAME"
git checkout -b "$BRANCH_NAME"

# Add "Started:" date to issue file
TODAY=$(date +%Y-%m-%d)
if grep -q "^- \*\*Started:\*\*" "$ISSUE_FILE"; then
    echo "⚠️  Issue already has Started date - skipping"
else
    # Find the Created line and add Started after it
    sed -i "/^- \*\*Created:\*\*/a - **Started:** $TODAY" "$ISSUE_FILE"
    echo "✅ Added Started date: $TODAY"
fi

# Stage the issue file
git add "$ISSUE_FILE"

# Pre-commit retry loop
echo "🔧 Running pre-commit hooks..."
while true; do
    if pre-commit run; then
        break
    fi
    echo "⚠️  Pre-commit made changes - re-staging..."
    git add -u
done

# Create initial commit
COMMIT_MSG="docs: start work on issue #${ISSUE_NUM}"
git commit -m "$COMMIT_MSG"
echo "✅ Committed: $COMMIT_MSG"

# Push with -u flag
echo "📤 Pushing to fork..."
git push -u fork "$BRANCH_NAME"

# Create draft PR
echo "📋 Creating draft PR..."
PR_TITLE="Issue #${ISSUE_NUM}: [WIP] ${ISSUE_SLUG}"
PR_BODY="Work in progress - implementing issue #${ISSUE_NUM}"

gh pr create --draft \
  --title "$PR_TITLE" \
  --body "$PR_BODY"

# Get PR number
PR_NUMBER=$(gh pr list --head "$BRANCH_NAME" --json number --jq '.[0].number')

if [ -z "$PR_NUMBER" ]; then
    echo "❌ Failed to get PR number"
    exit 1
fi

# Output results (parsed by skill)
echo ""
echo "STATUS:WORKSPACE_READY"
echo "BRANCH:$BRANCH_NAME"
echo "PR_NUMBER:$PR_NUMBER"
echo "ISSUE_FILE:$ISSUE_FILE"
echo "ISSUE_NUM:$ISSUE_NUM"
echo "ISSUE_SLUG:$ISSUE_SLUG"
