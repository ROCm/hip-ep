#!/bin/bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# PR monitoring script for /resolve-ci skill
# Monitors PR status every 30 seconds and auto-fixes common issues
# Only returns to skill when complex issues need intelligent analysis

set -euo pipefail

# Auto-detect PR from current branch
CURRENT_BRANCH=$(git branch --show-current)

echo "🔍 Detecting PR for branch: $CURRENT_BRANCH"

PR_INFO=$(gh pr list --head "$CURRENT_BRANCH" --json number,state,isDraft --jq '.[0]' 2>&1 || echo "ERROR")

if [ "$PR_INFO" = "ERROR" ]; then
    echo "❌ Failed to fetch PR information"
    exit 1
fi

PR_NUMBER=$(echo "$PR_INFO" | jq -r '.number')
PR_STATE=$(echo "$PR_INFO" | jq -r '.state')
IS_DRAFT=$(echo "$PR_INFO" | jq -r '.isDraft')

# Validate PR exists
if [ -z "$PR_NUMBER" ] || [ "$PR_NUMBER" = "null" ]; then
    echo "❌ No PR found for branch: $CURRENT_BRANCH"
    exit 1
fi

# Validate PR is open
if [ "$PR_STATE" != "OPEN" ]; then
    echo "ℹ️  PR #$PR_NUMBER is $PR_STATE (not OPEN)"
    exit 0
fi

# Validate PR is not draft
if [ "$IS_DRAFT" = "true" ]; then
    echo "❌ PR #$PR_NUMBER is still DRAFT"
    echo "Mark it ready for review first: gh pr ready $PR_NUMBER"
    exit 1
fi

echo "✅ Found PR #$PR_NUMBER (OPEN, ready for review)"
echo ""
echo "🔍 Starting monitoring loop..."
echo ""

CYCLE_COUNT=0

while true; do
    CYCLE_COUNT=$((CYCLE_COUNT + 1))
    echo "[Cycle $CYCLE_COUNT] Checking PR status..."

    # Get PR status
    PR_DATA=$(gh pr view "$PR_NUMBER" --json merged,mergeable,statusCheckRollup,autoMergeRequest 2>&1 || echo "ERROR")

    if [ "$PR_DATA" = "ERROR" ]; then
        echo "❌ Failed to fetch PR data"
        exit 1
    fi

    # Check if merged
    PR_MERGED=$(echo "$PR_DATA" | jq -r '.merged')
    if [ "$PR_MERGED" = "true" ]; then
        echo "✅ PR #$PR_NUMBER merged successfully!"
        echo "STATUS:MERGED"
        exit 0
    fi

    # Check mergeable status
    MERGEABLE=$(echo "$PR_DATA" | jq -r '.mergeable')
    if [ "$MERGEABLE" = "CONFLICTING" ]; then
        echo "⚠️  Merge conflicts detected"
        echo "STATUS:NEEDS_FIX_CONFLICTS"
        exit 0
    fi

    # Check CI status
    CI_CONCLUSIONS=$(echo "$PR_DATA" | jq -r '.statusCheckRollup[]? | select(.conclusion != null and .conclusion != "SUCCESS" and .conclusion != "SKIPPED") | .conclusion' 2>/dev/null || true)

    if [ -n "$CI_CONCLUSIONS" ]; then
        echo "❌ CI failures detected:"
        echo "$PR_DATA" | jq -r '.statusCheckRollup[]? | select(.conclusion != "SUCCESS" and .conclusion != "SKIPPED" and .conclusion != null) | "  - \(.name): \(.conclusion)"' 2>/dev/null || true

        # Check if it's a pre-commit failure (auto-fixable)
        PRECOMMIT_FAILED=$(echo "$PR_DATA" | jq -r '.statusCheckRollup[]? | select(.name == "pre-commit" and .conclusion == "FAILURE") | .name' 2>/dev/null || true)

        if [ -n "$PRECOMMIT_FAILED" ]; then
            echo ""
            echo "🔧 Pre-commit failure detected - auto-fixing..."

            # Run pre-commit to fix issues
            if pre-commit run --all-files; then
                echo "✅ Pre-commit passed after auto-fix"
            else
                echo "⚠️  Pre-commit made changes - staging and committing..."
            fi

            # Stage all changes
            git add -u

            # Check if there are changes to commit
            if git diff --cached --quiet; then
                echo "ℹ️  No changes to commit after pre-commit run"
            else
                # Commit and push
                git commit -m "style: apply pre-commit fixes"
                git push fork "$CURRENT_BRANCH"
                echo "✅ Pre-commit fixes pushed - restarting monitoring..."
            fi

            # Continue monitoring immediately (don't wait 30s)
            continue
        fi

        # Complex CI failure - needs intelligent analysis
        echo "STATUS:NEEDS_FIX_CI"
        exit 0
    fi

    # Check if CI is still running
    CI_PENDING=$(echo "$PR_DATA" | jq -r '.statusCheckRollup[]? | select(.status == "IN_PROGRESS" or .status == "QUEUED") | .name' 2>/dev/null || true)

    if [ -n "$CI_PENDING" ]; then
        echo "⏳ CI still running:"
        echo "$PR_DATA" | jq -r '.statusCheckRollup[]? | select(.status == "IN_PROGRESS" or .status == "QUEUED") | "  - \(.name): \(.status)"' 2>/dev/null || true
    else
        # All checks passed, enable auto-merge if not already
        AUTO_MERGE=$(echo "$PR_DATA" | jq -r '.autoMergeRequest != null')

        if [ "$AUTO_MERGE" = "false" ]; then
            echo ""
            echo "✅ All checks passed - enabling auto-merge (squash)..."

            if gh pr merge "$PR_NUMBER" --auto --squash; then
                echo "✅ Auto-merge enabled successfully"
                echo "   Continuing to monitor for cascade conflicts..."
            else
                echo "❌ Failed to enable auto-merge"
                echo "STATUS:AUTO_MERGE_FAILED"
                exit 0
            fi
        else
            echo "✅ All checks passed - auto-merge enabled, waiting for merge..."
        fi
    fi

    # Wait 30 seconds before next check
    echo "⏳ Waiting 30 seconds before next check..."
    echo ""
    sleep 30
done
