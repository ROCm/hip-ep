##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# PR Review Command Script
# Usage: tools/pr-review/review_pr.ps1 <PR_NUMBER>
# Example: tools/pr-review/review_pr.ps1 437
#
# This script automates the PR review process by:
# - Fetching the PR branch
# - Finding the merge-base with origin/main
# - Showing commits, diffs, and statistics
# - Checking if rebase is needed
# - Providing next-step commands

param(
    [Parameter(Mandatory=$false, Position=0)]
    [int]$PRNumber
)

# Get script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)

# If no PR number provided, prompt for it
if (-not $PRNumber) {
    $PRNumber = Read-Host "Enter PR number to review"
    if (-not $PRNumber) {
        Write-Host "Error: PR number is required" -ForegroundColor Red
        exit 1
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PR Review Script for PR #$PRNumber" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Fetch the PR
Write-Host "Step 1: Fetching PR #$PRNumber..." -ForegroundColor Yellow
git fetch origin pull/$PRNumber/head:pr-$PRNumber
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to fetch PR. Exiting." -ForegroundColor Red
    exit 1
}
Write-Host "✓ PR fetched successfully" -ForegroundColor Green
Write-Host ""

# Step 2: Update origin/main
Write-Host "Step 2: Fetching latest origin/main..." -ForegroundColor Yellow
git fetch origin main
Write-Host "✓ origin/main updated" -ForegroundColor Green
Write-Host ""

# Step 3: Find merge-base
Write-Host "Step 3: Finding merge-base..." -ForegroundColor Yellow
$mergeBase = git merge-base pr-$PRNumber origin/main
Write-Host "Merge-base: $mergeBase" -ForegroundColor Cyan
Write-Host ""

# Step 4: Show commit log
Write-Host "Step 4: Commits in PR #$PRNumber" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
git log $mergeBase..pr-$PRNumber --oneline --name-only
Write-Host ""

# Step 5: Show statistics
Write-Host "Step 5: File changes statistics" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
git diff $mergeBase pr-$PRNumber --stat
Write-Host ""

# Step 6: Show what's on main but not in PR
Write-Host "Step 6: Commits on origin/main not in PR" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
$mainCommits = git log $mergeBase..origin/main --oneline
if ($mainCommits) {
    Write-Host $mainCommits
    Write-Host ""
    Write-Host "⚠️  PR may need rebasing" -ForegroundColor Yellow
} else {
    Write-Host "None - PR is up to date with main" -ForegroundColor Green
}
Write-Host ""

# Step 7: View PR details
Write-Host "Step 7: PR Details" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
gh pr view $PRNumber
Write-Host ""

# Step 8: Show visual graph
Write-Host "Step 8: Visual commit graph" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
git log --graph --oneline --all --decorate -10
Write-Host ""

# Step 9: Ask for next steps
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Review Commands Available:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "View full diff:" -ForegroundColor Yellow
Write-Host "  git diff $mergeBase pr-$PRNumber" -ForegroundColor White
Write-Host ""
Write-Host "Checkout PR branch:" -ForegroundColor Yellow
Write-Host "  git checkout pr-$PRNumber" -ForegroundColor White
Write-Host ""
Write-Host "View specific file:" -ForegroundColor Yellow
Write-Host "  git show pr-$PRNumber:<file-path>" -ForegroundColor White
Write-Host ""
Write-Host "Submit review (using helper script):" -ForegroundColor Yellow
Write-Host "  tools/pr-review/submit_review.ps1 $PRNumber approve" -ForegroundColor White
Write-Host "  tools/pr-review/submit_review.ps1 $PRNumber changes" -ForegroundColor White
Write-Host "  tools/pr-review/submit_review.ps1 $PRNumber comment" -ForegroundColor White
Write-Host ""
Write-Host "Submit review (manual):" -ForegroundColor Yellow
Write-Host "  gh pr review $PRNumber --approve --body-file <review.md>" -ForegroundColor White
Write-Host "  gh pr review $PRNumber --request-changes --body-file <review.md>" -ForegroundColor White
Write-Host ""
Write-Host "Check PR status:" -ForegroundColor Yellow
Write-Host "  gh pr view $PRNumber" -ForegroundColor White
Write-Host ""

# Optional: Prompt to checkout the branch
Write-Host "========================================" -ForegroundColor Cyan
$checkout = Read-Host "Do you want to checkout pr-$PRNumber for detailed review? (y/n)"
if ($checkout -eq 'y' -or $checkout -eq 'Y') {
    Write-Host "Stashing local changes..." -ForegroundColor Yellow
    git stash
    git checkout pr-$PRNumber
    Write-Host "✓ Checked out pr-$PRNumber" -ForegroundColor Green
    Write-Host ""
    Write-Host "When done reviewing:" -ForegroundColor Yellow
    Write-Host "  1. git checkout main" -ForegroundColor White
    Write-Host "  2. git stash pop" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "To create and submit your review:" -ForegroundColor Cyan
    Write-Host "  Option 1 (Easy): Use the submit script" -ForegroundColor Yellow
    Write-Host "    tools/pr-review/submit_review.ps1 $PRNumber approve" -ForegroundColor White
    Write-Host ""
    Write-Host "  Option 2 (Manual):" -ForegroundColor Yellow
    Write-Host "    1. Copy doc/pr-review/review_template.md to pr${PRNumber}_review.md" -ForegroundColor White
    Write-Host "    2. Edit pr${PRNumber}_review.md with your review" -ForegroundColor White
    Write-Host "    3. Run: gh pr review $PRNumber --approve --body-file pr${PRNumber}_review.md" -ForegroundColor White
    Write-Host ""
}
