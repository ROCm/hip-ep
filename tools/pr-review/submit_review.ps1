##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Submit PR Review Script
# Usage: tools/pr-review/submit_review.ps1 <PR_NUMBER> <STATUS>
# Status: approve, changes, comment
# Example: tools/pr-review/submit_review.ps1 999 approve

param(
    [Parameter(Mandatory=$false, Position=0)]
    [int]$PRNumber,

    [Parameter(Mandatory=$false, Position=1)]
    [ValidateSet('approve', 'changes', 'comment')]
    [string]$Status = 'approve'
)

# Get script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$templatePath = Join-Path $projectRoot "doc/pr-review/review_template.md"

# Prompt for PR number if not provided
if (-not $PRNumber) {
    $PRNumber = Read-Host "Enter PR number"
    if (-not $PRNumber) {
        Write-Host "Error: PR number is required" -ForegroundColor Red
        exit 1
    }
}

# Prompt for status if not provided
if (-not $Status) {
    Write-Host "Select review status:" -ForegroundColor Cyan
    Write-Host "  1. Approve" -ForegroundColor Green
    Write-Host "  2. Request Changes" -ForegroundColor Yellow
    Write-Host "  3. Comment Only" -ForegroundColor White
    $choice = Read-Host "Enter choice (1-3)"

    switch ($choice) {
        "1" { $Status = "approve" }
        "2" { $Status = "changes" }
        "3" { $Status = "comment" }
        default {
            Write-Host "Invalid choice. Defaulting to approve." -ForegroundColor Yellow
            $Status = "approve"
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Submit Review for PR #$PRNumber" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if review file exists
$reviewFile = "pr${PRNumber}_review.md"
$useFile = $false

if (Test-Path $reviewFile) {
    Write-Host "✓ Found existing review file: $reviewFile" -ForegroundColor Green
    $useExisting = Read-Host "Use this file for review? (y/n)"
    if ($useExisting -eq 'y' -or $useExisting -eq 'Y') {
        $useFile = $true
    }
}

if (-not $useFile) {
    # Check if template exists
    if (Test-Path $templatePath) {
        Write-Host "Copying template to $reviewFile..." -ForegroundColor Yellow
        Copy-Item $templatePath $reviewFile
        Write-Host "✓ Template copied from: doc/pr-review/review_template.md" -ForegroundColor Green
        Write-Host ""
        Write-Host "Please edit $reviewFile with your review" -ForegroundColor Yellow
        Write-Host "Press Enter when ready to submit..." -ForegroundColor Yellow
        $open = Read-Host "Open in editor now? (y/n)"
        if ($open -eq 'y' -or $open -eq 'Y') {
            notepad $reviewFile
        }
        Read-Host "Press Enter to continue"
    } else {
        Write-Host "⚠️  Template not found at: $templatePath" -ForegroundColor Yellow
        Write-Host "Please enter your review message:" -ForegroundColor Yellow
        $reviewBody = Read-Host "Review message"
        if (-not $reviewBody) {
            Write-Host "Error: Review message is required" -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Submitting review..." -ForegroundColor Yellow

# Submit based on status
if ($useFile -or (Test-Path $reviewFile)) {
    # Use file for review
    switch ($Status) {
        "approve" {
            gh pr review $PRNumber --approve --body-file $reviewFile
        }
        "changes" {
            gh pr review $PRNumber --request-changes --body-file $reviewFile
        }
        "comment" {
            gh pr review $PRNumber --comment --body-file $reviewFile
        }
    }
} else {
    # Use inline message
    switch ($Status) {
        "approve" {
            gh pr review $PRNumber --approve --body $reviewBody
        }
        "changes" {
            gh pr review $PRNumber --request-changes --body $reviewBody
        }
        "comment" {
            gh pr review $PRNumber --comment --body $reviewBody
        }
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ Review submitted successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "View PR: gh pr view $PRNumber" -ForegroundColor Cyan
    Write-Host "Or visit: " -NoNewline -ForegroundColor Cyan

    # Get PR URL
    $prUrl = gh pr view $PRNumber --json url --jq .url
    Write-Host $prUrl -ForegroundColor White

    Write-Host ""
    $cleanup = Read-Host "Delete review file $reviewFile? (y/n)"
    if ($cleanup -eq 'y' -or $cleanup -eq 'Y') {
        Remove-Item $reviewFile -ErrorAction SilentlyContinue
        Write-Host "✓ Review file deleted" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "✗ Failed to submit review" -ForegroundColor Red
    Write-Host "Check the error above and try again" -ForegroundColor Yellow
}
