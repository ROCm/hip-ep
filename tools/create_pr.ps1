##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Define your variables
$ErrorActionPreference = "Stop"

$token = "$Env:MY_GHE_TOKEN"
$branch = "$Env:MY_BRANCH"
$body = "$Env:MY_BODY"
$title = "$Env:MY_TITLE"
$repoOwner = "VitisAI"
$repoName = "vaip"
$baseBranch = "cp_dev"
$apiUrl = "https://gitenterprise.xilinx.com/api/v3/repos/$repoOwner/$repoName/pulls"

# Create headers
$headers = @{
    Authorization = "token $token"
    Accept        = "application/vnd.github.v3+json"
    "Content-Type"= "application/json"
}

# Function to check if a PR exists for the given base and head branches
function Get-ExistingPr {
    param (
        [string]$baseBranch,
        [string]$headBranch
    )

    $prApiUrl = "${apiUrl}?base=${baseBranch}&head=VitisAI:${headBranch}"
    Write-Debug "Checking for existing PRs with base $baseBranch and head $headBranch..."
    Write-Debug "API URL: $prApiUrl"
    Write-Debug "API URL: $apiUrl"
    $prResponse = Invoke-RestMethod -Uri $prApiUrl -Method Get -Headers $headers
    return $prResponse
}

# Check if a PR already exists for the given branches
$existingPrs = Get-ExistingPr -baseBranch $baseBranch -headBranch $branch
Write-Debug "Checking for existing PRs with base $existingPRs.Count and head $headBranch..."

if ($null -eq $existingPrs) {
    # If no PR exists, create a new one
    Write-Host "No existing PR found. Creating a new PR..."

    # Define the pull request body for creation
    $body = @{
        title = "$title"
        body  = "$body"
        head  = "$branch"
        base  = "$baseBranch"
    } | ConvertTo-Json -Depth 10

    # Send the POST request to create the PR
    $response = Invoke-RestMethod -Uri $apiUrl -Method Post -Headers $headers -Body $body

    Write-Host "PR created successfully."

    # Add comment to original MorphiZen PR only when creating new PR
    if ($Env:MORPHIZEN_PR_NUMBER) {
        Write-Host "Adding comment to MorphiZen PR #$($Env:MORPHIZEN_PR_NUMBER)..."

        $morphizenCommentUrl = "https://gitenterprise.xilinx.com/api/v3/repos/VitisAI/MorphiZen/issues/$($Env:MORPHIZEN_PR_NUMBER)/comments"

        # Create comment text without emoji or special characters
        $commentText = "**VAIP Verification PR Created**" + "`n`n" + "Verification PR: " + $response.html_url + "`n`n" + "This PR will test the MorphiZen changes in the VAIP environment."

        $commentBody = @{
            body = $commentText
        } | ConvertTo-Json -Compress

        # Debug: Show the JSON payload being sent
        Write-Host "Debug - Comment JSON payload: $commentBody"
        Write-Host "Debug - Comment URL: $morphizenCommentUrl"

        try {
            $commentResponse = Invoke-RestMethod -Uri $morphizenCommentUrl -Method Post -Headers $headers -Body $commentBody
            Write-Host "Comment added to MorphiZen PR successfully: $($commentResponse.html_url)"
        }
        catch {
            Write-Host "Failed to add comment to MorphiZen PR. Error details:"
            Write-Host "Status Code: $($_.Exception.Response.StatusCode)"
            Write-Host "Status Description: $($_.Exception.Response.StatusDescription)"

            # Try to get response body for more details
            if ($_.Exception.Response) {
                $responseStream = $_.Exception.Response.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($responseStream)
                $responseBody = $reader.ReadToEnd()
                Write-Host "Response Body: $responseBody"
            }

            Write-Warning "Full Error: $($_.Exception.Message)"
        }
    }
} else {
    # If a PR already exists, update it
    $prNumber = $existingPrs.number
    $prTitle = $existingPrs.title
    Write-Host "PR already exists with number: $prNumber  $prTitle. Updating PR..."

    $prUpdateBody = @{
        title = "$title"
        body  = "$body"
    } | ConvertTo-Json -Depth 10

    $prApiUrl = "https://gitenterprise.xilinx.com/api/v3/repos/$repoOwner/$repoName/pulls/$prNumber"
    $response = Invoke-RestMethod -Uri $prApiUrl -Method Patch -Headers $headers -Body $prUpdateBody
    Write-Host "PR updated successfully."
}

write-host "PR url: $($response.html_url)"
Write-Host "PR body: $($response.body)"

# Output the result
# $response | Format-List
