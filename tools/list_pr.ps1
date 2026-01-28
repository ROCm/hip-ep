##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Define your variables
$ErrorActionPreference = "Stop"
$token = "$Env:MY_GHE_TOKEN"
$repoOwner = "MorphiZen"
$repoName = "vaip"
$apiUrl = "https://gitenterprise.xilinx.com/api/v3/repos/$repoOwner/$repoName/pulls"
$apiUrl = "https://gitenterprise.xilinx.com/api/v3/repos/$repoOwner/$repoName/pulls?state=open"

$headers = @{
    Authorization = "token $token"
    Accept        = "application/vnd.github.v3+json"
    "Content-Type"= "application/json"
}

$response = Invoke-RestMethod -Uri $apiUrl -Method Get -Headers $headers

# Output the result
# format list of pr nicely
# Output the list of PRs nicely
if ($response.Count -eq 0) {
    Write-Host "No open pull requests found."
} else {
    Write-Host "Open Pull Requests:"
    foreach ($pr in $response) {
        Write-Host "----------------------------------------"
        Write-Host "PR Number   : $($pr.number)"
        Write-Host "Title       : $($pr.title)"
        Write-Host "Author      : $($pr.user.login)"
        Write-Host "Created At  : $($pr.created_at)"
        Write-Host "URL         : $($pr.html_url)"
    }
    Write-Host "----------------------------------------"
}
