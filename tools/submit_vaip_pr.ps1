Write-Host "Running PowerShell version $($PSVersionTable.PSVersion)"
# Set-PSDebug -Trace 1
$ErrorActionPreference = "Stop"
# Set environment variables if it is not set
if (-Not $Env:VAIP_REMOTE_BRANCH) {
    $Env:VAIP_REMOTE_BRANCH = "cp_dev"
}

$SCRIPT_DIR = $PSScriptRoot
$PROJECT_DIR = (Resolve-Path "$SCRIPT_DIR\..").Path
$W = (Resolve-Path "$PROJECT_DIR\..").Path
$VAIP_DIR = "$W\vaip"

# Clone VAIP repository if it doesn't exist
if (-Not (Test-Path $VAIP_DIR)) {
    git clone git@gitenterprise.xilinx.com:VitisAI/vaip.git $VAIP_DIR
}
Write-Host "Syncing VAIP repository"
git -C $VAIP_DIR fetch origin ${Env:VAIP_REMOTE_BRANCH}
git -C $PROJECT_DIR fetch --all

# Get old and new commit IDs
$old_commit_id = git -C $VAIP_DIR show "origin/${Env:VAIP_REMOTE_BRANCH}:cmake/deps.txt" | Select-String -Pattern 'MorphiZen;' | ForEach-Object {
    ($_ -split ';')[2].Trim()
}
Write-Host "Old commit ID = $old_commit_id"

$new_commit_id = git -C "$W\MorphiZen" rev-parse origin/main
Write-Host "New commit ID = $new_commit_id"

# Check if update is needed
if ($old_commit_id -eq $new_commit_id) {
    Write-Host "No update"
    exit 0
}

# Prepare branch for update
$branch_name = "br_update_morphizen_from_$($old_commit_id.Substring(0, 8))"
Write-Host "Branch name = $branch_name"

git --git-dir="$VAIP_DIR/.git" --work-tree="$VAIP_DIR" checkout --detach --force "origin/${Env:VAIP_REMOTE_BRANCH}"

# Update deps.txt
(git --git-dir="$VAIP_DIR/.git" --work-tree="$VAIP_DIR" show  "origin/${Env:VAIP_REMOTE_BRANCH}:cmake/deps.txt") `
     -replace "(morphizen;https://gitenterprise\.xilinx\.com/VitisAI/MorphiZen\.git;)[a-f0-9]+", "`${1}$new_commit_id" |
    Set-Content "$VAIP_DIR/cmake/deps.txt"

git --git-dir="$VAIP_DIR/.git" --work-tree="$VAIP_DIR" add cmake/deps.txt

# Commit changes
$title = "update morphizen from $($old_commit_id.Substring(0, 8)) to $($new_commit_id.Substring(0, 8))"
$change_log = git --git-dir="$PROJECT_DIR/.git" --work-tree="$PROJECT_DIR" log `
    --date=short --reverse `
    --pretty=format:"  - %h %s (by %an @ %ad)" "$old_commit_id..$new_commit_id" `
    --date-order |
    ForEach-Object { $_ -replace "#(\d+)", "VitisAI/MorphiZen#`${1}" }
$change_log = $change_log -join "`n"
$body = "Change Log`n`n$change_log`n`n"
$msg = "$title`n`n$body"
git --git-dir="$VAIP_DIR/.git" --work-tree="$VAIP_DIR" commit -am "$msg"
git --git-dir="$VAIP_DIR/.git" --work-tree="$VAIP_DIR" push --force origin "HEAD:refs/heads/$branch_name"

$Env:MY_TITLE = "$title"
$Env:MY_BODY = "$body"
$Env:MY_BRANCH = "$branch_name"
powershell "$SCRIPT_DIR/create_pr.ps1"