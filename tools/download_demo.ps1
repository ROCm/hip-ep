Write-Host "Download morphizen-demo ..."
$ErrorActionPreference = "Stop"
$morphizenDemoPath = "$Env:VAI_RT_WORKSPACE/morphizen-demo"
if (-Not (Test-Path -Path $morphizenDemoPath)) {
    Write-Host "morphizen-demo Directory does not exist. Cloning the repository..."
    git clone git@gitenterprise.xilinx.com:VitisAI/morphizen-demo.git --branch dev --single-branch --depth 1 $morphizenDemoPath
}
Write-Host "morphizen-demo Directory has exist. Using the latest commit..."
$currentDirectory = Get-Location
Set-Location -Path $morphizenDemoPath
git fetch --depth 1 origin dev
git reset --hard FETCH_HEAD
git clean -fdx
Set-Location -Path $currentDirectory
$HEAD = git rev-parse HEAD
Write-Host "using morphizen-demo commit ID $HEAD"
