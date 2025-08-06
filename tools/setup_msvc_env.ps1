##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# for fix  No CMAKE_C_COMPILER could be found
# vswhere.exe is the officially recommended tool for finding the Visual C++ compilers and linkers.
# It is typically located in the following directory:
# %ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe

# REF Doc: https://github.com/microsoft/vswhere
# ref doc: https://github.com/microsoft/vswhere/wiki/Find-VC

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at: $vswhere"
}
$installPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath

if (-not $installPath -or -not (Test-Path $installPath)) {
    throw "Visual Studio installation path not found or invalid: $installPath"
}

# Here is not use common7\tools\vsdevcmd.bat, because vsdevcmd.bat default --arch is x86, but we need x64
# so we use vcvars64.bat,  vcvars64.bat is the recommended way to set up the environment for x64 builds
$vsdevcmdPath = Join-Path $installPath 'VC\Auxiliary\Build\vcvars64.bat'
if (Test-Path $vsdevcmdPath) {
    cmd /s /c """$vsdevcmdPath"" $args && set" | Where-Object { $_ -match '^(.*?)=(.*)$' } | ForEach-Object {
        Write-Host "Setting environment variable: $($Matches[1]) = $($Matches[2])"
        $null = New-Item -Force -Path "Env:\$($Matches[1])" -Value $Matches[2]
    }
} else {
    throw "vsdevcmd.bat not found at: $vsdevcmdPath"
}
