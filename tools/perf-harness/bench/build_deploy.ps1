##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
## Build the runtime artifacts and deploy them where the benchmark drivers load
## them from, so an A/B is one command and cannot silently measure a stale DLL.
##
## Why this exists as a script rather than a documented copy step. The perf
## harness measures whatever is in $HIPEP_BIN, which for the vlm driver is a
## venv's onnxruntime\capi -- not the cmake install prefix. Nothing connects the
## two, so a hand-run build followed by a forgotten copy reports the previous
## DLL's number as the new one. That failure is silent and produces a plausible
## result, which is the worst kind. Deploying prints each file's hash so a
## comparison against the previous run's hash proves the binary actually moved.
##
## Three artifacts carry the code the harness profiles:
##   hipgpu.dll                  lib/Runtime (gqa.cpp, matmul_nbits.cpp)
##   custom_kernels_<arch>.dll   lib/Runtime/Kernels/hip (gqa_kernel.hip,
##                               matmul_nbits_kernel.hip)
##   hip-compiler.dll            lib/Conversion, lib/Dialect -- everything that
##                               decides what the runtime is asked to execute
## They are deployed as a set on purpose. hipgpu and custom_kernels share the
## extern "C" kernel ABI, so a mixed pair is only accidentally correct -- and a
## mixed pair was in fact deployed on this machine (hipgpu from one build,
## custom_kernels from an eight-hour-older one), which is unattributable as a
## baseline. hip-compiler is here because omitting it produced the opposite and
## more insidious failure: a conversion-only change measured as an exact no-op,
## the stale compiler having emitted the old graph for a correctly rebuilt
## runtime. Anything that lowers a graph belongs in this list.

[CmdletBinding()]
param(
  # Ninja build directory. Defaults to the sibling layout build.py documents:
  # <workspace>/build/<repo>.
  [string]$BuildDir,
  [string]$Config = 'Release',
  [switch]$SkipBuild,
  [switch]$SkipDeploy,
  # Build everything instead of just the two deployed artifacts. Slower; needed
  # when a change touches the MLIR tools or the lit suite.
  [switch]$All,
  [int]$Parallel = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path (Split-Path -Parent $PSScriptRoot) 'common.ps1')

if (-not $BuildDir) {
  $env:HIPEP_BUILD_DIR | Out-Null
  $fromEnv = [Environment]::GetEnvironmentVariable('HIPEP_BUILD_DIR')
  if ($fromEnv) {
    $BuildDir = $fromEnv
  } else {
    $repoRoot = $HarnessEnv.RepoRoot
    $BuildDir = Join-Path (Split-Path -Parent $repoRoot) "build\$(Split-Path -Leaf $repoRoot)"
  }
}
if (-not (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))) {
  throw "No CMakeCache.txt in '$BuildDir'. Configure first (python build.py), or pass -BuildDir."
}
$BuildDir = (Resolve-Path $BuildDir).Path

# ---------------------------------------------------------------------------
# MSVC environment
# ---------------------------------------------------------------------------
# build.py deliberately does not source vcvars: with the Visual Studio generator
# CMake finds MSVC itself, and with Ninja it expects the caller to already be in
# a developer prompt. This build tree is Ninja, so a plain shell links against
# nothing and fails on kernel32.lib. Import the environment here instead of
# requiring the caller to remember, and only when cl.exe is genuinely absent so
# an already-correct developer prompt is left untouched.
function Import-MsvcEnv {
  if (Get-Command cl.exe -EA SilentlyContinue) {
    Write-Host "    MSVC env: already present"
    return
  }
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; cannot locate MSVC." }
  $vsPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
  if (-not $vsPath) { throw "vswhere found no VS install with the x64 C++ toolset." }
  $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
  if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under '$vsPath'." }

  # Run vcvars in a child cmd and lift the resulting environment back out. The
  # marker separates vcvars' own banner from the variable dump; without it a
  # banner line can parse as a KEY=VALUE pair. cmd's `echo X && set` emits the
  # marker with a trailing space, so compare trimmed.
  $marker = '___VCVARS_ENV___'
  $out = & cmd.exe /c "call `"$vcvars`" >nul 2>&1 && echo $marker && set"
  $seen = $false
  $n = 0
  foreach ($line in $out) {
    if (-not $seen) { if ($line.Trim() -eq $marker) { $seen = $true }; continue }
    $i = $line.IndexOf('=')
    if ($i -lt 1) { continue }
    Set-Item -Path "Env:$($line.Substring(0, $i))" -Value $line.Substring($i + 1)
    $n++
  }
  if (-not (Get-Command cl.exe -EA SilentlyContinue)) {
    throw "Imported $n vars from vcvars64.bat but cl.exe is still not on PATH."
  }
  Write-Host "    MSVC env: imported $n vars from $vcvars"
}

# ---------------------------------------------------------------------------
# Which artifacts, and where they land
# ---------------------------------------------------------------------------
# The custom-kernels target name carries the arch, which is a cache variable
# rather than something to hardcode: this repo builds one DLL per arch so a
# consumer does not load code for archs it will never run.
$arch = (Select-String -Path (Join-Path $BuildDir 'CMakeCache.txt') `
    -Pattern '^HIP_ARCHITECTURES:STRING=(.+)$').Matches[0].Groups[1].Value
$arch = ($arch -split ';')[0].Trim()
if (-not $arch) { throw "Could not read HIP_ARCHITECTURES from the cmake cache." }

$artifacts = @('hipgpu.dll', "custom_kernels_$arch.dll", 'hip-compiler.dll')

if (-not $SkipBuild) {
  Write-Host ">>> build [$Config] $BuildDir"
  Import-MsvcEnv
  if ($Parallel -le 0) { $Parallel = [Environment]::ProcessorCount }
  $cmd = @('--build', $BuildDir, '--config', $Config, '--parallel', "$Parallel")
  if (-not $All) {
    # Target names, not file names: the EP target is named by morphizen's
    # versioned unique id and only its OUTPUT_NAME is hipgpu, so ask cmake for
    # the file and let ninja resolve the producing target.
    $cmd += @('--target')
    foreach ($a in $artifacts) { $cmd += "bin/$a" }
  }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & cmake @cmd
  if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }
  Write-Host ("    built in {0:N1} min" -f $sw.Elapsed.TotalMinutes)
}

if (-not $SkipDeploy) {
  $dest = $HarnessEnv.Bin
  Write-Host ">>> deploy -> $dest"
  foreach ($a in $artifacts) {
    $src = Join-Path $BuildDir "bin\$a"
    if (-not (Test-Path $src)) { throw "missing build output: $src" }
    Copy-Item $src (Join-Path $dest $a) -Force
    $h = (Get-FileHash (Join-Path $dest $a) -Algorithm SHA256).Hash.Substring(0, 16)
    $t = (Get-Item $src).LastWriteTime.ToString('MM-dd HH:mm:ss')
    Write-Host ("    {0,-32} {1}  {2}" -f $a, $h, $t)
  }
}
