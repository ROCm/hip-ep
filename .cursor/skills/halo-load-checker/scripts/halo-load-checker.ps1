##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
param(
    [string]$Filter = "",
    [switch]$GpuOnly
)
$ErrorActionPreference = "SilentlyContinue"
function Write-Section($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}
function Get-DefaultBuildDir {
    if ($env:BUILD_DIR) { return $env:BUILD_DIR }
    $workspace = if ($env:WORKSPACE) { $env:WORKSPACE } else { Join-Path $env:USERPROFILE "workspace" }
    return Join-Path $workspace "build\hip-ep"
}
function Get-TherockCandidates {
    $buildDir = Get-DefaultBuildDir
    @(
        $env:THEROCK_DIST,
        (Join-Path $buildDir "_therock")
    ) | Where-Object { $_ -and (Test-Path (Join-Path $_ "bin")) } | ForEach-Object {
        (Resolve-Path -LiteralPath $_).Path
    }
}
function Write-WindowsGpuUtilization {
    try {
        $samples = Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop
        $all = @($samples.CounterSamples)
        if (-not $all.Count) {
            throw "No GPU engine counters"
        }
        $compute = $all | Where-Object { $_.InstanceName -match 'engtype_3D|Compute|CUDA' }
        if (-not $compute.Count) { $compute = $all }
        $max = ($compute | Measure-Object -Property CookedValue -Maximum).Maximum
        Write-Host ("Windows GPU engine utilization (snapshot max 3D/compute): {0:N1}%" -f $max)
        $top = $compute | Sort-Object CookedValue -Descending | Select-Object -First 3
        foreach ($s in $top) {
            if ($s.CookedValue -gt 0.5) {
                $short = ($s.InstanceName -replace '^.*?pid_\d+_', '')
                Write-Host ("  {0:N1}%  {1}" -f $s.CookedValue, $short)
            }
        }
    } catch {
        Write-Host "GPU utilization %: unavailable via Windows perf counters."
        Write-Host "Use Task Manager -> Performance -> GPU for live engine utilization."
    }
}
function Write-HipInfoSummary {
    param([string]$HipInfoExe)
    $out = (& $HipInfoExe 2>&1 | Out-String)
    if (-not $out.Trim()) {
        Write-Host "hipInfo produced no output."
        return
    }
    $name = $null
    $arch = $null
    $memTotal = $null
    $memFree = $null
    if ($out -match '(?m)^Name:\s+(.+)$') { $name = $Matches[1].Trim() }
    if ($out -match '(?m)^gcnArchName:\s+(\S+)') { $arch = $Matches[1].Trim() }
    if ($out -match '(?m)^memInfo\.total:\s+(.+)$') { $memTotal = $Matches[1].Trim() }
    if ($out -match '(?m)^memInfo\.free:\s+(.+)$') { $memFree = $Matches[1].Trim() }
    Write-Host "hipInfo (device memory at snapshot):"
    if ($name) { Write-Host "  Name: $name" }
    if ($arch) { Write-Host "  gcnArchName: $arch" }
    if ($memTotal) { Write-Host "  memInfo.total: $memTotal" }
    if ($memFree) { Write-Host "  memInfo.free: $memFree" }
    if ($memFree -match '\((\d+)%\)') {
        $pctFree = [int]$Matches[1]
        Write-Host ("  VRAM/unified memory in use (approx): {0}%" -f (100 - $pctFree))
    }
}
function Write-ClinfoSummary {
    param([string]$ClinfoExe)
    $out = (& $ClinfoExe 2>&1 | Out-String)
    if (-not $out.Trim()) {
        Write-Host "clinfo produced no output."
        return
    }
    $deviceCount = $null
    $board = $null
    if ($out -match 'Number of devices:\s+(\d+)') { $deviceCount = $Matches[1] }
    if ($out -match 'Board name:\s+(.+)') { $board = $Matches[1].Trim() }
    Write-Host "OpenCL (clinfo):"
    if ($null -ne $deviceCount) { Write-Host "  devices: $deviceCount" }
    if ($board) { Write-Host "  board: $board" }
}
function Write-HaloGpuSection {
    $therockCandidates = @(Get-TherockCandidates)
    foreach ($root in $therockCandidates) {
        $bin = Join-Path $root "bin"
        $rocm = Join-Path $bin "rocm-smi.exe"
        $amd = Join-Path $bin "amd-smi.exe"
        if (Test-Path $rocm) {
            Write-Host "THEROCK_DIST: $root (rocm-smi)"
            & $rocm 2>&1
            return
        }
        if (Test-Path $amd) {
            Write-Host "THEROCK_DIST: $root (amd-smi)"
            & $amd monitor 2>&1
            return
        }
    }
    $fallbackRoot = $therockCandidates | Select-Object -First 1
    if ($fallbackRoot) {
        $bin = Join-Path $fallbackRoot "bin"
        $hipInfo = Join-Path $bin "hipInfo.exe"
        $clinfo = Join-Path $bin "clinfo.exe"
        Write-Host "THEROCK_DIST: $fallbackRoot"
        Write-Host "Note: rocm-smi/amd-smi not in this THERock bundle; using hipInfo/clinfo + Windows GPU counters."
        if (Test-Path $hipInfo) {
            Write-HipInfoSummary -HipInfoExe $hipInfo
        } else {
            Write-Host "hipInfo.exe not found under $bin"
        }
        if (Test-Path $clinfo) {
            Write-ClinfoSummary -ClinfoExe $clinfo
        }
        Write-WindowsGpuUtilization
        return
    }
    Write-Host "No THEROCK _therock/bin found (configure hip-ep to populate build/hip-ep/_therock)."
    Write-Host "Use Task Manager -> Performance -> GPU for activity."
}
if ($GpuOnly) {
    Write-Section "GPU (AMD)"
    Write-HaloGpuSection
    exit 0
}
Write-Section "Time / host"
Write-Host (Get-Date -Format "yyyy-MM-dd HH:mm:ss K")
Write-Host $env:COMPUTERNAME
Write-Section "CPU"
try {
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 2 |
        Select-Object -ExpandProperty CounterSamples |
        Select-Object -Last 1
    Write-Host ("Processor (_Total): {0:N1}%" -f $cpu.CookedValue)
} catch {
    Write-Host "CPU counter unavailable"
}
Write-Section "Memory"
$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$usedGB = [math]::Round($totalGB - $freeGB, 2)
$pctUsed = if ($totalGB -gt 0) { [math]::Round(100 * $usedGB / $totalGB, 1) } else { 0 }
Write-Host "Used: ${usedGB} GB / ${totalGB} GB (${pctUsed}% used, ${freeGB} GB free)"
Write-Section "GPU (AMD)"
Write-HaloGpuSection
Write-Section "Top processes (CPU)"
Get-Process | Sort-Object CPU -Descending | Select-Object -First 12 Name, Id,
    @{N='CPU_s';E={[math]::Round($_.CPU, 1)}},
    @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB, 1)}} |
    Format-Table -AutoSize
Write-Section "Top processes (working set)"
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 12 Name, Id,
    @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB, 1)}},
    @{N='CPU_s';E={[math]::Round($_.CPU, 1)}} |
    Format-Table -AutoSize
if ($Filter) {
    Write-Section "Filter: $Filter"
    Get-Process | Where-Object { $_.Name -like "*$Filter*" -or $_.Path -like "*$Filter*" } |
        Select-Object Name, Id, CPU, @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB, 1)}}, Path |
        Format-Table -AutoSize
}
Write-Section "Build / benchmark hints"
@(
    "model_benchmark.exe",
    "onnxruntime_perf_test.exe",
    "hip-compiler.exe",
    "cmake.exe",
    "MSBuild.exe",
    "cl.exe"
) | ForEach-Object {
    $p = Get-Process -Name ($_ -replace '\.exe$','') -ErrorAction SilentlyContinue
    if ($p) { Write-Host "RUNNING: $_ (count $($p.Count))" }
}
