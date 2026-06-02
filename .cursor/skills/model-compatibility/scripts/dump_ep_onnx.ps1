# Dump EP input ONNX graph (onnx.onnx) via VOE test_onnx_runner + pass.init only.

# Usage:

#   .\dump_ep_onnx.ps1 -ModelPath "D:\path\to\model.onnx"

#   .\dump_ep_onnx.ps1 -ModelPath "D:\path\to\model.onnx" -DumpFileName "onnx.onnx"

# Dump output defaults to the same directory as the input model.

#

# Uses minimal only_init_config.json (init pass + VAIML target). Forces target via

# XLNX_CONFIG_TARGET_NAME and provider option "target" to avoid mepTable/X2 auto-discovery.



param(

    [Parameter(Mandatory = $true, Position = 0)]

    [string]$ModelPath,



    [string]$VoePackageRoot = "D:\Users\mingyue\cp_dev\AIESW-31511\onnx-rt-1250\onnx-rt",

    [string]$DumpDirectory = "",

    [string]$DumpFileName = "",

    [string]$VaipConfigPath = "",

    [string]$VaipTarget = "VAIML"

)



$ErrorActionPreference = "Stop"



# ProviderPath avoids "Microsoft.PowerShell.Core\FileSystem::" prefix (breaks native EXE on UNC).
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath

if ([string]::IsNullOrWhiteSpace($DumpDirectory)) {

    $DumpDirectory = [System.IO.Path]::GetDirectoryName($ModelPath)

}



$VoeBin = Join-Path $VoePackageRoot "bin"

$RunnerExe = Join-Path $VoeBin "test_onnx_runner.exe"



if (-not (Test-Path -LiteralPath $RunnerExe)) {

    throw "test_onnx_runner.exe not found: $RunnerExe"

}



if ([string]::IsNullOrWhiteSpace($VaipConfigPath)) {

    $VaipConfigPath = Join-Path $PSScriptRoot "only_init_config.json"

}

if (-not (Test-Path -LiteralPath $VaipConfigPath)) {

    $fallbackConfig = "D:\Users\mingyue\cp_dev\source\test_onnx_runner\win_scripts\vaip_config.json"

    if (Test-Path -LiteralPath $fallbackConfig) {

        $VaipConfigPath = $fallbackConfig

    }

}

$VaipConfigPath = (Resolve-Path -LiteralPath $VaipConfigPath).ProviderPath



if ([string]::IsNullOrWhiteSpace($DumpFileName)) {

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)

    $DumpFileName = "${baseName}_onnx.onnx"

}



New-Item -ItemType Directory -Force -Path $DumpDirectory | Out-Null



# EP reads VITISAI_EP_JSON_CONFIG relative to process cwd (VOE bin).

$VaipConfigInBin = Join-Path $VoeBin "only_init_config.json"

Copy-Item -LiteralPath $VaipConfigPath -Destination $VaipConfigInBin -Force



$providerOption = "a|a pass.init.enable_dump|1 pass.init.directory|$DumpDirectory pass.init.filename|$DumpFileName target|$VaipTarget b|b"

$dumpPath = Join-Path $DumpDirectory $DumpFileName



Write-Host "Model:           $ModelPath"

Write-Host "VOE bin:         $VoeBin"

Write-Host "Dump directory:  $DumpDirectory"

Write-Host "Dump filename:   $DumpFileName"

Write-Host "Expected output: $dumpPath"

Write-Host "VAIP config:     $VaipConfigPath"

Write-Host "  -> copied to:  $VaipConfigInBin"

Write-Host "VITISAI_EP_JSON_CONFIG=only_init_config.json"

Write-Host "XLNX_CONFIG_TARGET_NAME=$VaipTarget"

Write-Host "provider target=$VaipTarget (init pass dump only)"

Write-Host ""



$prevVaipConfig = $env:VITISAI_EP_JSON_CONFIG

$prevProviderOpt = $env:XLNX_EXTERNAL_PROVIDER_OPTION

$prevVaipTarget = $env:XLNX_CONFIG_TARGET_NAME



Push-Location $VoeBin

try {

    $env:VITISAI_EP_JSON_CONFIG = "only_init_config.json"

    $env:XLNX_CONFIG_TARGET_NAME = $VaipTarget

    $env:XLNX_EXTERNAL_PROVIDER_OPTION = $providerOption

    & ".\test_onnx_runner.exe" $ModelPath

    $exitCode = $LASTEXITCODE

}

finally {

    Pop-Location

    if ($null -eq $prevVaipConfig) {

        Remove-Item Env:VITISAI_EP_JSON_CONFIG -ErrorAction SilentlyContinue

    } else {

        $env:VITISAI_EP_JSON_CONFIG = $prevVaipConfig

    }

    if ($null -eq $prevProviderOpt) {

        Remove-Item Env:XLNX_EXTERNAL_PROVIDER_OPTION -ErrorAction SilentlyContinue

    } else {

        $env:XLNX_EXTERNAL_PROVIDER_OPTION = $prevProviderOpt

    }

    if ($null -eq $prevVaipTarget) {

        Remove-Item Env:XLNX_CONFIG_TARGET_NAME -ErrorAction SilentlyContinue

    } else {

        $env:XLNX_CONFIG_TARGET_NAME = $prevVaipTarget

    }

}



if (-not (Test-Path -LiteralPath $dumpPath)) {

    if ($exitCode -ne 0) {

        throw "test_onnx_runner.exe failed with exit code $exitCode and dump file was not created: $dumpPath"

    }

    throw "Dump file was not created: $dumpPath"

}



$size = (Get-Item -LiteralPath $dumpPath).Length

if ($size -le 0) {

    throw "Dump file is empty: $dumpPath"

}



Write-Host ""

if ($exitCode -ne 0) {

    Write-Host "WARN: test_onnx_runner.exe exited with code $exitCode, but dump file exists." -ForegroundColor Yellow

    Write-Host "      (pass.init dump often completes before inference; treating as success.)"

}

Write-Host "OK: dumped EP ONNX graph ($size bytes)"

Write-Host "    $dumpPath"


