# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Compiles hip_ep_add_f32.hip to a standalone AMDGPU HSA code object for
# hip-ep-dx12-driver.exe (Dx12Runner / AMD Cross-Compile D3D12 API).
#
# Requires a ROCm/HIP device compiler (hipcc, or clang++ with amdgcn target
# support) on PATH. This targets the AMDGPU backend directly, which is why
# it is a separate script rather than a CMake target in this tree's normal
# host-toolchain build.
#
# Usage:
#   ./build_add_kernel.ps1 -Arch gfx1100
#   hip-ep-dx12-driver.exe --kernel hip_ep_add_f32.gfx1100.elf

param(
    [string]$Arch = "gfx1100",
    [string]$Output = "$PSScriptRoot/hip_ep_add_f32.$Arch.elf"
)

$ErrorActionPreference = "Stop"

if(-not (Get-Command hipcc -ErrorAction SilentlyContinue)) {
    throw "hipcc not found on PATH. Install a ROCm/HIP toolchain with AMDGPU device-compile support."
}

# --cuda-device-only: emit only the AMDGPU device code object, no host stub.
hipcc --offload-arch=$Arch --cuda-device-only -c "$PSScriptRoot/hip_ep_add_f32.hip" -o $Output

Write-Host "Wrote $Output"