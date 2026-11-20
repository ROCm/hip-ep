#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Theoretical occupancy and the per-dispatch bound class.

Occupancy is the resident-wave ceiling a kernel's own VGPR / LDS / workgroup
usage allows, as a percentage of the hardware maximum (the same idea rocprof
calls theoretical occupancy). It is a resource ceiling, not a measurement.

gfx1151 (RDNA3.5) constants, checked against LLVM's AMDGPU IsaInfo for the
gfx1151 feature set: 1536 VGPRs per SIMD at a 24-register granule (wave32),
16 waves per SIMD, 512-byte LDS granule. The residency unit is the WGP (4
SIMDs, 128 KB LDS): the pinned toolchain compiles every kernel in WGP mode
(.amdhsa_workgroup_processor_mode 1). Modelling a CU instead halves both the
wave slots and the LDS pool and misattributes the binding resource. SGPRs do
not limit occupancy on gfx10+.
"""

from __future__ import annotations

import math

GFX1151 = {
    "max_waves_per_simd": 16,
    "simds_per_wgp": 4,
    "vgpr_file_per_simd": 1536,
    "vgpr_gran": 24,
    "lds_per_wgp": 131072,
    "lds_gran": 512,
    "max_wg_per_wgp": 64,
}

OVERHEAD_DUR_US = 3.0  # shorter dispatches are launch-overhead sensitive
LOW_OCC_PCT = 50.0
MEM_ROOF_FRAC = 0.7  # >= this fraction of the roofline is memory-bound
MEM_LOW_FRAC = 0.3


def _roundup(x: int, g: int) -> int:
    return int(math.ceil(x / g) * g) if x > 0 else 0


def theoretical(
    vgpr: int,
    sgpr: int,
    lds: int,
    workgroup: int,
    wave_size: int = 32,
    hw: dict = GFX1151,
) -> tuple[float, str, int]:
    """(occupancy %, binding resource or '', resident waves per WGP)."""
    simds = hw["simds_per_wgp"]
    max_waves = hw["max_waves_per_simd"] * simds
    waves_per_wg = max(1, math.ceil(workgroup / wave_size)) if workgroup > 0 else 1
    if vgpr and vgpr > 0:
        simd_waves = min(
            hw["max_waves_per_simd"],
            hw["vgpr_file_per_simd"] // max(1, _roundup(vgpr, hw["vgpr_gran"])),
        )
    else:
        simd_waves = hw["max_waves_per_simd"]
    limits = {
        "VGPR": (simd_waves * simds) // waves_per_wg,
        "LDS": (hw["lds_per_wgp"] // max(1, _roundup(lds, hw["lds_gran"])))
        if lds and lds > 0
        else hw["max_wg_per_wgp"],
        "workgroup": hw["max_wg_per_wgp"],
    }
    wgs = max(0, min(limits.values()))
    waves = min(max_waves, wgs * waves_per_wg)
    pct = round(100.0 * waves / max_waves, 1) if max_waves else 0.0
    limiter = min(limits, key=lambda k: limits[k])
    return pct, ("" if pct >= 99.9 else limiter), waves


def classify(
    d, spm_present: bool, mem_reliable: bool, roofline_gbps: float = 256.0
) -> tuple[str, str]:
    """(bound class, auditable reason). Never infers a bandwidth that was not measured."""
    dur = d.dur_us
    if 0 < dur < OVERHEAD_DUR_US and d.gap_before_us > dur:
        return "overhead", f"dur {dur:.1f}us < gap {d.gap_before_us:.1f}us"
    occ = d.occ_pct
    note = (
        f"occ {occ:.0f}% {d.occ_limiter}-limited"
        if d.occ_limiter
        else f"occ {occ:.0f}%"
    )
    if spm_present and mem_reliable and d.mem_gbps > 0:
        frac = d.mem_gbps / roofline_gbps if roofline_gbps else 0.0
        if frac >= MEM_ROOF_FRAC:
            return "memory-bound", f"mem {d.mem_gbps:.0f} GB/s ({frac * 100:.0f}% roof)"
        if occ and occ < LOW_OCC_PCT and frac < MEM_LOW_FRAC:
            return "latency/low-occupancy", f"{note}, mem {d.mem_gbps:.0f} GB/s"
        return (
            "compute-bound",
            f"mem {d.mem_gbps:.0f} GB/s ({frac * 100:.0f}% roof), {note}",
        )
    if occ and occ < LOW_OCC_PCT:
        return "latency/low-occupancy", note
    return (
        "compute-or-memory (undetermined)",
        "no SPM" if not spm_present else "SPM window too short",
    )
