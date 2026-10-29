#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Static (theoretical) occupancy + bottleneck classification.

Occupancy here is the *theoretical* resident-wave ceiling a kernel's own resource
usage (VGPRs / LDS / workgroup size) allows, expressed as a percentage of the
hardware maximum -- the same idea rocprof reports as "theoretical occupancy". It is
derived by hand from the per-dispatch resource counts (see AMD GPUOpen "Occupancy
explained" and the ROCm occupancy-math blog), NOT measured; a low value means the
kernel leaves latency-hiding headroom on the table and names which resource binds.

We deliberately do not infer an *achieved* per-SIMD occupancy per dispatch (SQTT
gives total wavefronts + a global occupancy band, not a clean per-dispatch per-SIMD
number), so this stays a resource-ceiling metric.

Constants are for gfx1151 (RDNA3.5), verified against LLVM's AMDGPU backend
(the authoritative occupancy source), traced to the exact gfx1151 feature set:
  - Feature1536VGPRs is in FeatureISAVersion11_5_1 (gfx1151), and FeatureGFX11
    carries FeatureAddressableLocalMemorySize65536 + FeatureGFX10_3Insts.
  - vgpr_file_per_simd 1536 / vgpr_gran 24  -> IsaInfo::getTotalNumVGPRs /
    getVGPRAllocGranule (Feature1536VGPRs, wave32).
  - max_waves_per_simd 16                   -> IsaInfo::getMaxWavesPerEU
    (hasGFX10_3Insts ? 16 : 20).
  - lds_gran 512                            -> getAddressableLocalMemorySize
    (=65536) and its 128 encoding blocks (65536/128 = 512 B).
The VGPR ceiling below mirrors IsaInfo::getNumWavesPerEUWithNumVGPRs exactly, and
SGPRs are intentionally omitted (isSGPROccupancyLimited is false on gfx10+).
Occupancy is theoretical (a resource ceiling), not measured.

The residency unit is the WGP, not the CU. RDNA splits these: in CU mode a
workgroup is confined to one CU (2 SIMDs, 64 KB LDS), in WGP mode it spreads
over the whole workgroup processor (4 SIMDs, 128 KB LDS). The compiler picks,
and for this project it picks WGP -- compiling any kernel with the pinned
toolchain at --offload-arch=gfx1151 emits `.amdhsa_workgroup_processor_mode 1`,
which is also clang's default for gfx10+ absent -mcumode. The runtime assumes
the same: the LDS-barrier comment in matmul_nbits_kernel.hip turns on a
workgroup being able to span the two CUs of a WGP, and the config table reasons
about "two workgroups sharing a WGP".

Modelling a CU instead understates the pool twice over and is not a uniform
scaling error, because the two ceilings move differently: it halves the wave
slots a workgroup's own waves have to fit into, so any workgroup above 14 waves
came out as *zero* resident workgroups -- a 512-thread kernel that plainly ran
was reported at 0% occupancy -- while it also halved the LDS pool, which shifted
the binding resource. Both numbers and the limiter attribution were wrong.
"""

from __future__ import annotations

import math

# gfx1151 / RDNA3.5 occupancy limits (wave32). Verified vs LLVM IsaInfo (see above).
GFX1151 = {
    "max_waves_per_simd": 16,  # getMaxWavesPerEU (GFX10_3Insts)
    "simds_per_wgp": 4,  # WGP mode: 2 CUs x 2 SIMD32. 2 would be CU mode.
    "vgpr_file_per_simd": 1536,  # getTotalNumVGPRs, Feature1536VGPRs (wave32)
    "vgpr_gran": 24,  # getVGPRAllocGranule, Feature1536VGPRs (wave32)
    # Physical LDS per WGP. Distinct from the 64 KB a *single* workgroup can
    # address (getAddressableLocalMemorySize), which is why a 60 KB workgroup is
    # legal yet only two of them fit.
    "lds_per_wgp": 131072,
    "lds_gran": 512,  # 65536 / 128 LDS encoding blocks
    # Concurrent workgroup slots per WGP. Set to the wave ceiling so single-wave
    # workgroups can still reach full occupancy; the real HW value is higher than
    # needed for wave32 and only matters as an upper bound.
    "max_wg_per_wgp": 64,
}

# Classification thresholds.
OVERHEAD_DUR_US = 3.0  # dispatches this short are launch/overhead-sensitive
LOW_OCC_PCT = 50.0  # below this theoretical occupancy => latency-hiding headroom
MEM_ROOF_FRAC = 0.7  # mem_gbps >= this * roofline => memory-bound
MEM_LOW_FRAC = 0.3  # mem_gbps below this * roofline supports latency-bound call


def _roundup(x: int, g: int) -> int:
    return int(math.ceil(x / g) * g) if x > 0 else 0


def theoretical_occupancy(vgpr, sgpr, lds, workgroup, wave_size=32, hw=GFX1151):
    """Return (occ_pct, limiter, waves_per_wgp).

    occ_pct: theoretical resident waves as % of the hardware max waves/WGP.
    limiter: 'VGPR' | 'LDS' | 'workgroup' | '' (empty at full occupancy).
    waves_per_wgp: the theoretical resident-wave count per WGP.
    """
    simds = hw["simds_per_wgp"]
    max_waves_wgp = hw["max_waves_per_simd"] * simds
    waves_per_wg = max(1, math.ceil(workgroup / wave_size)) if workgroup > 0 else 1

    # VGPR ceiling (per SIMD -> per WGP).
    if vgpr and vgpr > 0:
        waves_simd_vgpr = min(
            hw["max_waves_per_simd"],
            hw["vgpr_file_per_simd"] // max(1, _roundup(vgpr, hw["vgpr_gran"])),
        )
    else:
        waves_simd_vgpr = hw["max_waves_per_simd"]
    wg_by_vgpr = (waves_simd_vgpr * simds) // waves_per_wg

    # LDS ceiling (per WGP, shared by every workgroup resident on it).
    if lds and lds > 0:
        wg_by_lds = hw["lds_per_wgp"] // max(1, _roundup(lds, hw["lds_gran"]))
    else:
        wg_by_lds = hw["max_wg_per_wgp"]

    wg_by_slots = hw["max_wg_per_wgp"]

    cands = {"VGPR": wg_by_vgpr, "LDS": wg_by_lds, "workgroup": wg_by_slots}
    wg_wgp = max(0, min(cands.values()))
    waves_wgp = min(max_waves_wgp, wg_wgp * waves_per_wg)
    occ_pct = round(100.0 * waves_wgp / max_waves_wgp, 1) if max_waves_wgp else 0.0

    limiter = min(cands, key=lambda k: cands[k])
    if occ_pct >= 99.9:
        limiter = ""
    return occ_pct, limiter, waves_wgp


def classify_bound(d, *, spm_present, mem_reliable, roofline_gbps=256.0):
    """Return (bound_class, bound_reason) for a dispatch.

    Degraded (no/unreliable SPM) value space: overhead | latency/low-occupancy |
    'compute-or-memory (undetermined)'. Full (SPM present + reliable) adds real
    memory-bound / compute-bound. We never fabricate a bandwidth to force the call.
    """
    dur = d.dur_us
    # overhead-bound: tiny kernel dominated by the idle gap around it.
    if 0 < dur < OVERHEAD_DUR_US and d.gap_before_us > dur:
        return "overhead", f"dur {dur:.1f}us < gap {d.gap_before_us:.1f}us"

    occ = d.occ_pct
    occ_note = (
        f"occ {occ:.0f}% {d.occ_limiter}-limited"
        if d.occ_limiter
        else f"occ {occ:.0f}%"
    )

    if spm_present and mem_reliable and d.mem_gbps > 0:
        frac = d.mem_gbps / roofline_gbps if roofline_gbps else 0.0
        if frac >= MEM_ROOF_FRAC:
            return "memory-bound", f"mem {d.mem_gbps:.0f} GB/s ({frac * 100:.0f}% roof)"
        if occ and occ < LOW_OCC_PCT and frac < MEM_LOW_FRAC:
            return "latency/low-occupancy", f"{occ_note}, mem {d.mem_gbps:.0f} GB/s"
        return (
            "compute-bound",
            f"mem {d.mem_gbps:.0f} GB/s ({frac * 100:.0f}% roof), {occ_note}",
        )

    # SPM absent or per-kernel BW not trustworthy: no memory-vs-compute call.
    if occ and occ < LOW_OCC_PCT:
        return "latency/low-occupancy", occ_note
    reason = "no SPM" if not spm_present else "SPM window too short"
    return "compute-or-memory (undetermined)", reason
