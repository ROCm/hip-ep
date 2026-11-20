#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Unit checks for the RGP decoder layers that can be tested without a GPU.

The corpus tests run when HIPEP_RGP_CORPUS points at a directory of .rgp files
(see README: "Decoder parity"); otherwise they are skipped.
"""

import io
import os
from pathlib import Path

import pytest

from hipep_perf.rgp import container, elf, kernels, occupancy, symbols

CORPUS = (
    Path(os.environ.get("HIPEP_RGP_CORPUS", ""))
    if os.environ.get("HIPEP_RGP_CORPUS")
    else None
)


def test_kernel_names():
    k = kernels.parse(
        "Cijk_Ailk_Bljk_HHS_BH_Bias_MT128x48x32_MI16x16x1_SK3_GSU2_WG64_2_1_ISA1151"
    )
    assert (k.library, k.family, k.dtype, k.tile) == (
        "tensile",
        "gemm",
        "f16",
        "128x48x32",
    )
    assert k.label == "hgemm 128x48x32 wmma16x16 wg64x2 gsu3 +bias"
    # Inherited from the reference decoder and kept for output parity: casts such
    # as "(int)64" from itanium_demangler are not read as numeric config.
    k = kernels.parse("matmul_nbits_gemv_dp4a_kernel<(int)64, (int)2, (int)1>")
    assert k.family == "matmul_nbits_gemv_dp4a" and "config" not in k.params
    assert kernels.parse("foo_kernel<64, 2>").params["config"] == "64,2"
    assert kernels.parse("rope_kernel<__half>").dtype == "f16"
    assert kernels.parse("0x1234").family == "unresolved"


def test_simplify():
    assert symbols.simplify("void ns::foo<int, 2>(float*, int)") == "foo<int, 2>"
    assert symbols.simplify("(anonymous namespace)::bar(int)") == "bar"


def test_default_op_map_and_overrides():
    from hipep_perf.analysis import perfcommon

    m = perfcommon.op_family_map()
    assert perfcommon.op_for_family("matmul_nbits_gemv_dp4a", m) == "matmul_nbits"
    m = perfcommon.op_family_map(["gqa+=matmul_nbits_gemv_dp4a"])
    assert perfcommon.op_for_family("matmul_nbits_gemv_dp4a", m) == "gqa"


def test_occupancy_wgp_model():
    # 512 threads with modest VGPRs must not come out at zero resident workgroups.
    pct, lim, waves = occupancy.theoretical(vgpr=64, sgpr=128, lds=0, workgroup=512)
    assert pct > 0 and waves >= 16
    pct, lim, _ = occupancy.theoretical(vgpr=8, sgpr=128, lds=60 * 1024, workgroup=256)
    assert lim == "LDS"


@pytest.mark.skipif(CORPUS is None, reason="set HIPEP_RGP_CORPUS")
def test_elf_reader_matches_pyelftools():
    ELFFile = pytest.importorskip("elftools.elf.elffile").ELFFile

    def reference(blob: bytes):
        fns = []
        ef = ELFFile(io.BytesIO(blob))
        st = ef.get_section_by_name(".symtab") or ef.get_section_by_name(".dynsym")
        for s in st.iter_symbols() if st else []:
            if (
                s["st_info"]["type"] == "STT_FUNC"
                and s.name
                and not s.name.startswith("__hip_cuid")
            ):
                fns.append((s["st_value"], s["st_size"] or 0x100, s.name))
        return sorted(fns)

    checked = 0
    for rgp in sorted(CORPUS.glob("*.rgp")):
        c = container.Container.open(rgp)
        for ch in c.find("CodeObject"):
            blob = c.data(ch)
            assert elf.functions(blob) == reference(blob), rgp.name
            checked += 1
    assert checked > 0
