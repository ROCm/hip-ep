#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Decoder for Radeon GPU Profiler captures (.rgp), RDP- or ktrace-produced.

Layers, bottom up:

* ``container``  RDF chunk container (libamdrdf format), zstd via the stdlib.
* ``elf``        AMDGPU code-object symbol tables.
* ``tokens``     SQTT hardware token stream for gfx11/gfx11.5 (a port of AMD's
                 rocprof-trace-decoder token parser).
* ``sqtt``       per-shader-engine decode, clock alignment, wave attribution.
* ``markers``    RGP SQTT markers (dispatch grids, barriers; Mesa ac_sqtt.h).
* ``spm``        streaming performance counter chunks.
* ``symbols``    dispatch PC -> kernel name.
* ``kernels``    kernel-name decoding into op family / dtype / tile.
* ``occupancy``  theoretical occupancy and the per-dispatch bound class.
* ``build``      assembles a ``model.Capture``; ``formats`` writes it out.
"""
