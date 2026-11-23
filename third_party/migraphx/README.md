# MIGraphX kernel headers (vendored)

The header-only GPU kernel library that MIGraphX compiles at run time, limited to
the files that `winograd_conv.hpp` (the fp16 Winograd F(2x2, 3x3) WMMA kernel)
includes. hip-ep embeds these files into the compiler and builds one Winograd
kernel per convolution shape with hiprtc while it compiles a model; see
`docs/design/winograd-conv.md`.

- Source: https://github.com/ROCm/AMDMIGraphX, `src/targets/gpu/kernels/include`
- Commit: `013f7fe7c` ("Prevent FP16 GroupNorm overflow with float intermediates (#5474)")
- License: MIT, see `LICENSE` in this directory.

The files are byte-identical to upstream. To update them, copy the same file list
from a newer MIGraphX commit, update the commit above, and rerun the Winograd LIT
and numeric tests.
