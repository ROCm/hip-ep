<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Winograd F(2x2, 3x3) fp16 convolution

With `HIPDNN_EP_ROCMLIR=1` and `HIPDNN_EP_WINOGRAD=1`, eligible fp16 3x3
convolutions run as one fused Winograd F(2x2, 3x3) kernel instead of a rocMLIR
conv kernel. The kernel is MIGraphX's `winograd_conv_f23_wmma`
(`third_party/migraphx`, MIT). It needs WMMA, so it is enabled for gfx1151 and
the gfx12 family only.

## Why a kernel per shape

The MIGraphX kernel reads every tensor extent and stride as a compile-time
constant, and its index math depends on them. hip-ep therefore builds one
kernel per convolution shape and epilogue while it compiles the model, the way
the MIGraphX JIT does, instead of shipping a few precompiled kernels:

1. `hip-winograd-conv` (first in the rocMLIR fusion passes) rewrites an eligible
   `hip.conv`, with the LeakyRelu or ReLU that alone consumes it, into
   `hip.winograd_conv`.
2. After the fusion passes, `compileWinogradKernels`
   (`lib/Compiler/WinogradKernel.cpp`) writes the kernel source for each
   distinct (shape, bias, activation): an `args.hpp` with the extents, the
   kernel template with one configuration, and the epilogue. hiprtc compiles it
   against the embedded headers, and the code object, symbol and launch
   geometry are stored on the op.
3. `hip.winograd_conv` lowers to `wrap_winograd_conv`. On its first call the
   runtime rewrites the constant filter into the kernel's store (C innermost:
   `T = G g` as `[4, 3, K, C]`, or the raw `g` as `[3, 3, K, C]`) and loads the
   code object, both kept in the op state. Each call then launches the conv
   kernel and its fixup kernel with (input, store, bias, output).

## Edge fixup

The kernel loads each input row of a 2x2 output tile with one 4-wide vector
load that starts at column -1, and relies on the buffer range check to read
zeros for the padding. Two of those loads cross an end of the input buffer:
row 0 of image 0, channel 0, starts before it, and the last row of the last
image and channel ends after it. A buffer load that crosses the range returns
0 in every lane, so the in-range columns are lost too: the top-left 2x2 tile
of image 0 and the bottom-right tiles of the last image come out wrong.
Loading those rows per element inside the kernel spills registers (the kernel
already uses 256 VGPRs) and halves its speed. Instead, the generated code
object also holds `<name>_fixup`, which recomputes those at most 12 positions
per output channel as a direct convolution from the filter store, with the
same epilogue, and the runtime launches it right after the conv kernel. One
32-lane group computes one (position, output channel) pair and splits the
input channels across its lanes; a serial channel loop per thread would cost
about as much as the conv kernel at a few hundred channels. The vendored
headers stay byte-identical to upstream.

hiprtc is resolved at call time, like the HIP runtime in
`RocMlirKernelCompiler.cpp`, so it adds no load-time dependency. The headers
are embedded as byte arrays by `lib/Compiler/CMakeLists.txt`.

## Eligibility and gate

A conv qualifies when it is fp16, NCHW, rank 4 with static shapes, a 3x3 filter
that is a model constant, stride 1, dilation 1, padding 1 on every side and
group 1. It is left to rocMLIR when its input is an upsample, transpose,
reshape or slice, which the rocMLIR conv reads as a view.

The gate is MIGraphX's NCHW fp16 heuristic (`winograd_f23_profitable`), minus
the band where hip-ep's tuned rocMLIR conv is faster: both channel counts at
least 511 with a spatial side of at most 32. That band came from timing the
Winograd kernel against hip-ep's tuned rocMLIR convs on the conv shapes of a
set of fp16 image models.

## Configuration

`chooseWinogradConfig` picks one of MIGraphX's fp16 configurations by shape: four
waves with fold-at-end and the `T` store by default, three output blocks with
the raw filter for a narrow output over a large image, and eight waves for
large channel counts. Each build takes about 1-2 s, so the configurations are
not tuned per model.

## Cache identity

`HIPDNN_EP_WINOGRAD` changes which convolutions run as rocMLIR kernels, so the
artifact target gets a `+winograd` suffix.

## Not covered

- fp32 Winograd (`winograd_conv_fp32.hpp`).
- A residual add or other pointwise ops after the activation: they stay
  separate ops after the `hip.winograd_conv`.
- The default (non-rocMLIR) pipeline.
