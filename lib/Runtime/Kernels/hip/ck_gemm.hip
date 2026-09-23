/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Composable Kernel GEMM instances, behind a narrow C ABI so callers compiled
// outside hipcc never see CK's device templates. fp16 operands go through the
// WMMA op; gfx11's WMMA has no fp32 type, so fp32 operands go through the
// FMA-based DL pipeline instead. bf16 and fp64 operands go to hip_ref_gemm_run.
//
// Arguments arrive in hipBLASLt's column-major convention. A column-major
// (rows x cols, ld=rows) buffer is bit-identical to a row-major
// (cols x rows, ld=rows) one, so a column-major D = op(A) op(B) becomes CK's
// row-major E = A' B' with M'=n, N'=m, K'=k and the operands swapped:
//
//   A' = B buffer, always Row  (op(B) transposed is rejected below)
//   B' = A buffer, Row when transA=='N', Col when transA=='T'
//   E  = D buffer, Row, StrideE = ldd
//
// bias is length m, i.e. indexed by E's column, so it enters as a Ds tensor
// with StrideD = 0 (broadcast along M').
//
// Tile configurations are copied verbatim from CK's own tuned gfx11 WMMA lists
// in library/src/tensor_operation_instance/gpu/gemm_universal/
// device_gemm_wmma_universal_f16_f16_f16/{mk_kn,mk_nk}_mn.hpp, plus the
// single-relaxation variants noted below.
//
// There is deliberately no all-scalar catch-all instance: one measures 30-50x
// slower than the tuned instances, which is worse than the reference fallback a
// caller reaches by way of the non-zero return.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Empty_Tuple = ck::Tuple<>;
using Row_Tuple   = ck::Tuple<Row>;
using F16_Tuple   = ck::Tuple<F16>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;
using Scale       = ck::tensor_operation::element_wise::Scale;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmMNKPadding =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;
// The DL pipeline pads M and N only; GemmDefault would demand M % MPerBlock ==
// 0 && N % NPerBlock == 0, which real shapes almost never satisfy.
static constexpr auto GemmMNPadding =
    ck::tensor_operation::device::GemmSpecialization::MNPadding;
static constexpr auto Intrawave = ck::BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = ck::BlockGemmPipelineScheduler::Interwave;
static constexpr auto V1        = ck::BlockGemmPipelineVersion::v1;
static constexpr auto V3        = ck::BlockGemmPipelineVersion::v3;

// ---------------------------------------------------------------- tile sets

// A Row / B Row. EV = E scalar-per-vector (8 for f16 out, 4 for f32 out).
#define TKN_1(EV)                                                                           \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V1
#define TKN_2(EV)                                                                          \
    128, 128, 64, 64, 8, 8, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 32, 1, 4>,             \
        S<EV, 8, 8>, Intrawave, V1
#define TKN_3(EV)                                                                           \
    256, 128, 256, 64, 8, 8, 16, 16, 4, 4, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 8, 0, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V1
#define TKN_4(EV)                                                                           \
    256, 128, 128, 64, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, \
        S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 1, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V1
#define TKN_5(EV)                                                                         \
    128, 64, 64, 32, 8, 8, 16, 16, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 4, 1, 1, 1, S<1, 32, 1, 2>,            \
        S<EV, 8, 8>, Intrawave, V1
#define TKN_6(EV)                                                                        \
    64, 32, 64, 64, 8, 8, 16, 16, 2, 2, S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 16, 1, 4>,           \
        S<EV, 8, 8>, Interwave, V1
#define TKN_7(EV)                                                                           \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, \
        S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 1, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V3
// AK1=2 / SrcScalarPerVector=2: CK's own low-alignment entry, the only one that
// accepts a K not divisible by 8.
#define TKN_8(EV)                                                                          \
    128, 128, 64, 64, 2, 8, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, 1, \
        S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 4>,             \
        S<EV, 8, 8>, Intrawave, V1

// Not from CK's list: MPerBlock 64 with NPerBlock left at 128, so a shape whose
// whole N' fits inside one tile still reads A once but spreads over twice the
// workgroups. Serves attention's A @ V, where N' is the head dim.
#define TKN_M64(EV)                                                                         \
    256, 64, 128, 64, 8, 8, 16, 16, 2, 2, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,  \
        S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V1

// A Row / B Col.
#define TNK_1(EV)                                                                           \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, \
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V1
#define TNK_2(EV)                                                                          \
    128, 128, 64, 64, 8, 8, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 4>,             \
        S<EV, 8, 8>, Intrawave, V1
#define TNK_3(EV)                                                                           \
    256, 128, 256, 64, 8, 8, 16, 16, 4, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, \
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V1
#define TNK_4(EV)                                                                           \
    256, 128, 128, 64, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V1
#define TNK_5(EV)                                                                         \
    128, 64, 80, 64, 8, 8, 16, 16, 1, 5, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<8, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 64, 1, 2>,            \
        S<EV, 8, 8>, Intrawave, V1
#define TNK_6(EV)                                                                        \
    64, 32, 64, 64, 8, 8, 16, 16, 2, 2, S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 16, 1, 4>,           \
        S<EV, 8, 8>, Intrawave, V1
#define TNK_7(EV)                                                                           \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, \
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V3
#define TNK_8(EV)                                                                          \
    128, 128, 64, 64, 2, 8, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 2, 2, 0, \
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 4>,             \
        S<EV, 8, 8>, Intrawave, V1

// Not from CK's list. These relax exactly one alignment requirement each and
// keep the surrounding tile, pipeline and LDS vector widths intact, so that a
// shape blocked on one dimension does not pay for a fully scalar pipeline.
//
// A-side global load scalar, everything else 8-wide: for odd k, where A' is Row
// and vectorises along K'.
#define TKN_A1a(EV)                                                                         \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 0, \
        S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 32, 1, 8>, S<EV, 8, 8>, \
        Intrawave, V1
#define TKN_A1b(EV)                                                                        \
    128, 128, 64, 64, 8, 8, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 1, 8, 0, \
        S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 32, 1, 4>,             \
        S<EV, 8, 8>, Intrawave, V1

// K1 and both global loads at 4 instead of 8: for k divisible by 4 but not 8.
#define TNK_K4(EV)                                                                         \
    128, 128, 64, 64, 4, 4, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 4, 4, 0, \
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 4, 4, 0, 1, 1, S<1, 32, 1, 4>,             \
        S<EV, 8, 8>, Intrawave, V1

// E store narrowed, A and B left at 8: for an m (= CK's N') that is odd or only
// 2-aligned. EV is ignored; the width is the point of the variant.
#define TNK_E2a(EV)                                                                         \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, \
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 32, 1, 8>, S<2, 8, 8>,  \
        Intrawave, V1
#define TNK_E2b(EV)                                                                        \
    128, 128, 64, 64, 8, 8, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 4>,             \
        S<2, 8, 8>, Intrawave, V1
#define TNK_E1a(EV)                                                                         \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, \
        S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 32, 1, 8>, S<1, 8, 8>,  \
        Intrawave, V1
#define TNK_E1b(EV)                                                                        \
    128, 128, 64, 64, 8, 8, 16, 16, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 1, S<1, 32, 1, 4>,             \
        S<1, 8, 8>, Intrawave, V1

// The same E-store relaxation for the B-Row tiles. On those, B' already loads
// scalar (BBlockTransferSrcScalarPerVector is 1) and A' vectorises along K',
// so the CShuffle store is the only thing an m (= CK's N') indivisible by 8
// blocks. Without these, every B-Row shape whose m is not a multiple of 8 is
// refused by all instances and falls to the reference GEMM. EV is ignored; the
// width is the point of the variant.
#define TKN_E4a(EV)                                                                         \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 32, 1, 8>, S<4, 8, 8>,  \
        Intrawave, V1
#define TKN_E2a(EV)                                                                         \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 32, 1, 8>, S<2, 8, 8>,  \
        Intrawave, V1
#define TKN_E1a(EV)                                                                         \
    256, 128, 128, 32, 8, 8, 16, 16, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, \
        S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 0, 1, 1, S<1, 32, 1, 8>, S<1, 8, 8>,  \
        Intrawave, V1

// ------------------------------------------------- fp32 tile sets (DL / FMA)
//
// Copied from CK's tuned f32 lists in library/src/tensor_operation_instance/
// gpu/gemm/device_gemm_dl_f32_f32_f32_{mk_kn,mk_nk}_mn_instance.cpp, one entry
// per distinct (BlockSize, MPerBlock, NPerBlock, K0PerBlock) geometry. The
// dropped entries differ only in thread-cluster arrangement; on the 12 whisper
// fp32 shapes keeping them buys at most 1.3%, inside the run-to-run spread.
//
// The DLKN / DLNK tiles have K1 = 2, and the DL pipeline does not pad K, so they
// serve only K % (2 * K0PerBlock) == 0 -- multiples of 8. The DLK1 tiles trade
// that vector width for K1 = 1 and serve any K divisible by their K0PerBlock
// (20 or 4).
//
// The trailing CThreadTransferDstScalarPerVector is supplied by the entry macro
// instead of the tile, because runDl needs it to reject an N the store cannot
// land on exactly.

// A Row / B Row.
#define DLKN_1                                                                              \
    256, 128, 128, 16, 2, 4, 4, 1, S<8, 2>, S<8, 2>, S<8, 1, 1, 2>, S<2, 1, 128, 1>,        \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<2, 1, 4, 2>, S<8, 1, 32, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_2                                                                              \
    128, 128, 64, 16, 2, 4, 4, 1, S<8, 2>, S<4, 2>, S<8, 1, 2, 2>, S<2, 1, 64, 1>,          \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<2, 1, 8, 2>, S<8, 1, 8, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,          \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_3                                                                              \
    128, 64, 128, 16, 2, 4, 4, 1, S<4, 2>, S<8, 2>, S<8, 1, 2, 2>, S<2, 1, 32, 1>,          \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<2, 1, 8, 2>, S<8, 1, 16, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_4                                                                              \
    64, 64, 64, 8, 2, 4, 4, 1, S<4, 2>, S<4, 2>, S<4, 1, 2, 2>, S<2, 1, 32, 1>,             \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<2, 1, 4, 2>, S<4, 1, 16, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_5                                                                              \
    64, 16, 64, 16, 2, 1, 4, 1, S<4, 2>, S<4, 2>, S<4, 1, 1, 2>, S<4, 1, 16, 1>,            \
        S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<4, 1, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 1, 2>,          \
        S<4, 1, 4, 2>, S<4, 1, 16, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_6                                                                              \
    64, 64, 16, 16, 2, 4, 1, 1, S<4, 2>, S<4, 2>, S<4, 1, 4, 2>, S<4, 1, 16, 1>,            \
        S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<4, 1, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 1, 2>,          \
        S<1, 1, 4, 2>, S<16, 1, 4, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_7                                                                              \
    16, 16, 16, 16, 2, 2, 2, 1, S<2, 2>, S<2, 2>, S<4, 1, 4, 2>, S<4, 1, 4, 1>,             \
        S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<4, 1, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 1, 2>,          \
        S<4, 1, 4, 2>, S<4, 1, 4, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,          \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_8                                                                              \
    64, 8, 64, 32, 2, 1, 2, 1, S<4, 1>, S<8, 2>, S<4, 1, 1, 2>, S<8, 1, 8, 1>,              \
        S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<4, 1, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 1, 2>,          \
        S<8, 1, 4, 2>, S<4, 1, 16, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_9                                                                              \
    64, 64, 8, 32, 2, 2, 1, 1, S<8, 2>, S<4, 1>, S<8, 1, 4, 2>, S<4, 1, 16, 1>,             \
        S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<4, 1, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 1, 2>,          \
        S<1, 1, 4, 2>, S<32, 1, 2, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLKN_10                                                                             \
    8, 8, 8, 4, 2, 1, 2, 1, S<4, 1>, S<2, 1>, S<4, 1, 1, 2>, S<1, 1, 8, 1>, S<0, 3, 1, 2>,  \
        S<0, 3, 1, 2>, S<4, 1, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 1, 2>, S<1, 1, 4, 2>,          \
        S<4, 1, 2, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>, S<0, 3, 1, 2>,          \
        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5

// A Row / B Col, same ten geometries.
#define DLNK_1                                                                              \
    256, 128, 128, 16, 2, 4, 4, 1, S<8, 2>, S<8, 2>, S<8, 1, 1, 2>, S<2, 1, 128, 1>,        \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<8, 1, 1, 2>, S<2, 1, 128, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,        \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_2                                                                              \
    128, 128, 64, 16, 2, 4, 4, 1, S<8, 2>, S<4, 2>, S<8, 1, 2, 2>, S<2, 1, 64, 1>,          \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<8, 1, 2, 2>, S<2, 1, 32, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,         \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_3                                                                              \
    128, 64, 128, 16, 2, 4, 4, 1, S<4, 2>, S<8, 2>, S<8, 1, 2, 2>, S<2, 1, 32, 1>,          \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<8, 1, 2, 2>, S<2, 1, 64, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,         \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_4                                                                              \
    64, 64, 64, 8, 2, 4, 4, 1, S<4, 2>, S<4, 2>, S<4, 1, 2, 2>, S<2, 1, 32, 1>,             \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<4, 1, 2, 2>, S<2, 1, 32, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,         \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_5                                                                              \
    64, 16, 64, 16, 2, 1, 4, 1, S<4, 2>, S<4, 2>, S<4, 1, 1, 2>, S<4, 1, 16, 1>,            \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<4, 1, 4, 2>, S<4, 1, 16, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,         \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_6                                                                              \
    64, 64, 16, 16, 2, 4, 1, 1, S<4, 2>, S<4, 2>, S<4, 1, 4, 2>, S<4, 1, 16, 1>,            \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<4, 1, 1, 2>, S<4, 1, 16, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,         \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_7                                                                              \
    16, 16, 16, 16, 2, 2, 2, 1, S<2, 2>, S<2, 2>, S<4, 1, 4, 2>, S<4, 1, 4, 1>,             \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<4, 1, 4, 2>, S<4, 1, 4, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,          \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_8                                                                              \
    64, 8, 64, 32, 2, 1, 2, 1, S<4, 1>, S<8, 2>, S<4, 1, 1, 2>, S<8, 1, 8, 1>,              \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<8, 1, 4, 2>, S<4, 1, 16, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,         \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_9                                                                              \
    64, 64, 8, 32, 2, 2, 1, 1, S<8, 2>, S<4, 1>, S<8, 1, 4, 2>, S<4, 1, 16, 1>,             \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,          \
        S<4, 1, 1, 2>, S<8, 1, 8, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,          \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5
#define DLNK_10                                                                             \
    8, 8, 8, 4, 2, 1, 2, 1, S<4, 1>, S<2, 1>, S<4, 1, 1, 2>, S<1, 1, 8, 1>, S<1, 2, 0, 3>,  \
        S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<4, 1, 1, 2>,          \
        S<1, 1, 8, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>,          \
        S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5

// A Row / B Row, K1 = 1.
#define DLK1_1                                                                              \
    64, 64, 64, 20, 1, 4, 4, 1, S<4, 2>, S<4, 2>, S<10, 1, 2, 1>, S<2, 1, 32, 1>,           \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<2, 1, 1, 1>, S<1, 2, 0, 3>, S<1, 1, 1, 1>,          \
        S<5, 1, 4, 1>, S<4, 1, 16, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 1>, S<0, 1, 2, 3, 4, 5>, 5
#define DLK1_2                                                                              \
    64, 16, 64, 20, 1, 1, 4, 1, S<4, 2>, S<4, 2>, S<5, 1, 1, 1>, S<4, 1, 16, 1>,            \
        S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 1, 1>, S<0, 3, 1, 2>, S<1, 1, 1, 1>,          \
        S<5, 1, 4, 1>, S<4, 1, 16, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 1>, S<0, 1, 2, 3, 4, 5>, 5
#define DLK1_3                                                                              \
    64, 16, 64, 4, 1, 1, 4, 1, S<4, 2>, S<4, 2>, S<1, 1, 1, 1>, S<4, 1, 16, 1>,             \
        S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 1, 1>, S<0, 3, 1, 2>, S<1, 1, 1, 1>,          \
        S<1, 1, 4, 1>, S<4, 1, 16, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 1>, S<0, 1, 2, 3, 4, 5>, 5

// ComputeTypeA/B are the trailing pair. They default to ADataType/BDataType,
// but naming them keeps the f32-output combos from instantiating GetWmma<float>.
#define MAKE_OP(ABDATA, ALAY, BLAY, DSLAY, DSDATA, EDATA, CDEOP, EV, TILE)                \
    ck::tensor_operation::device::DeviceBatchedGemmMultiD_Wmma_CShuffleV3<ALAY,           \
                                                                         BLAY,           \
                                                                         DSLAY,          \
                                                                         Row,            \
                                                                         ABDATA,         \
                                                                         ABDATA,         \
                                                                         DSDATA,         \
                                                                         EDATA,          \
                                                                         F32,            \
                                                                         F32,            \
                                                                         PassThrough,    \
                                                                         PassThrough,    \
                                                                         CDEOP,          \
                                                                         GemmMNKPadding, \
                                                                         TILE(EV),       \
                                                                         ABDATA,         \
                                                                         ABDATA>

// Variadic because the DL tiles above are comma lists, which a single-parameter
// macro would see as too many arguments.
#define MAKE_DL_OP(BLAY, ...)                                                             \
    ck::tensor_operation::device::DeviceBatchedGemmMultipleD_Dl<Row,                      \
                                                                BLAY,                     \
                                                                Empty_Tuple,              \
                                                                Row,                      \
                                                                F32,                      \
                                                                F32,                      \
                                                                F32,                      \
                                                                Empty_Tuple,              \
                                                                F32,                      \
                                                                PassThrough,              \
                                                                PassThrough,              \
                                                                Scale,                    \
                                                                GemmMNPadding,            \
                                                                __VA_ARGS__>

#define DL_BLOCK_SIZE_(BLOCK, ...) BLOCK
#define DL_BLOCK_SIZE(...) DL_BLOCK_SIZE_(__VA_ARGS__)

// ------------------------------------------------------------------ runners

struct RunArgs {
  hipStream_t stream;
  const void *a;
  const void *b;
  const void *bias;
  void *e;
  int64_t m, n, k, batch;
  int64_t strideA, strideB, strideE;
  int64_t batchStrideA, batchStrideB, batchStrideE;
  float alpha;
};

int64_t elemBytes(int dtype) { return dtype == HIP_DTYPE_FLOAT16 ? 2 : 4; }

// Element span of one matrix: the last row starts at (rows - 1) * ld.
int64_t matrixSpan(int64_t rows, int64_t cols, int64_t ld) {
  return (rows - 1) * ld + cols;
}

// CK checks the contiguous extent against the vector width but not the leading
// dimension, so a non-packed ld must keep every row start at the widest vector
// any tile loads or stores (8 elements).
bool ldAligned(int64_t ld, int64_t packed) { return ld == packed || ld % 8 == 0; }

template <typename CDEOp>
CDEOp makeCde(float) {
  return CDEOp{};
}
template <>
Scale makeCde<Scale>(float alpha) {
  return Scale{alpha};
}

template <typename Op, typename CDEOp>
int runNoBias(const RunArgs &r) {
  auto arg = Op::MakeArgument(
      r.a, r.b, {}, r.e, (ck::index_t)r.m, (ck::index_t)r.n, (ck::index_t)r.k,
      (ck::index_t)r.batch, (ck::index_t)r.strideA, (ck::index_t)r.strideB, {},
      (ck::index_t)r.strideE, (ck::index_t)r.batchStrideA,
      (ck::index_t)r.batchStrideB, {}, (ck::index_t)r.batchStrideE,
      PassThrough{}, PassThrough{}, makeCde<CDEOp>(r.alpha), 1);
  if (!Op::IsSupportedArgument(arg)) {
    return 1;
  }
  auto inv = Op::MakeInvoker();
  inv.Run(arg, StreamConfig{r.stream, false});
  return 0;
}

template <typename Op, typename CDEOp>
int runBias(const RunArgs &r) {
  // StrideDs = 0 broadcasts the length-N' bias vector along M'.
  auto arg = Op::MakeArgument(
      r.a, r.b, {r.bias}, r.e, (ck::index_t)r.m, (ck::index_t)r.n,
      (ck::index_t)r.k, (ck::index_t)r.batch, (ck::index_t)r.strideA,
      (ck::index_t)r.strideB, {0}, (ck::index_t)r.strideE,
      (ck::index_t)r.batchStrideA, (ck::index_t)r.batchStrideB, {0},
      (ck::index_t)r.batchStrideE, PassThrough{}, PassThrough{},
      makeCde<CDEOp>(r.alpha), 1);
  if (!Op::IsSupportedArgument(arg)) {
    return 1;
  }
  auto inv = Op::MakeInvoker();
  inv.Run(arg, StreamConfig{r.stream, false});
  return 0;
}

// DeviceBatchedGemmMultipleD_Dl::Invoker::Run minus its unconditional
// std::cout of the grid descriptors. The op does not expose BlockSize, so the
// entry macro supplies it.
template <typename Op, int BlockSize>
void launchDl(const typename Op::Argument &arg, hipStream_t stream) {
  using GG = typename Op::GridwiseGemm;
  using ck::integral_constant;
  constexpr auto I0 = ck::Number<0>{};
  constexpr auto I1 = ck::Number<1>{};
  const ck::index_t gridSize =
      GG::CalculateGridSize(arg.e_grid_desc_m_n_.GetLength(I0),
                            arg.e_grid_desc_m_n_.GetLength(I1)) *
      arg.Batch_;
  auto launch = [&](auto mainLoop, auto doubleTail) {
    const auto kernel = ck::tensor_operation::device::kernel_gemm_dl_multiple_d<
        GG, F32, typename GG::DsGridPointer, F32, PassThrough, PassThrough,
        Scale, typename Op::AGridDesc_K0_M0_M1_K1,
        typename Op::BGridDesc_K0_N0_N1_K1,
        typename Op::DsGridDesc_M0_M10_M11_N0_N10_N11,
        typename Op::EGridDesc_M0_M10_M11_N0_N10_N11,
        typename Op::ComputePtrOffsetOfStridedBatch,
        typename Op::DefaultBlock2CTileMap, decltype(mainLoop)::value,
        decltype(doubleTail)::value>;
    launch_and_time_kernel(StreamConfig{stream, false}, kernel, dim3(gridSize),
                           dim3(BlockSize), 0, arg.p_a_grid_, arg.p_b_grid_,
                           arg.p_ds_grid_, arg.p_e_grid_, arg.Batch_,
                           arg.a_element_op_, arg.b_element_op_,
                           arg.cde_element_op_, arg.a_grid_desc_k0_m0_m1_k1_,
                           arg.b_grid_desc_k0_n0_n1_k1_,
                           arg.ds_grid_desc_m0_m10_m11_n0_n10_n11_,
                           arg.e_grid_desc_m0_m10_m11_n0_n10_n11_,
                           arg.compute_ptr_offset_of_batch_,
                           arg.block_2_ctile_map_);
  };
  const auto K0 = arg.a_grid_desc_k0_m0_m1_k1_.GetLength(I0);
  const bool mainLoop = GG::CalculateHasMainKBlockLoop(K0);
  const bool doubleTail = GG::CalculateHasDoubleTailKBlockLoop(K0);
  if (mainLoop && doubleTail) {
    launch(integral_constant<bool, true>{}, integral_constant<bool, true>{});
  } else if (mainLoop) {
    launch(integral_constant<bool, true>{}, integral_constant<bool, false>{});
  } else if (doubleTail) {
    launch(integral_constant<bool, false>{}, integral_constant<bool, true>{});
  } else {
    launch(integral_constant<bool, false>{}, integral_constant<bool, false>{});
  }
}

// MakeArgument takes no KBatch and the DL multi-D epilogue cannot broadcast a
// stride-0 D along M', so there is no bias variant of this.
template <typename Op, int EVec, int BlockSize>
int runDl(const RunArgs &r) {
  // CK's own DL instances run GemmDefault, which forces N % NPerBlock == 0 and
  // so can never leave a partial vector at E's right edge. GemmMNPadding pads
  // N for the GEMM but the CShuffle store still writes EVec elements at a time
  // and IsSupportedArgument checks neither the extent nor the row stride, so a
  // partial vector there silently spills into the next row.
  if (r.n % EVec != 0 || r.strideE % EVec != 0) {
    return 1;
  }
  auto arg = Op::MakeArgument(
      r.a, r.b, {}, r.e, (ck::index_t)r.m, (ck::index_t)r.n, (ck::index_t)r.k,
      (ck::index_t)r.batch, (ck::index_t)r.strideA, (ck::index_t)r.strideB, {},
      (ck::index_t)r.strideE, (ck::index_t)r.batchStrideA,
      (ck::index_t)r.batchStrideB, {}, (ck::index_t)r.batchStrideE,
      PassThrough{}, PassThrough{}, Scale{r.alpha});
  if (!Op::IsSupportedArgument(arg)) {
    return 1;
  }
  launchDl<Op, BlockSize>(arg, r.stream);
  return 0;
}

// ----------------------------------------------------------------- registry

enum Combo {
  kNN_F16 = 0,      // transA=N, f16 out, no bias  -> A'/B' both Row
  kTN_F32 = 1,      // transA=T, f32 out           -> B' Col, Scale epilogue
  kTN_F16_BIAS = 2, // transA=T, f16 out, bias     -> B' Col, Add epilogue
  kNN_F16_BIAS = 3, // transA=N, f16 out, bias     -> A'/B' both Row, Add
  kTN_F16 = 4,      // transA=T, f16 out, no bias  -> B' Col
  kNN_F32 = 5,      // fp32 operands, transA=N     -> DL, A'/B' both Row
  kTN_F32AB = 6,    // fp32 operands, transA=T     -> DL, B' Col
};

using RunFn = int (*)(const RunArgs &);

struct Entry {
  int combo;
  // Persisted by the offline LUT: must stay unique and must not change when
  // entries are reordered.
  const char *name;
  RunFn fn;
};

#define OP_A(TILE) MAKE_OP(F16, Row, Row, Empty_Tuple, Empty_Tuple, F16, PassThrough, 8, TILE)
#define OP_B(TILE) MAKE_OP(F16, Row, Col, Empty_Tuple, Empty_Tuple, F32, Scale, 4, TILE)
#define OP_C(TILE) MAKE_OP(F16, Row, Col, Row_Tuple, F16_Tuple, F16, Add, 8, TILE)
#define OP_D(TILE) MAKE_OP(F16, Row, Row, Row_Tuple, F16_Tuple, F16, Add, 8, TILE)
#define OP_E(TILE) MAKE_OP(F16, Row, Col, Empty_Tuple, Empty_Tuple, F16, PassThrough, 8, TILE)

#define ENT_A(TILE) {kNN_F16, "NN_F16:" #TILE, &runNoBias<OP_A(TILE), PassThrough>},
#define ENT_B(TILE) {kTN_F32, "TN_F32:" #TILE, &runNoBias<OP_B(TILE), Scale>},
#define ENT_C(TILE) {kTN_F16_BIAS, "TN_F16_BIAS:" #TILE, &runBias<OP_C(TILE), Add>},
#define ENT_D(TILE) {kNN_F16_BIAS, "NN_F16_BIAS:" #TILE, &runBias<OP_D(TILE), Add>},
#define ENT_E(TILE) {kTN_F16, "TN_F16:" #TILE, &runNoBias<OP_E(TILE), PassThrough>},

#define ENT_F(TILE, EV)                                                                   \
    {kNN_F32, "NN_F32:" #TILE,                                                            \
     &runDl<MAKE_DL_OP(Row, TILE, EV), EV, DL_BLOCK_SIZE(TILE)>},
#define ENT_G(TILE, EV)                                                                   \
    {kTN_F32AB, "TN_F32AB:" #TILE,                                                        \
     &runDl<MAKE_DL_OP(Col, TILE, EV), EV, DL_BLOCK_SIZE(TILE)>},

const Entry kEntries[] = {
    // 0-13: NN, f16 out, no bias
    ENT_A(TKN_1) ENT_A(TKN_2) ENT_A(TKN_3) ENT_A(TKN_4) ENT_A(TKN_5)
    ENT_A(TKN_6) ENT_A(TKN_7) ENT_A(TKN_8) ENT_A(TKN_A1a) ENT_A(TKN_A1b)
    ENT_A(TKN_E4a) ENT_A(TKN_E2a) ENT_A(TKN_E1a) ENT_A(TKN_M64)

    // 14-26: TN, f32 out, Scale
    ENT_B(TNK_1) ENT_B(TNK_2) ENT_B(TNK_3) ENT_B(TNK_4) ENT_B(TNK_5)
    ENT_B(TNK_6) ENT_B(TNK_7) ENT_B(TNK_8) ENT_B(TNK_E2a) ENT_B(TNK_E2b)
    ENT_B(TNK_E1a) ENT_B(TNK_E1b) ENT_B(TNK_K4)

    // 27-37: TN, f16 out, bias
    ENT_C(TNK_1) ENT_C(TNK_2) ENT_C(TNK_3) ENT_C(TNK_4) ENT_C(TNK_5)
    ENT_C(TNK_6) ENT_C(TNK_7) ENT_C(TNK_8) ENT_C(TNK_K4)
    ENT_C(TNK_E2a) ENT_C(TNK_E1a)

    // 38-48: NN, f16 out, bias
    ENT_D(TKN_1) ENT_D(TKN_2) ENT_D(TKN_3) ENT_D(TKN_4) ENT_D(TKN_5)
    ENT_D(TKN_6) ENT_D(TKN_7) ENT_D(TKN_8)
    ENT_D(TKN_E4a) ENT_D(TKN_E2a) ENT_D(TKN_E1a)

    // 49-58: TN, f16 out, no bias
    ENT_E(TNK_1) ENT_E(TNK_2) ENT_E(TNK_3) ENT_E(TNK_4) ENT_E(TNK_6)
    ENT_E(TNK_7) ENT_E(TNK_8) ENT_E(TNK_K4) ENT_E(TNK_E2a) ENT_E(TNK_E1a)

    // 59-68: NN, fp32 operands, no bias
    ENT_F(DLKN_1, 4) ENT_F(DLKN_2, 4) ENT_F(DLKN_3, 4) ENT_F(DLKN_4, 4)
    ENT_F(DLKN_5, 4) ENT_F(DLKN_6, 1) ENT_F(DLKN_7, 2) ENT_F(DLKN_8, 2)
    ENT_F(DLKN_9, 1) ENT_F(DLKN_10, 2)

    // 69-78: TN, fp32 operands, no bias
    ENT_G(DLNK_1, 4) ENT_G(DLNK_2, 4) ENT_G(DLNK_3, 4) ENT_G(DLNK_4, 4)
    ENT_G(DLNK_5, 4) ENT_G(DLNK_6, 1) ENT_G(DLNK_7, 2) ENT_G(DLNK_8, 2)
    ENT_G(DLNK_9, 1) ENT_G(DLNK_10, 2)

    // 79-81: NN, fp32 operands, no bias, K1 = 1
    ENT_F(DLK1_1, 4) ENT_F(DLK1_2, 4) ENT_F(DLK1_3, 4)
};

constexpr int kNumEntries = (int)(sizeof(kEntries) / sizeof(kEntries[0]));

} // namespace

extern "C" int hip_ck_gemm_num_instances(void) { return kNumEntries; }

extern "C" const char *hip_ck_gemm_instance_name(int instance) {
  if (instance < 0 || instance >= kNumEntries) {
    return nullptr;
  }
  return kEntries[instance].name;
}

extern "C" int hip_ck_gemm_run(void *stream, int instance, const void *A,
                               const void *B, const void *bias, void *D,
                               int64_t m, int64_t n, int64_t k, int64_t batch,
                               int transA, int transB, int abDtype, int dDtype,
                               float alpha, int64_t lda, int64_t ldb,
                               int64_t ldd, int64_t strideA, int64_t strideB,
                               int64_t strideD) {
  if (instance < 0 || instance >= kNumEntries) {
    return 1;
  }
  if (transB != 0) {
    return 1;
  }

  int combo;
  if (abDtype == HIP_DTYPE_FLOAT16 && dDtype == HIP_DTYPE_FLOAT32) {
    if (!transA || bias) {
      return 1;
    }
    combo = kTN_F32;
  } else if (abDtype == HIP_DTYPE_FLOAT16 && dDtype == HIP_DTYPE_FLOAT16) {
    if (bias) {
      combo = transA ? kTN_F16_BIAS : kNN_F16_BIAS;
    } else {
      combo = transA ? kTN_F16 : kNN_F16;
    }
  } else if (abDtype == HIP_DTYPE_FLOAT32 && dDtype == HIP_DTYPE_FLOAT32) {
    if (bias) {
      return 1;
    }
    combo = transA ? kTN_F32AB : kNN_F32;
  } else {
    return 1;
  }
  if (kEntries[instance].combo != combo) {
    return 1;
  }
  const bool isDl = (combo == kNN_F32 || combo == kTN_F32AB);
  // Only the f32-output WMMA combo and the DL combos apply alpha (Scale); the
  // f16-output epilogues are PassThrough / Add.
  if (!isDl && combo != kTN_F32 && alpha != 1.0f) {
    return 1;
  }
  if (batch <= 0 || m <= 0 || n <= 0 || k <= 0) {
    return 1;
  }
  const int64_t idx[] = {m, n, k, batch, lda, ldb, ldd, strideA, strideB, strideD};
  for (int64_t v : idx) {
    if (v < 0 || v > INT32_MAX) {
      return 1;
    }
  }

  RunArgs r;
  r.stream = static_cast<hipStream_t>(stream);
  r.a = B; // A' = B buffer
  r.b = A; // B' = A buffer
  r.bias = bias;
  r.e = D;
  r.m = n; // M' = n
  r.n = m; // N' = m
  r.k = k;
  r.batch = batch;
  r.strideA = ldb;
  r.strideB = lda;
  r.strideE = ldd;
  r.batchStrideA = strideB;
  r.batchStrideB = strideA;
  r.batchStrideE = strideD;
  r.alpha = alpha;

  if (!isDl) {
    const bool bpCol = (transA != 0);
    if (!ldAligned(r.strideA, r.k) || !ldAligned(r.strideB, bpCol ? r.k : r.n) ||
        !ldAligned(r.strideE, r.n)) {
      return 1;
    }
  }

  const int64_t abBytes = elemBytes(abDtype);
  const int64_t dBytes = elemBytes(dDtype);
  // The DL gridwise op validates each tensor against 2 GB rather than the SRD's
  // 4 GiB, and it does so per batch, so a batch that individually fits can
  // still overrun the window in one launch. Split against the tighter bound.
  const int64_t kMaxSpanBytes = isDl ? 0x80000000LL : 0x100000000LL;
  const int64_t spanA = matrixSpan(r.m, r.k, r.strideA) * abBytes;
  const int64_t spanB = (transA ? matrixSpan(r.n, r.k, r.strideB)
                                : matrixSpan(r.k, r.n, r.strideB)) *
                        abBytes;
  const int64_t spanE = matrixSpan(r.m, r.n, r.strideE) * dBytes;
  const int64_t spans[3] = {spanA, spanB, spanE};
  const int64_t perBatch[3] = {r.batchStrideA * abBytes,
                               r.batchStrideB * abBytes,
                               r.batchStrideE * dBytes};
  int64_t maxBatch = r.batch;
  for (int i = 0; i < 3; ++i) {
    if (spans[i] > kMaxSpanBytes) {
      return 1;
    }
    if (perBatch[i] <= 0) {
      continue;
    }
    const int64_t lim = 1 + (kMaxSpanBytes - spans[i]) / perBatch[i];
    maxBatch = lim < maxBatch ? lim : maxBatch;
  }

  // CK checks hipGetLastError() after each launch, which reports whatever
  // error is pending on the thread rather than only this launch's. Consume a
  // pending one so an unrelated caller's handled failure cannot be read as
  // this shape being unsupported.
  (void)hipGetLastError();

  try {
    if (maxBatch < r.batch) {
      RunArgs seg = r;
      for (int64_t off = 0; off < r.batch; off += maxBatch) {
        const int64_t rest = r.batch - off;
        seg.batch = rest < maxBatch ? rest : maxBatch;
        seg.a = static_cast<const char *>(r.a) + off * r.batchStrideA * abBytes;
        seg.b = static_cast<const char *>(r.b) + off * r.batchStrideB * abBytes;
        seg.e = static_cast<char *>(r.e) + off * r.batchStrideE * dBytes;
        // The bias is broadcast along the batch, so it keeps its base pointer.
        const int rc = kEntries[instance].fn(seg);
        if (rc != 0) {
          return rc;
        }
      }
      return 0;
    }
    return kEntries[instance].fn(r);
  } catch (...) {
    return 1;
  }
}
