/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Composable Kernel GEMM instances, behind a narrow C ABI so callers compiled
// outside hipcc never see CK's device templates. fp16 and bf16 go through the
// WMMA op; fp32 has no WMMA path and goes through the DL op instead.
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
// single-relaxation variants noted below. bf16 reuses them unchanged: CK's
// bf16 lists carry the same numbers. The fp32 DL tiles come from
// gpu/gemm/device_gemm_dl_f32_f32_f32_{mk_kn,mk_nk}_mn_instance.cpp.
//
// There is deliberately no all-scalar catch-all instance: one measures 30-50x
// slower than the tuned instances, which is worse than the hipBLASLt fallback a
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

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Empty_Tuple = ck::Tuple<>;
using Row_Tuple   = ck::Tuple<Row>;
using F16_Tuple   = ck::Tuple<F16>;
using BF16_Tuple  = ck::Tuple<BF16>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;
using Scale       = ck::tensor_operation::element_wise::Scale;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmMNKPadding =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;
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

// ------------------------------------------------------------- fp32 DL tiles

// A Row / B Row.
#define DKN_1()                                                                            \
    256, 128, 128, 16, 2, 4, 4, 1, S<8, 2>, S<8, 2>, S<8, 1, 1, 2>, S<2, 1, 128, 1>,       \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,         \
        S<2, 1, 4, 2>, S<8, 1, 32, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,        \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5, 1
#define DKN_2()                                                                            \
    256, 128, 128, 16, 2, 4, 4, 1, S<4, 4>, S<4, 4>, S<8, 1, 1, 2>, S<2, 1, 128, 1>,       \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,         \
        S<2, 1, 4, 2>, S<8, 1, 32, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,        \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5, 1
#define DKN_3()                                                                            \
    256, 128, 128, 16, 2, 4, 4, 1, S<2, 8>, S<2, 8>, S<8, 1, 1, 2>, S<2, 1, 128, 1>,       \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,         \
        S<2, 1, 4, 2>, S<8, 1, 32, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,        \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5, 1
#define DKN_4()                                                                            \
    128, 128, 64, 16, 2, 4, 4, 1, S<8, 2>, S<4, 2>, S<8, 1, 2, 2>, S<2, 1, 64, 1>,         \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,         \
        S<2, 1, 8, 2>, S<8, 1, 8, 1>, S<0, 3, 1, 2>, S<0, 3, 1, 2>, S<1, 1, 4, 1>,         \
        S<0, 3, 1, 2>, S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>, 5, 1

// A Row / B Col.
#define DNK_1()                                                                            \
    256, 128, 128, 16, 2, 4, 4, 1, S<8, 2>, S<8, 2>, S<8, 1, 1, 2>, S<2, 1, 128, 1>,       \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,         \
        S<8, 1, 1, 2>, S<2, 1, 128, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,       \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5, 1
#define DNK_2()                                                                            \
    256, 128, 128, 16, 2, 4, 4, 1, S<4, 4>, S<4, 4>, S<8, 1, 1, 2>, S<2, 1, 128, 1>,       \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,         \
        S<8, 1, 1, 2>, S<2, 1, 128, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,       \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5, 1
#define DNK_3()                                                                            \
    256, 128, 128, 16, 2, 4, 4, 1, S<2, 8>, S<2, 8>, S<8, 1, 1, 2>, S<2, 1, 128, 1>,       \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,         \
        S<8, 1, 1, 2>, S<2, 1, 128, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,       \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5, 1
#define DNK_4()                                                                            \
    128, 128, 64, 16, 2, 4, 4, 1, S<8, 2>, S<4, 2>, S<8, 1, 2, 2>, S<2, 1, 64, 1>,         \
        S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>, S<1, 2, 0, 3>, S<1, 1, 1, 2>,         \
        S<8, 1, 2, 2>, S<2, 1, 32, 1>, S<1, 2, 0, 3>, S<1, 2, 0, 3>, S<4, 1, 1, 2>,        \
        S<1, 2, 0, 3>, S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>, 5, 1

// MNPadding lets arbitrary M/N run (real fp32 shapes are rarely 128-multiples);
// its padded C write forces CThreadTransferDstScalarPerVector to 1, already
// baked into the DKN_/DNK_ tile tails. K is still not padded, so K must stay a
// multiple of K0PerBlock*K1 or IsSupportedArgument rejects and the caller falls
// back. Tiles are CK's own MNPadding (irregular) list, verbatim.
#define MAKE_DL_OP(ALAY, BLAY, DSLAY, DSDATA, CDEOP, TILE)                             \
    ck::tensor_operation::device::DeviceBatchedGemmMultipleD_Dl<ALAY,                  \
                                                               BLAY,                  \
                                                               DSLAY,                 \
                                                               Row,                   \
                                                               F32,                   \
                                                               F32,                   \
                                                               F32,                   \
                                                               DSDATA,                \
                                                               F32,                   \
                                                               PassThrough,           \
                                                               PassThrough,           \
                                                               CDEOP,                 \
                                                               GemmMNPadding,         \
                                                               TILE()>

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

// The DL op's MakeArgument is the WMMA one minus the trailing KBatch.
template <typename Op, typename CDEOp>
int runNoBiasDl(const RunArgs &r) {
  auto arg = Op::MakeArgument(
      r.a, r.b, {}, r.e, (ck::index_t)r.m, (ck::index_t)r.n, (ck::index_t)r.k,
      (ck::index_t)r.batch, (ck::index_t)r.strideA, (ck::index_t)r.strideB, {},
      (ck::index_t)r.strideE, (ck::index_t)r.batchStrideA,
      (ck::index_t)r.batchStrideB, {}, (ck::index_t)r.batchStrideE,
      PassThrough{}, PassThrough{}, makeCde<CDEOp>(r.alpha));
  if (!Op::IsSupportedArgument(arg)) {
    return 1;
  }
  auto inv = Op::MakeInvoker();
  inv.Run(arg, StreamConfig{r.stream, false});
  return 0;
}

// ----------------------------------------------------------------- registry

enum Combo {
  kNN_F16 = 0,      // transA=N, f16 out, no bias  -> A'/B' both Row
  kTN_F32 = 1,      // transA=T, f32 out, alpha!=1 -> B' Col, Scale epilogue
  kTN_F16_BIAS = 2, // transA=T, f16 out, bias     -> B' Col, Add epilogue
  kNN_F16_BIAS = 3, // transA=N, f16 out, bias     -> A'/B' both Row, Add

  kNN_BF16 = 4,      // bf16 in, mirrors kNN_F16
  kTN_F32_BF16 = 5,  // bf16 in, f32 out, mirrors kTN_F32
  kTN_BF16_BIAS = 6, // bf16 in, mirrors kTN_F16_BIAS
  kNN_BF16_BIAS = 7, // bf16 in, mirrors kNN_F16_BIAS

  kNN_F32 = 8,     // f32 in and out, no bias, Scale epilogue
  kTN_F32_F32 = 9, // f32 in and out, no bias, B' Col, Scale
};

using RunFn = int (*)(const RunArgs &);

struct Entry {
  int combo;
  RunFn fn;
};

#define OP_A(TILE) MAKE_OP(F16, Row, Row, Empty_Tuple, Empty_Tuple, F16, PassThrough, 8, TILE)
#define OP_B(TILE) MAKE_OP(F16, Row, Col, Empty_Tuple, Empty_Tuple, F32, Scale, 4, TILE)
#define OP_C(TILE) MAKE_OP(F16, Row, Col, Row_Tuple, F16_Tuple, F16, Add, 8, TILE)
#define OP_D(TILE) MAKE_OP(F16, Row, Row, Row_Tuple, F16_Tuple, F16, Add, 8, TILE)

#define OP_E(TILE) MAKE_OP(BF16, Row, Row, Empty_Tuple, Empty_Tuple, BF16, PassThrough, 8, TILE)
#define OP_F(TILE) MAKE_OP(BF16, Row, Col, Empty_Tuple, Empty_Tuple, F32, Scale, 4, TILE)
#define OP_G(TILE) MAKE_OP(BF16, Row, Col, Row_Tuple, BF16_Tuple, BF16, Add, 8, TILE)
#define OP_H(TILE) MAKE_OP(BF16, Row, Row, Row_Tuple, BF16_Tuple, BF16, Add, 8, TILE)

#define OP_I(TILE) MAKE_DL_OP(Row, Row, Empty_Tuple, Empty_Tuple, Scale, TILE)
#define OP_J(TILE) MAKE_DL_OP(Row, Col, Empty_Tuple, Empty_Tuple, Scale, TILE)

#define ENT_A(TILE) {kNN_F16, &runNoBias<OP_A(TILE), PassThrough>},
#define ENT_B(TILE) {kTN_F32, &runNoBias<OP_B(TILE), Scale>},
#define ENT_C(TILE) {kTN_F16_BIAS, &runBias<OP_C(TILE), Add>},
#define ENT_D(TILE) {kNN_F16_BIAS, &runBias<OP_D(TILE), Add>},

#define ENT_E(TILE) {kNN_BF16, &runNoBias<OP_E(TILE), PassThrough>},
#define ENT_F(TILE) {kTN_F32_BF16, &runNoBias<OP_F(TILE), Scale>},
#define ENT_G(TILE) {kTN_BF16_BIAS, &runBias<OP_G(TILE), Add>},
#define ENT_H(TILE) {kNN_BF16_BIAS, &runBias<OP_H(TILE), Add>},

#define ENT_I(TILE) {kNN_F32, &runNoBiasDl<OP_I(TILE), Scale>},
#define ENT_J(TILE) {kTN_F32_F32, &runNoBiasDl<OP_J(TILE), Scale>},

const Entry kEntries[] = {
    // 0-9: NN, f16 out, no bias
    ENT_A(TKN_1) ENT_A(TKN_2) ENT_A(TKN_3) ENT_A(TKN_4) ENT_A(TKN_5)
    ENT_A(TKN_6) ENT_A(TKN_7) ENT_A(TKN_8) ENT_A(TKN_A1a) ENT_A(TKN_A1b)

    // 10-22: TN, f32 out, Scale
    ENT_B(TNK_1) ENT_B(TNK_2) ENT_B(TNK_3) ENT_B(TNK_4) ENT_B(TNK_5)
    ENT_B(TNK_6) ENT_B(TNK_7) ENT_B(TNK_8) ENT_B(TNK_E2a) ENT_B(TNK_E2b)
    ENT_B(TNK_E1a) ENT_B(TNK_E1b) ENT_B(TNK_K4)

    // 23-31: TN, f16 out, bias
    ENT_C(TNK_1) ENT_C(TNK_2) ENT_C(TNK_3) ENT_C(TNK_4) ENT_C(TNK_5)
    ENT_C(TNK_6) ENT_C(TNK_7) ENT_C(TNK_8) ENT_C(TNK_K4)

    // 32-39: NN, f16 out, bias
    ENT_D(TKN_1) ENT_D(TKN_2) ENT_D(TKN_3) ENT_D(TKN_4) ENT_D(TKN_5)
    ENT_D(TKN_6) ENT_D(TKN_7) ENT_D(TKN_8)

    // 40-43: NN, bf16 out, no bias
    ENT_E(TKN_1) ENT_E(TKN_2) ENT_E(TKN_3) ENT_E(TKN_4)

    // 44-47: TN, bf16 in, f32 out, Scale
    ENT_F(TNK_1) ENT_F(TNK_2) ENT_F(TNK_3) ENT_F(TNK_4)

    // 48-51: TN, bf16 out, bias
    ENT_G(TNK_1) ENT_G(TNK_2) ENT_G(TNK_3) ENT_G(TNK_4)

    // 52-55: NN, bf16 out, bias
    ENT_H(TKN_1) ENT_H(TKN_2) ENT_H(TKN_3) ENT_H(TKN_4)

    // 56-59: NN, f32 in and out, no bias
    ENT_I(DKN_1) ENT_I(DKN_2) ENT_I(DKN_3) ENT_I(DKN_4)

    // 60-63: TN, f32 in and out, no bias
    ENT_J(DNK_1) ENT_J(DNK_2) ENT_J(DNK_3) ENT_J(DNK_4)
};

constexpr int kNumEntries = (int)(sizeof(kEntries) / sizeof(kEntries[0]));

} // namespace

extern "C" int hip_ck_gemm_num_instances(void) { return kNumEntries; }

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
    } else if (transA) {
      return 1;
    } else {
      combo = kNN_F16;
    }
  } else if (abDtype == HIP_DTYPE_BFLOAT16 && dDtype == HIP_DTYPE_FLOAT32) {
    if (!transA || bias) {
      return 1;
    }
    combo = kTN_F32_BF16;
  } else if (abDtype == HIP_DTYPE_BFLOAT16 && dDtype == HIP_DTYPE_BFLOAT16) {
    if (bias) {
      combo = transA ? kTN_BF16_BIAS : kNN_BF16_BIAS;
    } else if (transA) {
      return 1;
    } else {
      combo = kNN_BF16;
    }
  } else if (abDtype == HIP_DTYPE_FLOAT32 && dDtype == HIP_DTYPE_FLOAT32) {
    // The DL op does not broadcast a stride-0 D along M', so an fp32 bias would
    // only land on the first row; those shapes fall back rather than get a
    // wrong result. fp32 without bias is served.
    if (bias) {
      return 1;
    }
    combo = transA ? kTN_F32_F32 : kNN_F32;
  } else {
    return 1;
  }
  if (kEntries[instance].combo != combo) {
    return 1;
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

  try {
    return kEntries[instance].fn(r);
  } catch (...) {
    return 1;
  }
}
