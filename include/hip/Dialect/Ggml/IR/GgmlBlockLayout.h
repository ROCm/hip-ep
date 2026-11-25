/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlBlockLayout.h - The bytes a block of each GGML type takes ------===//
//
// Needs nothing but LLVM's ADT: the split frontend reads it without the
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_GGML_IR_GGMLBLOCKLAYOUT_H
#define HIP_DIALECT_GGML_IR_GGMLBLOCKLAYOUT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdint>
#include <optional>

namespace mlir::ggml {

/// Bytes per stored block and elements per block.
///
/// Quantized types are block-based, and a block is always a whole number of
/// bytes even when an element is not: q4_K averages four and a half bits, but
/// two hundred and fifty-six of them occupy exactly one hundred and
/// forty-four bytes. That is what lets a stride stay in bytes.
struct BlockLayout {
  int64_t typeSize;
  int64_t blockElements;
};

/// The block geometry of a GGML encoding, named as `ggml_type_name` names it.
///
/// This is a second spelling of what ggml knows -- `ggml_type_size` and
/// `ggml_blck_size` -- and has to be, because nothing here links ggml. The
/// numbers are those of the block layouts GGUF files store, which do not
/// change; `test/unit/hipep_backend` checks the K-quants' against the
/// kernels' (hip/kquant_abi.h).
inline std::optional<BlockLayout> getBlockLayout(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<BlockLayout>>(name)
      // Dense.
      .Case("f32", BlockLayout{4, 1})
      .Case("f16", BlockLayout{2, 1})
      .Case("bf16", BlockLayout{2, 1})
      .Case("f64", BlockLayout{8, 1})
      .Case("i8", BlockLayout{1, 1})
      .Case("i16", BlockLayout{2, 1})
      .Case("i32", BlockLayout{4, 1})
      .Case("i64", BlockLayout{8, 1})
      // Round-number quantization, one scale per block.
      .Case("q4_0", BlockLayout{18, 32})
      .Case("q4_1", BlockLayout{20, 32})
      .Case("q5_0", BlockLayout{22, 32})
      .Case("q5_1", BlockLayout{24, 32})
      .Case("q8_0", BlockLayout{34, 32})
      .Case("q8_1", BlockLayout{36, 32})
      .Case("q1_0", BlockLayout{18, 128})
      .Case("q2_0", BlockLayout{18, 64})
      // K-quants: a super-block of 256 with per-sub-block scales.
      .Case("q2_K", BlockLayout{84, 256})
      .Case("q3_K", BlockLayout{110, 256})
      .Case("q4_K", BlockLayout{144, 256})
      .Case("q5_K", BlockLayout{176, 256})
      .Case("q6_K", BlockLayout{210, 256})
      .Case("q8_K", BlockLayout{292, 256})
      // I-quants: codebook lookups, same super-block.
      .Case("iq1_s", BlockLayout{50, 256})
      .Case("iq1_m", BlockLayout{56, 256})
      .Case("iq2_xxs", BlockLayout{66, 256})
      .Case("iq2_xs", BlockLayout{74, 256})
      .Case("iq2_s", BlockLayout{82, 256})
      .Case("iq3_xxs", BlockLayout{98, 256})
      .Case("iq3_s", BlockLayout{110, 256})
      .Case("iq4_nl", BlockLayout{18, 32})
      .Case("iq4_xs", BlockLayout{136, 256})
      // Ternary.
      .Case("tq1_0", BlockLayout{54, 256})
      .Case("tq2_0", BlockLayout{66, 256})
      // Microscaling floating point.
      .Case("mxfp4", BlockLayout{17, 32})
      .Case("nvfp4", BlockLayout{36, 64})
      // OCP 8-bit float, one byte an element.
      .Case("f8_e4m3", BlockLayout{1, 1})
      .Default(std::nullopt);
}

} // namespace mlir::ggml

#endif // HIP_DIALECT_GGML_IR_GGMLBLOCKLAYOUT_H
