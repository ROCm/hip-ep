/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace mlir_compilation {

// Pure, dependency-free helper for the OGA `past_present_share_buffer` output
// shape override. Shared by BOTH output-staging paths so they can never
// diverge:
//   * classic marshal_output_tensors (3-arg ABI) -- applied before GetOutput
//   * the output-allocator callback (2-arg ABI)   -- applied before GetOutput
//
// Why this exists: OGA binds the SAME OrtValue to past_key_values.N.{key,value}
// (input) and present.N.{key,value} (output). The compiler/DimSource resolves
// the present tensor's sequence dim from the tight current length (e.g. 7),
// but ORT only returns the pre-allocated shared buffer (preserving the
// past==present pointer identity that in-place GQA append relies on) if we ask
// for the buffer's real, larger shape. So before GetOutput we bump each
// DYNAMIC present dim up to the matching past input's dim when the past is
// strictly larger.
//
// Contract:
//   compiled_dims[0..rank)  -- the compiled metadata shape; -1 marks a dim that
//                              was dynamic at compile time (the ONLY dims we
//                              may override; static dims are architecture
//                              constants like batch/num_heads/head_dim and must
//                              never change).
//   past_dims[0..past_rank) -- the matching past input's actual runtime shape.
//   out_dims[0..rank)       -- working shape, mutated in place.
//
// Rules (must match the prior inline logic in marshal_output_tensors):
//   * rank mismatch (past_rank != rank)            -> no-op, return false.
//   * compiled_dims[d] != -1 (static dim)          -> leave out_dims[d] alone.
//   * past_dims[d] > out_dims[d] on a dynamic dim  -> out_dims[d] =
//   past_dims[d].
// The strictly-greater test is the proxy for shared-buffer mode: separate-
// buffer (past_present_share_buffer=false) has past = prev_total < curr_total,
// so the override correctly does nothing there.
//
// Returns true iff any dim was changed (callers use it only for logging).
//
// Before: compiled=[1,8,-1,64] past=[1,8,128,64] out=[1,8,7,64]
// After:  out=[1,8,128,64]   (only dim 2 -- dynamic and past>out -- changed)
inline bool apply_present_share_buffer_override(const int64_t *compiled_dims,
                                                const int64_t *past_dims,
                                                int64_t *out_dims, size_t rank,
                                                size_t past_rank) {
  if (past_rank != rank)
    return false;
  bool changed = false;
  for (size_t d = 0; d < rank; ++d) {
    if (compiled_dims[d] != -1)
      continue; // static dim: never override
    if (past_dims[d] > out_dims[d]) {
      out_dims[d] = past_dims[d];
      changed = true;
    }
  }
  return changed;
}

} // namespace mlir_compilation
