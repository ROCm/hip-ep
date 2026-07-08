/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ConstantMetadata.h - Externalized-constant metadata contract ------===//
//
// Single source of truth for the per-constant metadata that
// hip-externalize-constants (producer) hands to generate-interface (consumer).
//
// The metadata rides the module as one attribute:
//
//   hipdnn.constants = [
//     { offset = 0,  size = 64, kind = 0 },                       // full
//     sidecar { offset = 64, size = 32, kind = 1, splat_value = ..,       //
//     splat
//       splat_elem_size = 4 },
//     { offset = 0,  size = 16, kind = 2, file_path = "..",       // file-ref
//       file_offset = 1048576 },
//     { offset = 0,  size = 96, kind = 3, sidecar_offset = 128 }  // mem-addr
//   ] : one DictionaryAttr per externalized constant, in constant-index order.
//
// This replaces the earlier set of index-synced parallel `DenseI64ArrayAttr`s
// (constant_sizes/offsets/source_kinds/splat_*/file_*/sidecar_offset): each
// constant's descriptor is now a single self-describing entry, so producer and
// consumer cannot drift out of index lockstep and new fields are additive keys
// rather than a new parallel array.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_TRANSFORMS_CONSTANTMETADATA_H
#define HIP_DIALECT_TRANSFORMS_CONSTANTMETADATA_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace hip {

/// How an externalized constant's bytes are obtained at model load. Shared by
/// the producer (hip-externalize-constants) and the consumer
/// (generate-interface). The integer values are ABI -- generate-interface maps
/// them into the flatbuffer `ConstantInfo.source` union -- so do NOT renumber.
enum class ConstantSourceKind : int32_t {
  None = 0,    ///< bytes live in the full sidecar at `offset`
  Splat = 1,   ///< tile the single element (`splat_value` / `splat_elem_size`)
  FileRef = 2, ///< fread from (`file_path`, `file_offset`) at upload time
  Sidecar = 3, ///< mem-addr bytes packed into the partial sidecar
               ///< (`sidecar_offset`)
};

/// Attribute + dictionary-key names for the `hipdnn.constants` metadata.
/// Sharing them keeps the encode (externalize) and decode (generate-interface)
/// sides in lockstep. `offset`/`size`/`kind` are always present; the remaining
/// keys appear only on entries of the matching kind.
namespace constant_meta {
inline constexpr llvm::StringLiteral kConstantsAttr = "hipdnn.constants";
inline constexpr llvm::StringLiteral kConstantsFileAttr = "hip.constants_file";
inline constexpr llvm::StringLiteral kOffset = "offset";
inline constexpr llvm::StringLiteral kSize = "size";
inline constexpr llvm::StringLiteral kKind = "kind";
inline constexpr llvm::StringLiteral kSplatValue = "splat_value";
inline constexpr llvm::StringLiteral kSplatElemSize = "splat_elem_size";
inline constexpr llvm::StringLiteral kFilePath = "file_path";
inline constexpr llvm::StringLiteral kFileOffset = "file_offset";
inline constexpr llvm::StringLiteral kSidecarOffset = "sidecar_offset";
} // namespace constant_meta

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_TRANSFORMS_CONSTANTMETADATA_H
