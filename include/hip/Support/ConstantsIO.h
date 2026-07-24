/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Shared emit logic for the constants.bin file.
//
// Both the ONNX->HIP conversion pass (standalone hip-compiler offline path)
// and the MorphiZen EP level-1 pass (EPContext export path) produce the same
// constants.bin layout. This header provides a single implementation they
// call, preserving the streaming-with-1MB-tile pattern for splat constants so
// no full-size expansion buffer is allocated during write.

#ifndef HIP_SUPPORT_CONSTANTSIO_H
#define HIP_SUPPORT_CONSTANTSIO_H

#include <cstdint>
#include <string>
#include <vector>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir {
namespace hip {

/// One constant's placement and source in the constants.bin layout.
///
/// Three source kinds are supported (mutually exclusive, decided by which
/// fields are non-default):
///   * mem-addr / inline:  `file_path` empty, `splat_elem_size == 0` →
///                         `data` points at a `size`-byte buffer (MLIR raw
///                         data, ORT mmap address, or caller-owned).
///   * splat:              `file_path` empty, `splat_elem_size > 0` →
///                         `data` points at a single element's bytes that
///                         the writer tile-expands to `size` bytes.
///   * file-ref:           `file_path` non-empty → writer fopens the file
///                         and freads `size` bytes from `file_offset`. The
///                         `data` pointer is ignored. This is the path that
///                         lets the EP avoid mmap'ing multi-GB external
///                         data into the system page cache.
///
/// `data` lifetime (when used) is the caller's responsibility and must
/// cover the call to `writeConstantsBinToFileSystem`.
struct ConstantEntry {
  std::string name;
  int64_t offset = 0;          // aligned offset in constants.bin
  int64_t size = 0;            // total byte size after splat expansion
  const void *data = nullptr;  // mmap addr / rawData.data() / owned buffer
  int64_t splat_elem_size = 0; // 0 = data is already `size` bytes; >0 = splat
  std::string file_path;       // non-empty: read from disk on demand
  int64_t file_offset = 0;     // byte offset within file_path
};

/// Write a constants.bin with alignment padding and tiled splat expansion
/// to a FileSystem entry. Streams via the writer's fwrite; for splat entries
/// a 1 MB scratch tile buffer is reused across the iterated writes, so peak
/// host memory is bounded regardless of individual constant size.
///
/// Returns true on success; false if the writer could not be opened or a
/// splat entry has invalid element size.
bool writeConstantsBinToFileSystem(morphizen::FileSystem *fs,
                                   const std::string &filename,
                                   const std::vector<ConstantEntry> &entries,
                                   int64_t totalBlobSize);

} // namespace hip
} // namespace mlir

#endif // HIP_SUPPORT_CONSTANTSIO_H
