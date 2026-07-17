/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Support/ConstantsIO.h"
#include "morphizen-foundation/file_io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

namespace mlir {
namespace hip {

// Fills the gap between `pos` and `targetOffset` with zeros using a small
// shared buffer; advances `pos`.
static bool writePaddingTo(morphizen::FileWriter *writer, int64_t &pos,
                           int64_t targetOffset) {
  static constexpr size_t kZeroChunk = 4096;
  static const char zeros[kZeroChunk] = {};
  if (targetOffset < pos)
    return false;
  int64_t padding = targetOffset - pos;
  while (padding > 0) {
    size_t chunk = static_cast<size_t>(
        std::min<int64_t>(padding, static_cast<int64_t>(kZeroChunk)));
    writer->fwrite(zeros, chunk);
    padding -= chunk;
  }
  pos = targetOffset;
  return true;
}

// Tile-fills a 1 MB scratch buffer with the single element value, then
// writes it repeatedly until `size` bytes have been emitted. Preserves the
// historical OnnxToHip splat-write behavior, bounding peak host memory
// independent of individual constant size.
static void writeSplat(morphizen::FileWriter *writer, const void *elemBytes,
                       int64_t elemSize, int64_t size) {
  static constexpr size_t kSplatChunk = 1024 * 1024;
  size_t elem = static_cast<size_t>(elemSize);
  size_t bufSize =
      (std::min(static_cast<size_t>(size), kSplatChunk) / elem) * elem;
  if (bufSize == 0)
    bufSize = elem;
  std::vector<char> buf(bufSize);
  for (size_t i = 0; i < bufSize; i += elem)
    std::memcpy(buf.data() + i, elemBytes, elem);

  size_t remaining = static_cast<size_t>(size);
  while (remaining > 0) {
    size_t toWrite = std::min(remaining, bufSize);
    writer->fwrite(buf.data(), toWrite);
    remaining -= toWrite;
  }
}

namespace {
// Keeps FILE* handles open across consecutive entries that read from the
// same external-data file. With ORT-style external data layouts most or
// all tensors of a model share one weights.data file, so caching avoids
// hundreds of fopen/fclose round trips.
struct FileHandleCache {
  ~FileHandleCache() {
    for (auto &kv : handles)
      if (kv.second)
        std::fclose(kv.second);
  }
  std::FILE *get(const std::string &path) {
    auto it = handles.find(path);
    if (it != handles.end())
      return it->second;
    std::FILE *f = std::fopen(path.c_str(), "rb");
    handles.emplace(path, f);
    return f;
  }
  std::unordered_map<std::string, std::FILE *> handles;
};
} // namespace

// Streams `size` bytes from `file_path` at `file_offset` through a fixed
// 1 MB buffer into the writer. Bounded peak host memory, identical pattern
// to writeSplat so the constants.bin emission stays predictable.
static bool writeFileRef(FileHandleCache &cache, morphizen::FileWriter *writer,
                         const std::string &file_path, int64_t file_offset,
                         int64_t size) {
  static constexpr size_t kReadChunk = 1024 * 1024;
  std::FILE *f = cache.get(file_path);
  if (!f)
    return false;
#ifdef _WIN32
  if (_fseeki64(f, file_offset, SEEK_SET) != 0)
    return false;
#else
  if (std::fseek(f, static_cast<long>(file_offset), SEEK_SET) != 0)
    return false;
#endif
  std::vector<char> buf(
      std::min<size_t>(static_cast<size_t>(size), kReadChunk));
  size_t remaining = static_cast<size_t>(size);
  while (remaining > 0) {
    size_t toRead = std::min(remaining, buf.size());
    size_t got = std::fread(buf.data(), 1, toRead, f);
    if (got != toRead)
      return false;
    writer->fwrite(buf.data(), got);
    remaining -= got;
  }
  return true;
}

bool writeConstantsBinToFileSystem(morphizen::FileSystem *fs,
                                   const std::string &filename,
                                   const std::vector<ConstantEntry> &entries,
                                   int64_t totalBlobSize) {
  if (!fs)
    return false;
  auto writer = fs->create_writer_template(filename.c_str());
  if (!writer)
    return false;

  FileHandleCache fileCache;

  int64_t pos = 0;
  for (const auto &e : entries) {
    if (!writePaddingTo(writer.get(), pos, e.offset))
      return false;
    if (!e.file_path.empty()) {
      if (!writeFileRef(fileCache, writer.get(), e.file_path, e.file_offset,
                        e.size))
        return false;
    } else if (e.splat_elem_size > 0) {
      writeSplat(writer.get(), e.data, e.splat_elem_size, e.size);
    } else {
      writer->fwrite(e.data, static_cast<size_t>(e.size));
    }
    pos += e.size;
  }
  // Trailing padding up to the declared total size, in case the final
  // constant ended before the aligned blob boundary. Typically a no-op.
  if (totalBlobSize > pos)
    writePaddingTo(writer.get(), pos, totalBlobSize);
  return true;
}

} // namespace hip
} // namespace mlir
