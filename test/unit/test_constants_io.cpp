/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// writeConstantsBinToFileSystem with row-padded entries: each dense source row
// (from memory or from a file) is written followed by its zero pad, so the
// full constants file holds the padded layout while only one bounded buffer
// is ever held. Compares the written file byte for byte with the expected
// layout, including a file source larger than the 1 MB read buffer.

#include "hip/Support/ConstantsIO.h"
#include "hip/Support/DiskFileSystem.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char *what) {
  std::printf("[%s] %s\n", cond ? " OK " : "FAIL", what);
  if (!cond)
    ++g_failures;
}

std::vector<char> rowsOf(int64_t rows, int64_t rowBytes, int seed) {
  std::vector<char> v(static_cast<size_t>(rows * rowBytes));
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = static_cast<char>((i * 31 + seed) & 0xff);
  return v;
}

// The padded image of `dense`, as the runtime and the kernels expect it.
std::vector<char> padded(const std::vector<char> &dense, int64_t rowBytes,
                         int64_t rowPitch) {
  const int64_t rows = static_cast<int64_t>(dense.size()) / rowBytes;
  std::vector<char> v(static_cast<size_t>(rows * rowPitch), 0);
  for (int64_t r = 0; r < rows; ++r)
    std::copy_n(dense.data() + r * rowBytes, rowBytes, v.data() + r * rowPitch);
  return v;
}

std::vector<char> readFile(const std::filesystem::path &p) {
  std::ifstream ifs(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(ifs),
          std::istreambuf_iterator<char>()};
}

} // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "hip_constants_io_test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  // A: memory source, 3 rows of 48 B at pitch 64.
  const auto denseA = rowsOf(3, 48, 1);
  // B: dense neighbour, 100 bytes.
  const auto denseB = rowsOf(1, 100, 2);
  // C: file source, 600 rows of 2048 B at pitch 2304 (1.2 MB of source, more
  // than one 1 MB read chunk), stored after 512 unrelated leading bytes.
  const auto denseC = rowsOf(600, 2048, 3);
  const fs::path srcPath = dir / "weights.data";
  {
    std::ofstream ofs(srcPath, std::ios::binary);
    std::vector<char> lead(512, 'x');
    ofs.write(lead.data(), lead.size());
    ofs.write(denseC.data(), denseC.size());
  }

  std::vector<mlir::hip::ConstantEntry> entries(3);
  entries[0].offset = 0;
  entries[0].size = 3 * 64;
  entries[0].data = denseA.data();
  entries[0].row_bytes = 48;
  entries[0].row_pitch = 64;
  entries[1].offset = 256;
  entries[1].size = 100;
  entries[1].data = denseB.data();
  entries[2].offset = 384;
  entries[2].size = 600 * 2304;
  entries[2].file_path = srcPath.string();
  entries[2].file_offset = 512;
  entries[2].row_bytes = 2048;
  entries[2].row_pitch = 2304;
  const int64_t total = 384 + 600 * 2304;

  mlir::hip::DiskFileSystem diskFs(dir.string().c_str());
  check(mlir::hip::writeConstantsBinToFileSystem(&diskFs, "out.bin", entries,
                                                 total),
        "row-padded memory and file entries are written");
  std::vector<char> expected(static_cast<size_t>(total), 0);
  const auto pa = padded(denseA, 48, 64);
  std::copy(pa.begin(), pa.end(), expected.begin());
  std::copy(denseB.begin(), denseB.end(), expected.begin() + 256);
  const auto pc = padded(denseC, 2048, 2304);
  std::copy(pc.begin(), pc.end(), expected.begin() + 384);
  check(readFile(dir / "out.bin") == expected,
        "file matches the padded layout byte for byte");

  auto bad = entries;
  bad[0].row_pitch = 32; // smaller than row_bytes
  check(!mlir::hip::writeConstantsBinToFileSystem(&diskFs, "bad1.bin", bad,
                                                  total),
        "row_pitch < row_bytes is rejected");
  bad = entries;
  bad[0].size = 3 * 64 + 1; // not a whole number of rows
  check(!mlir::hip::writeConstantsBinToFileSystem(&diskFs, "bad2.bin", bad,
                                                  total),
        "size not a multiple of row_pitch is rejected");
  bad = entries;
  bad[2].file_offset = 512 + 4096; // source shorter than 600 rows
  check(!mlir::hip::writeConstantsBinToFileSystem(&diskFs, "bad3.bin", bad,
                                                  total),
        "short file source is rejected");

  fs::remove_all(dir);
  if (g_failures) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  return 0;
}
