/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Unit test for -hipsr-externalize-constants that exercises Phase 2 (the
// sidecar write), which the LIT tests cannot: LIT runs hip-mlir-opt with no
// injected FileSystem, so it only sees the Phase 1 IR mutation (offset/size).
// Here we inject an in-memory FileSystem and assert the actual bytes written --
// the only way to catch entry-field bugs (e.g. file_source vs inline routing)
// that are invisible in the IR because they produce identical offset/size.
//
// Plain main() (no GTest): matches the other MLIR-side unit tests and avoids a
// GTest dependency that is not present in the compiler build.

#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Support/DiskFileSystem.h"

#include "morphizen-foundation/file_io.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, llvm::StringRef what) {
  if (cond) {
    llvm::outs() << "[ OK ] " << what << "\n";
  } else {
    llvm::errs() << "[FAIL] " << what << "\n";
    ++g_failures;
  }
}

// Seeks to `offset` in `file` and reads `size` bytes back -- an fseek
// round-trip used to verify that each constant's data really lives at the
// offset stamped on its op. Returns empty on short read / seek failure.
std::vector<char> readAt(const std::filesystem::path &file, int64_t offset,
                         int64_t size) {
  std::ifstream ifs(file, std::ios::binary);
  ifs.seekg(offset);
  std::vector<char> buf(static_cast<size_t>(size));
  ifs.read(buf.data(), size);
  if (!ifs || ifs.gcount() != size) {
    return {};
  }
  return buf;
}

// Verifies an in-memory sidecar `blob` against the ops (in walk order) and
// their expected bytes, driven by the offset/size stamped on each op: the data
// at each stamped offset must be that constant's bytes, sizes must match,
// offsets must be 64-aligned / monotonic / non-overlapping, gap and trailing
// bytes zero, and the total length the aligned end of the last constant.
void verifyLayout(const std::vector<char> &blob,
                  const std::vector<mlir::hipsr::ConstantOp> &ops,
                  const std::vector<std::vector<char>> &expected,
                  const std::string &label) {
  check(ops.size() == expected.size(), label + ": constant count");
  if (ops.size() != expected.size()) {
    return;
  }

  int64_t prevEnd = 0, lastOffset = 0, lastSize = 0;
  bool layout = true, bytes = true, len = true, gaps = true;
  for (size_t i = 0; i < ops.size(); ++i) {
    mlir::hipsr::ConstantOp c = ops[i];
    if (!c.getOffsetAttr() || !c.getSizeAttr()) {
      layout = false;
      break;
    }
    int64_t off = c.getOffsetAttr().getInt();
    int64_t sz = c.getSizeAttr().getInt();

    if (sz != static_cast<int64_t>(expected[i].size())) {
      len = false;
    }
    if (off % 64 != 0 || off < prevEnd) {
      layout = false;
    }
    for (int64_t g = prevEnd; g < off && g < static_cast<int64_t>(blob.size());
         ++g) {
      if (blob[g] != 0) {
        gaps = false;
      }
    }
    if (off + sz <= static_cast<int64_t>(blob.size())) {
      std::vector<char> slice(blob.begin() + off, blob.begin() + off + sz);
      if (slice != expected[i]) {
        bytes = false;
      }
    } else {
      bytes = false;
    }
    prevEnd = off + sz;
    lastOffset = off;
    lastSize = sz;
  }
  for (int64_t g = prevEnd; g < static_cast<int64_t>(blob.size()); ++g) {
    if (blob[g] != 0) {
      gaps = false;
    }
  }

  check(len, label + ": each stamped size == data length");
  check(layout, label + ": offsets 64-aligned, monotonic, non-overlapping");
  check(bytes, label + ": data at each stamped offset matches");
  check(gaps, label + ": gap and trailing bytes are zero");
  check(static_cast<int64_t>(blob.size()) ==
            llvm::alignTo(lastOffset + lastSize, 64),
        label + ": blob length == aligned total");
}

// FileSystem that captures every write into an in-memory buffer per filename,
// so a test can inspect the exact sidecar bytes. create_reader is unused --
// writeConstantsBinToFileSystem reads file_source data via std::fopen on the
// file path directly, not through the FileSystem.
class CapturingFileSystem : public morphizen::FileSystem {
public:
  class Writer : public morphizen::FileWriter {
  public:
    explicit Writer(std::vector<char> *buf) : buf_(buf) {}
    std::size_t fwrite(const void *data, std::size_t size) const override {
      const char *p = static_cast<const char *>(data);
      buf_->insert(buf_->end(), p, p + size);
      return size;
    }

  private:
    std::vector<char> *buf_;
  };

  morphizen::FileReader *create_reader(const char *) override {
    return nullptr;
  }
  morphizen::FileWriter *create_writer(const char *path) override {
    return new Writer(&files[path]);
  }
  void destroy_reader(morphizen::FileReader *r) override { delete r; }
  void destroy_writer(morphizen::FileWriter *w) override { delete w; }

  std::map<std::string, std::vector<char>> files;
};

// FileSystem whose writer can never be created, to drive the failure path.
class FailingFileSystem : public morphizen::FileSystem {
public:
  morphizen::FileReader *create_reader(const char *) override {
    return nullptr;
  }
  morphizen::FileWriter *create_writer(const char *) override {
    return nullptr;
  }
  void destroy_reader(morphizen::FileReader *) override {}
  void destroy_writer(morphizen::FileWriter *) override {}
};

// Loads the parsed dialects, injects `fs`, runs the pass. Returns the module
// and sets `ok` to whether the pass succeeded.
struct Harness {
  mlir::MLIRContext ctx;

  Harness() {
    ctx.loadDialect<mlir::hipsr::HipsrDialect, mlir::func::FuncDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> run(const std::string &ir,
                                        morphizen::FileSystem *fs, bool &ok) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
    if (!module) {
      llvm::errs() << "failed to parse IR\n";
      ok = false;
      return module;
    }
    ctx.getLoadedDialect<mlir::hipsr::HipsrDialect>()->setFileSystem(fs);

    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::hipsr::createHipsrExternalizeConstantsPass());
    ok = mlir::succeeded(pm.run(*module));
    return module;
  }
};

// Inline value: the raw dense bytes land in the sidecar at offset 0, and the op
// is stamped with that offset/size.
void testInlineValueBytesWritten() {
  Harness h;
  CapturingFileSystem fs;
  bool ok = false;
  auto module = h.run(R"mlir(
    func.func @f() -> tensor<4xi8> {
      %0 = hipsr.constant {value = dense<[10, 20, 30, 40]> : tensor<4xi8>} : tensor<4xi8>
      return %0 : tensor<4xi8>
    }
  )mlir",
                      &fs, ok);
  check(ok, "inline: pass succeeds");
  if (!ok || !module) {
    return;
  }

  mlir::hipsr::ConstantOp c;
  module->walk([&](mlir::hipsr::ConstantOp op) { c = op; });
  check(c && c.getOffsetAttr() && c.getOffsetAttr().getInt() == 0,
        "inline: offset == 0");
  check(c && c.getSizeAttr() && c.getSizeAttr().getInt() == 4,
        "inline: size == 4");

  // The blob is padded up to the 64-byte aligned total, so 4 data bytes at the
  // start followed by zeros.
  const std::vector<char> &blob = fs.files["constants.bin"];
  check(blob.size() == 64, "inline: blob padded to 64");
  check(blob.size() == 64 && static_cast<uint8_t>(blob[0]) == 10 &&
            static_cast<uint8_t>(blob[3]) == 40 &&
            static_cast<uint8_t>(blob[4]) == 0,
        "inline: blob bytes match dense value, rest zero");
}

// mem_source: the writer must copy the bytes at the raw address (entry.data),
// not treat it as a file. Covers the getDataValues mem-pointer branch.
void testMemSourceBytesWritten() {
  std::vector<uint8_t> host = {50, 60, 70};
  auto addr = reinterpret_cast<uintptr_t>(host.data());

  Harness h;
  CapturingFileSystem fs;
  bool ok = false;
  auto module = h.run("func.func @f() -> tensor<3xi8> {\n"
                      "  %0 = hipsr.constant {source = #hipsr.mem_source<" +
                          std::to_string(addr) +
                          ", 3>} : tensor<3xi8>\n"
                          "  return %0 : tensor<3xi8>\n"
                          "}\n",
                      &fs, ok);
  check(ok, "mem_source: pass succeeds");
  if (!ok) {
    return;
  }

  const std::vector<char> &blob = fs.files["constants.bin"];
  check(blob.size() == 64, "mem_source: blob padded to 64");
  check(blob.size() == 64 && static_cast<uint8_t>(blob[0]) == 50 &&
            static_cast<uint8_t>(blob[2]) == 70,
        "mem_source: blob bytes copied from address");
}

// file_source: the sidecar bytes come from the referenced file at the given
// offset. Covers the file-ref branch.
void testFileSourceBytesStreamed() {
  auto path = std::filesystem::temp_directory_path() /
              "hipsr_externalize_test_weights.bin";
  {
    std::ofstream ofs(path, std::ios::binary);
    const char bytes[] = {1, 2, 3, 4, 5};
    ofs.write(bytes, sizeof(bytes));
  }

  Harness h;
  CapturingFileSystem fs;
  bool ok = false;
  auto module = h.run("func.func @f() -> tensor<5xi8> {\n"
                      "  %0 = hipsr.constant {source = #hipsr.file_source<\"" +
                          path.generic_string() +
                          "\", 0, 5>} : tensor<5xi8>\n"
                          "  return %0 : tensor<5xi8>\n"
                          "}\n",
                      &fs, ok);
  check(ok, "file_source: pass succeeds");

  const std::vector<char> &blob = fs.files["constants.bin"];
  check(blob.size() == 64, "file_source: blob padded to 64");
  check(blob.size() == 64 && static_cast<uint8_t>(blob[0]) == 1 &&
            static_cast<uint8_t>(blob[4]) == 5,
        "file_source: blob bytes streamed from file");

  std::filesystem::remove(path);
}

// Two constants: the second is padded to the next 64-byte boundary, and the
// blob is padded up to the aligned total. Checks alignment + gap/trailing
// zeros.
void testCumulativeAlignmentAndPadding() {
  Harness h;
  CapturingFileSystem fs;
  bool ok = false;
  // Non-splat values: getDataValues is a MorphiZen-only contract that does not
  // support splat-optimized DenseElementsAttr, so use distinct bytes.
  auto module = h.run(R"mlir(
    func.func @f() -> (tensor<4xi8>, tensor<8xi8>) {
      %0 = hipsr.constant {value = dense<[1, 2, 3, 4]> : tensor<4xi8>} : tensor<4xi8>
      %1 = hipsr.constant {value = dense<[5, 6, 7, 8, 9, 10, 11, 12]> : tensor<8xi8>} : tensor<8xi8>
      return %0, %1 : tensor<4xi8>, tensor<8xi8>
    }
  )mlir",
                      &fs, ok);
  check(ok, "cumulative: pass succeeds");
  if (!ok || !module) {
    return;
  }

  std::vector<mlir::hipsr::ConstantOp> ops;
  module->walk([&](mlir::hipsr::ConstantOp c) { ops.push_back(c); });
  std::vector<std::vector<char>> expected = {
      {1, 2, 3, 4},
      {5, 6, 7, 8, 9, 10, 11, 12},
  };
  // Offset-driven: verifies the data sits at the stamped offset (0 then aligned
  // 64), gap/trailing zeros, and total length alignTo(72, 64) = 128.
  verifyLayout(fs.files["constants.bin"], ops, expected, "cumulative");
}

// Multiple functions in one module share a single cumulative offset and a
// single sidecar write. This is the module-scoped contract: a per-function pass
// would restart offsets at 0 and overwrite the file per function.
void testMultiFunctionSharesOneSidecar() {
  Harness h;
  CapturingFileSystem fs;
  bool ok = false;
  auto module = h.run(R"mlir(
    func.func @a() -> tensor<4xi8> {
      %0 = hipsr.constant {value = dense<[1, 2, 3, 4]> : tensor<4xi8>} : tensor<4xi8>
      return %0 : tensor<4xi8>
    }
    func.func @b() -> tensor<8xi8> {
      %0 = hipsr.constant {value = dense<[5, 6, 7, 8, 9, 10, 11, 12]> : tensor<8xi8>} : tensor<8xi8>
      return %0 : tensor<8xi8>
    }
  )mlir",
                      &fs, ok);
  check(ok, "multi-func: pass succeeds");
  if (!ok || !module) {
    return;
  }

  // One shared file with both functions' constants, not one overwrite per
  // function.
  check(fs.files.size() == 1, "multi-func: exactly one sidecar written");

  std::vector<mlir::hipsr::ConstantOp> ops;
  module->walk([&](mlir::hipsr::ConstantOp c) { ops.push_back(c); });
  std::vector<std::vector<char>> expected = {
      {1, 2, 3, 4},
      {5, 6, 7, 8, 9, 10, 11, 12},
  };
  // Offset-driven: @a's constant at 0, @b's at aligned 64 in the shared blob.
  verifyLayout(fs.files["constants.bin"], ops, expected, "multi-func");
}

// The core round-trip: several constants of mixed kinds / dtypes / sizes chosen
// so offsets need real (non-coincidental) padding. The sidecar is written to a
// real file; each constant is read back by fseek-ing to the offset stamped on
// its op and comparing the bytes and length. This is what catches an
// offset/size stamped != data actually placed bug, which the per-byte hardcoded
// checks cannot.
void testOffsetDrivenReadBack() {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "hipsr_ext_readback";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  // Source file for the file_source constant: 200 bytes, src[i] == i, so the
  // window [50, 150) is bytes 50..149. A nonzero file_offset also checks that
  // the writer reads from the right place in the source.
  fs::path srcFile = dir / "weights.bin";
  {
    std::ofstream ofs(srcFile, std::ios::binary);
    for (int i = 0; i < 200; ++i) {
      char b = static_cast<char>(i);
      ofs.write(&b, 1);
    }
  }

  // Live host buffer for the mem_source constant.
  std::vector<uint8_t> memHost(7);
  for (int i = 0; i < 7; ++i) {
    memHost[i] = static_cast<uint8_t>(200 + i);
  }
  auto addr = reinterpret_cast<uintptr_t>(memHost.data());

  // Expected bytes, in declaration order (== module walk order).
  std::vector<std::vector<char>> expected;
  expected.push_back(
      {1, 2, 3, 4, 5, 6, 7, 8}); // c0 i32 little-endian raw bytes
  std::vector<char> e1;          // c1 file window [50, 150)
  for (int i = 50; i < 150; ++i) {
    e1.push_back(static_cast<char>(i));
  }
  expected.push_back(std::move(e1));
  expected.push_back({91, 92, 93}); // c2
  std::vector<char> e3;             // c3 mem buffer
  for (int i = 0; i < 7; ++i) {
    e3.push_back(static_cast<char>(memHost[i]));
  }
  expected.push_back(std::move(e3));

  // c0's i32 values whose little-endian bytes are {1..4} and {5..8}.
  std::string ir =
      "func.func @f() -> (tensor<2xi32>, tensor<100xi8>, tensor<3xi8>, "
      "tensor<7xi8>) {\n"
      "  %0 = hipsr.constant {value = dense<[67305985, 134678021]> : "
      "tensor<2xi32>} : tensor<2xi32>\n"
      "  %1 = hipsr.constant {source = #hipsr.file_source<\"" +
      srcFile.generic_string() +
      "\", 50, 100>} : tensor<100xi8>\n"
      "  %2 = hipsr.constant {value = dense<[91, 92, 93]> : tensor<3xi8>} : "
      "tensor<3xi8>\n"
      "  %3 = hipsr.constant {source = #hipsr.mem_source<" +
      std::to_string(addr) +
      ", 7>} : tensor<7xi8>\n"
      "  return %0, %1, %2, %3 : tensor<2xi32>, tensor<100xi8>, tensor<3xi8>, "
      "tensor<7xi8>\n"
      "}\n";

  Harness h;
  mlir::hip::DiskFileSystem diskFs(dir.string().c_str());
  bool ok = false;
  auto module = h.run(ir, &diskFs, ok);
  check(ok, "readback: pass succeeds");
  if (!ok || !module) {
    fs::remove_all(dir, ec);
    return;
  }

  std::vector<mlir::hipsr::ConstantOp> ops;
  module->walk([&](mlir::hipsr::ConstantOp c) { ops.push_back(c); });
  check(ops.size() == expected.size(), "readback: 4 constants stamped");
  if (ops.size() != expected.size()) {
    fs::remove_all(dir, ec);
    return;
  }

  fs::path sidecar = dir / "constants.bin";
  int64_t prevEnd = 0;
  int64_t lastOffset = 0, lastSize = 0;
  bool layoutOk = true;
  bool bytesOk = true;
  bool lenOk = true;
  for (size_t i = 0; i < ops.size(); ++i) {
    mlir::hipsr::ConstantOp c = ops[i];
    if (!c.getOffsetAttr() || !c.getSizeAttr()) {
      layoutOk = false;
      break;
    }
    int64_t off = c.getOffsetAttr().getInt();
    int64_t sz = c.getSizeAttr().getInt();

    // Stamped size must equal the true data length (not element count).
    if (sz != static_cast<int64_t>(expected[i].size())) {
      lenOk = false;
    }
    // 64-aligned, monotonic, non-overlapping.
    if (off % 64 != 0 || off < prevEnd) {
      layoutOk = false;
    }
    // The data at the stamped offset is this constant's bytes.
    if (readAt(sidecar, off, sz) != expected[i]) {
      bytesOk = false;
    }
    prevEnd = off + sz;
    lastOffset = off;
    lastSize = sz;
  }
  check(lenOk, "readback: each stamped size == true byte length");
  check(layoutOk, "readback: offsets 64-aligned, monotonic, non-overlapping");
  check(bytesOk, "readback: bytes at each stamped offset match source data");

  // Total file length is the aligned end of the last constant (derived, not
  // hardcoded).
  auto fileLen = static_cast<int64_t>(fs::file_size(sidecar, ec));
  check(!ec && fileLen == llvm::alignTo(lastOffset + lastSize, 64),
        "readback: sidecar length == aligned total");

  fs::remove_all(dir, ec);
}

// A write failure (writer cannot be created) surfaces as pass failure.
void testWriteFailureFailsPass() {
  Harness h;
  // The pass emits an error diagnostic on write failure; swallow it so the
  // expected failure does not clutter the test log.
  mlir::ScopedDiagnosticHandler handler(
      &h.ctx, [](mlir::Diagnostic &) { return mlir::success(); });
  FailingFileSystem fs;
  bool ok = true;
  h.run(R"mlir(
    func.func @f() -> tensor<4xi8> {
      %0 = hipsr.constant {value = dense<[1, 2, 3, 4]> : tensor<4xi8>} : tensor<4xi8>
      return %0 : tensor<4xi8>
    }
  )mlir",
        &fs, ok);
  check(!ok, "write failure: pass fails");
}

} // namespace

int main() {
  testInlineValueBytesWritten();
  testMemSourceBytesWritten();
  testFileSourceBytesStreamed();
  testCumulativeAlignmentAndPadding();
  testMultiFunctionSharesOneSidecar();
  testOffsetDrivenReadBack();
  testWriteFailureFailsPass();

  if (g_failures == 0) {
    llvm::outs() << "All hipsr-externalize-constants unit tests passed.\n";
    return 0;
  }
  llvm::errs() << g_failures << " check(s) failed.\n";
  return 1;
}
