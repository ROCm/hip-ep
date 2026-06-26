/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit test for the HIP memory-space attribute and the device/host
// operand predicates (lib/Dialect/IR/HipDialect.cpp).
//
// LIT covers the textual parse/print round-trip and the op-verifier
// diagnostics; this test exercises the parts LIT cannot reach directly:
//   * the C++ attribute API (MemorySpaceAttr::get / getKind)
//   * mlir::parseAttribute of "#hip.mem<...>" and print-back
//   * isDeviceCompatibleMemRef / isHostCompatibleMemRef across device, host,
//     no-space (transitional), legacy integer-space (transitional), and
//     non-memref inputs.
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

std::string printAttr(mlir::Attribute attr) {
  std::string s;
  llvm::raw_string_ostream os(s);
  attr.print(os);
  return os.str();
}

} // namespace

int main() {
  using namespace mlir;
  using namespace mlir::hip;

  MLIRContext ctx;
  ctx.loadDialect<HipDialect>();
  Builder b(&ctx);
  Type f32 = b.getF32Type();

  // --- Attribute C++ API -------------------------------------------------
  auto deviceAttr = MemorySpaceAttr::get(&ctx, MemorySpaceKind::Device);
  auto hostAttr = MemorySpaceAttr::get(&ctx, MemorySpaceKind::Host);
  auto pinnedAttr = MemorySpaceAttr::get(&ctx, MemorySpaceKind::Pinned);
  auto managedAttr = MemorySpaceAttr::get(&ctx, MemorySpaceKind::Managed);
  CHECK(deviceAttr.getKind() == MemorySpaceKind::Device);
  CHECK(hostAttr.getKind() == MemorySpaceKind::Host);
  CHECK(pinnedAttr.getKind() == MemorySpaceKind::Pinned);
  CHECK(managedAttr.getKind() == MemorySpaceKind::Managed);
  // Uniqued: same kind -> same instance.
  CHECK(deviceAttr == MemorySpaceAttr::get(&ctx, MemorySpaceKind::Device));
  CHECK(deviceAttr != hostAttr);

  // --- Parse + print round-trip ------------------------------------------
  CHECK(printAttr(deviceAttr) == "#hip.mem<device>");
  CHECK(printAttr(hostAttr) == "#hip.mem<host>");
  CHECK(printAttr(pinnedAttr) == "#hip.mem<pinned>");
  CHECK(printAttr(managedAttr) == "#hip.mem<managed>");

  Attribute parsed = parseAttribute("#hip.mem<managed>", &ctx);
  CHECK(parsed && isa<MemorySpaceAttr>(parsed));
  if (auto p = dyn_cast_or_null<MemorySpaceAttr>(parsed))
    CHECK(p.getKind() == MemorySpaceKind::Managed);

  // --- Predicates: explicit device / host --------------------------------
  auto devMR = MemRefType::get({4}, f32, /*layout=*/MemRefLayoutAttrInterface{},
                               deviceAttr);
  auto hostMR = MemRefType::get({4}, f32, /*layout=*/MemRefLayoutAttrInterface{},
                                hostAttr);
  CHECK(isDeviceCompatibleMemRef(devMR));
  CHECK(!isHostCompatibleMemRef(devMR));
  CHECK(isHostCompatibleMemRef(hostMR));
  CHECK(!isDeviceCompatibleMemRef(hostMR));

  // A pinned/managed space is neither device nor host.
  auto pinnedMR = MemRefType::get(
      {4}, f32, /*layout=*/MemRefLayoutAttrInterface{}, pinnedAttr);
  CHECK(!isDeviceCompatibleMemRef(pinnedMR));
  CHECK(!isHostCompatibleMemRef(pinnedMR));

  // --- Predicates: transitional (no hip space) ---------------------------
  // Plain memref with no memory space — accepted by BOTH while the pipeline
  // has not yet stamped spaces (kAcceptUnspecifiedMemorySpace == true).
  auto plainMR = MemRefType::get({4}, f32);
  CHECK(isDeviceCompatibleMemRef(plainMR));
  CHECK(isHostCompatibleMemRef(plainMR));

  // Legacy integer memory space (the `, 1` form) — also accepted by both.
  auto intMR = MemRefType::get({4}, f32, /*layout=*/MemRefLayoutAttrInterface{},
                               b.getI64IntegerAttr(1));
  CHECK(isDeviceCompatibleMemRef(intMR));
  CHECK(isHostCompatibleMemRef(intMR));

  // --- Predicates: non-memref inputs are never compatible ----------------
  CHECK(!isDeviceCompatibleMemRef(f32));
  CHECK(!isHostCompatibleMemRef(f32));
  auto tensorTy = RankedTensorType::get({4}, f32);
  CHECK(!isDeviceCompatibleMemRef(tensorTy));
  CHECK(!isHostCompatibleMemRef(tensorTy));

  if (g_failures == 0) {
    std::fprintf(stderr, "test_memory_space: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "test_memory_space: %d check(s) failed\n", g_failures);
  return 1;
}
