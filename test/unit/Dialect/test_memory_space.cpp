/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit test for the HIP memory-space attribute and the per-space
// operand predicates (lib/Dialect/IR/HipDialect.cpp).
//
// LIT covers the textual parse/print round-trip and the op-verifier
// diagnostics; this test exercises the parts LIT cannot reach directly:
//   * the C++ attribute API (MemorySpaceAttr::get / getKind) for all 4 spaces
//   * mlir::parseAttribute of "#hip.mem<...>" and print-back
//   * the 4 predicates is{Device,Host,Pinned,Managed}CompatibleMemRef across
//     each explicit space (exact match + cross-rejection), no-space
//     (transitional), legacy integer-space (transitional), and non-memref
//     inputs.
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

  // --- Predicates: each explicit space (exact match + cross-rejection) ---
  // One memref per space; each predicate must accept ONLY its own space.
  auto mk = [&](MemorySpaceAttr s) {
    return MemRefType::get({4}, f32, /*layout=*/MemRefLayoutAttrInterface{}, s);
  };
  auto devMR = mk(deviceAttr);
  auto hostMR = mk(hostAttr);
  auto pinnedMR = mk(pinnedAttr);
  auto managedMR = mk(managedAttr);

  CHECK(isDeviceCompatibleMemRef(devMR));
  CHECK(!isHostCompatibleMemRef(devMR));
  CHECK(!isPinnedCompatibleMemRef(devMR));
  CHECK(!isManagedCompatibleMemRef(devMR));

  CHECK(isHostCompatibleMemRef(hostMR));
  CHECK(!isDeviceCompatibleMemRef(hostMR));
  CHECK(!isPinnedCompatibleMemRef(hostMR));
  CHECK(!isManagedCompatibleMemRef(hostMR));

  CHECK(isPinnedCompatibleMemRef(pinnedMR));
  CHECK(!isDeviceCompatibleMemRef(pinnedMR));
  CHECK(!isHostCompatibleMemRef(pinnedMR));
  CHECK(!isManagedCompatibleMemRef(pinnedMR));

  CHECK(isManagedCompatibleMemRef(managedMR));
  CHECK(!isDeviceCompatibleMemRef(managedMR));
  CHECK(!isHostCompatibleMemRef(managedMR));
  CHECK(!isPinnedCompatibleMemRef(managedMR));

  // --- Predicates: transitional (no hip space) ---------------------------
  // Plain memref with no memory space — accepted by ALL four while the
  // pipeline has not yet stamped spaces (kAcceptUnspecifiedMemorySpace).
  auto plainMR = MemRefType::get({4}, f32);
  CHECK(isDeviceCompatibleMemRef(plainMR));
  CHECK(isHostCompatibleMemRef(plainMR));
  CHECK(isPinnedCompatibleMemRef(plainMR));
  CHECK(isManagedCompatibleMemRef(plainMR));

  // Legacy integer memory space (the `, 1` form) — also accepted by all four.
  auto intMR = MemRefType::get({4}, f32, /*layout=*/MemRefLayoutAttrInterface{},
                               b.getI64IntegerAttr(1));
  CHECK(isDeviceCompatibleMemRef(intMR));
  CHECK(isHostCompatibleMemRef(intMR));
  CHECK(isPinnedCompatibleMemRef(intMR));
  CHECK(isManagedCompatibleMemRef(intMR));

  // --- Predicates: non-memref inputs are never compatible ----------------
  auto tensorTy = RankedTensorType::get({4}, f32);
  for (Type t : {f32, static_cast<Type>(tensorTy)}) {
    CHECK(!isDeviceCompatibleMemRef(t));
    CHECK(!isHostCompatibleMemRef(t));
    CHECK(!isPinnedCompatibleMemRef(t));
    CHECK(!isManagedCompatibleMemRef(t));
  }

  if (g_failures == 0) {
    std::fprintf(stderr, "test_memory_space: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "test_memory_space: %d check(s) failed\n", g_failures);
  return 1;
}
