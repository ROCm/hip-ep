/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Convolution tile selection, with no GPU: the choice is integer arithmetic
// over a shape and a CU count, so all of it can be checked here.
//
// The invariant that matters is monotonicity. The heuristic this replaced
// picked its tile by walking a ladder and taking the first rung producing at
// least `deviceUnitCount()` blocks, which means a device with more CUs
// rejects more rungs and falls further down -- to narrower row blocks, each of
// which re-gathers the im2col matrix once more. That is backwards, and it is
// how a kernel tuned on a 16-CU part came to spend 38.8% of resnet50's
// convolution time in a 16-row tile on a 20-CU one.
//
// So the first test here sweeps CU counts across every shape in the models and
// asserts the row block never narrows as the device grows. It is the property
// that makes a heuristic tuned on one part trustworthy on another, and it is
// checkable without hardware, which is the whole reason the selection was
// pulled out of the .hip file.
//
// The rest pin the specific shapes that regressed, so a future retune has to
// notice it is undoing this one.

#include "conv_tile_select.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using conv_tile::Choice;
using conv_tile::Kind;
using conv_tile::Shape;

int failures = 0;

// The CU counts worth holding the selection to. The lower bound is not a
// rounding-off: below about eight units a tile producing four blocks really
// does leave most of the device idle, and trading gather for parallelism
// really is the right call, so monotonicity is not the property one wants
// there. No part this EP runs on is that small -- the two it ships on are 16
// and 20 -- and the upper bound is generous room above them.
constexpr int64_t kMinUnits = 8;
constexpr int64_t kMaxUnits = 128;

void require(bool condition, const char *expression, int line) {
  if (condition)
    return;
  std::fprintf(stderr, "requirement failed at line %d: %s\n", line, expression);
  ++failures;
}

#define REQUIRE(expr) require((expr), #expr, __LINE__)

const char *kindName(Kind k) {
  switch (k) {
  case Kind::Depthwise: return "depthwise";
  case Kind::Wmma:      return "wmma";
  case Kind::Tiled:     return "tiled";
  case Kind::Direct:    return "direct";
  }
  return "?";
}

void describe(const char *what, const Shape &s, const Choice &c) {
  std::fprintf(stderr,
               "  %-28s M=%-5lld Nout=%-8lld K=%-6lld -> %s %dx%d\n", what,
               (long long)s.MPerGroup, (long long)s.Nout, (long long)s.K,
               kindName(c.kind), c.bm, c.bn);
}

// A convolution as the models express it, reduced to what selection reads.
Shape conv(int64_t cinPerGroup, int64_t mPerGroup, int64_t nout, int64_t kvol,
           int64_t batch = 1, int64_t group = 1, bool fp16 = true) {
  Shape s;
  s.CinPerGroup = cinPerGroup;
  s.MPerGroup = mPerGroup;
  s.Nout = nout;
  s.kvol = kvol;
  s.K = cinPerGroup * kvol;
  s.N = batch;
  s.group = group;
  s.elemBytes = fp16 ? 2 : 4;
  s.wmmaEligible = fp16;
  return s;
}

// Every distinct convolution shape in the guard set, by the numbers that
// selection actually reads. Drawn from the models, not invented: resnet50,
// esrgan, mobilenet, ssd-resnet34, detr, bevformer, sam2.1 and simplebev.
std::vector<Shape> modelShapes() {
  std::vector<Shape> v;
  // resnet50, all four stages.
  v.push_back(conv(3, 64, 112 * 112, 49));
  v.push_back(conv(64, 64, 56 * 56, 1));
  v.push_back(conv(64, 64, 56 * 56, 9));
  v.push_back(conv(64, 256, 56 * 56, 1));
  v.push_back(conv(256, 64, 56 * 56, 1));
  v.push_back(conv(256, 128, 56 * 56, 1));
  v.push_back(conv(128, 128, 28 * 28, 9));
  v.push_back(conv(128, 512, 28 * 28, 1));
  v.push_back(conv(512, 128, 28 * 28, 1));
  v.push_back(conv(512, 256, 28 * 28, 1));
  v.push_back(conv(256, 256, 14 * 14, 9));
  v.push_back(conv(256, 1024, 14 * 14, 1));
  v.push_back(conv(1024, 256, 14 * 14, 1));
  v.push_back(conv(1024, 512, 14 * 14, 1));
  v.push_back(conv(512, 512, 7 * 7, 9));
  v.push_back(conv(512, 2048, 7 * 7, 1));
  v.push_back(conv(2048, 512, 7 * 7, 1));
  // esrgan: 3x3 stride 1 throughout, on a very large plane.
  v.push_back(conv(3, 64, 250 * 250, 9));
  v.push_back(conv(64, 32, 250 * 250, 9));
  v.push_back(conv(192, 64, 250 * 250, 9));
  v.push_back(conv(64, 64, 1000 * 1000, 9));
  v.push_back(conv(64, 3, 1000 * 1000, 9));
  // Depthwise, from mobilenet and mb1-ssd.
  v.push_back(conv(1, 1, 112 * 112, 9, 1, 32));
  v.push_back(conv(1, 1, 14 * 14, 9, 1, 576));
  // bevformer / detr FPN.
  v.push_back(conv(2048, 256, 15 * 25, 1));
  v.push_back(conv(1024, 256, 29 * 50, 1));
  v.push_back(conv(256, 256, 15 * 25, 9));
  // sam2.1 encoder, and a batched case.
  v.push_back(conv(3, 96, 256 * 256, 49));
  v.push_back(conv(96, 96, 64 * 64, 9, 4));
  // simplebev is fp32, which cannot use the matrix cores.
  v.push_back(conv(128, 128, 200 * 200, 9, 1, 1, false));
  v.push_back(conv(64, 64, 100 * 100, 9, 1, 1, false));
  // Degenerate extremes.
  v.push_back(conv(1, 1, 1, 1));
  v.push_back(conv(2048, 4096, 1, 1));
  v.push_back(conv(8, 8, 4, 9));
  return v;
}

// --------------------------------------------------------------------------

// The invariant the old ladder violated. A wider row block re-gathers the
// im2col matrix fewer times, so it is never the wrong answer on a *larger*
// device: whatever made it affordable at 16 CUs is still true at 64. If the
// selection narrows as `units` grows it is buying parallelism with bandwidth,
// which is the trade that produced the regression.
void testRowBlockIsMonotoneInUnits() {
  for (const Shape &s : modelShapes()) {
    int prevBm = 0;
    Choice prev{};
    for (int64_t units = kMinUnits; units <= kMaxUnits; ++units) {
      const Choice c = conv_tile::select(s, units);
      if (c.kind == Kind::Depthwise)
        break; // not a tiling decision, and independent of units
      const int bm = c.kind == Kind::Direct ? 1 : c.bm;
      if (bm < prevBm) {
        std::fprintf(stderr,
                     "row block narrowed as units grew: M=%lld Nout=%lld "
                     "K=%lld went %s %dx%d -> %s %dx%d at units=%lld\n",
                     (long long)s.MPerGroup, (long long)s.Nout, (long long)s.K,
                     kindName(prev.kind), prev.bm, prev.bn, kindName(c.kind),
                     c.bm, c.bn, (long long)units);
        ++failures;
        break;
      }
      prevBm = bm;
      prev = c;
    }
  }
}

// A device with more CUs must never be given a *worse* grid than a smaller one
// would get for the same shape -- the block count must not fall.
void testBlockCountDoesNotFallAsUnitsGrow() {
  for (const Shape &s : modelShapes()) {
    for (int64_t units = kMinUnits + 1; units <= kMaxUnits; ++units) {
      const Choice lo = conv_tile::select(s, units - 1);
      const Choice hi = conv_tile::select(s, units);
      if (lo.kind != Kind::Wmma && lo.kind != Kind::Tiled)
        continue;
      if (hi.kind != Kind::Wmma && hi.kind != Kind::Tiled)
        continue;
      const int64_t bl = conv_tile::ceilDiv(s.MPerGroup, lo.bm) *
                         conv_tile::ceilDiv(s.Nout, lo.bn);
      const int64_t bh = conv_tile::ceilDiv(s.MPerGroup, hi.bm) *
                         conv_tile::ceilDiv(s.Nout, hi.bn);
      REQUIRE(bh >= bl);
      if (bh < bl)
        std::fprintf(stderr, "  at units=%lld M=%lld Nout=%lld\n",
                     (long long)units, (long long)s.MPerGroup,
                     (long long)s.Nout);
    }
  }
}

// Selection must be stable across the CU counts of the parts this ships on.
// The two differ by four CUs and nothing else; a shape whose tile changes
// between them is a shape whose performance is not portable.
void testAgreesAcrossShippingParts() {
  int differ = 0;
  for (const Shape &s : modelShapes()) {
    const Choice a = conv_tile::select(s, 16); // 8050S
    const Choice b = conv_tile::select(s, 20); // 8060S
    if (!(a == b)) {
      ++differ;
      describe("differs at 16 CUs", s, a);
      describe("           20 CUs", s, b);
    }
  }
  REQUIRE(differ == 0);
}

// The shapes that regressed. resnet50's stage-4 convolutions have 512 output
// channels and a 7x7 output; the old ladder gave them a 16-row tile on a
// 20-CU part, which is 32 passes over the input where 4 would do.
void testDeepNarrowOutputsGetAWideRowBlock() {
  const Shape s512x49 = conv(2048, 512, 7 * 7, 1);
  const Shape s512x49k9 = conv(512, 512, 7 * 7, 9);
  const Shape s128x784 = conv(512, 128, 28 * 28, 1);
  const Shape s512x196 = conv(1024, 512, 14 * 14, 1);

  for (int64_t units : {8, 12, 16, 20, 32, 40, 64}) {
    for (const Shape &s : {s512x49, s512x49k9, s128x784, s512x196}) {
      const Choice c = conv_tile::select(s, units);
      REQUIRE(c.kind == Kind::Wmma);
      // Anything with at least 32 output channels has better uses for a block
      // than a 16-row tile: doubling the row block halves the gather.
      REQUIRE(c.bm >= 32);
      if (c.bm < 32)
        std::fprintf(stderr, "  units=%lld M=%lld Nout=%lld got %dx%d\n",
                     (long long)units, (long long)s.MPerGroup,
                     (long long)s.Nout, c.bm, c.bn);
    }
  }
}

// The narrow rung still has to exist. esrgan's last convolution produces three
// channels; there is nothing for a wide row block to do, and a 16-row tile
// over a million output positions covers the device on its columns alone.
void testGenuinelyNarrowOutputsStillGetTheNarrowTile() {
  const Shape s = conv(64, 3, 1000 * 1000, 9);
  for (int64_t units : {8, 16, 20, 32, 64}) {
    const Choice c = conv_tile::select(s, units);
    REQUIRE(c.kind == Kind::Wmma);
    REQUIRE(c.bm == 16);
    if (c.bm != 16)
      describe("narrow output", s, c);
  }
}

// Depthwise has no output-channel reuse to tile, and is decided before any
// tile is priced.
void testDepthwiseIsNotTiled() {
  for (int64_t units : {1, 16, 20, 64}) {
    REQUIRE(conv_tile::select(conv(1, 1, 112 * 112, 9, 1, 32), units).kind ==
            Kind::Depthwise);
    REQUIRE(conv_tile::select(conv(1, 1, 14 * 14, 9, 1, 576), units).kind ==
            Kind::Depthwise);
  }
  // group == 1 is not depthwise however narrow it is.
  REQUIRE(conv_tile::select(conv(1, 1, 4096, 9), 20).kind != Kind::Depthwise);
}

// fp32 and bf16 have no matrix path; they must land on the scalar ladder.
void testNonHalfNeverPicksMatrixCores() {
  for (int64_t units : {1, 8, 16, 20, 64}) {
    for (const Shape &s : modelShapes()) {
      if (s.wmmaEligible)
        continue;
      REQUIRE(conv_tile::select(s, units).kind != Kind::Wmma);
    }
  }
}

// Every choice must be one the kernel can actually be launched with.
void testChoiceIsAlwaysInstantiable() {
  for (const Shape &s : modelShapes()) {
    for (int64_t units = 1; units <= 128; ++units) {
      const Choice c = conv_tile::select(s, units);
      if (c.kind == Kind::Depthwise || c.kind == Kind::Direct)
        continue;
      const conv_tile::Tile *table =
          c.kind == Kind::Wmma ? conv_tile::kWmmaTiles : conv_tile::kScalarTiles;
      const int n = c.kind == Kind::Wmma ? conv_tile::kNumWmmaTiles
                                         : conv_tile::kNumScalarTiles;
      bool found = false;
      for (int i = 0; i < n; ++i)
        if (table[i].bm == c.bm && table[i].bn == c.bn &&
            table[i].tm == c.tm && table[i].tn == c.tn)
          found = true;
      REQUIRE(found);
    }
  }
}

// A shape must not be selected differently just because it was described with
// a batch of 4 rather than four separate calls: the batch multiplies the grid,
// it does not change what a block does.
void testBatchOnlyAddsParallelism() {
  for (int64_t units : {16, 20, 64}) {
    const Choice one = conv_tile::select(conv(96, 96, 64 * 64, 9, 1), units);
    const Choice four = conv_tile::select(conv(96, 96, 64 * 64, 9, 4), units);
    // More parallelism must never buy a narrower row block.
    REQUIRE(four.bm >= one.bm);
  }
}

void testNoDegenerateShapeIsRejected() {
  for (int64_t units : {1, 20, 64}) {
    for (const Shape &s : modelShapes()) {
      const Choice c = conv_tile::select(s, units);
      REQUIRE(c.kind == Kind::Depthwise || c.kind == Kind::Direct ||
              (c.bm > 0 && c.bn > 0 && c.tm > 0 && c.tn > 0));
    }
  }
}

// HIPDNN_EP_CONV_TILE names a tile and resolve() finds it, which is how the
// sweep that calibrated the thresholds forces a tile selection rejects. Two
// entries now share a block shape and differ only in the register tile, so a
// name has to be able to say which -- otherwise a sweep row silently measures
// the other one and the number it reports is attributed to the wrong tile.
void testEveryTileIsReachableByName() {
  for (bool fp16 : {true, false}) {
    const Shape s = conv(64, 128, 56 * 56, 9, 1, 1, fp16);
    const conv_tile::Tile *table =
        fp16 ? conv_tile::kWmmaTiles : conv_tile::kScalarTiles;
    const int n = fp16 ? conv_tile::kNumWmmaTiles : conv_tile::kNumScalarTiles;
    const Kind want = fp16 ? Kind::Wmma : Kind::Tiled;
    for (int i = 0; i < n; ++i) {
      Choice c{};
      REQUIRE(conv_tile::resolve(s, table[i].bm, table[i].bn, table[i].tm,
                                 table[i].tn, c));
      REQUIRE(c.kind == want);
      REQUIRE(c.bm == table[i].bm && c.bn == table[i].bn &&
              c.tm == table[i].tm && c.tn == table[i].tn);
    }
    // The two-field form is still accepted, and takes the first entry listed
    // for that block shape -- the one selection would have picked.
    for (int i = 0; i < n; ++i) {
      Choice c{};
      REQUIRE(conv_tile::resolve(s, table[i].bm, table[i].bn, 0, 0, c));
      int first = 0;
      while (table[first].bm != table[i].bm || table[first].bn != table[i].bn)
        ++first;
      REQUIRE(c.tm == table[first].tm && c.tn == table[first].tn);
    }
    // A name in the other dtype's table is refused rather than resolved into
    // a kernel this dtype has no instantiation for.
    Choice c{};
    REQUIRE(!conv_tile::resolve(s, 7, 7, 0, 0, c));
  }
}

} // namespace

int main() {
  testRowBlockIsMonotoneInUnits();
  testBlockCountDoesNotFallAsUnitsGrow();
  testAgreesAcrossShippingParts();
  testDeepNarrowOutputsGetAWideRowBlock();
  testGenuinelyNarrowOutputsStillGetTheNarrowTile();
  testDepthwiseIsNotTiled();
  testNonHalfNeverPicksMatrixCores();
  testChoiceIsAlwaysInstantiable();
  testBatchOnlyAddsParallelism();
  testNoDegenerateShapeIsRejected();
  testEveryTileIsReachableByName();

  if (failures) {
    std::fprintf(stderr, "\n%d requirement(s) failed\n", failures);
    return 1;
  }
  std::printf("conv tile selection: all checks passed\n");
  return 0;
}
