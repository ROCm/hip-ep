/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ===========================================================================
// Which kernel and which tile a convolution shape should run on
// ===========================================================================
//
// Split out of conv_kernel.hip and deliberately free of HIP: the choice is
// integer arithmetic over a shape, so it can be compiled and tested natively.
// That matters because the choice is not obviously right, and a heuristic that
// can only be exercised by running convolutions on a GPU does not get
// exercised. test/runtime/test_conv_tile_select.cpp is the consequence.
//
// What this replaces, and why
// ---------------------------
// The original was a ladder of tile shapes, each guarded by "is this tile's
// row block fully live, and does it produce at least `units` blocks". The
// first tile to clear both won, and a second pass dropped the block-count
// requirement if none did.
//
// Insisting on covering the device *before* considering what a tile costs gets
// the priority backwards, and on a part with more CUs it inverts. The im2col
// matrix is never materialised, so it is re-gathered once per tile row;
// halving the row block doubles the dominant cost of the kernel. A coverage
// test rejects wide tiles precisely when the output is small -- which is when
// the shape has few blocks under *any* tile -- so the ladder walks down to a
// narrow tile and pays that doubling to buy parallelism it cannot use.
//
// The last rung made it worse: unlike the others it did not require its rows
// to be live, so it caught every shape that reached it, whatever its channel
// count. On this 8060S (20 CUs) resnet50's six 512-channel stage-4
// convolutions land there and re-gather the im2col matrix 32 times where a
// 128-row tile would do it 4 times; they measured 111 ms, 38.8% of the model's
// entire convolution time, at 7 GB/s. On a 16-CU part the same shapes clear
// the 32-row rung and never reach it. Thirteen of resnet50's convolutions
// change tile between 16 and 20 CUs, in both directions.
//
// How the replacement was arrived at
// ----------------------------------
// Not by reasoning. Every tile below was forced, one at a time, on all 297
// distinct non-depthwise convolution shapes in the 20-model guard set and
// timed on the part (tools/perf-harness/bench/conv_microbench.py tile-sweep).
// That gives the cost of every candidate on every shape, so a policy can be
// scored against the best-possible per-shape choice rather than argued about.
// Summed over the guard set, weighted by how often each shape occurs:
//
//   old coverage ladder                381.1 ms
//   this policy                        331.8 ms
//   best possible per-shape choice     328.5 ms
//
// An explicit traffic-and-parallelism cost model -- gather proportional to
// tile rows, filter to tile columns, a saturation term, fitted constants --
// was also built and scored: 341.9 ms at its best fit, worse than the four
// comparisons below despite five free parameters. It is not here because it
// did not earn its complexity. The thresholds are what the measurements say,
// and the sweep that produced them is checked in, so a retune re-runs it
// rather than re-deriving it.
//
// The property the ladder could not offer is that the choice is monotone in
// the CU count: giving the device more CUs must never select a narrower row
// block. This policy gets that in the strongest available form -- it does not
// consult the CU count at all. Adding CUs does not add memory bandwidth, and
// on these shapes the tile choice is a bandwidth decision, so there is
// nothing for the CU count to say. `units` is still taken, and still ignored,
// to keep that deliberate rather than accidental; see select().

#ifndef HIPDNN_EP_CONV_TILE_SELECT_H
#define HIPDNN_EP_CONV_TILE_SELECT_H

#include <cstdint>

namespace conv_tile {

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

// Only what the choice depends on. Everything here is per-group where the
// distinction matters, because a group is what a block works inside.
struct Shape {
  int64_t MPerGroup = 1;   // output channels in a group: the GEMM's M
  int64_t Nout = 1;        // output positions: the GEMM's N
  int64_t K = 1;           // CinPerGroup * kvol: the contraction
  int64_t N = 1;           // batch
  int64_t group = 1;
  int64_t CinPerGroup = 1;
  int64_t kvol = 1;
  int32_t elemBytes = 2;
  bool wmmaEligible = false; // fp16 with the matrix-core builtin available
};

enum class Kind {
  Depthwise,
  Wmma,
  Tiled,   // scalar register-tiled
  Direct,  // one thread per output element
};

struct Choice {
  Kind kind = Kind::Direct;
  int bm = 0;
  int bn = 0;
  int tm = 0; // wave tile M (wmma) / register tile M (tiled)
  int tn = 0;
};

inline bool operator==(const Choice &a, const Choice &b) {
  return a.kind == b.kind && a.bm == b.bm && a.bn == b.bn && a.tm == b.tm &&
         a.tn == b.tn;
}

// ---------------------------------------------------------------------------
// The tiles that exist
// ---------------------------------------------------------------------------
//
// Every entry is instantiated in conv_kernel.hip; this table and those
// instantiations have to agree, which the unit test checks by walking the
// table and the kernel enforces by static_assert. The table says only what
// can be launched -- which shapes *should* use what is select()'s business,
// and is kept there so there is one place to read the policy.
//
// Entries select() never picks are still listed, and still instantiated,
// because HIPDNN_EP_CONV_TILE resolves against this table: re-running the
// sweep that produced the thresholds requires being able to force a tile the
// policy rejects. They are cheap to keep and the sweep is useless without
// them.

struct Tile {
  int bm, bn, tm, tn;
};

// fp16, matrix cores. `tm`/`tn` are the wave tile in 16x16 units, so they set
// the accumulator count (tm*tn*8 VGPRs) and, with the block shape, the thread
// count -- which is why the same block shape appears at more than one wave
// tile. The narrow-wave entries exist to trade accumulators for occupancy: a
// 128x256 block is 128 accumulator VGPRs at 4x4 and 64 at 2x4, for the same
// tile and twice the threads.
constexpr Tile kWmmaTiles[] = {
    {128, 256, 4, 4},
    {256, 128, 4, 2},
    {128, 128, 4, 2},
    {64, 256, 2, 4},
    {32, 256, 2, 2},
    {16, 256, 1, 2},
    {128, 256, 2, 4},
    {128, 128, 2, 2},
    {64, 256, 2, 2},
};
constexpr int kNumWmmaTiles =
    static_cast<int>(sizeof(kWmmaTiles) / sizeof(kWmmaTiles[0]));

// fp32 and bf16, which have no matrix path. The register tile is BM*BN/256,
// since every one of these launches 256 threads (the staging assignment in
// conv_tiled_kernel requires BM and BN to divide that way), so the block shape
// and the accumulator count are not independent choices here.
constexpr Tile kScalarTiles[] = {
    {128, 128, 8, 8},
    {64, 128, 4, 8},
    {128, 64, 8, 4},
    {64, 64, 4, 4},
    {32, 128, 2, 8},
    {64, 32, 4, 2},
    {32, 64, 2, 4},
    {32, 32, 2, 2},
};
constexpr int kNumScalarTiles =
    static_cast<int>(sizeof(kScalarTiles) / sizeof(kScalarTiles[0]));

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------
//
// Read the ladder in select() alongside these. Each is a measured crossover,
// and the comment on each says what crosses.

// Output channels at which the widest usable row block starts paying for
// itself outright. Above this a 128-row tile's rows are all live and it
// re-gathers the im2col matrix a quarter as often as a 32-row one.
constexpr int64_t kWideRowBlockMinM = 128;

// The two rungs between: enough output channels for this row block, not
// enough for the next one up. 64x256 also issues more matrix work per
// fragment load than 32x256 -- 2x4 is 8 WMMAs off 6 where 2x2 is 4 off 4 --
// and that ratio cannot be had at 32 rows, where the B-stage mapping pins the
// block at 256 threads.
constexpr int64_t kMidRowBlockMinM = 64;
constexpr int64_t kNarrowRowBlockMinM = 32;

// The contraction length past which a wide row block wins *even though most
// of its rows are dead*. This is the one threshold that is not about how many
// output channels there are, and it is the interesting one.
//
// Gather traffic is proportional to the number of tile rows, so it scales
// with K; wasted matrix lanes are proportional to the row block and do not.
// Past a long enough contraction the gather dominates by enough that halving
// the tile rows is worth an eightfold increase in wasted matrix work. The ssd
// detection heads are exactly this: 24 output channels over a 4608-long
// contraction measured 0.539 ms on a 16-row tile and 0.266 on a 128-row one,
// which throws away 104 of its 128 rows. Below the threshold the gather is
// cheap and the narrow tile is simply the right size -- mobilenet's 24
// channels over a 144-long contraction want the 16-row tile.
constexpr int64_t kGatherDominatesK = 2048;

// Output positions below which the *column* block is the one overhanging, and
// the trade runs the other way: prefer 256 rows by 128 columns to 128 by 256.
//
// This is kGatherDominatesK's argument on the other axis. A 256-wide column
// block serving 49 output positions wastes four fifths of its columns, and
// nothing about a wider column block pays for that when there is no width to
// cover -- whereas the taller row block halves the tile rows, and so halves
// the gather. Of the 31 shapes in the guard set below this threshold, 28
// improve by 6-21%: resnet50's stage-4 7x7 outputs, mobilenet's 7x7 tail, the
// ssd detection heads, convnext's 7x7 downsample. The three that do not have
// a single output position, where the whole question is moot and the loss is
// 3% of 0.15 ms; excluding them would be fitting the guard set rather than
// reading it.
constexpr int64_t kFewOutputPositions = 64;

// The same crossover on the scalar path, where it decides tiled against
// direct rather than which tile. simplebev's single- and two-channel fp32
// convolutions over a 40000-position plane and a 128-long contraction
// measured 0.081 ms direct against 0.214 on a 64x64 tile: 63 of 64 dead rows
// is not worth a short gather. Anything longer goes tiled, because
// conv_direct has no reuse at all.
constexpr int64_t kScalarGatherDominatesK = 256;

// ---------------------------------------------------------------------------

inline int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Depthwise is not a tiling decision. group == C means every group contracts
// over one input channel and M is 1, so there is no output-channel reuse for a
// register tile to exploit, and the contraction is shared by the whole block
// -- which the dedicated kernel stages once and conv_direct re-decodes on
// every tap.
constexpr int64_t kMaxDepthwiseK = 128;

inline bool isDepthwise(const Shape &s) {
  return s.CinPerGroup == 1 && s.MPerGroup == 1 && s.kvol <= kMaxDepthwiseK &&
         s.group > 1;
}

// Resolve a named tile against the table the shape's dtype would actually
// use. Two tables share the 128x128 block shape, so a name is only unambiguous
// once the dtype has chosen a table -- resolving it against the wrong one hands
// fp32 a matrix-core tile it cannot launch, which silently fell through to
// conv_direct and cost a 28x slowdown while the sweep was being calibrated.
//
// `tm`/`tn` of 0 means "whichever register tile the table lists first for this
// block shape", which is what a two-field name like 128x256 asks for. The
// matrix table now carries the same block shape at two wave tiles, so a sweep
// wanting the narrow one has to say which, and passes all four.
inline bool resolve(const Shape &s, int bm, int bn, int tm, int tn,
                    Choice &out) {
  const Tile *table = s.wmmaEligible ? kWmmaTiles : kScalarTiles;
  const int n = s.wmmaEligible ? kNumWmmaTiles : kNumScalarTiles;
  const Kind kind = s.wmmaEligible ? Kind::Wmma : Kind::Tiled;
  for (int i = 0; i < n; ++i) {
    if (table[i].bm != bm || table[i].bn != bn) continue;
    if (tm && (table[i].tm != tm || table[i].tn != tn)) continue;
    out = Choice{kind, table[i].bm, table[i].bn, table[i].tm, table[i].tn};
    return true;
  }
  return false;
}

// The choice.
//
// `units` is the device's CU count, and is deliberately unused. The old
// heuristic's central mistake was scaling a memory decision with the CU count:
// adding CUs does not add bandwidth, so a heuristic that lets the device size
// widen the grid will trade bandwidth for parallelism as the part grows --
// narrower row block, more tile rows, more passes over the input. That is the
// trade that made this 8060S regress harder than the 8050S the kernel was
// tuned on. Ignoring `units` is what makes the monotonicity the unit test
// asserts hold by construction rather than by luck. The parameter stays so
// that a future variant needing it does not have to re-thread it, and so this
// paragraph has somewhere to live.
inline Choice select(const Shape &s, int64_t units) {
  (void)units;

  if (isDepthwise(s)) return Choice{Kind::Depthwise, 0, 0, 0, 0};

  const int64_t M = s.MPerGroup;

  if (s.wmmaEligible) {
    // Every rung from 64 output channels up now asks for the same tile, wide
    // shapes included. That is not what the gather term predicts -- a 64-row
    // block over M=512 is eight tile rows where 128x128 needs four, so twice
    // the im2col traffic -- and it measured the other way round until there
    // was a narrow wave tile to measure against.
    //
    // The wave tile, not the block shape, was the binding constraint. A 2x2
    // wave tile holds 32 accumulator VGPRs where 2x4 holds 64, which takes
    // 64x256 from 146 VGPR / 9 waves per SIMD to 97 / 12, and spreads the same
    // block over 512 threads instead of 256 so the staging has twice the lanes
    // to hide behind. Over the guard set that is worth more than the doubled
    // gather: the wide rung goes 200.2 ms to 176.3, and it wins in every M
    // bucket from 128 up and every output-size bucket below 16k positions, so
    // it is not an average concealing a split. The rungs stay separate because
    // the thresholds are what the monotonicity argument rests on, and because
    // the next retune should not have to rediscover that they were ever
    // distinct.
    if (M >= kWideRowBlockMinM || s.K >= kGatherDominatesK) {
      // Few enough output positions that 256 columns would be mostly overhang,
      // so the columns go into the row block instead.
      if (s.Nout < kFewOutputPositions)
        return Choice{Kind::Wmma, 128, 128, 2, 2};
      return Choice{Kind::Wmma, 64, 256, 2, 2};
    }
    if (M >= kMidRowBlockMinM) return Choice{Kind::Wmma, 64, 256, 2, 2};
    if (M >= kNarrowRowBlockMinM) return Choice{Kind::Wmma, 32, 256, 2, 2};
    // Genuinely narrower than any other row block, and a short enough
    // contraction that the gather does not pay for overhang: esrgan's 64->3
    // output convolution, mobilenet's 24-channel expansions.
    return Choice{Kind::Wmma, 16, 256, 1, 2};
  }

  // fp32 and bf16. Only one rung survived the sweep: 128x128's 8x8 register
  // tile does 64 FMAs per 16 LDS reads against 64x64's 16 per 8, but it needs
  // 256 VGPRs and caps occupancy at 25%, and it lost on every shape in the
  // guard set -- simplebev's 200x200 fp32 convolutions measured 0.713 ms on
  // it against 0.214 on 64x64. 32x128 lost everywhere too.
  //
  // Narrower register tiles do not help here, which is the opposite of the
  // matrix path and worth recording so it is not retried. 64x32 and 32x64 run
  // at 137 VGPR / 10 waves and 32x32 at 101 / 12, against 64x64's 177 / 8, and
  // all three lose on simplebev's 200x200 shapes: the 3x3 measures 12.68 ms on
  // 64x64 against 16.03, 16.93 and 20.35. Unlike the matrix tiles these were
  // never accumulator-bound -- a 4x4 register tile is 16 VGPRs of the 177 --
  // so the occupancy is bought by giving up the LDS reuse that was paying for
  // itself. The aggregate says otherwise only if the extract's Nout==1
  // shape-inference fallbacks are left in, and those are not shapes the model
  // runs.
  //
  // All of them stay instantiated for the sweep; none is selected.
  if (M >= kMidRowBlockMinM || s.K >= kScalarGatherDominatesK)
    return Choice{Kind::Tiled, 64, 64, 4, 4};

  return Choice{Kind::Direct, 0, 0, 0, 0};
}

} // namespace conv_tile

#endif // HIPDNN_EP_CONV_TILE_SELECT_H
