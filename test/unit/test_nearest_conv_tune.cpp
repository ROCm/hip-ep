/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GPU-free checks for the grouped log2 nearest conv lookup: exact GEMM,
// closer spatial neighbor, hard mismatches, and a tile that does not fit.

#include "nearest_conv_tune.h"
#include "problem_cache_generated.h"

#include "flatbuffers/flatbuffers.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int gFailures = 0;

void expect(bool cond, const char *message) {
  if (cond)
    return;
  std::fprintf(stderr, "FAIL: %s\n", message);
  ++gFailures;
}

std::string key(const char *conv) {
  return std::string("gfx1151\t20\t1\t") + conv;
}

const char *kBase =
    "conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
    "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1";

using mlir::hip::ProblemCacheConvEntry;

} // namespace

int main() {
  const std::string exact = key(kBase);
  const std::string spatial96 = key(
      "conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 96 -W 96 -k 32 "
      "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1");
  const std::string spatial128 = key(
      "conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 128 -W 128 -k 32 "
      "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1");
  const std::string padded = key(
      "conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
      "-y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1");
  const std::string fp16 = key(
      "convfp16 -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
      "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1");
  const std::string tiny = key(
      "conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 4 -W 2 -k 2 "
      "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1");

  const std::string tile32 = "gemm:mPerBlock=32,nPerBlock=32,kPerBlock=4";
  const std::string tile96 = "gemm:mPerBlock=32,nPerBlock=64,kPerBlock=4";
  const std::string tile128 = "gemm:mPerBlock=32,nPerBlock=128,kPerBlock=4";

  std::vector<ProblemCacheConvEntry> table = {
      {exact, tile32},
      {spatial96, tile96},
      {spatial128, tile128},
  };

  auto hit = mlir::hip::findNearestConvTune(exact, table);
  expect(hit.has_value() && hit->exact && hit->solution == tile32,
         "exact GEMM returns that row");

  const std::string query64 = key(
      "conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
      "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1");
  std::vector<ProblemCacheConvEntry> spatial = {
      {spatial96, tile96},
      {spatial128, tile128},
  };
  hit = mlir::hip::findNearestConvTune(query64, spatial);
  expect(hit.has_value() && !hit->exact && hit->solution == tile96 &&
             hit->distance > 0.0f,
         "H/W=64 picks the 96 row over 128");

  hit = mlir::hip::findNearestConvTune(padded, table);
  expect(hit.has_value() && !hit->exact && hit->solution == tile32,
         "same stride and layout prefers the closer output size");

  hit = mlir::hip::findNearestConvTune(fp16, table);
  expect(!hit.has_value(), "convfp16 rejects conv rows");

  hit = mlir::hip::findNearestConvTune(tiny, table);
  expect(!hit.has_value(), "k=2 rejects mPerBlock=32");

  hit = mlir::hip::findNearestConvTune("gfx1151\t20\t1\tgemm -t i8", table);
  expect(!hit.has_value(), "a gemm problem with no exact row misses");

  const std::string gemmKey = "gfx1151\t2\t1\t-t f32 -m 1 -n 1000 -k 1280";
  const std::string gemmTile = "gemm:mPerBlock=64,nPerBlock=64,kPerBlock=16";
  std::vector<ProblemCacheConvEntry> withGemm = table;
  withGemm.push_back({gemmKey, gemmTile});
  hit = mlir::hip::findNearestConvTune(gemmKey, withGemm);
  expect(hit.has_value() && hit->exact && hit->solution == gemmTile,
         "an exact gemm key returns its stored tile");

  const std::string otherArch = std::string("gfx1100\t20\t1\t") + kBase;
  hit = mlir::hip::findNearestConvTune(otherArch, table);
  expect(!hit.has_value(), "a different arch is a different group");

  const std::string stride2 = key(
      "conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
      "-y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1");
  hit = mlir::hip::findNearestConvTune(stride2, table);
  expect(!hit.has_value(), "a different stride is a different group");

  const std::string splitK = exact + " -supportsSplitK false";
  hit = mlir::hip::findNearestConvTune(splitK, table);
  expect(!hit.has_value(), "a fusion suffix does not match a bare row");

  const std::string oversized = "gemm:mPerBlock=64,nPerBlock=32,kPerBlock=4";
  std::vector<ProblemCacheConvEntry> exactOversized = {{exact, oversized}};
  hit = mlir::hip::findNearestConvTune(exact, exactOversized);
  expect(hit.has_value() && hit->exact && hit->solution == oversized,
         "an exact problem returns its stored tile past the derived GEMM");

  std::vector<ProblemCacheConvEntry> illegalNeighbor = {
      {spatial96, oversized},
      {spatial128, tile128},
  };
  hit = mlir::hip::findNearestConvTune(query64, illegalNeighbor);
  expect(hit.has_value() && !hit->exact && hit->solution == tile128,
         "a borrowed oversized tile is skipped for the next nearest");

  hit = mlir::hip::findNearestConvTune(query64, spatial, {tile96});
  expect(hit.has_value() && hit->solution == tile128,
         "a rejected solution yields the next nearest row");

  namespace fbs = hipdnn_ep::rocmlir_autotune::fbs;
  flatbuffers::FlatBufferBuilder builder;
  auto tileOff = builder.CreateString(tile32);
  auto reduceOff = builder.CreateString("{\"algo\":\"block\"}");
  auto gemmOff = builder.CreateString(gemmTile);
  std::vector<flatbuffers::Offset<fbs::RocmlirTuneConfig>> configs = {
      fbs::CreateRocmlirTuneConfig(builder, tileOff),
      fbs::CreateRocmlirTuneConfig(builder, reduceOff),
      fbs::CreateRocmlirTuneConfig(builder, gemmOff),
  };
  auto convProblem = builder.CreateString(exact);
  auto reduceProblem = builder.CreateString("{\"lens\":[1,2]}");
  auto gemmProblem = builder.CreateString(gemmKey);
  std::vector<flatbuffers::Offset<fbs::RocmlirTunePoint>> points = {
      fbs::CreateRocmlirTunePoint(builder, convProblem, 0, 32, 4096, 144),
      fbs::CreateRocmlirTunePoint(builder, reduceProblem, 1, 0, 0, 0),
      fbs::CreateRocmlirTunePoint(builder, gemmProblem, 2, 0, 0, 0),
  };
  auto gpuArch = builder.CreateString("gfx1151");
  auto kernelAbi = builder.CreateString("rocmlir-conv-v1");
  auto modelKey = builder.CreateString("");
  std::vector<flatbuffers::Offset<fbs::RocmlirTuneFallback>> fallbacks;
  auto table = fbs::CreateRocmlirAutotuneLut(
      builder, 1, gpuArch, 0, kernelAbi, modelKey, 1.0f, 1.0f, 1.0f,
      builder.CreateVector(configs), builder.CreateVector(points),
      builder.CreateVector(fallbacks));
  fbs::FinishRocmlirAutotuneLutBuffer(builder, table);

  mlir::hip::ProblemCacheLut loaded;
  std::string error;
  expect(mlir::hip::loadProblemCacheLut(builder.GetBufferPointer(),
                                        builder.GetSize(), loaded, error),
         "RCAL buffer loads");
  expect(loaded.entries.size() == 2 && loaded.entries[0].solution == tile32 &&
             loaded.entries[1].solution == gemmTile && loaded.weightM == 1.0f,
         "loader keeps conv and gemm rows and drops a non-gemm config");

  const uint8_t garbage[] = {1, 2, 3, 4};
  expect(!mlir::hip::loadProblemCacheLut(garbage, sizeof(garbage), loaded,
                                         error),
         "a non-RCAL buffer is rejected");

  if (gFailures != 0) {
    std::fprintf(stderr, "%d nearest-conv-tune checks failed\n", gFailures);
    return 1;
  }
  return 0;
}
