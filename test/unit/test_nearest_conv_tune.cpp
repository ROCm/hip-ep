/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GPU-free checks for the grouped log2 nearest lookup: conv, gemm,
// attention, and gemm-elementwise-gemm.

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

std::string gemmProblem(const char *dtype, int m, int n, int k,
                        const char *transB = "false") {
  return "gfx1151\t20\t1\t-t " + std::string(dtype) +
         " -out_datatype " + dtype + " -transA false -transB " + transB +
         " -transO false -g 1 -m " + std::to_string(m) + " -n " +
         std::to_string(n) + " -k " + std::to_string(k) +
         " -supportsSplitK true";
}

std::string attnProblem(int seqQ, int seqK, const char *causal = "false",
                        int head = 64) {
  return "gfx1151\t20\t1\t-t f16 -transQ false -transK false -transV false "
         "-transO false -causal " +
         std::string(causal) +
         " -return_lse false -split_kv 1 -num_heads_q 8 -num_heads_kv 8 "
         "-g 1 -seq_len_q " +
         std::to_string(seqQ) + " -seq_len_k " + std::to_string(seqK) +
         " -head_dim_qk " + std::to_string(head) + " -head_dim_v " +
         std::to_string(head) +
         " -with-attn-scale false -with-attn-bias false -transBias false "
         "-supportsSplitK true";
}

std::string gegProblem(int m, int n, int k, int gemmO,
                       const char *transC = "false") {
  return "gfx1151\t20\t1\t-t f16 -transA false -transB false -transC " +
         std::string(transC) + " -transO false -g 1 -m " + std::to_string(m) +
         " -n " + std::to_string(n) + " -k " + std::to_string(k) + " -gemmO " +
         std::to_string(gemmO) + " -supportsSplitK true";
}

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

  const std::string gemmNear = "gemm:v4:32,64,16,1,1,4,16,1,2,0,0";
  const std::string gemmFar = "gemm:mPerBlock=32,nPerBlock=64,kPerBlock=16";
  const std::string gemmIllegal = "gemm:v4:128,64,16,1,1,4,16,1,2,0,0";
  const std::string gemmQuery = gemmProblem("f16", 96, 128, 256);
  std::vector<ProblemCacheConvEntry> gemms = {
      {gemmProblem("f16", 80, 128, 256), gemmIllegal},
      {gemmProblem("f16", 64, 128, 256), gemmNear},
      {gemmProblem("f16", 256, 128, 256), gemmFar},
  };
  hit = mlir::hip::findNearestConvTune(gemmQuery, gemms);
  expect(hit.has_value() && !hit->exact && hit->solution == gemmNear,
         "gemm M=96 skips an oversized closer tile for M=64");

  hit = mlir::hip::findNearestConvTune(gemmProblem("f32", 96, 128, 256), gemms);
  expect(!hit.has_value(), "gemm f32 does not borrow an f16 row");

  hit = mlir::hip::findNearestConvTune(gemmProblem("f16", 96, 128, 256, "true"),
                                      gemms);
  expect(!hit.has_value(), "gemm transB is a different group");

  int64_t dimM = 0, dimN = 0, dimK = 0;
  expect(mlir::hip::implicitGemmDims(gemmQuery, dimM, dimN, dimK) &&
             dimM == 96 && dimN == 128 && dimK == 256,
         "gemm problem string reports M, N, and K");

  const std::string attnNear = "attn:v1:64,32,32,1,1,4,16,1,1,0,0";
  const std::string attnIllegal = "attn:v1:256,32,32,1,1,4,16,1,1,0,0";
  const std::string attnQuery = attnProblem(200, 128);
  std::vector<ProblemCacheConvEntry> attns = {
      {attnProblem(180, 128), attnIllegal},
      {attnProblem(128, 128), attnNear},
      {attnProblem(512, 128), "attn:v1:32,32,32,1,1,4,16,1,1,0,0"},
  };
  hit = mlir::hip::findNearestConvTune(attnQuery, attns);
  expect(hit.has_value() && !hit->exact && hit->solution == attnNear,
         "attention seq 200 skips an oversized closer tile for seq 128");

  hit = mlir::hip::findNearestConvTune(attnProblem(200, 128, "true"), attns);
  expect(!hit.has_value(), "attention causal is a different group");

  hit = mlir::hip::findNearestConvTune(attnProblem(200, 128, "false", 128),
                                      attns);
  expect(!hit.has_value(), "attention head dim is a different group");

  hit = mlir::hip::findNearestConvTune(attnQuery, gemms);
  expect(!hit.has_value(), "attention does not borrow a gemm row");

  const std::string gegNear = "attn:v1:32,64,16,1,1,4,16,1,1,0,0";
  const std::string gegIllegal = "attn:v6:32,64,128,16,1,1,4,16,1,1,0,0";
  const std::string gegQuery = gegProblem(96, 128, 64, 64);
  std::vector<ProblemCacheConvEntry> gegs = {
      {gegProblem(80, 128, 64, 64), gegIllegal},
      {gegProblem(64, 128, 64, 64), gegNear},
      {gegProblem(256, 128, 64, 32), "attn:v1:16,64,16,1,1,4,16,1,1,0,0"},
  };
  hit = mlir::hip::findNearestConvTune(gegQuery, gegs);
  expect(hit.has_value() && !hit->exact && hit->solution == gegNear,
         "gemm-elementwise-gemm skips a second-GEMM tile that does not fit");

  hit = mlir::hip::findNearestConvTune(gegProblem(96, 128, 64, 64, "true"),
                                      gegs);
  expect(!hit.has_value(), "gemm-elementwise-gemm transC is a different group");

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
  auto lutOff = fbs::CreateRocmlirAutotuneLut(
      builder, 1, gpuArch, 0, kernelAbi, modelKey, 1.0f, 1.0f, 1.0f,
      builder.CreateVector(configs), builder.CreateVector(points),
      builder.CreateVector(fallbacks));
  fbs::FinishRocmlirAutotuneLutBuffer(builder, lutOff);

  mlir::hip::ProblemCacheLut loaded;
  std::string error;
  expect(mlir::hip::loadProblemCacheLut(builder.GetBufferPointer(),
                                        builder.GetSize(), loaded, error),
         "RCAL buffer loads");
  expect(loaded.entries.size() == 2 && loaded.entries[0].solution == tile32 &&
             loaded.entries[1].solution == gemmTile && loaded.weightM == 1.0f &&
             loaded.gpuArch == "gfx1151",
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
