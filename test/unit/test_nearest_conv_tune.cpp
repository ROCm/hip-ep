/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GPU-free checks for the grouped log2 nearest lookup: conv, gemm,
// attention, and gemm-elementwise-gemm.

#include "nearest_conv_tune.h"
#include "problem_cache_generated.h"

#include "flatbuffers/flatbuffers.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <system_error>
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
  return "gfx1151\t20\t1\t-t " + std::string(dtype) + " -out_datatype " +
         dtype + " -transA false -transB " + transB +
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
  const std::string spatial96 =
      key("conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 96 -W 96 -k 32 "
          "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1");
  const std::string spatial128 =
      key("conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 128 -W 128 -k 32 "
          "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1");
  const std::string padded =
      key("conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
          "-y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1");
  const std::string fp16 = key(
      "convfp16 -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
      "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1");
  const std::string tiny =
      key("conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 4 -W 2 -k 2 "
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

  const std::string query64 =
      key("conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
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

  const std::string grouped =
      key("conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
          "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4");
  int64_t groupedM = 0, groupedN = 0, groupedK = 0;
  expect(mlir::hip::implicitGemmDims(grouped, groupedM, groupedN, groupedK) &&
             groupedM == 8,
         "grouped conv M is K divided by the group count");
  hit = mlir::hip::findNearestConvTune(grouped, table);
  expect(!hit.has_value(), "grouped conv M=8 rejects mPerBlock=32");

  const std::string positionalOversized = "gemm:v4:32,32,4,1,1,4,16,1,2,0,0";
  std::vector<ProblemCacheConvEntry> positional = {
      {spatial96, positionalOversized},
  };
  hit = mlir::hip::findNearestConvTune(tiny, positional);
  expect(!hit.has_value(), "a positional conv tile larger than M is rejected");

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

  const std::string stride2 =
      key("conv -F 1 -f NGC01 -I N01GC -O N01GC -n 1 -c 16 -H 64 -W 64 -k 32 "
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

  hit =
      mlir::hip::findNearestConvTune(gegProblem(96, 128, 64, 64, "true"), gegs);
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

  {
    // An infinite or NaN weight in the buffer makes the distance NaN for a row
    // that shares the axis, so the loader refuses it like a non-positive one.
    const auto withWeights = [&](float m, float n, float k) {
      flatbuffers::FlatBufferBuilder fb;
      auto arch = fb.CreateString("gfx1151");
      auto abi = fb.CreateString("rocmlir-conv-v1");
      auto model = fb.CreateString("");
      std::vector<flatbuffers::Offset<fbs::RocmlirTuneConfig>> noConfigs;
      std::vector<flatbuffers::Offset<fbs::RocmlirTunePoint>> noPoints;
      std::vector<flatbuffers::Offset<fbs::RocmlirTuneFallback>> noFallbacks;
      auto root = fbs::CreateRocmlirAutotuneLut(
          fb, 1, arch, 0, abi, model, m, n, k, fb.CreateVector(noConfigs),
          fb.CreateVector(noPoints), fb.CreateVector(noFallbacks));
      fbs::FinishRocmlirAutotuneLutBuffer(fb, root);
      mlir::hip::ProblemCacheLut lut;
      std::string weightError;
      return mlir::hip::loadProblemCacheLut(fb.GetBufferPointer(), fb.GetSize(),
                                            lut, weightError);
    };
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    expect(withWeights(1.0f, 1.0f, 1.0f), "finite weights load");
    expect(!withWeights(inf, 1.0f, 1.0f), "an infinite weight is rejected");
    expect(!withWeights(1.0f, nan, 1.0f), "a NaN weight is rejected");
    expect(!withWeights(1.0f, 1.0f, 0.0f), "a zero weight is rejected");
  }

  const uint8_t garbage[] = {1, 2, 3, 4};
  expect(
      !mlir::hip::loadProblemCacheLut(garbage, sizeof(garbage), loaded, error),
      "a non-RCAL buffer is rejected");

  // The external table is the LUT JSON that --enable-problem-cache-rw reads
  // and writes. Build the same document as the RCAL buffer above, in that
  // file's own shape (points carry m/n/k, which the loader ignores).
  const auto jsonText = [&](llvm::json::Object root) {
    std::string text;
    llvm::raw_string_ostream os(text);
    os << llvm::json::Value(std::move(root));
    return text;
  };
  const auto dumpDoc = [&] {
    llvm::json::Array jsonConfigs;
    for (const char *perf :
         {tile32.c_str(), "{\"algo\":\"block\"}", gemmTile.c_str()})
      jsonConfigs.push_back(llvm::json::Object{{"perf_config", perf}});
    llvm::json::Array jsonPoints;
    jsonPoints.push_back(llvm::json::Object{
        {"problem", exact}, {"config", 0}, {"m", 32}, {"n", 4096}, {"k", 144}});
    jsonPoints.push_back(llvm::json::Object{{"problem", "{\"lens\":[1,2]}"},
                                            {"config", 1},
                                            {"m", 0},
                                            {"n", 0},
                                            {"k", 0}});
    jsonPoints.push_back(llvm::json::Object{
        {"problem", gemmKey}, {"config", 2}, {"m", 0}, {"n", 0}, {"k", 0}});
    return llvm::json::Object{{"schema_version", 1},
                              {"gpu_arch", "gfx1151"},
                              {"rocm_version", 0},
                              {"kernel_abi", "rocmlir-conv-v1"},
                              {"model_key", ""},
                              {"weight_m", 1.0},
                              {"weight_n", 1.0},
                              {"weight_k", 1.0},
                              {"configs", std::move(jsonConfigs)},
                              {"points", std::move(jsonPoints)},
                              {"fallbacks", llvm::json::Array()}};
  };

  // `loaded` was reset by the rejected buffer above, so reload the RCAL table
  // to compare against.
  mlir::hip::ProblemCacheLut fromRcal;
  expect(mlir::hip::loadProblemCacheLut(builder.GetBufferPointer(),
                                        builder.GetSize(), fromRcal, error) &&
             fromRcal.entries.size() == 2,
         "the RCAL buffer reloads for the parity check");
  mlir::hip::ProblemCacheLut fromJson;
  expect(
      mlir::hip::loadProblemCacheLutJson(jsonText(dumpDoc()), fromJson, error),
      "a dumped LUT JSON loads");
  expect(fromJson.entries.size() == fromRcal.entries.size() &&
             fromJson.entries.size() == 2 &&
             fromJson.entries[0].problem == fromRcal.entries[0].problem &&
             fromJson.entries[0].solution == fromRcal.entries[0].solution &&
             fromJson.entries[1].problem == fromRcal.entries[1].problem &&
             fromJson.entries[1].solution == fromRcal.entries[1].solution &&
             fromJson.gpuArch == fromRcal.gpuArch &&
             fromJson.weightM == fromRcal.weightM &&
             fromJson.weightN == fromRcal.weightN &&
             fromJson.weightK == fromRcal.weightK,
         "the JSON loader matches the RCAL loader on the same table");

  {
    llvm::json::Object doc = dumpDoc();
    doc["weight_m"] = 2.0;
    doc["weight_n"] = 0.5;
    mlir::hip::ProblemCacheLut weighted;
    expect(mlir::hip::loadProblemCacheLutJson(jsonText(std::move(doc)),
                                              weighted, error) &&
               weighted.weightM == 2.0f && weighted.weightN == 0.5f &&
               weighted.weightK == 1.0f,
           "JSON weights are read, and an absent one stays 1");
  }
  {
    llvm::json::Object doc = dumpDoc();
    doc.erase("schema_version");
    doc.erase("weight_m");
    doc.erase("weight_n");
    doc.erase("weight_k");
    doc.erase("gpu_arch");
    mlir::hip::ProblemCacheLut defaulted;
    expect(mlir::hip::loadProblemCacheLutJson(jsonText(std::move(doc)),
                                              defaulted, error) &&
               defaulted.weightM == 1.0f && defaulted.gpuArch.empty() &&
               defaulted.entries.size() == 2,
           "absent schema_version, weights, and gpu_arch take the defaults");
  }
  {
    // A point naming a config that does not exist is skipped, not an error,
    // as in the RCAL loader.
    llvm::json::Object doc = dumpDoc();
    llvm::json::Array &pts = *doc.getArray("points");
    pts.push_back(
        llvm::json::Object{{"problem", "out-of-range"}, {"config", 99}});
    pts.push_back(llvm::json::Object{{"problem", "negative"}, {"config", -1}});
    mlir::hip::ProblemCacheLut skipped;
    expect(mlir::hip::loadProblemCacheLutJson(jsonText(std::move(doc)), skipped,
                                              error) &&
               skipped.entries.size() == 2,
           "points with a bad config index are skipped");
  }

  const auto rejects = [&](llvm::json::Object doc, const char *message) {
    mlir::hip::ProblemCacheLut bad;
    expect(!mlir::hip::loadProblemCacheLutJson(jsonText(std::move(doc)), bad,
                                               error) &&
               bad.entries.empty(),
           message);
  };
  {
    llvm::json::Object doc = dumpDoc();
    doc["schema_version"] = 2;
    rejects(std::move(doc), "an unsupported JSON schema_version is rejected");
  }
  {
    llvm::json::Object doc = dumpDoc();
    doc["schema_version"] = "1";
    rejects(std::move(doc), "a non-integer schema_version is rejected");
  }
  {
    llvm::json::Object doc = dumpDoc();
    doc["kernel_abi"] = "other-abi";
    rejects(std::move(doc), "a different kernel_abi is rejected");
  }
  {
    llvm::json::Object doc = dumpDoc();
    doc.erase("kernel_abi");
    rejects(std::move(doc), "a missing kernel_abi is rejected");
  }
  {
    llvm::json::Object doc = dumpDoc();
    doc["weight_k"] = 0.0;
    rejects(std::move(doc), "a non-positive JSON weight is rejected");
  }
  {
    llvm::json::Object doc = dumpDoc();
    doc["weight_n"] = "heavy";
    rejects(std::move(doc), "a non-numeric JSON weight is rejected");
  }
  {
    // Positive as a double, zero as the float it is stored in. The FlatBuffer
    // loader rejects a zero weight, so this one must be rejected too.
    llvm::json::Object doc = dumpDoc();
    doc["weight_n"] = 1e-50;
    rejects(std::move(doc), "a JSON weight that rounds to zero is rejected");
  }
  {
    // Outside the finite range of a float: converting it is undefined, and an
    // infinite weight breaks the distance.
    llvm::json::Object doc = dumpDoc();
    doc["weight_k"] = 1e40;
    rejects(std::move(doc), "a JSON weight above the float range is rejected");
  }
  {
    // 1e50 is finite as a double and narrows to infinity as a float.
    llvm::json::Object doc = dumpDoc();
    doc["weight_m"] = 1e50;
    rejects(std::move(doc), "a JSON weight that overflows a float is rejected");
  }
  {
    // The largest float is still a finite weight, and one more step is not.
    llvm::json::Object doc = dumpDoc();
    doc["weight_n"] = static_cast<double>(std::numeric_limits<float>::max());
    mlir::hip::ProblemCacheLut widest;
    expect(mlir::hip::loadProblemCacheLutJson(jsonText(std::move(doc)), widest,
                                              error) &&
               std::isfinite(widest.weightN),
           "the largest float is accepted as a JSON weight");
  }
  {
    llvm::json::Object doc = dumpDoc();
    doc["weight_m"] = 3e38;
    mlir::hip::ProblemCacheLut large;
    expect(mlir::hip::loadProblemCacheLutJson(jsonText(std::move(doc)), large,
                                              error) &&
               std::isfinite(large.weightM) && large.weightM > 0.0f,
           "a large JSON weight that still fits a float is accepted");
  }
  {
    llvm::json::Object doc = dumpDoc();
    doc["weight_m"] = 1e-30;
    mlir::hip::ProblemCacheLut small;
    expect(mlir::hip::loadProblemCacheLutJson(jsonText(std::move(doc)), small,
                                              error) &&
               small.weightM > 0.0f,
           "a small JSON weight that still fits a float is accepted");
  }
  {
    mlir::hip::ProblemCacheLut bad;
    expect(!mlir::hip::loadProblemCacheLutJson("", bad, error),
           "empty JSON text is rejected");
    expect(!mlir::hip::loadProblemCacheLutJson("{not json", bad, error),
           "malformed JSON text is rejected");
    expect(!mlir::hip::loadProblemCacheLutJson("[1, 2]", bad, error),
           "a JSON array at the top level is rejected");
  }

  // The path overload. A real dump round-trips, and a FlatBuffer written to the
  // same path is no longer an accepted external file.
  {
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::string stem = "test_nearest_conv_tune_" +
                             std::to_string(reinterpret_cast<uintptr_t>(&dir));
    const std::string jsonPath = (dir / (stem + ".json")).string();
    const std::string fbPath = (dir / (stem + ".fb")).string();
    {
      std::ofstream out(jsonPath, std::ios::binary);
      out << jsonText(dumpDoc());
    }
    {
      std::ofstream out(fbPath, std::ios::binary);
      out.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
                static_cast<std::streamsize>(builder.GetSize()));
    }
    mlir::hip::ProblemCacheLut fromFile;
    expect(mlir::hip::loadProblemCacheLutJsonFile(jsonPath, fromFile, error) &&
               fromFile.entries.size() == 2,
           "the JSON path overload loads a dumped file");
    expect(!mlir::hip::loadProblemCacheLutJsonFile(fbPath, fromFile, error) &&
               error.find(fbPath) != std::string::npos,
           "a .fb file is rejected as an external problem cache");
    expect(!mlir::hip::loadProblemCacheLutJsonFile(
               dir.string() + "/no-such-" + stem + ".json", fromFile, error) &&
               fromFile.entries.empty(),
           "a missing external problem cache reports an error");
    std::error_code ignored;
    std::filesystem::remove(jsonPath, ignored);
    std::filesystem::remove(fbPath, ignored);
  }

  // Rewriting the external file: merge winners into the LUT JSON, keep what is
  // there, and replace the file without losing the old one on a failure.
  using mlir::hip::ProblemCacheWinner;
  const std::string tile16 = "gemm:mPerBlock=16,nPerBlock=16,kPerBlock=8";
  const std::string tileBig = "gemm:mPerBlock=128,nPerBlock=64,kPerBlock=8";
  const auto gemmAt = [](int m) {
    return "gfx1151\t2\t1\t-t f32 -m " + std::to_string(m) + " -n 1000 -k 1280";
  };
  const auto parseDoc = [](const std::string &text) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return llvm::json::Value(nullptr);
    }
    return std::move(*parsed);
  };
  // problem -> perf_config for every point of a LUT JSON document.
  const auto pointMap = [&](const std::string &text) {
    std::map<std::string, std::string> rows;
    llvm::json::Value doc = parseDoc(text);
    const llvm::json::Object *root = doc.getAsObject();
    if (!root || !root->getArray("points") || !root->getArray("configs"))
      return rows;
    const llvm::json::Array &cfgs = *root->getArray("configs");
    for (const llvm::json::Value &p : *root->getArray("points")) {
      const llvm::json::Object *o = p.getAsObject();
      auto problem = o->getString("problem");
      auto index = o->getInteger("config");
      if (problem && index && *index >= 0 &&
          static_cast<size_t>(*index) < cfgs.size())
        rows[problem->str()] = cfgs[static_cast<size_t>(*index)]
                                   .getAsObject()
                                   ->getString("perf_config")
                                   .value_or("")
                                   .str();
    }
    return rows;
  };

  {
    // A new document for the device, with the dump's own fields.
    std::string merged;
    expect(mlir::hip::mergeProblemCacheJson(
               "", "gfx1151", {{exact, tile32}, {gemmAt(8), tile16}}, merged,
               error),
           "merging into nothing starts a new table");
    llvm::json::Value doc = parseDoc(merged);
    const llvm::json::Object *root = doc.getAsObject();
    expect(root != nullptr && root->getString("gpu_arch") == "gfx1151" &&
               root->getString("kernel_abi") == "rocmlir-conv-v1" &&
               root->getInteger("schema_version") == 1 &&
               root->getArray("fallbacks") != nullptr,
           "a new table carries the LUT header fields");
    mlir::hip::ProblemCacheLut reread;
    expect(mlir::hip::loadProblemCacheLutJson(merged, reread, error) &&
               reread.entries.size() == 2 && reread.gpuArch == "gfx1151",
           "the merged table loads with the same loader");
    int64_t dm = 0, dn = 0, dk = 0;
    mlir::hip::implicitGemmDims(exact, dm, dn, dk);
    bool dimsMatch = false;
    if (root && root->getArray("points"))
      for (const llvm::json::Value &p : *root->getArray("points")) {
        const llvm::json::Object *o = p.getAsObject();
        if (o->getString("problem") == exact)
          dimsMatch = o->getInteger("m") == dm && o->getInteger("n") == dn &&
                      o->getInteger("k") == dk;
      }
    expect(dimsMatch, "a winner is stored with its implicit GEMM dims");

    std::string again;
    expect(mlir::hip::mergeProblemCacheJson(
               merged, "gfx1151", {{exact, tile32}, {gemmAt(8), tile16}}, again,
               error) &&
               again == merged,
           "merging the same winners again changes nothing");
  }
  {
    // Existing data is kept: a point the loader would skip, other top-level
    // fields, and points the winners do not touch.
    llvm::json::Object doc = dumpDoc();
    doc["weight_m"] = 2.0;
    doc["model_key"] = "keep-me";
    const std::string base = jsonText(std::move(doc));

    std::string merged;
    expect(mlir::hip::mergeProblemCacheJson(
               base, "gfx1151",
               {{exact, tileBig}, {gemmAt(24), tile32}, {gemmAt(48), tile32}},
               merged, error),
           "merging into an existing table works");
    const auto rows = pointMap(merged);
    expect(rows.size() == 5, "winners add two new problems to the three kept");
    expect(rows.count(exact) && rows.at(exact) == tileBig,
           "a winner replaces the perfConfig of the same problem");
    expect(rows.count(gemmKey) && rows.at(gemmKey) == gemmTile,
           "an untouched point is kept");
    expect(rows.count("{\"lens\":[1,2]}") &&
               rows.at("{\"lens\":[1,2]}") == "{\"algo\":\"block\"}",
           "a point the loader skips is not deleted by a rewrite");
    expect(rows.count(gemmAt(24)) && rows.at(gemmAt(24)) == tile32,
           "a new winner is added");
    llvm::json::Value out = parseDoc(merged);
    const llvm::json::Object *root = out.getAsObject();
    expect(root && root->getString("model_key") == "keep-me" &&
               root->getNumber("weight_m") == 2.0,
           "weights and model_key are carried over");
    // tile32 is used by two new winners and tileBig by one: still one entry
    // each.
    size_t tile32Configs = 0;
    if (root && root->getArray("configs"))
      for (const llvm::json::Value &c : *root->getArray("configs"))
        if (c.getAsObject()->getString("perf_config") == tile32)
          ++tile32Configs;
    expect(tile32Configs == 1, "a perfConfig shared by points is stored once");
    // Points are written sorted by problem.
    std::vector<std::string> order;
    if (root && root->getArray("points"))
      for (const llvm::json::Value &p : *root->getArray("points"))
        order.push_back(
            p.getAsObject()->getString("problem").value_or("").str());
    expect(std::is_sorted(order.begin(), order.end()),
           "points are written sorted by problem");
  }
  {
    std::string merged;
    expect(!mlir::hip::mergeProblemCacheJson(jsonText(dumpDoc()), "gfx942",
                                             {{exact, tile32}}, merged,
                                             error) &&
               merged.empty() && error.find("gfx1151") != std::string::npos,
           "a table for another gpu_arch is not rewritten");
    expect(!mlir::hip::mergeProblemCacheJson(
               "{not json", "gfx1151", {{exact, tile32}}, merged, error) &&
               merged.empty(),
           "an unparsable existing table is not replaced");
    llvm::json::Object bad = dumpDoc();
    bad["kernel_abi"] = "other-abi";
    expect(!mlir::hip::mergeProblemCacheJson(jsonText(std::move(bad)),
                                             "gfx1151", {{exact, tile32}},
                                             merged, error),
           "an unsupported existing table is not replaced");
    expect(mlir::hip::mergeProblemCacheJson("  \n", "gfx1151",
                                            {{exact, tile32}}, merged, error) &&
               pointMap(merged).size() == 1,
           "a blank existing file is treated as a new table");
    expect(mlir::hip::mergeProblemCacheJson(
               "", "gfx1151", {{"", tile32}, {exact, ""}}, merged, error) &&
               pointMap(merged).empty(),
           "a winner with no problem or no perfConfig is ignored");
  }
  {
    // A rewrite replaces the file, so a document it cannot represent is refused
    // whole. Nothing is dropped from the result, and `configs` and `points` are
    // never rebuilt from the winners alone.
    const auto refused = [&](llvm::json::Object doc, const char *needle,
                             const char *what) {
      std::string merged;
      std::string why;
      expect(!mlir::hip::mergeProblemCacheJson(jsonText(std::move(doc)),
                                               "gfx1151", {{exact, tile32}},
                                               merged, why) &&
                 merged.empty() && why.find(needle) != std::string::npos &&
                 why.find("refusing to rewrite") != std::string::npos,
             what);
    };
    const auto withPoints = [&](llvm::json::Value point) {
      llvm::json::Object doc = dumpDoc();
      llvm::json::Array points;
      points.push_back(std::move(point));
      doc["points"] = std::move(points);
      return doc;
    };
    const auto withConfigs = [&](llvm::json::Value config) {
      llvm::json::Object doc = dumpDoc();
      llvm::json::Array configs;
      configs.push_back(std::move(config));
      doc["configs"] = std::move(configs);
      return doc;
    };
    const auto withFallbacks = [&](llvm::json::Value fallback) {
      llvm::json::Object doc = dumpDoc();
      llvm::json::Array fallbacks;
      fallbacks.push_back(std::move(fallback));
      doc["fallbacks"] = std::move(fallbacks);
      return doc;
    };

    llvm::json::Object noPoints = dumpDoc();
    noPoints.erase("points");
    refused(std::move(noPoints), "points",
            "a table with no points array is refused");
    llvm::json::Object noConfigs = dumpDoc();
    noConfigs.erase("configs");
    refused(std::move(noConfigs), "configs",
            "a table with no configs array is refused");
    llvm::json::Object pointsNotArray = dumpDoc();
    pointsNotArray["points"] = "not an array";
    refused(std::move(pointsNotArray), "points",
            "a table whose points are not an array is refused");
    llvm::json::Object configsNotArray = dumpDoc();
    configsNotArray["configs"] = 5;
    refused(std::move(configsNotArray), "configs",
            "a table whose configs are not an array is refused");

    refused(withConfigs(5), "configs[0]",
            "a config that is not an object is refused");
    refused(withConfigs(llvm::json::Object{{"other", "x"}}), "configs[0]",
            "a config with no perf_config is refused");
    refused(withConfigs(llvm::json::Object{{"perf_config", 3}}), "configs[0]",
            "a config whose perf_config is not a string is refused");

    refused(withPoints(5), "points[0]",
            "a point that is not an object is refused");
    refused(withPoints(llvm::json::Object{{"config", 0}}), "points[0]",
            "a point with no problem is refused");
    refused(withPoints(llvm::json::Object{{"problem", exact}}), "points[0]",
            "a point with no config is refused");
    refused(withPoints(llvm::json::Object{{"problem", exact}, {"config", "0"}}),
            "points[0]", "a point whose config is not an integer is refused");
    refused(withPoints(llvm::json::Object{{"problem", exact}, {"config", -1}}),
            "points[0]", "a point with a negative config is refused");
    refused(withPoints(llvm::json::Object{{"problem", exact}, {"config", 99}}),
            "points[0]", "a point whose config is out of range is refused");
    refused(
        withPoints(llvm::json::Object{
            {"problem", exact}, {"config", 0}, {"m", "x"}, {"n", 1}, {"k", 2}}),
        "points[0]", "a point with a non-integer m is refused");

    llvm::json::Object fallbacksNotArray = dumpDoc();
    fallbacksNotArray["fallbacks"] = "not an array";
    refused(std::move(fallbacksNotArray), "fallbacks",
            "fallbacks that are not an array are refused");
    refused(withFallbacks(5), "fallbacks[0]",
            "a fallback that is not an object is refused");
    refused(withFallbacks(llvm::json::Object{{"kind", "x"}}), "fallbacks[0]",
            "a fallback with no config is refused");
    refused(withFallbacks(llvm::json::Object{{"kind", "x"}, {"config", 99}}),
            "fallbacks[0]",
            "a fallback whose config is out of range is refused");
  }
  {
    // Fallbacks index into `configs`, which a rewrite renumbers. A config that
    // only a fallback uses stays, and every fallback is renumbered with the
    // points, so it still names the same perfConfig.
    const std::string fallbackOnly =
        "gemm:mPerBlock=16,nPerBlock=16,kPerBlock=4";
    llvm::json::Object doc = dumpDoc();
    llvm::json::Array configs;
    for (const std::string &perf : {fallbackOnly, tile32, gemmTile})
      configs.push_back(llvm::json::Object{{"perf_config", perf}});
    llvm::json::Array points;
    points.push_back(llvm::json::Object{{"problem", exact}, {"config", 1}});
    points.push_back(llvm::json::Object{{"problem", gemmKey}, {"config", 2}});
    llvm::json::Array fallbacks;
    fallbacks.push_back(llvm::json::Object{
        {"kind", "prefill"}, {"config", 0}, {"note", "keep"}});
    fallbacks.push_back(llvm::json::Object{{"kind", "decode"}, {"config", 1}});
    fallbacks.push_back(llvm::json::Object{{"kind", "shared"}, {"config", 2}});
    doc["configs"] = std::move(configs);
    doc["points"] = std::move(points);
    doc["fallbacks"] = std::move(fallbacks);
    const std::string base = jsonText(std::move(doc));

    // The winner moves `exact` off tile32, so only a fallback uses tile32 now,
    // and config 0 was never used by a point.
    std::string merged;
    expect(mlir::hip::mergeProblemCacheJson(base, "gfx1151", {{exact, tileBig}},
                                            merged, error),
           "a table with fallbacks is rewritten");
    llvm::json::Value out = parseDoc(merged);
    const llvm::json::Object *root = out.getAsObject();
    const llvm::json::Array *outConfigs =
        root != nullptr ? root->getArray("configs") : nullptr;
    const llvm::json::Array *outFallbacks =
        root != nullptr ? root->getArray("fallbacks") : nullptr;
    expect(outConfigs != nullptr && outFallbacks != nullptr &&
               outFallbacks->size() == 3,
           "the rewritten table keeps all three fallbacks");
    std::vector<std::string> fallbackPerf;
    std::vector<std::string> fallbackKind;
    bool inRange = true;
    if (outConfigs != nullptr && outFallbacks != nullptr)
      for (const llvm::json::Value &value : *outFallbacks) {
        const llvm::json::Object *record = value.getAsObject();
        std::optional<int64_t> index =
            record != nullptr ? record->getInteger("config") : std::nullopt;
        if (!index || *index < 0 ||
            static_cast<size_t>(*index) >= outConfigs->size()) {
          inRange = false;
          continue;
        }
        fallbackPerf.push_back((*outConfigs)[static_cast<size_t>(*index)]
                                   .getAsObject()
                                   ->getString("perf_config")
                                   .value_or("")
                                   .str());
        fallbackKind.push_back(record->getString("kind").value_or("").str());
      }
    expect(inRange, "every fallback still indexes a config");
    expect(
        fallbackPerf ==
                std::vector<std::string>({fallbackOnly, tile32, gemmTile}) &&
            fallbackKind ==
                std::vector<std::string>({"prefill", "decode", "shared"}),
        "each fallback keeps its perfConfig and its place after renumbering");
    expect(outFallbacks != nullptr && outFallbacks->size() == 3 &&
               (*outFallbacks)[0].getAsObject()->getString("note") == "keep",
           "fields of a fallback other than config are kept");
    const auto rows = pointMap(merged);
    expect(rows.size() == 2 && rows.at(exact) == tileBig &&
               rows.at(gemmKey) == gemmTile,
           "points keep their perfConfigs when the configs are renumbered");
    std::set<std::string> distinct;
    if (outConfigs != nullptr)
      for (const llvm::json::Value &config : *outConfigs)
        distinct.insert(
            config.getAsObject()->getString("perf_config").value_or("").str());
    expect(outConfigs != nullptr && distinct.size() == outConfigs->size() &&
               distinct.count(gemmTile) == 1,
           "a perfConfig used by a point and a fallback is stored once");

    std::string again;
    expect(mlir::hip::mergeProblemCacheJson(merged, "gfx1151",
                                            {{exact, tileBig}}, again, error) &&
               again == merged,
           "rewriting a table with fallbacks again changes nothing");

    // Without a `fallbacks` key the rewrite does not add one.
    llvm::json::Object plain = dumpDoc();
    plain.erase("fallbacks");
    expect(mlir::hip::mergeProblemCacheJson(jsonText(std::move(plain)),
                                            "gfx1151", {{gemmAt(8), tile16}},
                                            merged, error) &&
               parseDoc(merged).getAsObject() != nullptr &&
               parseDoc(merged).getAsObject()->get("fallbacks") == nullptr &&
               pointMap(merged).size() == 4,
           "a table with no fallbacks key is rewritten without adding one");
  }
  {
    // A problem can have several rows. A lookup returns the first one it can
    // use, so a later row is the alternative when the first is filtered out
    // (a perfConfig the loader skips) or rejected (it failed to compile). A
    // rewrite keeps every row of a problem the winners do not touch, in file
    // order, so it never turns a usable lookup into a miss.
    const std::string skipped = "{\"algo\":\"block\"}";
    const auto build =
        [&](const std::vector<std::pair<std::string, std::string>> &rows) {
          llvm::json::Object doc = dumpDoc();
          llvm::json::Array configs;
          std::map<std::string, int64_t> index;
          llvm::json::Array points;
          for (const auto &[problem, perf] : rows) {
            auto found = index.find(perf);
            if (found == index.end()) {
              found =
                  index.emplace(perf, static_cast<int64_t>(index.size())).first;
              configs.push_back(llvm::json::Object{{"perf_config", perf}});
            }
            points.push_back(llvm::json::Object{{"problem", problem},
                                                {"config", found->second}});
          }
          doc["configs"] = std::move(configs);
          doc["points"] = std::move(points);
          return jsonText(std::move(doc));
        };
    // Every row in file order, repeats included.
    const auto rowsOf = [&](const std::string &text) {
      std::vector<std::pair<std::string, std::string>> rows;
      llvm::json::Value doc = parseDoc(text);
      const llvm::json::Object *root = doc.getAsObject();
      if (!root || !root->getArray("points") || !root->getArray("configs"))
        return rows;
      const llvm::json::Array &cfgs = *root->getArray("configs");
      for (const llvm::json::Value &p : *root->getArray("points")) {
        const llvm::json::Object *o = p.getAsObject();
        auto problem = o->getString("problem");
        auto index = o->getInteger("config");
        if (problem && index && *index >= 0 &&
            static_cast<size_t>(*index) < cfgs.size())
          rows.emplace_back(problem->str(), cfgs[static_cast<size_t>(*index)]
                                                .getAsObject()
                                                ->getString("perf_config")
                                                .value_or("")
                                                .str());
      }
      return rows;
    };
    using Rows = std::vector<std::pair<std::string, std::string>>;
    const auto lookup = [&](const std::string &text,
                            const std::vector<std::string> &rejected) {
      mlir::hip::ProblemCacheLut lut;
      std::string loadError;
      std::string solution;
      if (mlir::hip::loadProblemCacheLutJson(text, lut, loadError))
        if (auto hit =
                mlir::hip::findNearestConvTune(exact, lut.entries, rejected))
          solution = hit->solution;
      return solution;
    };

    // The first row is one the loader skips, so the lookup uses the second.
    const std::string filteredFirst =
        build({{exact, skipped}, {exact, tile32}});
    std::string merged;
    expect(mlir::hip::mergeProblemCacheJson(filteredFirst, "gfx1151", {},
                                            merged, error),
           "a table with a repeated problem is rewritten");
    expect(rowsOf(merged) == Rows({{exact, skipped}, {exact, tile32}}),
           "a repeated problem keeps both rows, in file order");
    expect(lookup(filteredFirst, {}) == tile32 && lookup(merged, {}) == tile32,
           "a filtered first row still falls through to the second after a "
           "rewrite");

    // The first row compiles badly, so the caller rejects it and takes the
    // next. Both rows have to survive for that to work after a rewrite.
    const std::string rejectedFirst =
        build({{exact, tile32}, {exact, tileBig}});
    expect(mlir::hip::mergeProblemCacheJson(rejectedFirst, "gfx1151", {},
                                            merged, error) &&
               rowsOf(merged) == Rows({{exact, tile32}, {exact, tileBig}}),
           "a repeated problem with two usable rows keeps both");
    expect(lookup(merged, {}) == tile32 &&
               lookup(merged, {tile32}) == tileBig &&
               lookup(rejectedFirst, {tile32}) == tileBig,
           "a rejected first row still yields the second after a rewrite");
    std::string again;
    expect(
        mlir::hip::mergeProblemCacheJson(merged, "gfx1151", {}, again, error) &&
            again == merged,
        "rewriting a table with a repeated problem again changes nothing");

    // Winners for other problems leave the repeated problem alone.
    expect(mlir::hip::mergeProblemCacheJson(
               rejectedFirst, "gfx1151", {{gemmAt(8), tile16}}, merged, error),
           "a winner for another problem is merged");
    Rows keptForExact;
    Rows keptForOther;
    for (const auto &row : rowsOf(merged))
      (row.first == exact ? keptForExact : keptForOther).push_back(row);
    expect(keptForExact == Rows({{exact, tile32}, {exact, tileBig}}) &&
               keptForOther == Rows({{gemmAt(8), tile16}}) &&
               lookup(merged, {tile32}) == tileBig,
           "a winner for another problem keeps the alternatives");

    // A winner is a fresh measurement of its own problem, so it replaces all
    // of that problem's rows rather than piling up beside them.
    expect(mlir::hip::mergeProblemCacheJson(rejectedFirst, "gfx1151",
                                            {{exact, tile16}}, merged, error) &&
               rowsOf(merged) == Rows({{exact, tile16}}),
           "a winner replaces every row of its problem");
  }
  {
    // The file overload across compiles: create, accumulate, replace, and keep
    // the old file when the rewrite is refused.
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        ("test_nearest_conv_tune_rw_" +
         std::to_string(reinterpret_cast<uintptr_t>(&error)));
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    const std::string file = (dir / "nested" / "cache.json").string();

    expect(mlir::hip::updateProblemCacheJsonFile(file, "gfx1151",
                                                 {{exact, tile32}}, error),
           "the first compile creates the file and its directory");
    expect(mlir::hip::updateProblemCacheJsonFile(
               file, "gfx1151", {{gemmAt(8), tile16}, {gemmAt(16), tile16}},
               error),
           "a second compile updates the same file");
    expect(mlir::hip::updateProblemCacheJsonFile(file, "gfx1151",
                                                 {{exact, tileBig}}, error),
           "a third compile replaces one of its entries");

    mlir::hip::ProblemCacheLut reloaded;
    expect(mlir::hip::loadProblemCacheLutJsonFile(file, reloaded, error) &&
               reloaded.entries.size() == 3 && reloaded.gpuArch == "gfx1151",
           "the rewritten file loads and holds every compile's winners");
    std::string text;
    {
      std::ifstream in(file, std::ios::binary);
      text.assign(std::istreambuf_iterator<char>(in),
                  std::istreambuf_iterator<char>());
    }
    const auto rows = pointMap(text);
    expect(rows.size() == 3 && rows.at(exact) == tileBig &&
               rows.at(gemmAt(8)) == tile16 && rows.at(gemmAt(16)) == tile16,
           "later compiles keep the earlier entries and replace by problem");

    size_t leftovers = 0;
    for (const auto &entry :
         std::filesystem::directory_iterator(dir / "nested"))
      if (entry.path().filename() != "cache.json")
        ++leftovers;
    expect(leftovers == 0, "no temporary file is left beside the cache");

    // A table for another device is refused and the file is left as it was.
    expect(!mlir::hip::updateProblemCacheJsonFile(
               file, "gfx942", {{gemmAt(99), tile32}}, error),
           "a file for another gpu_arch is not rewritten");
    std::string after;
    {
      std::ifstream in(file, std::ios::binary);
      after.assign(std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>());
    }
    expect(after == text, "a refused rewrite leaves the file untouched");

    // An existing file that is not a LUT is not clobbered.
    const std::string junk = (dir / "junk.json").string();
    {
      std::ofstream out(junk, std::ios::binary);
      out << "this is not json";
    }
    expect(!mlir::hip::updateProblemCacheJsonFile(junk, "gfx1151",
                                                  {{exact, tile32}}, error) &&
               error.find(junk) != std::string::npos,
           "a file that is not a LUT is not overwritten");
    {
      std::ifstream in(junk, std::ios::binary);
      std::string kept((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
      expect(kept == "this is not json", "the rejected file keeps its text");
    }

    // A table whose `points` array is missing is refused whole. Without the
    // check the rewrite would write the winners alone over it.
    const std::string partial = (dir / "partial.json").string();
    llvm::json::Object partialDoc = dumpDoc();
    partialDoc.erase("points");
    const std::string partialText = jsonText(std::move(partialDoc));
    {
      std::ofstream out(partial, std::ios::binary);
      out << partialText;
    }
    expect(!mlir::hip::updateProblemCacheJsonFile(partial, "gfx1151",
                                                  {{exact, tile32}}, error) &&
               error.find(partial) != std::string::npos &&
               error.find("points") != std::string::npos,
           "a table with no points array is not rewritten");
    {
      std::ifstream in(partial, std::ios::binary);
      std::string kept((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
      expect(kept == partialText, "the refused table keeps its bytes");
    }
    size_t partialTemps = 0;
    for (const auto &entry : std::filesystem::directory_iterator(dir))
      if (entry.path().filename().string().rfind("partial.json.", 0) == 0)
        ++partialTemps;
    expect(partialTemps == 0, "a refused table leaves no temporary file");

    // A symlink is followed to its target, which is created if it does not
    // exist yet. The link itself is never replaced by the file.
    {
      namespace fs = std::filesystem;
      const fs::path linkPath = dir / "dangling.json";
      std::error_code linkEc;
      fs::create_symlink("dangling_target.json", linkPath, linkEc);
      if (linkEc) {
        std::fprintf(stderr, "skipping symlink checks: %s\n",
                     linkEc.message().c_str());
      } else {
        expect(mlir::hip::updateProblemCacheJsonFile(
                   linkPath.string(), "gfx1151", {{exact, tile32}}, error),
               "a dangling symlink is written through");
        expect(fs::is_symlink(fs::symlink_status(linkPath)) &&
                   fs::is_regular_file(dir / "dangling_target.json"),
               "the link stays and its target is created");
        expect(mlir::hip::updateProblemCacheJsonFile(
                   linkPath.string(), "gfx1151", {{gemmAt(8), tile16}}, error),
               "a link whose target now exists is written through");
        mlir::hip::ProblemCacheLut viaLink;
        expect(
            fs::is_symlink(fs::symlink_status(linkPath)) &&
                mlir::hip::loadProblemCacheLutJsonFile(
                    (dir / "dangling_target.json").string(), viaLink, error) &&
                viaLink.entries.size() == 2,
            "later compiles keep updating the target, not the link");

        // A chain of dangling links, ending in a directory that is not there.
        fs::create_symlink("chain_mid.json", dir / "chain.json", linkEc);
        fs::create_symlink("chain_end/final.json", dir / "chain_mid.json",
                           linkEc);
        expect(!linkEc && mlir::hip::updateProblemCacheJsonFile(
                              (dir / "chain.json").string(), "gfx1151",
                              {{exact, tile32}}, error),
               "a chain of dangling symlinks is written through");
        expect(fs::is_symlink(fs::symlink_status(dir / "chain.json")) &&
                   fs::is_symlink(fs::symlink_status(dir / "chain_mid.json")) &&
                   fs::is_regular_file(dir / "chain_end" / "final.json"),
               "every link in the chain stays and the last target is created");

        // A loop cannot be resolved, so it is refused and left alone.
        fs::create_symlink("loop_b.json", dir / "loop_a.json", linkEc);
        fs::create_symlink("loop_a.json", dir / "loop_b.json", linkEc);
        const std::string loopPath = (dir / "loop_a.json").string();
        expect(!linkEc &&
                   !mlir::hip::updateProblemCacheJsonFile(
                       loopPath, "gfx1151", {{exact, tile32}}, error) &&
                   error.find(loopPath) != std::string::npos,
               "a symlink loop is refused");
        expect(fs::is_symlink(fs::symlink_status(dir / "loop_a.json")) &&
                   fs::is_symlink(fs::symlink_status(dir / "loop_b.json")),
               "a refused symlink loop is left in place");

        size_t temps = 0;
        for (const auto &entry : fs::recursive_directory_iterator(dir))
          if (entry.path().filename().string().find(".tmp") !=
              std::string::npos)
            ++temps;
        expect(temps == 0, "symlinked writes leave no temporary file");
      }
    }

    // A directory opens as a stream on Linux and then fails to read. That is
    // an error return, not an exception, for the load and for the rewrite.
    {
      const std::filesystem::path asDir = dir / "a_directory.json";
      std::filesystem::create_directories(asDir);
      mlir::hip::ProblemCacheLut unread;
      bool threw = false;
      bool loaded = true;
      bool updated = true;
      std::string loadError;
      std::string updateError;
      try {
        loaded = mlir::hip::loadProblemCacheLutJsonFile(asDir.string(), unread,
                                                        loadError);
        updated = mlir::hip::updateProblemCacheJsonFile(
            asDir.string(), "gfx1151", {{exact, tile32}}, updateError);
      } catch (...) {
        threw = true;
      }
      expect(!threw, "a directory path does not throw");
      expect(!loaded && !loadError.empty(),
             "loading a directory reports an error");
      expect(!updated && updateError.find(asDir.string()) != std::string::npos,
             "rewriting a directory reports an error");
      expect(std::filesystem::is_directory(asDir),
             "a refused directory is left in place");
    }

    // An empty file, as `touch` makes it, is a fresh table.
    const std::string blank = (dir / "blank.json").string();
    { std::ofstream out(blank, std::ios::binary); }
    expect(mlir::hip::updateProblemCacheJsonFile(blank, "gfx1151",
                                                 {{exact, tile32}}, error) &&
               mlir::hip::loadProblemCacheLutJsonFile(blank, reloaded, error) &&
               reloaded.entries.size() == 1,
           "an empty file is filled in as a new table");
    std::filesystem::remove_all(dir, ignored);
  }

  if (gFailures != 0) {
    std::fprintf(stderr, "%d nearest-conv-tune checks failed\n", gFailures);
    return 1;
  }
  return 0;
}
