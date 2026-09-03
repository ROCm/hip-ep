/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The offline-LUT policy, with no GPU and no measurement: a hand-built table in
// memory, then the questions that decide whether a shape gets a good config.
//
// What is worth pinning here is not "a lookup works" but the parts that are
// easy to break silently: the tier order, that a token-by-token walk never
// falls out of the table, that a stored split count is clamped to the splits
// with work, that a row whose wildcards disagree with its tier is refused, and
// that a geometry the grid never measured still lands on a row rather than on
// the compiled-in heuristic.

#include "gqa_autotune.h"
#include "gqa_autotune_generated.h"
#include "hip/env.h"
#include "morphizen-foundation/file_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char *expression, int line) {
  if (condition)
    return;
  std::fprintf(stderr, "requirement failed at line %d: %s\n", line, expression);
  std::exit(1);
}

#define REQUIRE(expr) require((expr), #expr, __LINE__)

void setModeEnv(const char *value) {
#ifdef _WIN32
  _putenv_s("HIPDNN_GQA_AUTOTUNE_MODE", value ? value : "");
#else
  if (value)
    setenv("HIPDNN_GQA_AUTOTUNE_MODE", value, 1);
  else
    unsetenv("HIPDNN_GQA_AUTOTUNE_MODE");
#endif
}

class ScopedModeEnv {
public:
  ScopedModeEnv()
      : original_(hipdnn_ep::env_string("HIPDNN_GQA_AUTOTUNE_MODE")) {
    setModeEnv(nullptr);
  }
  ~ScopedModeEnv() {
    setModeEnv(original_.empty() ? nullptr : original_.c_str());
  }

private:
  std::string original_;
};

class MemoryReader final : public morphizen::FileReader {
public:
  explicit MemoryReader(const std::vector<uint8_t> &bytes) : bytes_(bytes) {}

  size_t size() const override { return bytes_.size(); }
  void rewind() const override { offset_ = 0; }
  size_t fread(void *buffer, size_t size) const override {
    const size_t remaining = bytes_.size() - offset_;
    const size_t count = size < remaining ? size : remaining;
    if (count != 0)
      std::memcpy(buffer, bytes_.data() + offset_, count);
    offset_ += count;
    return count;
  }

private:
  const std::vector<uint8_t> &bytes_;
  mutable size_t offset_ = 0;
};

class MemoryFileSystem final : public morphizen::FileSystem {
public:
  explicit MemoryFileSystem(std::vector<uint8_t> bytes)
      : bytes_(std::move(bytes)) {}

  morphizen::FileReader *create_reader(const char *path) override {
    if (std::strcmp(path, hipdnn_ep::kGqaAutotuneFilename) != 0)
      return nullptr;
    return new MemoryReader(bytes_);
  }
  morphizen::FileWriter *create_writer(const char *) override {
    return nullptr;
  }
  void destroy_reader(morphizen::FileReader *reader) override { delete reader; }
  void destroy_writer(morphizen::FileWriter *writer) override { delete writer; }

private:
  std::vector<uint8_t> bytes_;
};

using Phase = mlir::hip::GqaTunePhase;
using Tier = mlir::hip::GqaTuneTier;
using Dtype = mlir::hip::GqaTuneKvDtype;
using Dim = mlir::hip::GqaTuneHeadDim;
using Seq = mlir::hip::GqaSeqBucket;
using Win = mlir::hip::GqaWindowClass;
using Par = mlir::hip::GqaParClass;
using Bat = mlir::hip::GqaBatchClass;
using Cfg = mlir::hip::GqaTuneConfig;
using Head = mlir::hip::GqaHeadCountClass;
using Row = mlir::hip::GqaTuneRow;

std::vector<uint8_t> makeLut() {
  std::vector<Row> rows;
  const auto decode = [&](Tier tier, uint8_t hpg, Par par, Seq skv, Cfg config,
                          uint8_t splits, Dim dim = Dim::D64,
                          Bat batch = Bat::Any, Win window = Win::NoWindow) {
    rows.push_back(Row(Phase::Decode, tier, Dtype::Any, dim, hpg, Head::Any,
                       par, batch, Seq::Any, skv, window, config, splits));
  };

  // heads-per-group 8 at head_dim 64, which is what gpt-oss runs: WMMA is
  // templated for this pair, so a row keyed on it may name it.
  decode(Tier::HeadGroup, 8, Par::Any, Seq::S128, Cfg::Wmma, 4);
  // PR #675 routes sliding-window decode to the fused kernel. A window is part
  // of the key: full attention at 128 keys cannot borrow this row.
  decode(Tier::HeadGroup, 8, Par::Any, Seq::S128, Cfg::WmmaBkv16, 4, Dim::D64,
         Bat::Any, Win::W128);
  // Bucket labels are four to the octave, so S768 answers (640, 768] and S896
  // answers (768, 896] -- a request at 800 is not in the same row as one at
  // 700.
  decode(Tier::HeadGroup, 8, Par::Any, Seq::S768, Cfg::Scalar, 6);
  decode(Tier::HeadGroup, 8, Par::Any, Seq::S896, Cfg::Scalar, 8);
  // Deliberately more splits than a short request can use, to pin the clamp:
  // S80 answers (64, 80], and at 70 keys only ceil(70/16) = 5 splits have work.
  decode(Tier::HeadGroup, 8, Par::Any, Seq::S80, Cfg::Scalar, 48);
  // A bucket wide enough that two lengths inside it clamp differently: S160
  // answers (128, 160], where ceil(len/16) is 9 at the bottom and 10 at the
  // top. Used to pin what the resolved-answer memo is allowed to remember.
  decode(Tier::HeadGroup, 8, Par::Any, Seq::S160, Cfg::Scalar, 10);
  // A batch*num_heads bucket beats the pooled row for the same lengths. This is
  // the only tier that can express "this head count wants something else", and
  // it is also how batch enters the key at all.
  decode(Tier::Geometry, 8, Par::P512, Seq::S128, Cfg::Scalar, 16);
  // The same parallelism at a named batch beats the one that pools batches: 512
  // work items as 64 sequences of 8 heads is not the same launch as one
  // sequence of 512, and only this row can say so.
  decode(Tier::Geometry, 8, Par::P512, Seq::S128, Cfg::WmmaBkv32, 4, Dim::D64,
         Bat::B64);
  // No geometry in the key, so no WMMA: the loader refuses a Length row that
  // names it, because this row answers the pairs WMMA is not templated for.
  decode(Tier::Length, 0, Par::Any, Seq::S1536, Cfg::Scalar, 10);
  decode(Tier::Length, 0, Par::Any, Seq::S1536, Cfg::Scalar, 10, Dim::D128);

  // A Fallback row wildcards the window too: it has to answer both a windowed
  // and an unwindowed request, which `Any` means and `NoWindow` does not.
  const auto add_fallback = [&](Phase phase, Dim dim, Cfg config,
                                uint8_t splits) {
    rows.push_back(Row(phase, Tier::Fallback, Dtype::Any, dim, 0, Head::Any,
                       Par::Any, Bat::Any, Seq::Any, Seq::Any, Win::Any, config,
                       splits));
  };
  add_fallback(Phase::Decode, Dim::D64, Cfg::Scalar, 12);
  add_fallback(Phase::PrefillV5, Dim::D64, Cfg::MT1_BKV32, 0);
  add_fallback(Phase::PrefillV8, Dim::D256, Cfg::ND4_MT1_BKV32, 0);

  // Rows the loader must refuse, each for a different reason. They are here so
  // that a table with a generator bug degrades to a coarser tier instead of
  // answering something narrower or wider than it claims.
  //
  //  - a Length row naming WMMA: unhonourable without heads-per-group.
  rows.push_back(Row(Phase::Decode, Tier::Length, Dtype::Any, Dim::D64, 0,
                     Head::Any, Par::Any, Bat::Any, Seq::Any, Seq::S2048,
                     Win::NoWindow, Cfg::Wmma, 8));
  //  - a HeadGroup row that also sets par: it would answer one head count while
  //    looking like it answers all of them.
  rows.push_back(Row(Phase::Decode, Tier::HeadGroup, Dtype::Any, Dim::D64, 8,
                     Head::Any, Par::P256, Bat::Any, Seq::S2048, Seq::S2048,
                     Win::NoWindow, Cfg::Scalar, 8));
  //  - a Fallback row that keys on a length: it would answer far less than the
  //    last resort is expected to.
  rows.push_back(Row(Phase::Decode, Tier::Fallback, Dtype::Any, Dim::D128, 0,
                     Head::Any, Par::Any, Bat::Any, Seq::Any, Seq::S4096,
                     Win::Any, Cfg::Scalar, 8));
  //  - a v7 phase naming a v8 config: the knobs would go to the wrong
  //  dispatcher.
  rows.push_back(Row(Phase::PrefillV7, Tier::Fallback, Dtype::Any, Dim::D128, 0,
                     Head::Any, Par::Any, Bat::Any, Seq::Any, Seq::Any,
                     Win::Any, Cfg::ND2_MT1_BKV32, 0));
  //  - a decode row with a split count outside 1..64.
  rows.push_back(Row(Phase::Decode, Tier::HeadGroup, Dtype::Any, Dim::D256, 4,
                     Head::Any, Par::Any, Bat::Any, Seq::Any, Seq::S2048,
                     Win::NoWindow, Cfg::Scalar, 200));

  // Prefill v5 is the one variant with a window in its key, and the one whose
  // window is not always NoWindow.
  const auto v5 = [&](Tier tier, Seq sq, Seq skv, Win window, Cfg config) {
    rows.push_back(Row(Phase::PrefillV5, tier, Dtype::Any, Dim::D64, 0,
                       Head::Any, Par::Any, Bat::Any, sq, skv, window, config,
                       0));
  };
  v5(Tier::Length, Seq::S256, Seq::S512, Win::NoWindow, Cfg::MT2_BKV64);
  v5(Tier::Length, Seq::S256, Seq::S3072, Win::NoWindow, Cfg::MT1_BKV32);
  v5(Tier::Length, Seq::S256, Seq::S3072, Win::W128, Cfg::MT2_BKV32);

  // An Int8 decode row beats the dtype-agnostic one for an Int8 request.
  // Nothing measured is dtype-specific yet; this pins the probe order so that
  // measuring one later does not need a schema change.
  rows.push_back(Row(Phase::Decode, Tier::HeadGroup, Dtype::Int8, Dim::D64, 8,
                     Head::Any, Par::Any, Bat::Any, Seq::Any, Seq::S128,
                     Win::NoWindow, Cfg::Scalar, 2));

  flatbuffers::FlatBufferBuilder builder;
  auto lut = mlir::hip::CreateGqaAutotuneLutDirect(
      builder, /*schema_version=*/6, /*gpu_arch=*/"", /*rocm_version=*/0,
      hipdnn_ep::kGqaKernelAbi, "unit-test", &rows);
  mlir::hip::FinishGqaAutotuneLutBuffer(builder, lut);
  return {builder.GetBufferPointer(),
          builder.GetBufferPointer() + builder.GetSize()};
}

// A table with exactly one answer in it, so two of them differ only in what
// they say about the same geometry. Used to pin that the process-wide memo keys
// on the table as well as the question.
std::vector<uint8_t> makeOneRowLut(uint8_t splits) {
  std::vector<Row> rows;
  rows.push_back(Row(Phase::Decode, Tier::HeadGroup, Dtype::Any, Dim::D64, 8,
                     Head::Any, Par::Any, Bat::Any, Seq::Any, Seq::S128,
                     Win::NoWindow, Cfg::Scalar, splits));
  flatbuffers::FlatBufferBuilder builder;
  auto lut = mlir::hip::CreateGqaAutotuneLutDirect(
      builder, /*schema_version=*/6, /*gpu_arch=*/"", /*rocm_version=*/0,
      hipdnn_ep::kGqaKernelAbi, "unit-test-one-row", &rows);
  mlir::hip::FinishGqaAutotuneLutBuffer(builder, lut);
  return {builder.GetBufferPointer(),
          builder.GetBufferPointer() + builder.GetSize()};
}

hipdnn_ep::GqaDecodeRequest decodeRequest(int num_heads, int kv_num_heads,
                                          int head_dim, int effective_skv,
                                          int batch = 1) {
  return {/*kv_dtype=*/0,    batch,
          num_heads,         kv_num_heads,
          head_dim,          effective_skv,
          /*max_splits=*/64, /*local_window=*/0};
}

// Load a real table and report what a shipped file resolves to, for the tiers a
// caller cares about. Given a path, that is all this binary does: the loader is
// the only thing that can tell you a generated table is loadable, and offline
// validation cannot -- validate_lut_json.py checks the JSON, and flatc will
// happily pack rows this loader refuses.
int inspect(const char *path) {
  std::vector<uint8_t> bytes;
  if (std::FILE *f = std::fopen(path, "rb")) {
    std::fseek(f, 0, SEEK_END);
    bytes.resize(static_cast<size_t>(std::ftell(f)));
    std::fseek(f, 0, SEEK_SET);
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size())
      bytes.clear();
    std::fclose(f);
  }
  if (bytes.empty()) {
    std::fprintf(stderr, "cannot read %s\n", path);
    return 1;
  }
  MemoryFileSystem fs(std::move(bytes));
  void *policy = hipdnn_ep::gqa_autotune_create(&fs);
  if (!policy)
    return 1;
  // A decode step and a prefill chunk on a geometry the table should know, plus
  // one it cannot: what tier answers is the interesting part.
  const struct {
    const char *name;
    int H, G, d;
  } probes[] = {
      {"Llama-3-8B", 32, 8, 128},           {"Qwen3-1.7B", 16, 8, 128},
      {"Llama-3.2-3B (HpG 3)", 24, 8, 128}, {"GLM-4-9B (HpG 16)", 32, 2, 128},
      {"Llama-2-7B (MHA)", 32, 32, 128},    {"gpt-oss (d64)", 64, 8, 64},
      {"Gemma-3-12B (d256)", 16, 8, 256}};
  for (const auto &p : probes) {
    for (int batch : {1, 8}) {
      const auto decode = hipdnn_ep::gqa_autotune_resolve_decode(
          policy, decodeRequest(p.H, p.G, p.d, 8193, batch));
      std::printf("  %-22s B=%d decode: %-10s %s splits=%d\n", p.name, batch,
                  hipdnn_ep::gqa_tune_source_name(decode.source),
                  decode.config.use_wmma ? "wmma  " : "scalar",
                  decode.config.splits);
    }
    const hipdnn_ep::GqaPrefillVariant variant =
        p.d == 64    ? hipdnn_ep::GqaPrefillVariant::V5
        : p.d == 128 ? hipdnn_ep::GqaPrefillVariant::V7
                     : hipdnn_ep::GqaPrefillVariant::V8;
    const hipdnn_ep::GqaPrefillRequest prefill{variant, 1,   p.H,  p.G,
                                               p.d,     512, 8193, 0};
    const auto got = hipdnn_ep::gqa_autotune_resolve_prefill(policy, prefill);
    std::printf("  %-22s     prefill: %-10s m_tiles=%d bkv=%d nw=%d mt=%d "
                "nd=%d\n",
                p.name, hipdnn_ep::gqa_tune_source_name(got.source),
                got.config.m_tiles, got.config.bkv, got.config.nw,
                got.config.mt, got.config.nd);
  }
  hipdnn_ep::gqa_autotune_destroy(policy);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1)
    return inspect(argv[1]);
  ScopedModeEnv mode_env;
  MemoryFileSystem fs(makeLut());
  void *policy = hipdnn_ep::gqa_autotune_create(&fs);
  REQUIRE(policy != nullptr);
  REQUIRE(hipdnn_ep::gqa_autotune_mode(policy) ==
          hipdnn_ep::GqaAutotuneMode::Lookup);

  hipdnn_ep::gqa_autotune_apply_provider_mode(policy, " OnLiNe ");
  REQUIRE(hipdnn_ep::gqa_autotune_mode(policy) ==
          hipdnn_ep::GqaAutotuneMode::Online);
  hipdnn_ep::gqa_autotune_destroy(policy);

  policy = hipdnn_ep::gqa_autotune_create(&fs);
  hipdnn_ep::gqa_autotune_apply_provider_mode(policy, "invalid");
  REQUIRE(hipdnn_ep::gqa_autotune_mode(policy) ==
          hipdnn_ep::GqaAutotuneMode::Lookup);
  hipdnn_ep::gqa_autotune_apply_provider_mode(policy, "");
  REQUIRE(hipdnn_ep::gqa_autotune_mode(policy) ==
          hipdnn_ep::GqaAutotuneMode::Lookup);
  hipdnn_ep::gqa_autotune_destroy(policy);

  setModeEnv("lookup");
  policy = hipdnn_ep::gqa_autotune_create(&fs);
  hipdnn_ep::gqa_autotune_apply_provider_mode(policy, "online");
  REQUIRE(hipdnn_ep::gqa_autotune_mode(policy) ==
          hipdnn_ep::GqaAutotuneMode::Lookup);
  hipdnn_ep::gqa_autotune_destroy(policy);

  // Even an invalid non-empty environment setting owns the highest-priority
  // slot and preserves the historical fallback to the build default.
  setModeEnv("invalid");
  policy = hipdnn_ep::gqa_autotune_create(&fs);
  hipdnn_ep::gqa_autotune_apply_provider_mode(policy, "online");
  REQUIRE(hipdnn_ep::gqa_autotune_mode(policy) ==
          hipdnn_ep::GqaAutotuneMode::Lookup);
  hipdnn_ep::gqa_autotune_destroy(policy);
  setModeEnv(nullptr);

  policy = hipdnn_ep::gqa_autotune_create(&fs);

  // 64:8:64, one sequence: batch*num_heads is 64, which no Geometry row names,
  // so the pooled heads-per-group row answers.
  auto decode = decodeRequest(64, 8, 64, 128);
  auto result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.use_wmma && result.config.splits == 4);

  // The same geometry at a deep context scans 128 keys under a sliding window.
  // Its W128 row, rather than the full-attention S16384 row, is selected.
  auto windowed = decodeRequest(64, 8, 64, 16384);
  windowed.local_window = 128;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, windowed);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.use_wmma && result.config.splits == 4 &&
          result.config.bkv == 16);

  // The same model at batch 8 is 512 work items, which the Geometry row names.
  // Nothing else about the request changed. Its 16 splits then clamp to the 8
  // that have work at 128 keys, so this pins the clamp at this tier too.
  auto batched = decodeRequest(64, 8, 64, 128, /*batch=*/8);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, batched);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Geometry);
  REQUIRE(!result.config.use_wmma && result.config.splits == 8);

  // 512 work items again, reached as 64 sequences of 8 heads rather than one of
  // 512. The batch-specific row wins over the one that pools batches, which is
  // the whole reason the key carries the batch as well as the product.
  auto many_seqs = decodeRequest(8, 1, 64, 128, /*batch=*/64);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, many_seqs);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Geometry);
  REQUIRE(result.config.use_wmma && result.config.splits == 4 &&
          result.config.bkv == 32);

  // The headline of keying on heads-per-group: 32:4:64 was never measured, and
  // it gets the row measured on 64:8:64 because both are 8 queries per KV head.
  // Under a key on head counts this was the last resort.
  auto same_ratio = decodeRequest(32, 4, 64, 128);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, same_ratio);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.use_wmma && result.config.splits == 4);

  // An Int8 cache prefers a row measured on Int8 over the shared one.
  auto int8 = decodeRequest(64, 8, 64, 128);
  int8.kv_dtype = 1;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, int8);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(!result.config.use_wmma && result.config.splits == 2);

  // Quarter-octave labels: 700 is in (640, 768] and 800 is in (768, 896]. Under
  // half-octave labels both landed in (768, 1024] and shared one row.
  decode.effective_skv = 700;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.splits == 6);
  decode.effective_skv = 800;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.splits == 8);

  // A stored split count is clamped to the splits that have work: the row holds
  // 48, but 70 keys is ceil(70/16) = 5 tiles.
  decode.effective_skv = 70;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.splits == 5);

  // Resolving is memoised per question so that a served model walks the tiers
  // once instead of once per layer per token. What the memo may hold is the
  // row, not one request's reading of it: 129 and 160 keys share a bucket and
  // therefore a row, but not a clamp, so each has to get its own split count
  // whichever of them asked first.
  auto low_in_bucket = decodeRequest(64, 8, 64, 129);
  auto high_in_bucket = decodeRequest(64, 8, 64, 160);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, low_in_bucket);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.splits == 9);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, high_in_bucket);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.splits == 10);
  // Asking again is the path every token after the first takes, and it is the
  // one that reads the memo rather than writing it.
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, low_in_bucket);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.splits == 9);

  // Nothing keyed on heads-per-group covers (1280, 1536], so the length-keyed
  // row answers -- a tier below the geometry ones but still measured, and above
  // the last resort. The WMMA row at that tier was refused on load, so this is
  // the scalar one.
  decode.effective_skv = 1500;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Length);
  REQUIRE(!result.config.use_wmma && result.config.splits == 10);

  // A heads-per-group the table has no rows for at all still gets that tier:
  // this is what a model outside the measured set lands on, and it is why the
  // tier exists. 24:8 is heads-per-group 3.
  auto unseen_ratio = decodeRequest(24, 8, 64, 1500);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, unseen_ratio);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Length);
  REQUIRE(!result.config.use_wmma && result.config.splits == 10);

  // Outside every length-keyed row, the last resort answers, and it comes from
  // the table rather than from the config compiled into gqa_autotune.cpp.
  decode.effective_skv = 40000;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Fallback);
  REQUIRE(!result.config.use_wmma && result.config.splits == 12);

  // The clamp applies to a Fallback row too: 40 keys is ceil(40/16) = 3 tiles.
  decode.effective_skv = 40;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Fallback);
  REQUIRE(result.config.splits == 3);

  // head_dim 128's Fallback row keyed on a length and was refused on load, and
  // there is no head_dim-agnostic row, so a d=128 request outside the length
  // rows reaches the compiled-in heuristic. That is the one path that means "no
  // usable table", which is why validate_lut_json.py warns when a phase has no
  // Fallback.
  auto d128 = decodeRequest(64, 8, 128, 40000);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, d128);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Heuristic);
  // And what it answers with is a split count that fills the machine rather
  // than a fixed number: 64 heads at 40000 keys has 2500 useful splits to
  // choose from and wants far fewer than that, but more than the 8 the old
  // fixed heuristic gave.
  REQUIRE(!result.config.use_wmma);
  REQUIRE(result.config.splits >= 1 && result.config.splits <= 64);

  // The rule scales with the work: one head with the same context has a
  // hundredth of the work items, so it takes more splits to fill the same
  // machine.
  auto one_head = decodeRequest(1, 1, 128, 40000);
  const auto few = hipdnn_ep::gqa_autotune_resolve_decode(policy, one_head);
  REQUIRE(few.source == hipdnn_ep::GqaTuneSource::Heuristic);
  REQUIRE(few.config.splits > result.config.splits);

  // What generation actually looks like: seq_kv grows by one token per step,
  // and every step has to come from the table. Row count at round lengths is a
  // poor measure of coverage; this is the property the bucketed tiers exist
  // for.
  for (int skv = 641; skv <= 768; ++skv) {
    decode.effective_skv = skv;
    result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
    REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
    // 6 splits is under ceil(641/16), so the clamp leaves the row alone.
    REQUIRE(result.config.splits == 6);
  }

  // A length past the longest label saturates onto it rather than falling out
  // of the table. A 512 k context is not a shape anyone runs on this part, and
  // the longest row measured is a better answer than the last resort.
  auto huge = decodeRequest(64, 8, 64, 400000);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, huge);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Fallback);
  REQUIRE(result.config.splits == 12);

  // A geometry whose head counts do not divide, or whose ratio is outside the
  // set the kernels are templated for, has no heads-per-group: the geometry
  // tiers are skipped rather than probed with a zero. (real/gqa.cpp keeps these
  // off the fused path in the first place; this is the belt to that braces.)
  auto ragged = decodeRequest(30, 8, 64, 1500);
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, ragged);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Length);
  auto too_wide = decodeRequest(64, 2, 64, 1500); // heads-per-group 32
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, too_wide);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Length);

  // An Int8 request with no Int8 row for its lengths still resolves: the probe
  // falls back to the dtype-agnostic row at the same tier, not to a coarser
  // tier.
  auto int8_long = decodeRequest(64, 8, 64, 700);
  int8_long.kv_dtype = 1;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, int8_long);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::HeadGroup);
  REQUIRE(result.config.splits == 6);

  // Prefill v5 keys on the window, and NoWindow is a value rather than a
  // wildcard: the same lengths with a 128-token window get a different row.
  hipdnn_ep::GqaPrefillRequest prefill{hipdnn_ep::GqaPrefillVariant::V5,
                                       1,
                                       32,
                                       8,
                                       64,
                                       /*seq_q=*/256,
                                       /*seq_kv=*/512,
                                       /*local_window=*/0};
  auto prefill_result =
      hipdnn_ep::gqa_autotune_resolve_prefill(policy, prefill);
  REQUIRE(prefill_result.source == hipdnn_ep::GqaTuneSource::Length);
  REQUIRE(prefill_result.config.m_tiles == 2 &&
          prefill_result.config.bkv == 64);

  // 3000 rounds up to S3072, the row above; 2500 would have landed in S2560,
  // which this table does not have.
  prefill.seq_kv = 3000;
  prefill_result = hipdnn_ep::gqa_autotune_resolve_prefill(policy, prefill);
  REQUIRE(prefill_result.source == hipdnn_ep::GqaTuneSource::Length);
  REQUIRE(prefill_result.config.m_tiles == 1 &&
          prefill_result.config.bkv == 32);

  prefill.local_window = 128;
  prefill_result = hipdnn_ep::gqa_autotune_resolve_prefill(policy, prefill);
  REQUIRE(prefill_result.source == hipdnn_ep::GqaTuneSource::Length);
  REQUIRE(prefill_result.config.m_tiles == 2 &&
          prefill_result.config.bkv == 32);

  // Most prefill rows carry no head geometry, so a head count nobody measured
  // resolves from the same rows. The prefill kernels are templated on head_dim
  // alone, which is why that is sound where it holds.
  prefill.num_heads = 48;
  prefill.kv_num_heads = 6;
  prefill.local_window = 0;
  prefill_result = hipdnn_ep::gqa_autotune_resolve_prefill(policy, prefill);
  REQUIRE(prefill_result.source == hipdnn_ep::GqaTuneSource::Length);
  REQUIRE(prefill_result.config.m_tiles == 1);

  // v8's whole policy is one row: one config wins every shape measured, so the
  // last resort is the answer rather than a compromise.
  const hipdnn_ep::GqaPrefillRequest v8{
      hipdnn_ep::GqaPrefillVariant::V8, 1, 16, 8, 256, 512, 20000, 0};
  prefill_result = hipdnn_ep::gqa_autotune_resolve_prefill(policy, v8);
  REQUIRE(prefill_result.source == hipdnn_ep::GqaTuneSource::Fallback);
  REQUIRE(prefill_result.config.nd == 4 && prefill_result.config.mt == 1 &&
          prefill_result.config.bkv == 32);

  // v7's Fallback row named a v8 config and was refused on load, so v7 has no
  // row at all here and reaches the heuristic.
  const hipdnn_ep::GqaPrefillRequest v7{
      hipdnn_ep::GqaPrefillVariant::V7, 1, 32, 8, 128, 512, 2048, 0};
  REQUIRE(hipdnn_ep::gqa_autotune_resolve_prefill(policy, v7).source ==
          hipdnn_ep::GqaTuneSource::Heuristic);

  // The memo outlives a session, so two models loaded into one process must not
  // be answered from each other's table. These two differ only in the split
  // count they hold for one geometry, which is exactly the collision a memo
  // keyed on the question alone would produce -- and it would produce it
  // silently, since both requests hash to the same question.
  MemoryFileSystem four_fs(makeOneRowLut(4));
  MemoryFileSystem eight_fs(makeOneRowLut(8));
  void *four = hipdnn_ep::gqa_autotune_create(&four_fs);
  void *eight = hipdnn_ep::gqa_autotune_create(&eight_fs);
  REQUIRE(four != nullptr && eight != nullptr);
  const auto shared = decodeRequest(64, 8, 64, 128);
  REQUIRE(hipdnn_ep::gqa_autotune_resolve_decode(four, shared).config.splits ==
          4);
  REQUIRE(hipdnn_ep::gqa_autotune_resolve_decode(eight, shared).config.splits ==
          8);
  // Asking the first one again has to give its own answer back, not the one the
  // second stored under the same question.
  REQUIRE(hipdnn_ep::gqa_autotune_resolve_decode(four, shared).config.splits ==
          4);
  // A fresh session over the same table -- the case the memo is process-wide
  // for, where it answers from what an earlier session resolved. Whether the
  // entry was reused is not observable from here; that it is still the right
  // answer is.
  MemoryFileSystem reload_fs(makeOneRowLut(4));
  void *reloaded = hipdnn_ep::gqa_autotune_create(&reload_fs);
  REQUIRE(
      hipdnn_ep::gqa_autotune_resolve_decode(reloaded, shared).config.splits ==
      4);
  hipdnn_ep::gqa_autotune_destroy(reloaded);
  hipdnn_ep::gqa_autotune_destroy(eight);
  hipdnn_ep::gqa_autotune_destroy(four);

  hipdnn_ep::gqa_autotune_destroy(policy);
  return 0;
}
