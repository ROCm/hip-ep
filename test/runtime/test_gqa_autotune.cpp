/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "gqa_autotune.h"
#include "gqa_autotune_generated.h"
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
  std::fprintf(stderr, "requirement failed at line %d: %s\n", line,
               expression);
  std::exit(1);
}

#define REQUIRE(expr) require((expr), #expr, __LINE__)

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

std::vector<uint8_t> makeLut() {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<mlir::hip::GqaTuneEntry>> entries;

  auto add_decode = [&](mlir::hip::GqaTuneMatch match, int skv, bool wmma,
                        int splits, int max_seq = 0,
                        int local_window = 0) {
    auto key = mlir::hip::CreateGqaTuneKey(
        builder, mlir::hip::GqaTunePhase::Decode, match,
        mlir::hip::GqaTuneKvDtype::Fp16, 1, 32, 8, 64,
        /*seq_q=*/1, skv, max_seq, local_window);
    auto config = mlir::hip::CreateGqaTuneConfig(
        builder, wmma, splits, 0, 0, 0, 0, 0);
    entries.push_back(mlir::hip::CreateGqaTuneEntry(builder, key, config));
  };
  add_decode(mlir::hip::GqaTuneMatch::Exact, 128, true, 4);
  add_decode(mlir::hip::GqaTuneMatch::Bucket, 128, false, 2);
  add_decode(mlir::hip::GqaTuneMatch::Bucket, 1024, false, 8);
  // Same effective window length, different cache stride and measured winner.
  add_decode(mlir::hip::GqaTuneMatch::Exact, 128, true, 8,
             /*max_seq=*/253, /*local_window=*/128);
  add_decode(mlir::hip::GqaTuneMatch::Exact, 128, false, 8,
             /*max_seq=*/2112, /*local_window=*/128);

  auto prefill_key = mlir::hip::CreateGqaTuneKey(
      builder, mlir::hip::GqaTunePhase::PrefillV5,
      mlir::hip::GqaTuneMatch::Exact, mlir::hip::GqaTuneKvDtype::Fp16,
      1, 32, 8, 64, /*seq_q=*/256, /*seq_kv=*/0, /*max_seq=*/0,
      /*local_window=*/0);
  auto prefill_config = mlir::hip::CreateGqaTuneConfig(
      builder, false, 0, /*m_tiles=*/2, /*bkv=*/64, 0, 0, 0);
  entries.push_back(
      mlir::hip::CreateGqaTuneEntry(builder, prefill_key, prefill_config));

  auto entries_vec = builder.CreateVector(entries);
  auto lut = mlir::hip::CreateGqaAutotuneLut(
      builder, /*schema_version=*/1, builder.CreateString(""),
      /*rocm_version=*/0, builder.CreateString(hipdnn_ep::kGqaKernelAbi),
      builder.CreateString("unit-test"), entries_vec);
  mlir::hip::FinishGqaAutotuneLutBuffer(builder, lut);
  return {builder.GetBufferPointer(),
          builder.GetBufferPointer() + builder.GetSize()};
}

} // namespace

int main() {
  MemoryFileSystem fs(makeLut());
  void *policy = hipdnn_ep::gqa_autotune_create(&fs);
  REQUIRE(policy != nullptr);
  REQUIRE(hipdnn_ep::gqa_autotune_mode(policy) ==
          hipdnn_ep::GqaAutotuneMode::Lookup);

  hipdnn_ep::GqaDecodeRequest decode{
      /*kv_dtype=*/0, 1, 32, 8, 64, 128,
      /*exact_length_known=*/true, 4096, 64, 0};
  auto result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Exact);
  REQUIRE(result.config.use_wmma && result.config.splits == 4);

  decode.exact_length_known = false;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Bucket);
  REQUIRE(!result.config.use_wmma && result.config.splits == 2);

  decode.exact_length_known = true;
  decode.effective_skv = 700;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Bucket);
  REQUIRE(!result.config.use_wmma && result.config.splits == 8);

  decode.effective_skv = 1500;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Heuristic);
  REQUIRE(result.config.use_wmma && result.config.splits == 8);

  decode.effective_skv = 253;
  decode.max_seq = 253;
  decode.local_window = 128;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Exact);
  REQUIRE(result.config.use_wmma && result.config.splits == 8);

  decode.effective_skv = 2112;
  decode.max_seq = 2112;
  result = hipdnn_ep::gqa_autotune_resolve_decode(policy, decode);
  REQUIRE(result.source == hipdnn_ep::GqaTuneSource::Exact);
  REQUIRE(!result.config.use_wmma && result.config.splits == 8);

  const hipdnn_ep::GqaPrefillRequest prefill{
      hipdnn_ep::GqaPrefillVariant::V5, 1, 32, 8, 64, 256, 512, 4096, 0};
  const auto prefill_result =
      hipdnn_ep::gqa_autotune_resolve_prefill(policy, prefill);
  REQUIRE(prefill_result.source == hipdnn_ep::GqaTuneSource::Exact);
  REQUIRE(prefill_result.config.m_tiles == 2);
  REQUIRE(prefill_result.config.bkv == 64);

  hipdnn_ep::gqa_autotune_destroy(policy);
  return 0;
}
