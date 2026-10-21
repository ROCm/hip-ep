// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// Diagnostic only. All model inference uses the configured GPU provider.
#include "ort_genai_c.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(OgaResult *result) {
  if (!result)
    return;
  std::string message = OgaResultGetError(result);
  OgaDestroyResult(result);
  throw std::runtime_error(message);
}
template <typename T, void (*Destroy)(T *)> struct Handle {
  T *value = nullptr;
  ~Handle() {
    if (value)
      Destroy(value);
  }
};
std::ifstream input(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("Cannot read " + path.string());
  return stream;
}
std::ofstream output(const std::filesystem::path &path, bool binary = false) {
  if (std::filesystem::exists(path))
    throw std::runtime_error("Output already exists: " + path.string());
  std::ofstream stream(path, std::ios::out | (binary ? std::ios::binary
                                                     : std::ios::openmode(0)));
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  return stream;
}

void run(int argc, char **argv) {
  if (argc != 5 && argc != 6)
    throw std::runtime_error(
        "Usage: capture_logits MODEL PROMPT OUT STEPS [REFERENCE_DIR]");
  const std::filesystem::path out = argv[3];
  const int steps = std::stoi(argv[4]);
  if (steps < 1)
    throw std::runtime_error("STEPS must be positive");
  auto promptFile = input(argv[2]);
  std::string prompt((std::istreambuf_iterator<char>(promptFile)), {});
  std::vector<int32_t> reference;
  if (argc == 6) {
    auto stream = input(std::filesystem::path(argv[5]) / "tokens.txt");
    reference.assign(std::istream_iterator<int32_t>(stream), {});
    if (reference.empty())
      throw std::runtime_error("Reference tokens are empty");
  }
  Handle<OgaModel, OgaDestroyModel> model;
  std::cout << "Creating GPU model for logits validation..." << std::endl;
  check(OgaCreateModel(argv[1], &model.value));
  Handle<OgaTokenizer, OgaDestroyTokenizer> tokenizer;
  check(OgaCreateTokenizer(model.value, &tokenizer.value));
  Handle<OgaSequences, OgaDestroySequences> promptTokens;
  check(OgaCreateSequences(&promptTokens.value));
  check(
      OgaTokenizerEncode(tokenizer.value, prompt.c_str(), promptTokens.value));
  const size_t count = OgaSequencesGetSequenceCount(promptTokens.value, 0);
  const auto *ids = OgaSequencesGetSequenceData(promptTokens.value, 0);
  if (argc == 6) {
    auto stream = input(std::filesystem::path(argv[5]) / "prompt-tokens.txt");
    std::vector<int32_t> expected((std::istream_iterator<int32_t>(stream)), {});
    if (expected != std::vector<int32_t>(ids, ids + count))
      throw std::runtime_error("Reference prompt tokens differ");
  }
  auto promptOut = output(out / "prompt-tokens.txt");
  for (size_t i = 0; i < count; ++i)
    promptOut << ids[i] << '\n';
  Handle<OgaGeneratorParams, OgaDestroyGeneratorParams> params;
  check(OgaCreateGeneratorParams(model.value, &params.value));
  check(OgaGeneratorParamsSetSearchNumber(params.value, "max_length",
                                          count + steps));
  check(OgaGeneratorParamsSetSearchBool(params.value, "do_sample", false));
  check(
      OgaGeneratorParamsSetSearchNumber(params.value, "repetition_penalty", 1));
  Handle<OgaGenerator, OgaDestroyGenerator> generator;
  check(OgaCreateGenerator(model.value, params.value, &generator.value));
  check(OgaGenerator_AppendTokenSequences(generator.value, promptTokens.value));
  auto logitsOut = output(out / "logits.f32", true);
  auto tokensOut = output(out / "tokens.txt");
  auto meta = output(out / "logits.tsv");
  meta << "step\tvocab\targmax\tforced_token\tmax_logit\n";
  std::vector<int32_t> generated;
  for (int step = 0; step < steps; ++step) {
    if (argc == 6 && step == static_cast<int>(reference.size()))
      break;
    if (OgaGenerator_IsDone(generator.value))
      break;
    Handle<OgaTensor, OgaDestroyTensor> tensor;
    check(OgaGenerator_GetLogits(generator.value, &tensor.value));
    OgaElementType type;
    size_t rank;
    check(OgaTensorGetType(tensor.value, &type));
    check(OgaTensorGetShapeRank(tensor.value, &rank));
    std::vector<int64_t> shape(rank);
    check(OgaTensorGetShape(tensor.value, shape.data(), rank));
    if (type != OgaElementType_float32 || rank != 3 || shape[0] != 1 ||
        shape[1] != 1 || shape[2] < 1)
      throw std::runtime_error("Expected FP32 [1,1,vocab] logits");
    void *data = nullptr;
    check(OgaTensorGetData(tensor.value, &data));
    auto *logits = static_cast<float *>(data);
    size_t vocab = shape[2];
    if (!std::all_of(logits, logits + vocab,
                     [](float v) { return std::isfinite(v); }))
      throw std::runtime_error("Non-finite model logits");
    int32_t best = std::max_element(logits, logits + vocab) - logits;
    int32_t next = argc == 6 ? reference[step] : best;
    if (next < 0 || static_cast<size_t>(next) >= vocab)
      throw std::runtime_error("Reference token is out of range");
    logitsOut.write(reinterpret_cast<const char *>(logits),
                    vocab * sizeof(float));
    meta << step << '\t' << vocab << '\t' << best << '\t' << next << '\t'
         << logits[best] << '\n';
    // Save raw logits first. Override selection only to keep histories equal.
    std::fill(logits, logits + vocab, -10000.0f);
    logits[next] = 10000.0f;
    check(OgaGenerator_SetLogits(generator.value, tensor.value));
    check(OgaGenerator_GenerateNextToken(generator.value));
    size_t actualCount = OgaGenerator_GetSequenceCount(generator.value, 0);
    auto *actual = OgaGenerator_GetSequenceData(generator.value, 0);
    if (actualCount != count + step + 1 || actual[actualCount - 1] != next)
      throw std::runtime_error("Forced token was not used");
    generated.push_back(next);
    tokensOut << next << '\n';
    std::cout << "Logits step " << step << ": argmax=" << best
              << ", history token=" << next << std::endl;
  }
  if (generated.empty() || (argc == 6 && generated.size() != reference.size()))
    throw std::runtime_error("Incomplete logits capture");
  const char *text = nullptr;
  check(OgaTokenizerDecode(tokenizer.value, generated.data(), generated.size(),
                           &text));
  auto textOut = output(out / "generated.txt");
  textOut << text << '\n';
  std::cout << "History text: " << text << std::endl;
  OgaDestroyString(text);
}
} // namespace
int main(int argc, char **argv) {
  int status = 0;
  try {
    run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    status = 1;
  }
  OgaShutdown();
  return status;
}
