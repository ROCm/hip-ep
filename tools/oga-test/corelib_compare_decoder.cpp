/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// corelib_compare_decoder: logit-level accuracy check of hip-ep's RyzenAI
// CoreLib (NPU) path against hip-ep's GPU-only path on an onnxruntime-genai
// decoder LLM.
//
// The tool is specific to onnxruntime-genai decoder ONNX files: models whose
// inputs include attention_mask and past_key_values.*, and whose outputs
// include logits and present.*. It will crash or produce wrong results on
// arbitrary ONNX models.
//
// Runs one prefill of a fixed, deterministic prompt and one decode step per
// configuration, sequentially (one session alive at a time), and compares the
// logits: cosine similarity, max abs difference and top-1 agreement over
// all positions. Exit code 0 = PASS, 1 = FAIL, 2 = error, 3 = INCONCLUSIVE
// (identical logits: nothing ran on the NPU).
//
// Build (Windows, MSVC, adjust paths as needed):
//   cl /std:c++17 /EHsc /O2 /I<ort-include> corelib_compare_decoder.cpp
//      /link /LIBPATH:<ort-lib> onnxruntime.lib
//
// Usage:
//   corelib_compare_decoder <model.onnx> <hipgpu.dll> [--tokens N]
//                           [--capacity C] [--cpu]
//                           [--min-cosine C] [--min-top1 F]

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

const char *kEpName = "hipgpu";

float half_to_float(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F,
           mant = h & 0x3FF, bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        --exp;
      }
      bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

std::vector<float> to_floats(const Ort::Value &v) {
  auto info = v.GetTensorTypeAndShapeInfo();
  size_t n = info.GetElementCount();
  std::vector<float> out(n);
  switch (info.GetElementType()) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    std::memcpy(out.data(), v.GetTensorData<float>(), n * 4);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
    auto *p = reinterpret_cast<const uint16_t *>(v.GetTensorRawData());
    for (size_t i = 0; i < n; ++i)
      out[i] = half_to_float(p[i]);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
    auto *p = reinterpret_cast<const uint16_t *>(v.GetTensorRawData());
    for (size_t i = 0; i < n; ++i) {
      uint32_t b = uint32_t(p[i]) << 16;
      std::memcpy(&out[i], &b, 4);
    }
    break;
  }
  default:
    throw std::runtime_error("logits: unsupported element type");
  }
  return out;
}

bool contains(const std::string &s, const char *sub) {
  std::string a = s, b = sub;
  std::transform(a.begin(), a.end(), a.begin(), ::tolower);
  return a.find(b) != std::string::npos;
}

// Logits of one configuration: [step][position * vocab].
struct Result {
  std::string label;
  std::vector<std::vector<float>> steps;
  int64_t vocab = 0;
};

// One forward pass: `seq` new tokens after `past` cached ones.
struct Step {
  int64_t past, seq;
};

// hip-ep runs decoder LLMs in ONNX Runtime GenAI's shared-buffer mode: each
// layer's KV cache is one [batch, kv_heads, capacity, head_dim] tensor bound
// as both past_key_values.N.* and present.N.*, and attention_mask spans the
// capacity with ones over the valid positions.
Ort::Value make_input(const std::string &name, const Ort::TypeInfo &type,
                      Step st, int64_t capacity,
                      const std::vector<int64_t> &tokens,
                      std::vector<std::vector<uint8_t>> &storage) {
  auto info = type.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = info.GetShape();
  const bool kv = contains(name, "past_key_values");
  const bool mask = contains(name, "attention_mask");
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] >= 0)
      continue;
    if (i == 0)
      shape[i] = 1;
    else if (kv || mask)
      shape[i] = capacity;
    else
      shape[i] = st.seq;
  }
  size_t n = 1;
  for (int64_t d : shape)
    n *= size_t(d);
  auto type_id = info.GetElementType();
  size_t elem = type_id == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64   ? 8
                : type_id == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? 4
                : type_id == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4
                                                                 : 2;
  storage.emplace_back(std::max<size_t>(n * elem, 1), 0);
  uint8_t *buf = storage.back().data();
  auto put = [&](size_t i, int64_t v) {
    if (elem == 8)
      reinterpret_cast<int64_t *>(buf)[i] = v;
    else
      reinterpret_cast<int32_t *>(buf)[i] = int32_t(v);
  };
  if (type_id == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      type_id == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    for (size_t i = 0; i < n; ++i) {
      if (contains(name, "input_ids"))
        put(i, tokens[size_t(st.past) + i]);
      else if (contains(name, "position"))
        put(i, st.past + int64_t(i));
      else if (mask)
        put(i, int64_t(i) < st.past + st.seq ? 1 : 0);
      else
        put(i, 1);
    }
  }
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor(mem, buf, storage.back().size(), shape.data(),
                                  shape.size(), type_id);
}

Result run_config(Ort::Env &env, const std::string &model,
                  const std::string &label, const char *corelib, bool cpu,
                  const std::vector<int64_t> &tokens, int64_t prompt_len,
                  int64_t capacity) {
  Ort::SessionOptions so;
  if (!cpu) {
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto &d : env.GetEpDevices())
      if (std::string(d.EpName()) == kEpName)
        devices.push_back(d);
    if (devices.empty())
      throw std::runtime_error("no hipgpu EP device");
    so.AppendExecutionProvider_V2(env, devices,
                                  std::unordered_map<std::string, std::string>{
                                      {"use_ryzenai_corelib_model", corelib}});
    so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  }
  std::printf("[%s] creating session ...\n", label.c_str());
  std::fflush(stdout);
  Ort::Session session(env, std::wstring(model.begin(), model.end()).c_str(),
                       so);
  Ort::AllocatorWithDefaultOptions alloc;
  auto cpu_mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<std::string> in_names, out_names;
  for (size_t i = 0; i < session.GetInputCount(); ++i)
    in_names.push_back(session.GetInputNameAllocated(i, alloc).get());
  for (size_t i = 0; i < session.GetOutputCount(); ++i)
    out_names.push_back(session.GetOutputNameAllocated(i, alloc).get());

  // KV buffers live for the whole session: prefill writes them, decode reads
  // and extends them.
  std::vector<std::vector<uint8_t>> kv_storage;
  kv_storage.reserve(in_names.size());
  std::map<std::string, Ort::Value> kv; // keyed by the present.* name
  for (size_t i = 0; i < in_names.size(); ++i) {
    if (!contains(in_names[i], "past_key_values"))
      continue;
    std::string present = in_names[i];
    present.replace(present.find("past_key_values"),
                    std::strlen("past_key_values"), "present");
    kv.emplace(present,
               make_input(in_names[i], session.GetInputTypeInfo(i),
                          {0, prompt_len}, capacity, tokens, kv_storage));
  }

  Result r;
  r.label = label;
  const Step steps[2] = {{0, prompt_len}, {prompt_len, 1}};
  for (const Step &st : steps) {
    std::vector<std::vector<uint8_t>> storage;
    storage.reserve(in_names.size());
    Ort::IoBinding binding(session);
    for (size_t i = 0; i < in_names.size(); ++i) {
      if (contains(in_names[i], "past_key_values")) {
        std::string present = in_names[i];
        present.replace(present.find("past_key_values"),
                        std::strlen("past_key_values"), "present");
        binding.BindInput(in_names[i].c_str(), kv.at(present));
      } else {
        binding.BindInput(in_names[i].c_str(),
                          make_input(in_names[i], session.GetInputTypeInfo(i),
                                     st, capacity, tokens, storage));
      }
    }
    std::string logits_name;
    for (const auto &o : out_names) {
      auto it = kv.find(o);
      if (it != kv.end()) {
        binding.BindOutput(o.c_str(), it->second);
      } else {
        binding.BindOutput(o.c_str(), cpu_mem);
        if (contains(o, "logits"))
          logits_name = o;
      }
    }
    session.Run(Ort::RunOptions{nullptr}, binding);
    auto names = binding.GetOutputNames();
    auto values = binding.GetOutputValues();
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] != logits_name)
        continue;
      auto shape = values[i].GetTensorTypeAndShapeInfo().GetShape();
      r.vocab = shape.back();
      // CoreLib computes only the last row's logits; the rest are zeros.
      // Compare only the last position — the one generation uses to pick the
      // next token — so the prefill and decode steps are judged the same way.
      auto all = to_floats(values[i]);
      r.steps.push_back(std::vector<float>(all.end() - r.vocab, all.end()));
    }
  }
  std::printf("[%s] done\n", label.c_str());
  return r;
}

struct Stats {
  double cosine, max_abs, ref_max;
  int positions, top1_match;
  // Over the positions whose top-1 differs: the largest gap, in the
  // reference's own logits, between its top token and the tested one's. A gap
  // within the logit noise (max|diff|) is a near-tie, not a wrong answer.
  double worst_gap;
};

Stats compare(const std::vector<float> &a, const std::vector<float> &b,
              int64_t vocab) {
  Stats s{};
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
    s.max_abs = std::max(s.max_abs, double(std::fabs(a[i] - b[i])));
    s.ref_max = std::max(s.ref_max, double(std::fabs(b[i])));
  }
  s.cosine = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
  s.positions = int(a.size() / size_t(vocab));
  for (int p = 0; p < s.positions; ++p) {
    auto *pa = a.data() + size_t(p) * vocab, *pb = b.data() + size_t(p) * vocab;
    auto ta = std::max_element(pa, pa + vocab) - pa;
    auto tb = std::max_element(pb, pb + vocab) - pb;
    if (ta == tb)
      ++s.top1_match;
    else
      s.worst_gap = std::max(s.worst_gap, double(pb[tb] - pb[ta]));
  }
  return s;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: corelib_compare_decoder <model.onnx> <hipgpu.dll> "
                 "[--tokens N] [--capacity C] [--cpu] "
                 "[--min-cosine C] [--min-top1 F]\n"
                 "\n"
                 "model.onnx must be an onnxruntime-genai decoder: inputs\n"
                 "attention_mask and past_key_values.*, output logits and\n"
                 "present.*. Arbitrary ONNX models are not supported.\n");
    return 2;
  }
  std::string model = argv[1], ep_lib = argv[2];
  int64_t prompt_len = 50, capacity = 64;
  bool with_cpu = false;
  double min_cosine = 0.99, min_top1 = 0.9;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--tokens" && i + 1 < argc)
      prompt_len = std::atoll(argv[++i]);
    else if (a == "--capacity" && i + 1 < argc)
      capacity = std::atoll(argv[++i]);
    else if (a == "--cpu")
      with_cpu = true;
    else if (a == "--min-cosine" && i + 1 < argc)
      min_cosine = std::atof(argv[++i]);
    else if (a == "--min-top1" && i + 1 < argc)
      min_top1 = std::atof(argv[++i]);
  }

  // Deterministic "prompt": token ids in a range every LLM vocabulary covers.
  std::vector<int64_t> tokens(size_t(prompt_len) + 1);
  uint32_t lcg = 12345;
  for (auto &t : tokens) {
    lcg = lcg * 1664525u + 1013904223u;
    t = 1000 + int64_t(lcg % 20000u);
  }

  if (capacity < prompt_len + 1) {
    std::fprintf(stderr, "--capacity must exceed --tokens\n");
    return 2;
  }
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "corelib_compare_decoder");
    env.RegisterExecutionProviderLibrary(
        kEpName, std::wstring(ep_lib.begin(), ep_lib.end()));

    std::vector<Result> results;
    results.push_back(run_config(env, model, "GPU", "0", false, tokens,
                                 prompt_len, capacity));
    results.push_back(run_config(env, model, "NPU", "1", false, tokens,
                                 prompt_len, capacity));
    if (with_cpu)
      results.push_back(run_config(env, model, "CPU", "0", true, tokens,
                                   prompt_len, capacity));

    const char *step_names[2] = {"prefill", "decode"};
    bool pass = true;
    std::printf("\nprompt %lld tokens, vocab %lld; thresholds: cosine >= "
                "%.4f per step, top-1 >= %.0f%% over all positions\n",
                (long long)prompt_len, (long long)results[0].vocab, min_cosine,
                min_top1 * 100);
    std::printf("%-12s %-8s %10s %12s %12s %10s %10s\n", "compare", "step",
                "cosine", "max|diff|", "max|ref|", "top-1", "worst gap");
    // Top-1 is judged over every position together: one decode position alone
    // would fail on any near-tie.
    auto report = [&](const Result &x, const Result &ref, bool gate) {
      int matched = 0, positions = 0;
      std::string name = x.label + " vs " + ref.label;
      for (size_t s = 0; s < x.steps.size(); ++s) {
        Stats st = compare(x.steps[s], ref.steps[s], ref.vocab);
        matched += st.top1_match;
        positions += st.positions;
        std::printf("%-12s %-8s %10.6f %12.4f %12.4f %4d/%-5d %10.4f\n",
                    name.c_str(), step_names[s], st.cosine, st.max_abs,
                    st.ref_max, st.top1_match, st.positions, st.worst_gap);
        if (gate && st.cosine < min_cosine)
          pass = false;
      }
      std::printf("%-12s %-8s %10s %12s %12s %4d/%-5d\n", name.c_str(), "all",
                  "", "", "", matched, positions);
      if (gate && double(matched) / positions < min_top1)
        pass = false;
    };
    report(results[1], results[0], true);
    if (with_cpu) {
      report(results[0], results[2], false);
      report(results[1], results[2], false);
    }
    // The NPU computes in BF16, so its logits never equal the GPU's bit for
    // bit; identical ones mean nothing ran on the NPU (no kernels, no NPU).
    bool identical = true;
    for (size_t s = 0; s < results[1].steps.size(); ++s)
      identical = identical && results[1].steps[s] == results[0].steps[s];
    if (identical) {
      std::printf("\nRESULT: INCONCLUSIVE (NPU vs GPU): the logits are "
                  "identical, so no op ran on the NPU; see the [corelib] "
                  "lines above\n");
      return 3;
    }
    std::printf("\nRESULT: %s (NPU vs GPU)\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
