/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/* GPU-free tests for whole-graph backend selection
 * (morphizen/ort-bridge/src/backend-routing.hpp): the gpu_backend option
 * values and every v1 MIGraphX eligibility rule. A rule change that is not
 * reflected here must also bump kRoutingPolicyVersion.
 */
#include "backend-routing.hpp"

#include <cstdio>
#include <string>

using namespace morphizen;

static int failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// A static CNN read from a file: the case MIGraphX v1 accepts.
static GraphSummary cnn() {
  GraphSummary g;
  g.op_types = {"Conv", "Relu", "GlobalAveragePool", "Gemm"};
  g.input_names = {"input"};
  g.has_model_path = true;
  return g;
}

static void testParseGpuBackend() {
  CHECK(parse_gpu_backend("hip") == GpuBackend::Hip, "hip");
  CHECK(parse_gpu_backend("migraphx") == GpuBackend::MIGraphX, "migraphx");
  CHECK(parse_gpu_backend("auto") == GpuBackend::Auto, "auto");
  // Values are exact: a typo must fail session creation, not pick a default.
  CHECK(!parse_gpu_backend(""), "empty is invalid");
  CHECK(!parse_gpu_backend("MIGraphX"), "case-sensitive");
  CHECK(!parse_gpu_backend("mgx"), "unknown value");
  for (auto b : {GpuBackend::Hip, GpuBackend::MIGraphX, GpuBackend::Auto})
    CHECK(parse_gpu_backend(to_string(b)) == b, "to_string round-trips");
}

static void testStaticCnnIsEligible() {
  const auto e = migraphx_eligibility(cnn());
  CHECK(e.eligible, "static CNN from a file is eligible");
  CHECK(e.reason.empty(), "no reason when eligible");
}

static void testRejections() {
  auto g = cnn();
  g.has_model_path = false;
  auto e = migraphx_eligibility(g);
  CHECK(!e.eligible && e.reason.find("model file path") != std::string::npos,
        "in-memory model is rejected: the plugin parses the file");

  g = cnn();
  g.static_inputs = false;
  CHECK(!migraphx_eligibility(g).eligible, "symbolic input dim is rejected");

  g = cnn();
  g.static_outputs = false;
  CHECK(!migraphx_eligibility(g).eligible, "run-time output shape is rejected");

  for (const char *op : {"GroupQueryAttention", "PagedAttention",
                         "SparseAttention", "LSTM", "GRU", "RNN"}) {
    g = cnn();
    g.op_types.push_back(op);
    e = migraphx_eligibility(g);
    CHECK(!e.eligible && e.reason.find(op) != std::string::npos,
          "stateful op is rejected and named");
  }

  g = cnn();
  g.input_names.push_back("past_key_values.0.key");
  e = migraphx_eligibility(g);
  CHECK(!e.eligible &&
            e.reason.find("past_key_values.0.key") != std::string::npos,
        "KV-cache input is rejected and named");

  // Only the "past_" prefix marks a KV cache.
  g = cnn();
  g.input_names.push_back("input_past_frame");
  CHECK(migraphx_eligibility(g).eligible, "'past_' elsewhere is allowed");
}

int main() {
  testParseGpuBackend();
  testStaticCnnIsEligible();
  testRejections();
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("backend routing: all tests passed\n");
  return 0;
}
