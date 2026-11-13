/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// CPU serialization fixture; timestamps, tensors and identities are synthetic.
#include "../../lib/Runtime/chrome_trace.h"
#include <limits>
#include <type_traits>

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    return 2;
  using Json = CallCaptureJson;
  static_assert(!std::is_constructible<Json, const void *>::value,
                "capture JSON must not export tensor addresses");
  Json input = Json::object({{"slot", 0},
                             {"role", "input\n\"quoted\""},
                             {"present", true},
                             {"dtype", "float16"},
                             {"shape", Json::dimensions({1, 8})},
                             {"layout", Json::object({{"contiguous", true}})}});
  Json output =
      Json::object({{"slot", 0},
                    {"role", "output"},
                    {"present", true},
                    {"dtype", "float16"},
                    {"shape", Json::dimensions({1, 8})},
                    {"layout", Json::object({{"contiguous", true}})}});
  OpCallCapture capture;
  capture.exampleOnly = true;
  capture.callId = "synthetic-call-1";
  capture.caseId = "synthetic-case";
  capture.graphId = "synthetic-graph";
  capture.windowId = "synthetic-window";
  capture.stage = "measured";
  capture.graph = Json::object(
      {{"name", "synthetic-graph"}, {"sha256", std::string(64, 'a')}});
  capture.window = Json::object({{"graph_id", capture.graphId},
                                 {"stage", capture.stage},
                                 {"workload_id", "synthetic-workload"},
                                 {"request_id", "synthetic-request"},
                                 {"phase", "fixture"},
                                 {"start_us", 100},
                                 {"end_us", 200}});
  capture.descriptor = Json::object(
      {{"graph_id", capture.graphId},
       {"op_type", "SyntheticUnknownOp"},
       {"execution", Json::object({{"backend", "HIP-EP"},
                                   {"entry", "synthetic_entry"},
                                   {"variant", "fixture"},
                                   {"config", Json::object({})}})},
       {"inputs",
        Json::array({input, Json::object({{"slot", 1}, {"present", false}})})},
       {"outputs", Json::array({output})},
       {"attributes", Json::object({})},
       {"parameters",
        Json::object({{"scalar", 0.125},
                      {"length", 8},
                      {"missing", nullptr},
                      {"invalid", std::numeric_limits<double>::infinity()}})}});
  ChromeTrace trace;
  trace.addOp("synthetic", "opaque-shape", 101, 0.125, 100, 1, 32, &capture);
  // Exact same case/window descriptors deduplicate across actual call IDs.
  capture.callId = "synthetic-call-2";
  trace.addOp("synthetic", "opaque-shape", 102, 0.250, 101, 1, 32, &capture);
  // Preserve an unannotated legacy record; a producer must reject it as
  // complete-call evidence rather than silently dropping the missing metadata.
  trace.addOp("legacy", "opaque", 103, 0.375, 102, 1, 32);
  if (argc == 3) {
    if (std::string(argv[2]) == "conflict") {
      capture.callId = "synthetic-call-conflict";
      capture.descriptor = Json::object({{"op_type", "conflicting-case"}});
    } else if (std::string(argv[2]) == "mixed") {
      capture.callId = "synthetic-call-mixed";
      capture.exampleOnly = false;
    } else {
      return 2;
    }
    trace.addOp("synthetic", "opaque", 104, 0.125, 103, 1, 32, &capture);
  }
  trace.write(argv[1], 256);
  return 0;
}
