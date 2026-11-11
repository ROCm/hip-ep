/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_ep_test {
namespace timing {

inline std::string jsonString(const std::string& value) {
  std::string out = "\"";
  for (unsigned char c : value) {
    switch (c) {
      case '\"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 32) {
          char escaped[7];
          std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
          out += escaped;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out + "\"";
}

struct CaseContext {
  std::string leaf, shape, op_entry;
};
struct CompletedRecord {
  uint64_t id;
  double median_ms;
  bool bound = false;
};
inline thread_local CaseContext currentCase;
inline thread_local std::vector<CompletedRecord> completedRecords;
inline thread_local uint64_t nextRecordId = 1;

// One case can time several implementations. Keep their IDs until the case's
// CPU-checked CSV row selects its exact returned result; never bind the most
// recent timing merely because it appeared last in the log.
class CaseScope {
 public:
  CaseScope(std::string leaf, std::string shape, std::string op_entry)
      : saved_(currentCase), saved_records_(std::move(completedRecords)) {
    currentCase = {std::move(leaf), std::move(shape), std::move(op_entry)};
    completedRecords.clear();
  }
  ~CaseScope() {
    currentCase = std::move(saved_);
    completedRecords = std::move(saved_records_);
  }
  CaseScope(const CaseScope&) = delete;
  CaseScope& operator=(const CaseScope&) = delete;
 private:
  CaseContext saved_;
  std::vector<CompletedRecord> saved_records_;
};

class EntryScope {
 public:
  explicit EntryScope(const char* entry) : saved_(currentCase.op_entry) {
    if (entry) currentCase.op_entry = entry;
  }
  ~EntryScope() { currentCase.op_entry = std::move(saved_); }
 private:
  std::string saved_;
};

inline uint64_t rememberResult(double median_ms) {
  const uint64_t id = nextRecordId++;
  completedRecords.push_back({id, median_ms, false});
  return id;
}

inline void emitCsvBinding(const std::string& shape, const std::string& config,
                           const std::string& verdict, double rel_l2, double ms) {
  if (!(ms > 0.0)) return;  // A declined/error case has no timing PASS.
  CompletedRecord* selected = nullptr;
  for (auto& record : completedRecords) {
    if (!record.bound && record.median_ms == ms) {
      if (selected) throw std::runtime_error("UT timing: ambiguous CSV result binding");
      selected = &record;
    }
  }
  if (!selected || shape != currentCase.shape || !std::isfinite(rel_l2))
    throw std::runtime_error("UT timing: CSV result has no matching case record");
  selected->bound = true;
  std::printf("UT_TIMING_BINDING {\"record_id\":%llu,\"leaf\":%s,"
              "\"shape\":%s,\"config\":%s,\"verdict\":%s,"
              "\"cpu_correct\":%s,\"rel_l2\":%.17g,\"median_ms\":%.17g}\n",
              static_cast<unsigned long long>(selected->id),
              jsonString(currentCase.leaf).c_str(), jsonString(shape).c_str(),
              jsonString(config).c_str(), jsonString(verdict).c_str(),
              verdict == "PASS" ? "true" : "false", rel_l2, ms);
  std::fflush(stdout);
}

}  // namespace timing
}  // namespace hipdnn_ep_test
