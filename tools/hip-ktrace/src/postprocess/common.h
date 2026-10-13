/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ktrace {
namespace fs = std::filesystem;

using CsvRow = std::unordered_map<std::string, std::string>;

class CsvReader {
public:
  explicit CsvReader(const fs::path &path);
  bool IsOpen() const;
  bool Next(CsvRow &row);
  const std::vector<std::string> &Header() const { return header_; }

private:
  std::vector<char> buffer_;
  std::ifstream file_;
  std::vector<std::string> header_;
};

class CsvWriter {
public:
  explicit CsvWriter(const fs::path &path);
  bool IsOpen() const;
  void Header(const std::vector<std::string> &values);
  void Row(const std::vector<std::string> &values,
           const std::vector<size_t> &string_columns = {});

private:
  std::vector<char> buffer_;
  std::ofstream file_;
};

struct Capture {
  fs::path prefix;
  fs::path input;
  fs::path legacy_ops;

  bool IsLegacy() const { return !legacy_ops.empty(); }
  fs::path Domain(const std::string &domain) const;
  fs::path AutoOutput(const std::string &suffix) const;
};

struct Phase {
  std::string tag;
  double begin_us = 0;
  double end_us = 0;
};

struct KernelInfo {
  std::string name;
  uint64_t func_addr = 0;
  int vgpr = 0;
  uint64_t scratch = 0;
  uint64_t static_lds = 0;
  int max_threads = 0;
  std::string binary_version;
  int occ_block_size = 0;
  uint64_t occ_dyn_lds = 0;
  int blocks_per_cu = 0;
  double occupancy_pct = 0;
};

std::vector<std::string> ParseCsvLine(const std::string &line);
std::string CsvQuote(const std::string &value);
std::string JsonEscape(const std::string &value);
std::string Lower(std::string value);
bool EndsWithInsensitive(const std::string &value, const std::string &suffix);
uint64_t ToU64(const std::string &value, uint64_t fallback = 0);
int64_t ToI64(const std::string &value, int64_t fallback = 0);
double ToDouble(const std::string &value, double fallback = 0);
std::string HumanBytes(double bytes);
std::string NowIso8601();
std::string Hostname();
std::string AbsoluteUtf8(const fs::path &path);

Capture ResolveCapture(const fs::path &source);
std::vector<Phase>
SegmentPhases(std::vector<std::pair<double, std::string>> marks,
              double gap_seconds = 0.5);
std::string FindCxxFilt();
std::unordered_map<std::string, std::string>
ResolveKernelNames(const std::vector<std::string> &raw_names);

int FinalizeCommand(const std::vector<std::string> &args);
int AnalyzeCommand(const std::vector<std::string> &args);
int ChromeCommand(const std::vector<std::string> &args);
int AllCommand(const std::vector<std::string> &args);
} // namespace ktrace
