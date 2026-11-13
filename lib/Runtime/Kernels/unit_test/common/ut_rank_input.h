/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "ut_rank.h"

namespace hipdnn_ep_test {
namespace timing {
struct RankRow {
  std::string shape;
  std::vector<std::string> configs;
  bool priority = false;
};
inline bool rankRequested() { const char* p = std::getenv("HIPDNN_UT_RANK"); return p && p[0]; }
inline std::vector<std::string> rankCsvFields(const std::string& line) {
  std::vector<std::string> out;
  std::string value;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '\r') continue;
    if (c == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') { value += '"'; ++i; }
      else quoted = !quoted;
    } else if (c == ',' && !quoted) { out.push_back(value); value.clear(); }
    else value += c;
  }
  if (quoted) throw std::invalid_argument("UT rank: unterminated CSV quote");
  out.push_back(value); return out;
}
inline std::vector<RankRow> readRankRows(const std::string& leaf) {
  std::ifstream in(std::getenv("HIPDNN_UT_RANK"));
  if (!in) throw std::invalid_argument("UT rank: cannot open candidate CSV");
  std::string line;
  if (!std::getline(in, line) || rankCsvFields(line) !=
      (std::vector<std::string>{"leaf", "shape", "priority", "config"}))
    throw std::invalid_argument("UT rank: expected leaf,shape,priority,config CSV header");
  std::vector<RankRow> rows;
  std::map<std::string, size_t> indices;
  while (std::getline(in, line)) {
    if (line.empty() || line == "\r") continue;
    const auto fields = rankCsvFields(line);
    if (fields.size() != 4 || fields[1].empty() || fields[3].empty() ||
        (fields[2] != "0" && fields[2] != "1"))
      throw std::invalid_argument("UT rank: malformed candidate CSV row");
    if (fields[0] != leaf) continue;
    auto it = indices.find(fields[1]);
    if (it == indices.end()) {
      it = indices.emplace(fields[1], rows.size()).first;
      rows.push_back({fields[1], {}, fields[2] == "1"});
    }
    auto& row = rows[it->second];
    row.priority = row.priority || fields[2] == "1";
    row.configs.push_back(fields[3]);
  }
  if (rows.empty()) throw std::invalid_argument("UT rank: candidate CSV contains no rows for this leaf");
  return rows;
}
inline std::map<std::string, int> rankShapeFields(const std::string& shape) {
  std::istringstream in(shape);
  std::string token;
  std::map<std::string, int> fields;
  while (std::getline(in, token, '_')) {
    size_t end = 0;
    while (end < token.size() && ((token[end] >= 'a' && token[end] <= 'z') ||
                                 (token[end] >= 'A' && token[end] <= 'Z'))) ++end;
    const std::string key = token.substr(0, end);
    if (key.empty() || fields.count(key)) throw std::invalid_argument("UT rank: invalid shape field");
    if (end == token.size()) {
      if (key != "sink" && key != "smooth") throw std::invalid_argument("UT rank: unknown shape flag");
      fields[key] = 1;
    } else {
      size_t consumed = 0;
      const std::string number = token.substr(end);
      const int value = std::stoi(number, &consumed);
      if (consumed != number.size() || value < 0) throw std::invalid_argument("UT rank: invalid shape integer");
      fields[key] = value;
    }
  }
  return fields;
}

inline int rankConfigInt(const std::string& config, const std::string& name) {
  std::istringstream in(config);
  std::string field;
  bool found = false;
  int result = 0;
  while (std::getline(in, field, ';')) {
    if (field.substr(0, name.size() + 1) != name + "=") continue;
    if (found) throw std::invalid_argument("UT rank: duplicate config field");
    found = true;
    const std::string number = field.substr(name.size() + 1);
    size_t end = 0;
    result = std::stoi(number, &end);
    if (end != number.size()) throw std::invalid_argument("UT rank: invalid config integer");
  }
  if (!found) throw std::invalid_argument("UT rank: missing config field " + name);
  return result;
}
}  // namespace timing
}  // namespace hipdnn_ep_test
