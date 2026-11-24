/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Hands CoreLib model sessions from the compile pass to the custom op that
// runs them; built into hipdnn-ep-corelib with or without CoreLib.

#include "corelib_model.h"

#include <map>
#include <mutex>

namespace hipdnn_ep::corelib {

namespace {
std::mutex g_mu;
std::map<std::string, std::shared_ptr<ModelSession>> &sessions() {
  static auto *m = new std::map<std::string, std::shared_ptr<ModelSession>>;
  return *m;
}
} // namespace

void put_session(const std::string &key, std::shared_ptr<ModelSession> s) {
  std::lock_guard<std::mutex> lock(g_mu);
  sessions()[key] = std::move(s);
}

std::shared_ptr<ModelSession> take_session(const std::string &key) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = sessions().find(key);
  if (it == sessions().end())
    return nullptr;
  auto s = std::move(it->second);
  sessions().erase(it);
  return s;
}

} // namespace hipdnn_ep::corelib
