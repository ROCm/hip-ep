/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#ifdef ENABLE_BOOST
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
namespace morphizen {
class WithFileLock {
public:
  WithFileLock(std::filesystem::path filename);
  ~WithFileLock();

private:
  boost::interprocess::file_lock lock_;
};
#else
#include <mutex>
namespace morphizen {
class WithFileLock {
public:
  WithFileLock(std::filesystem::path filename);
  ~WithFileLock();

private:
  std::lock_guard<std::mutex> lock_;
};
#endif
} // namespace morphizen
