/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/morphizen.hpp"
#include <memory>
namespace morphizen {
/**
 * @brief Get the peak working set size of the current process.
 *
 * @return The peak working (maximum resident) set size in bytes.
 */
size_t GetPeakWorkingSetSize();
MemUsageProto GetMemUsage();
/**
 * @brief Interface for measuring CPU usage.
 */
class ICPUUsage {
public:
  virtual ~ICPUUsage() = default;

  /**
   * @brief Get the average CPU utilization as a percentage.
   *
   * This function calculates the CPU usage percentage based on the system and
   * process times.
   *
   * @return The average CPU utilization as a percentage.
   */
  virtual float GetUsage() const = 0;

  /**
   * @brief Reset the CPU usage measurement.
   */
  virtual void Reset() = 0;
};

/**
 * @brief Create an instance of ICPUUsage.
 *
 * @return A unique pointer to the created ICPUUsage instance.
 */
std::unique_ptr<ICPUUsage> CreateICPUUsage();

} // namespace morphizen
