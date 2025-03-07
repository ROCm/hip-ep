/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include <cstddef>

#include "profile_utils.hpp"
#include <glog/logging.h>

namespace vaip_core {

/**
 * @brief Calculates the CPU usage percentage.
 *
 * This function calculates the CPU usage percentage by comparing the system
 * and process times. It returns the CPU usage as a float value between 0.0 and
 * 100.0.
 *
 * @return The CPU usage percentage.
 */
#ifdef _WIN32
#  include <Windows.h>
#  include <psapi.h>
size_t GetPeakWorkingSetSize() {
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return pmc.PeakWorkingSetSize;
  }

  return 0;
}
MemUsageProto GetMemUsage() {
  auto ret = MemUsageProto();
  constexpr auto kbytes = 1024;
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    ret.set_peak_memory_in_bytes((int64_t)pmc.PeakWorkingSetSize);
    ret.set_current_memory_in_bytes((int64_t)pmc.WorkingSetSize);
  }
  return ret;
}
/**
 * @brief Subtract two FILETIME values and return the result as a 64-bit
 * unsigned integer.
 *
 * This function subtracts two FILETIME values and returns the result as a
 * 64-bit unsigned integer.
 *
 * @param ft_a The first FILETIME value.
 * @param ft_b The second FILETIME value.
 * @return The difference between the two FILETIME values.
 */
static std::uint64_t SubtractFILETIME(const FILETIME& ft_a,
                                      const FILETIME& ft_b) {
  LARGE_INTEGER a, b;
  a.LowPart = ft_a.dwLowDateTime;
  a.HighPart = ft_a.dwHighDateTime;

  b.LowPart = ft_b.dwLowDateTime;
  b.HighPart = ft_b.dwHighDateTime;

  return a.QuadPart - b.QuadPart;
}

/**
 * @brief A class for calculating CPU usage on Windows.
 */
class CPUUsage : public ICPUUsage {
public:
  CPUUsage() { Reset(); }

  /**
   * @brief Get the CPU usage percentage.
   *
   * This function calculates the CPU usage percentage by comparing the system
   * and process times. It returns the CPU usage as a float value between 0.0
   * and 100.0.
   *
   * @return The CPU usage percentage.
   */
  float GetUsage() const override {
    FILETIME sys_idle_ft, sys_kernel_ft, sys_user_ft, proc_creation_ft,
        proc_exit_ft, proc_kernel_ft, proc_user_ft;
    GetSystemTimes(&sys_idle_ft, &sys_kernel_ft, &sys_user_ft);
    GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft,
                    &proc_kernel_ft, &proc_user_ft);

    std::uint64_t sys_kernel_ft_diff =
        SubtractFILETIME(sys_kernel_ft, sys_kernel_ft_);
    std::uint64_t sys_user_ft_diff =
        SubtractFILETIME(sys_user_ft, sys_user_ft_);

    std::uint64_t proc_kernel_diff =
        SubtractFILETIME(proc_kernel_ft, proc_kernel_ft_);
    std::uint64_t proc_user_diff =
        SubtractFILETIME(proc_user_ft, proc_user_ft_);

    std::uint64_t total_sys = sys_kernel_ft_diff + sys_user_ft_diff;
    std::uint64_t total_proc = proc_kernel_diff + proc_user_diff;
    return total_sys > 0
               ? static_cast<float>((double)(100.0 * total_proc) / total_sys)
               : 0.0f;
  }

  /**
   * @brief Resets the system and process times.
   */
  void Reset() override {
    FILETIME sys_idle_ft, proc_creation_ft, proc_exit_ft;
    GetSystemTimes(&sys_idle_ft, &sys_kernel_ft_, &sys_user_ft_);
    GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft,
                    &proc_kernel_ft_, &proc_user_ft_);
  }

private:
  FILETIME sys_kernel_ft_;  ///< The system kernel time.
  FILETIME sys_user_ft_;    ///< The system user time.
  FILETIME proc_kernel_ft_; ///< The process kernel time.
  FILETIME proc_user_ft_;   ///< The process user time.
};
#else
#  include <sys/resource.h>
#  include <sys/times.h>
std::size_t GetPeakWorkingSetSize() {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return static_cast<size_t>(rusage.ru_maxrss * 1024L);
}

/**
 * @brief Get the number of physical CPU cores.
 *
 * This function retrieves the number of physical CPU cores.
 *
 * @return The number of physical CPU cores.
 */
static int GetNumPhyCPUCores() {
  // TODO
  return 1;
}

/**
 * @brief A class for calculating CPU usage on Linux.
 */
class CPUUsage : public ICPUUsage {
public:
  CPUUsage() { Reset(); }

  /**
   * @brief Get the CPU usage percentage.
   *
   * This function calculates the CPU usage percentage by comparing the system
   * and process times.
   *
   * @return The CPU usage percentage.
   */
  float GetUsage() const override {
    struct tms time_sample;
    clock_t total_clock_now = times(&time_sample);
    if (total_clock_now <= total_clock_start_ ||
        time_sample.tms_stime < proc_sys_clock_start_ ||
        time_sample.tms_utime < proc_user_clock_start_) {
      // overflow detection
      return -1;
    } else {
      clock_t proc_total_clock_diff =
          (time_sample.tms_stime - proc_sys_clock_start_) +
          (time_sample.tms_utime - proc_user_clock_start_);
      clock_t total_clock_diff = total_clock_now - total_clock_start_;
      return static_cast<float>(100.0 * (double)proc_total_clock_diff /
                                (double)total_clock_diff / GetNumPhyCPUCores());
    }
  }

  void Reset() override {
    struct tms time_sample;
    total_clock_start_ = times(&time_sample);
    proc_sys_clock_start_ = time_sample.tms_stime;
    proc_user_clock_start_ = time_sample.tms_utime;
  }

private:
  clock_t total_clock_start_;     ///< The total clock time.
  clock_t proc_sys_clock_start_;  ///< The process system clock time.
  clock_t proc_user_clock_start_; ///< The process user clock time.
};
MemUsageProto GetMemUsage() {
  auto ret = MemUsageProto();
  return ret;
}
#endif

/**
 * @brief Create an instance of ICPUUsage.
 *
 * @return A unique pointer to ICPUUsage.
 */
std::unique_ptr<ICPUUsage> CreateICPUUsage() {
  return std::make_unique<CPUUsage>();
}

} // namespace vaip_core
