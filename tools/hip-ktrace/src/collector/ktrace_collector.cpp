/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "binary_journal.h"
#include "etw_collector.h"
#include "spool_writer.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
std::string Env(const char *name, const char *fallback) {
  const DWORD size = GetEnvironmentVariableA(name, nullptr, 0);
  if (!size)
    return fallback;
  std::string result(size - 1, '\0');
  GetEnvironmentVariableA(name, result.data(), size);
  return result;
}

} // namespace

int wmain(int argc, wchar_t **argv) {
  if (argc != 3 || std::wstring(argv[1]) != L"--session") {
    std::wcerr << L"usage: ktrace_collector --session <id>\n";
    return 2;
  }

  ktrace::ipc::Mapping mapping;
  std::wstring error;
  if (!ktrace::ipc::Open(argv[2], &mapping, &error)) {
    std::wcerr << L"ktrace_collector: " << error << L'\n';
    return 3;
  }

  // shutdown_requested is written only by the launcher, so a launcher that
  // exits without setting it (crash, forced kill) must end the capture too.
  const HANDLE launcher =
      OpenProcess(SYNCHRONIZE, FALSE, mapping.session->launcher_pid);
  auto shutdown_requested = [&] {
    return mapping.session->shutdown_requested.load(
               std::memory_order_acquire) != 0 ||
           (launcher && WaitForSingleObject(launcher, 0) == WAIT_OBJECT_0);
  };

  try {
    const std::string output_dir = Env("KTRACE_OUTPUT_DIR", ".");
    const std::string prefix = Env("KTRACE_OUTPUT_FILE", "ktrace");
    ktrace::collector::EtwCollector etw(output_dir, prefix);
    if (Env("KTRACE_ETW", "0") != "0") {
      std::string etw_error;
      if (!etw.Start(&etw_error))
        std::cerr << "[collector] ETW disabled: " << etw_error << '\n';
    }
    mapping.session->collector_ready.store(1, std::memory_order_release);
    while (
        !mapping.session->resume_timestamp_ns.load(std::memory_order_acquire) &&
        !shutdown_requested())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::filesystem::path journal_path =
        std::filesystem::path(output_dir) /
        ("." + prefix + "_" + std::to_string(mapping.session->launcher_pid) +
         ".ktrj");
    ktrace::collector::BinaryJournal journal(journal_path);
    std::set<std::uint32_t> process_ids;
    std::set<std::uint32_t> thread_ids;

    auto empty_since = std::chrono::steady_clock::time_point{};
    for (;;) {
      ktrace::ipc::Record record{};
      bool consumed = false;
      while (ktrace::ipc::Pop(&mapping, &record)) {
        journal.Write(record);
        process_ids.insert(record.process_id);
        if (record.thread_id)
          thread_ids.insert(record.thread_id);
        consumed = true;
      }
      if (shutdown_requested() && ktrace::ipc::Empty(mapping)) {
        const auto now = std::chrono::steady_clock::now();
        if (empty_since == std::chrono::steady_clock::time_point{})
          empty_since = now;
        if (now - empty_since >= std::chrono::milliseconds(100))
          break;
      } else {
        empty_since = {};
      }
      if (!consumed)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const std::uint64_t pushed =
        mapping.session->pushed.load(std::memory_order_relaxed);
    const std::uint64_t dropped =
        mapping.session->dropped.load(std::memory_order_relaxed);
    const std::uint64_t high_watermark =
        mapping.session->high_watermark.load(std::memory_order_relaxed);
    if (etw.Enabled()) {
      std::string etw_error;
      if (!etw.StopAndExport(process_ids, thread_ids, &etw_error))
        std::cerr << "[collector] ETW export failed: " << etw_error << '\n';
    }
    std::cout << "[collector] replaying binary journal ...\n";
    const auto replay_start = std::chrono::steady_clock::now();
    ktrace::collector::SpoolWriter writer(
        output_dir, prefix,
        mapping.session->resume_timestamp_ns.load(std::memory_order_acquire));
    journal.Replay(&writer);
    writer.Finish(pushed, dropped, high_watermark);
    journal.Remove();
    const double replay_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      replay_start)
            .count();
    std::cout << "[collector] journal replay " << replay_seconds << " s\n";
    std::cout << "[collector] records " << pushed << ", dropped " << dropped
              << ", ring peak " << high_watermark << '/'
              << ktrace::ipc::kRingCapacity << '\n';
    if (launcher)
      CloseHandle(launcher);
    ktrace::ipc::Close(&mapping);
    return dropped ? 4 : 0;
  } catch (const std::exception &exception) {
    std::cerr << "ktrace_collector: " << exception.what() << '\n';
    if (launcher)
      CloseHandle(launcher);
    ktrace::ipc::Close(&mapping);
    return 5;
  }
}
