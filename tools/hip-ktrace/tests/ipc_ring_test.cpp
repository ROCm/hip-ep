/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ktrace_ipc.h"

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  using namespace ktrace::ipc;
  const std::wstring name = L"test_" + std::to_wstring(GetCurrentProcessId()) +
                            L"_" + std::to_wstring(GetTickCount64());
  Mapping owner, peer;
  std::wstring error;
  if (!Create(name, &owner, &error) || !Open(name, &peer, &error)) {
    std::wcerr << error << L'\n';
    return 1;
  }

  constexpr std::uint32_t kThreads = 4;
  constexpr std::uint32_t kPerThread = 20000;
  std::vector<std::thread> producers;
  for (std::uint32_t thread = 0; thread < kThreads; ++thread) {
    producers.emplace_back([&, thread] {
      for (std::uint32_t i = 0; i < kPerThread; ++i) {
        Record record{};
        record.type = RecordType::api;
        record.process_id = thread;
        record.thread_id = i;
        if (!Push(&peer, record))
          std::abort();
      }
    });
  }
  for (auto &producer : producers)
    producer.join();

  std::uint64_t count = 0;
  Record record{};
  while (Pop(&owner, &record))
    ++count;
  if (count != std::uint64_t{kThreads} * kPerThread || !Empty(owner)) {
    std::cerr << "MPSC count mismatch: " << count << '\n';
    return 2;
  }

  for (std::uint32_t i = 0; i < kRingCapacity; ++i) {
    if (!Push(&peer, record)) {
      std::cerr << "ring filled early at " << i << '\n';
      return 3;
    }
  }
  if (Push(&peer, record) || owner.session->dropped.load() != 1) {
    std::cerr << "overflow was not reported\n";
    return 4;
  }

  Close(&peer);
  Close(&owner);
  std::cout << "ktrace IPC ring test passed\n";
  return 0;
}
