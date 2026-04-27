/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mm/mm_types.h"

/* Include the internal header directly (tests are allowed to). The path
   is relative — CMake will add the lib/MemoryManager directory. */
#include "handle_table.h"

#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

static void test_basic_ops() {
  HandleTable table;

  char dummy1, dummy2, dummy3;

  mm_handle_t h1 = table.insert(&dummy1, 100, MM_CLASS_GENERIC,
                                MM_LIFETIME_TRANSIENT, MM_DEVICE_GPU0);
  mm_handle_t h2 = table.insert(&dummy2, 200, MM_CLASS_WEIGHT,
                                MM_LIFETIME_STATIC, MM_DEVICE_GPU0);
  mm_handle_t h3 = table.insert(&dummy3, 300, MM_CLASS_ACTIVATION,
                                MM_LIFETIME_STEP, MM_DEVICE_GPU0);

  assert(h1 != MM_HANDLE_INVALID);
  assert(h2 != MM_HANDLE_INVALID);
  assert(h3 != MM_HANDLE_INVALID);
  assert(h1 != h2 && h2 != h3);
  assert(h1 < h2 && h2 < h3); /* monotonically increasing */
  assert(table.size() == 3);

  /* lookup */
  mm_alloc_info_t info;
  assert(table.lookup(h1, &info));
  assert(info.handle == h1);
  assert(info.ptr == &dummy1);
  assert(info.size == 100);
  assert(info.mem_class == MM_CLASS_GENERIC);

  assert(table.lookup(h2, &info));
  assert(info.mem_class == MM_CLASS_WEIGHT);
  assert(info.lifetime == MM_LIFETIME_STATIC);

  /* remove */
  assert(table.remove(h2));
  assert(table.size() == 2);
  assert(!table.lookup(h2, &info));

  /* double-free detection */
  assert(!table.remove(h2));

  /* invalid handle lookup */
  assert(!table.lookup(MM_HANDLE_INVALID, &info));
  assert(!table.lookup(9999, &info));

  /* clear */
  table.clear();
  assert(table.size() == 0);
  assert(!table.lookup(h1, nullptr));

  printf("  basic_ops: ok\n");
}

static void test_for_each() {
  HandleTable table;
  char dummies[5];
  for (int i = 0; i < 5; i++)
    table.insert(&dummies[i], (size_t)(i + 1) * 64, MM_CLASS_GENERIC,
                 MM_LIFETIME_TRANSIENT, MM_DEVICE_GPU0);

  size_t total = 0;
  int count = 0;
  table.for_each([&](const mm_alloc_info_t &info) {
    total += info.size;
    count++;
  });
  assert(count == 5);
  assert(total == (1 + 2 + 3 + 4 + 5) * 64);

  table.clear();
  printf("  for_each: ok\n");
}

static void test_concurrent_insert() {
  HandleTable table;
  const int threads = 4;
  const int per_thread = 1000;

  std::vector<std::thread> workers;
  for (int t = 0; t < threads; t++) {
    workers.emplace_back([&table, per_thread]() {
      char dummy;
      for (int i = 0; i < per_thread; i++)
        table.insert(&dummy, 64, MM_CLASS_GENERIC, MM_LIFETIME_TRANSIENT,
                     MM_DEVICE_GPU0);
    });
  }

  for (auto &w : workers)
    w.join();

  assert(table.size() == (size_t)(threads * per_thread));
  table.clear();
  printf("  concurrent_insert: ok\n");
}

int main() {
  printf("test_handle_table:\n");
  test_basic_ops();
  test_for_each();
  test_concurrent_insert();
  printf("test_handle_table: PASSED\n");
  return 0;
}
