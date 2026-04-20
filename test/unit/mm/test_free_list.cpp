#include "mm_free_list.h"
#include <gtest/gtest.h>
#include <set>
#include <thread>
#include <vector>

TEST(FreeListTest, PushPopLifo) {
    mm::FreeList fl;
    fl.init(16);

    fl.push(10);
    fl.push(20);
    fl.push(30);

    EXPECT_EQ(fl.pop(), 30u);
    EXPECT_EQ(fl.pop(), 20u);
    EXPECT_EQ(fl.pop(), 10u);

    fl.shutdown();
}

TEST(FreeListTest, PopEmptyReturnsMax) {
    mm::FreeList fl;
    fl.init(4);

    EXPECT_EQ(fl.pop(), UINT32_MAX);

    fl.shutdown();
}

TEST(FreeListTest, CountTracking) {
    mm::FreeList fl;
    fl.init(16);

    EXPECT_EQ(fl.count(), 0u);
    fl.push(1);
    fl.push(2);
    EXPECT_EQ(fl.count(), 2u);

    fl.pop();
    EXPECT_EQ(fl.count(), 1u);

    fl.pop();
    EXPECT_EQ(fl.count(), 0u);
    EXPECT_TRUE(fl.empty());

    fl.shutdown();
}

TEST(FreeListTest, ConcurrentPushPop) {
    mm::FreeList fl;
    constexpr uint32_t kPerThread = 1000;
    constexpr int kThreads = 4;
    fl.init(kPerThread * kThreads);

    /* Push kPerThread * kThreads items */
    std::vector<std::thread> pushers;
    for (int t = 0; t < kThreads; ++t) {
        pushers.emplace_back([&fl, t, kPerThread]() {
            for (uint32_t i = 0; i < kPerThread; ++i) {
                EXPECT_TRUE(fl.push(t * kPerThread + i));
            }
        });
    }
    for (auto &th : pushers)
        th.join();

    EXPECT_EQ(fl.count(), kPerThread * kThreads);

    /* Pop all items concurrently */
    std::vector<std::vector<uint32_t>> results(kThreads);
    std::vector<std::thread> poppers;
    for (int t = 0; t < kThreads; ++t) {
        poppers.emplace_back([&fl, &results, t, kPerThread]() {
            for (uint32_t i = 0; i < kPerThread; ++i) {
                uint32_t val = fl.pop();
                if (val != UINT32_MAX)
                    results[t].push_back(val);
            }
        });
    }
    for (auto &th : poppers)
        th.join();

    /* Verify no duplicates */
    std::set<uint32_t> all;
    for (const auto &v : results)
        for (uint32_t x : v)
            all.insert(x);

    EXPECT_EQ(all.size(), (size_t)(kPerThread * kThreads));

    fl.shutdown();
}
