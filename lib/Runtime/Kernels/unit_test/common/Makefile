# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror
TARGET ?= out/test_ut_timing_policy

.PHONY: test
test: $(TARGET)
	./$(TARGET)

$(TARGET): test_ut_timing_policy.cpp ut_timing_policy.h ut_timing_device_profiles.h ut_timing_context.h ut_rank.h ut_rank_input.h ut_gpu_frequency.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) test_ut_timing_policy.cpp -o $@
