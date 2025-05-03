/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "debug_logger.hpp"
#include "unit_test_env_params.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
//
#include "morphizen/vaip.hpp"
class ContextTest : public DebugLogger {};
