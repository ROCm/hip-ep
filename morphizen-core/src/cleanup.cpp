/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./cleanup.hpp"
namespace morphizen {
void deinitialize_onnxruntime_morphizen_ep() { cleanup_all(); }
} // namespace morphizen
