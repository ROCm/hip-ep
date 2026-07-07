/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
namespace morphizen {
struct SimpleConvPattern {
  std::shared_ptr<Pattern> input_;
  std::shared_ptr<Pattern> weight_fix_;
  std::shared_ptr<Pattern> bias_fix_; // optional
  std::shared_ptr<Pattern> conv_;
  std::shared_ptr<Pattern> relu_;     // optional
  std::shared_ptr<Pattern> end_fix_;
};

} // namespace morphizen
