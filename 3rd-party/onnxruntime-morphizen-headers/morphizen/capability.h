// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "vaip/custom_op.h"
namespace vaip {
using namespace ::onnxruntime;
std::unique_ptr<ComputeCapability>
XirSubgraphToComputeCapability1(const onnxruntime::GraphViewer& graph,
                                morphizen::ExecutionProvider* ep, size_t index);
std::vector<std::unique_ptr<ComputeCapability>> GetComputeCapabilityOps(
    const onnxruntime::GraphViewer& graph,
    morphizen::DllSafe<
        std::vector<std::unique_ptr<morphizen::ExecutionProvider>>>* ep,
    const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup);

} // namespace vaip
