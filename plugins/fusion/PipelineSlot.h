/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PLUGIN_PIPELINE_SLOT_H
#define HIP_PLUGIN_PIPELINE_SLOT_H

//===- PipelineSlot.h - Touchpoint C: plugin pipeline-slot registry -------===//
//
// Provides a lightweight mechanism for plugins to declare WHERE in the
// onnx-to-hip pipeline their passes should run, using named stable anchors
// (pass IDs) as reference points.
//
// Design constraints (from requirements doc §3 / §6 open questions):
//  - Must not require editing Pipelines.cpp for each new plugin.
//  - Anchors are stable public names; renaming breaks plugins.
//  - Plugin specifies (anchor_pass_id, position, pass_creator).
//  - The registry is a process-global singleton — plugins call
//    PipelineSlotRegistry::get().addSlot(...) from their entry points.
//  - applyPluginSlots() is called by the pipeline builder to insert
//    registered passes at the declared anchors.
//
// Stable anchor names (matching pass IDs in Pipelines.cpp):
//   "convert-onnx-to-hip"      — after ONNX→HIP, before bufferize
//   "one-shot-bufferize"       — before/after bufferize
//   "hip-optimize-mem-refs"    — HIP-specific buffer optimizations
//   "hip-promote-strided-hip-operands"
//   "hip-pool-allocs"
//   "hip-lower-allocs"
//   "hip-resolve-extern-constants"
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include <functional>
#include <string>
#include <vector>

namespace mlir {
namespace hip {

enum class SlotPosition { Before, After };

struct PipelineSlot {
  std::string anchorPassId; // stable pass ID string, e.g. "one-shot-bufferize"
  SlotPosition position;    // insert before or after the anchor
  std::function<std::unique_ptr<Pass>()> passCreator;
  std::string description;  // for diagnostics
};

/// Process-global registry of pipeline slots contributed by plugins.
/// Thread-safety: registration happens at plugin load time (single-threaded);
/// reads happen during pipeline construction.
class PipelineSlotRegistry {
public:
  static PipelineSlotRegistry &get() {
    static PipelineSlotRegistry instance;
    return instance;
  }

  void addSlot(std::string anchorPassId, SlotPosition position,
               std::function<std::unique_ptr<Pass>()> creator,
               std::string description = "") {
    slots_.push_back({std::move(anchorPassId), position, std::move(creator),
                      std::move(description)});
  }

  const std::vector<PipelineSlot> &slots() const { return slots_; }
  void clear() { slots_.clear(); }

private:
  std::vector<PipelineSlot> slots_;
};

/// Insert registered plugin passes into \p pm at their declared anchor points.
///
/// For each PipelineSlot, this function walks the passes already in \p pm,
/// finds the anchor by pass ID, and inserts the plugin pass immediately
/// before or after it. If the anchor is not found the slot is silently skipped
/// (the anchor may belong to a different pipeline stage).
///
/// Call this at the end of buildOnnxToHipPipelineTail (or equivalent) so that
/// all base passes are added before the plugin slots are resolved.
void applyPluginSlots(OpPassManager &pm,
                      const PipelineSlotRegistry &registry =
                          PipelineSlotRegistry::get());

} // namespace hip
} // namespace mlir

#endif // HIP_PLUGIN_PIPELINE_SLOT_H
