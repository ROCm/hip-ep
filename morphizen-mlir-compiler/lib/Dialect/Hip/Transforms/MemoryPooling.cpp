/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Memory Pooling Pass - Optimize buffer allocation with graph coloring
//===----------------------------------------------------------------------===//
// This pass implements buffer lifetime management:
// - Tracks all hip.alloc operations and assigns buffer indices
// - Analyzes buffer lifetimes using liveness analysis
// - Builds interference graph for overlapping lifetimes
// - Performs greedy graph coloring with first-fit offset assignment
// - Computes total pool size
// - Attaches metadata to module (pool_size, buffer_offsets, buffer_count)
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "morphizen-mlir-compiler/Dialect/Hip/IR/HipDialect.h"
#include "morphizen-mlir-compiler/Dialect/Hip/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::hip;

namespace {
// GPU memory alignment (4K page alignment)
static constexpr size_t GPU_BUFFER_ALIGNMENT = 4096;

// Align offset to boundary
static inline size_t alignOffset(size_t offset, size_t alignment) {
  return (offset + alignment - 1) / alignment * alignment;
}
} // namespace

// Static initializer to verify this file is being compiled and linked
namespace {
struct MemoryPoolingPassDebugInit {
  MemoryPoolingPassDebugInit() {
    llvm::errs() << "========================================\n";
    llvm::errs() << "MemoryPoolingPass.cpp loaded into binary!\n";
    llvm::errs() << "========================================\n";
  }
};
static MemoryPoolingPassDebugInit debugInit;
} // namespace

namespace {

// Buffer metadata for pooling analysis
struct BufferInfo {
  size_t index;        // Unique buffer index
  size_t sizeBytes;    // Buffer size in bytes
  AllocOp allocOp;     // The allocation operation
  Operation* firstUse; // First operation using this buffer
  Operation* lastUse;  // Last operation using this buffer
};

class MemoryPoolingPass
    : public PassWrapper<MemoryPoolingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryPoolingPass)

  StringRef getArgument() const final { return "memory-pooling"; }
  StringRef getDescription() const final {
    return "Optimize memory allocation using graph coloring for pool offsets";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::errs()
        << "[MemoryPooling] ========================================\n";
    llvm::errs() << "[MemoryPooling] Pass started\n";

    // Step 1: Collect all hip.alloc operations
    SmallVector<BufferInfo> buffers;
    if (failed(collectAllocations(module, buffers))) {
      llvm::errs() << "[MemoryPooling] ERROR: Failed to collect allocations\n";
      signalPassFailure();
      return;
    }

    llvm::errs() << "[MemoryPooling] Found " << buffers.size()
                 << " allocations\n";

    if (buffers.empty()) {
      llvm::errs() << "[MemoryPooling] No allocations found, skipping\n";
      llvm::errs()
          << "[MemoryPooling] ========================================\n";
      return;
    }

    // Step 2: Analyze liveness intervals
    if (failed(analyzeLiveness(module, buffers))) {
      signalPassFailure();
      return;
    }

    // Step 3: Build interference graph and compute pool offsets
    SmallVector<size_t> offsets(buffers.size());
    size_t poolSize = 0;
    if (failed(computePoolOffsets(buffers, offsets, poolSize))) {
      signalPassFailure();
      return;
    }

    // Step 4: Attach metadata to module
    attachMetadata(module, buffers, offsets, poolSize);

    // Calculate memory savings
    size_t totalIndividual = 0;
    for (const auto& buf : buffers) {
      totalIndividual += buf.sizeBytes;
    }
    size_t alignedPoolSize = alignOffset(poolSize, GPU_BUFFER_ALIGNMENT);
    size_t alignmentOverhead = alignedPoolSize - poolSize;
    double savingsPct =
        totalIndividual > 0
            ? 100.0 * (1.0 - double(alignedPoolSize) / totalIndividual)
            : 0.0;

    llvm::errs() << "[MemoryPooling] Pool size: " << alignedPoolSize
                 << " bytes "
                 << "(was " << totalIndividual << " bytes, saved " << savingsPct
                 << "%, alignment overhead: " << alignmentOverhead
                 << " bytes)\n";
    llvm::errs() << "[MemoryPooling] Processed " << buffers.size()
                 << " buffers\n";
  }

private:
  /// Collect all hip.alloc operations and compute their sizes
  LogicalResult collectAllocations(ModuleOp module,
                                   SmallVector<BufferInfo>& buffers) {
    size_t index = 0;

    llvm::errs() << "[MemoryPooling] Collecting allocations from module\n";

    // Walk all functions in the module
    size_t funcCount = 0;
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      funcCount++;
      llvm::errs() << "[MemoryPooling]   Scanning function: "
                   << funcOp.getName() << "\n";

      // Walk all hip.alloc operations in deterministic order
      funcOp.walk([&](AllocOp allocOp) {
        llvm::errs() << "[MemoryPooling]     Found hip.alloc operation\n";
        MemRefType memrefType = allocOp.getMemref().getType();

        // Compute buffer size
        auto sizeBytes = computeBufferSize(memrefType);
        if (failed(sizeBytes)) {
          allocOp->emitError("Cannot compute buffer size for dynamic memref");
          return WalkResult::interrupt();
        }

        // Create buffer info
        BufferInfo info;
        info.index = index++;
        info.sizeBytes = *sizeBytes;
        info.allocOp = allocOp;
        info.firstUse = allocOp.getOperation();
        info.lastUse = nullptr;

        buffers.push_back(info);
        return WalkResult::advance();
      });
    }

    return success();
  }

  /// Compute buffer size in bytes from memref type
  FailureOr<size_t> computeBufferSize(MemRefType memrefType) {
    Type elementType = memrefType.getElementType();
    size_t elementSize = getElementSize(elementType);

    if (elementSize == 0) {
      return failure();
    }

    // Check for dynamic shapes
    if (!memrefType.hasStaticShape()) {
      // Memory pooling requires compile-time known buffer sizes for:
      // - Graph coloring algorithm (computes static offsets)
      // - Module metadata (hipdnn.pool_size, hipdnn.buffer_offsets are
      // constants)
      // - Runtime pool allocation (single fixed-size pool)
      // See doc/design/DYNAMIC-SHAPE-DESIGN.md for design challenges
      return failure();
    }

    // Compute total size
    size_t totalElements = 1;
    for (int64_t dim : memrefType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        return failure();
      }
      totalElements *= dim;
    }

    return totalElements * elementSize;
  }

  /// Get element size in bytes
  size_t getElementSize(Type elementType) {
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      return floatType.getWidth() / 8;
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      return intType.getWidth() / 8;
    }
    return 0;
  }

  /// Analyze liveness intervals for each buffer
  LogicalResult analyzeLiveness(ModuleOp module,
                                SmallVector<BufferInfo>& buffers) {
    // Build a map from AllocOp to buffer index
    DenseMap<Operation*, size_t> allocToIndex;
    for (auto& buf : buffers) {
      allocToIndex[buf.allocOp.getOperation()] = buf.index;
    }

    // Analyze each function
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      Liveness liveness(funcOp);

      // For each buffer, find its last use
      for (auto& buf : buffers) {
        if (buf.allocOp->getParentOfType<func::FuncOp>() != funcOp) {
          continue;
        }

        Value buffer = buf.allocOp.getResult();
        Operation* lastUse = buf.allocOp.getOperation();

        // Find the last operation where this buffer is live
        for (Block& block : funcOp.getBlocks()) {
          for (Operation& op : block) {
            // Check if this operation uses the buffer
            for (Value operand : op.getOperands()) {
              if (operand == buffer) {
                lastUse = &op;
              }
            }
          }
        }

        buf.lastUse = lastUse;
      }
    }

    return success();
  }

  /// Check if two buffers have overlapping lifetimes
  bool buffersInterfere(const BufferInfo& a, const BufferInfo& b) {
    // If buffers are in different functions, they don't interfere
    if (a.allocOp->getParentOfType<func::FuncOp>() !=
        b.allocOp->getParentOfType<func::FuncOp>()) {
      return false;
    }

    // Simple interference check: if lifetimes overlap
    // We use operation ordering within the same block
    // For simplicity, assume all buffers in same function interfere
    // A more sophisticated analysis would use dominance and liveness

    // For now, use a conservative approach: buffers in the same function
    // interfere unless one is definitely dead before the other starts
    // This requires more sophisticated analysis, so we'll be conservative
    return true;
  }

  /// Compute pool offsets using greedy first-fit graph coloring
  LogicalResult computePoolOffsets(const SmallVector<BufferInfo>& buffers,
                                   SmallVector<size_t>& offsets,
                                   size_t& poolSize) {
    size_t numBuffers = buffers.size();
    offsets.resize(numBuffers);
    poolSize = 0;

    // Sort buffers by size (descending) for better packing
    SmallVector<size_t> sortedIndices;
    for (size_t i = 0; i < numBuffers; i++) {
      sortedIndices.push_back(i);
    }
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&](size_t a, size_t b) {
                return buffers[a].sizeBytes > buffers[b].sizeBytes;
              });

    // Track which buffers have been assigned
    SmallVector<bool> assigned(numBuffers, false);

    // Greedy first-fit allocation
    for (size_t idx : sortedIndices) {
      const auto& buffer = buffers[idx];

      // Try to find a suitable offset
      size_t candidateOffset = 0;
      bool foundSlot = false;

      // Try offsets at every existing allocation boundary
      SmallVector<size_t> boundaries = {0};
      for (size_t j = 0; j < numBuffers; j++) {
        if (!assigned[j] || j == idx) {
          continue;
        }
        boundaries.push_back(offsets[j]);
        boundaries.push_back(offsets[j] + buffers[j].sizeBytes);
      }
      std::sort(boundaries.begin(), boundaries.end());
      boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                       boundaries.end());

      // Try each boundary position
      for (size_t boundary : boundaries) {
        candidateOffset = alignOffset(boundary, GPU_BUFFER_ALIGNMENT);
        bool conflict = false;

        // Check if this offset conflicts with any interfering buffer
        for (size_t j = 0; j < numBuffers; j++) {
          if (j == idx || !assigned[j]) {
            continue;
          }

          // Check if buffers interfere
          if (!buffersInterfere(buffer, buffers[j])) {
            continue;
          }

          // Check for overlap: [candidateOffset, candidateOffset + size) vs
          // [offsets[j], offsets[j] + buffers[j].sizeBytes)
          size_t start1 = candidateOffset;
          size_t end1 = candidateOffset + buffer.sizeBytes;
          size_t start2 = offsets[j];
          size_t end2 = offsets[j] + buffers[j].sizeBytes;

          if (!(end1 <= start2 || end2 <= start1)) {
            conflict = true;
            break;
          }
        }

        if (!conflict) {
          foundSlot = true;
          break;
        }
      }

      // If no slot found at boundaries, append at the end
      if (!foundSlot) {
        candidateOffset = alignOffset(poolSize, GPU_BUFFER_ALIGNMENT);
      }

      // Assign offset
      offsets[idx] = candidateOffset;
      assigned[idx] = true;
      poolSize = std::max(poolSize, candidateOffset + buffer.sizeBytes);
    }

    return success();
  }

  /// Attach metadata to module and operations
  void attachMetadata(ModuleOp module, const SmallVector<BufferInfo>& buffers,
                      const SmallVector<size_t>& offsets, size_t poolSize) {
    OpBuilder builder(module.getContext());

    // Align final pool size
    size_t alignedPoolSize = alignOffset(poolSize, GPU_BUFFER_ALIGNMENT);

    // Store pool size
    module->setAttr("hipdnn.pool_size",
                    builder.getI64IntegerAttr(alignedPoolSize));

    // Store buffer count
    module->setAttr("hipdnn.buffer_count",
                    builder.getI64IntegerAttr(buffers.size()));

    // Store buffer sizes (for validation)
    SmallVector<Attribute> sizeAttrs;
    for (const auto& buf : buffers) {
      sizeAttrs.push_back(builder.getI64IntegerAttr(buf.sizeBytes));
    }
    module->setAttr("hipdnn.buffer_sizes", builder.getArrayAttr(sizeAttrs));

    // Store buffer offsets
    SmallVector<Attribute> offsetAttrs;
    for (size_t offset : offsets) {
      offsetAttrs.push_back(builder.getI64IntegerAttr(offset));
    }
    module->setAttr("hipdnn.buffer_offsets", builder.getArrayAttr(offsetAttrs));

    // CRITICAL: Attach buffer index as attribute to each hip.alloc operation
    // This allows HipToLLVM to find the index even if the operation is
    // cloned/replaced by later passes
    for (const auto& buf : buffers) {
      buf.allocOp->setAttr("hipdnn.buffer_index",
                           builder.getI64IntegerAttr(buf.index));
    }
  }
};

} // namespace

namespace mlir {
namespace hip {

std::unique_ptr<Pass> createMemoryPoolingPass() {
  llvm::errs() << "[DEBUG] createMemoryPoolingPass() called!\n";
  return std::make_unique<MemoryPoolingPass>();
}

} // namespace hip
} // namespace mlir
