# Async Execution Pipeline Design

## Overview

This document describes the async execution pipeline design for the ROCm custom op. The pipeline is designed to:

1. **Minimize host-device synchronization** - avoid blocking waits
2. **Overlap compute with memory transfers** - D2H can run while GPU kernels execute
3. **Keep intermediate tensors on GPU** - no unnecessary round-trips to host

## Memory Flow Architecture

### Traditional (Non-Fused) Execution

Without subgraph fusion, each operation requires separate H2D and D2H transfers:

```
[Conv1]                    [Conv2]
  │                          │
H2D(X) → Conv1 → D2H(T1) → H2D(T1) → Conv2 → D2H(Y)
  ↓                          ↓
Host ←─────────────────────────────────────────→ Host

Total transfers: 4 (2× H2D, 2× D2H)
```

### Fused Subgraph Execution

With subgraph fusion, intermediate tensor T1 stays on GPU:

```
[RocmSubgraph: Conv1 → Conv2]

H2D(X) → Conv1 → Conv2 → D2H(Y)
           │       ↑
           └───────┘
          T1 on GPU (no transfer)

Total transfers: 2 (1× H2D, 1× D2H)
```

## Execution Pipeline Design

### Phase 1: Async Input Upload

Upload all external inputs to GPU using async copies:

```cpp
void RocmCustomOp::UploadExternalInputs(OrtKernelContext* context) {
  for (const auto& node : subgraph_.nodes()) {
    for (const auto& input : node.inputs()) {
      if (input.has_external_name()) {
        const auto& name = input.external_name();
        
        // Get host pointer from ORT
        const OrtValue* ort_value = GetOrtInput(context, name);
        float* h_data = GetTensorData<float>(ort_value);
        size_t bytes = GetTensorBytes(ort_value);
        
        // Allocate device buffer (cached)
        float* d_data = AllocateDeviceBuffer(name, bytes);
        
        // Async H2D copy
        hipMemcpyAsync(d_data, h_data, bytes, 
                       hipMemcpyHostToDevice, stream_);
        
        external_input_buffers_[name] = d_data;
      }
    }
  }
}
```

### Phase 2: Execute Nodes in Topological Order

Execute each node, using internal tensor pool for intermediates:

```cpp
void RocmCustomOp::ExecuteSubgraph() {
  for (const auto& node : subgraph_.nodes()) {
    // Gather input pointers
    std::vector<float*> inputs;
    for (const auto& input : node.inputs()) {
      if (input.has_external_name()) {
        inputs.push_back(external_input_buffers_[input.external_name()]);
      } else {
        auto& ref = input.internal();
        inputs.push_back(
            intermediate_tensors_[ref.producer_node_id()][ref.output_index()]);
      }
    }
    
    // Allocate output buffers
    std::vector<float*> outputs;
    for (int i = 0; i < node.output_names_size(); ++i) {
      outputs.push_back(AllocateIntermediateBuffer(node.node_id(), i));
    }
    
    // Execute operation
    if (node.params().op_type() == "conv") {
      ExecuteConv(node.params().conv_params(), inputs, outputs);
    } else if (node.params().op_type() == "gemm") {
      ExecuteGemm(node.params().gemm_params(), inputs, outputs);
    }
    
    // Schedule async D2H for external outputs
    ScheduleExternalOutputCopies(node.node_id());
  }
}
```

### Phase 3: Async Output Download (Overlapped)

The key optimization: schedule D2H copies as soon as the producing node completes, while subsequent GPU kernels continue executing:

```cpp
void RocmCustomOp::ScheduleExternalOutputCopies(int32_t completed_node_id) {
  for (const auto& output : subgraph_.outputs()) {
    if (output.producer_node_id() == completed_node_id) {
      // Get device buffer
      float* d_data = 
          intermediate_tensors_[output.producer_node_id()][output.output_index()];
      
      // Get host buffer from ORT
      OrtValue* ort_output = GetOrtOutput(context_, output.name());
      float* h_data = GetTensorData<float>(ort_output);
      size_t bytes = GetOutputBytes(output.name());
      
      // Async D2H copy - overlaps with subsequent GPU work!
      hipMemcpyAsync(h_data, d_data, bytes,
                     hipMemcpyDeviceToHost, stream_);
    }
  }
}
```

### Phase 4: Final Synchronization

Wait for all operations to complete with timeout protection:

```cpp
void RocmCustomOp::SyncStream() {
  TimeoutStatus status = HipContext::instance().sync_stream_with_timeout();
  if (status == TimeoutStatus::TIMEOUT) {
    throw std::runtime_error("GPU operation timed out");
  }
}
```

## Timeline Visualization

### Case 1: Simple Sequential Subgraph

```
X → Conv1 → Conv2 → Y

Time →
┌────────────────────────────────────────────────────────┐
│ H2D(X) │ Conv1 │ Conv2 │ D2H(Y) │ sync │              │
└────────────────────────────────────────────────────────┘
                              ↑
                          Only 1 D2H needed
```

### Case 2: Branching Subgraph (Multiple External Outputs)

```
X → Conv1 → Conv2 → Y1
         ↘
          Gemm → Y2

Time →
┌──────────────────────────────────────────────────────────────┐
│ H2D(X) │ Conv1 │ Conv2 │ D2H(Y1) │ Gemm │ D2H(Y2) │ sync    │
└──────────────────────────────────────────────────────────────┘
                         ↑                 ↑
                    Overlapped!       Overlapped with sync prep
```

### Case 3: Multiple External Outputs from Same Node

```
X → Conv → Y1 (consumed externally)
         → Conv2 (uses Conv output) → Y2

RocmSubgraphProto:
  outputs: [
    { name: "Y1", producer_node_id: 0, output_index: 0 },
    { name: "Y2", producer_node_id: 1, output_index: 0 }
  ]

Time →
┌───────────────────────────────────────────────────────────────┐
│ H2D(X) │ Conv │ D2H(Y1) │ Conv2 │ D2H(Y2) │ sync             │
└───────────────────────────────────────────────────────────────┘
                ↑                   ↑
            D2H immediately    D2H overlapped with sync
```

## Memory Management

> **Note:** For detailed GPU memory layout, buffer categories, and resource lifecycle, see [08_ROCM_RESOURCE_MANAGEMENT.md](08_ROCM_RESOURCE_MANAGEMENT.md).

**Key buffer types for async execution:**

| Category | Role in Async Pipeline |
|----------|------------------------|
| External Input | H2D at start of pipeline |
| External Output | D2H overlapped with compute |
| Intermediate | GPU-only, enables fusion benefit |
| Weights | Pre-loaded, no transfer overhead |

## Performance Considerations

### When Async Overlap Helps Most

1. **Large output tensors** - D2H transfer time is significant
2. **Many operations in subgraph** - More opportunity for overlap
3. **Branching subgraphs** - Early outputs can copy while later ops run

### When Async Overlap Has Limited Benefit

1. **Single operation subgraph** - No subsequent GPU work to overlap with
2. **Small tensors** - Transfer time is negligible
3. **Final output only** - D2H overlaps only with sync (minimal benefit)

### Memory Bandwidth Considerations

On AMD GPUs, PCIe bandwidth can become a bottleneck:

- **PCIe 4.0 x16**: ~25 GB/s theoretical
- **HBM2e (Compute)**: ~1.5 TB/s

For compute-bound workloads, async D2H rarely impacts kernel execution. For bandwidth-bound workloads, careful scheduling is more important.

## Error Handling

### Timeout Detection

Each phase can timeout independently:

```cpp
void RocmCustomOp::Compute(...) {
  // Phase 1: H2D
  UploadExternalInputs(context);
  
  // Phase 2 + 3: Execute + D2H (interleaved)
  ExecuteSubgraph();
  
  // Phase 4: Final sync with timeout
  auto status = HipContext::instance().sync_stream_with_timeout();
  if (status != TimeoutStatus::SUCCESS) {
    LOG(ERROR) << "Subgraph execution failed: " 
               << (status == TimeoutStatus::TIMEOUT ? "timeout" : "error");
    throw std::runtime_error("ROCm subgraph execution failed");
  }
}
```

### Resource Cleanup on Error

```cpp
RocmCustomOp::~RocmCustomOp() {
  // Clean up intermediate buffers
  for (auto& [node_id, outputs] : intermediate_tensors_) {
    for (auto* ptr : outputs) {
      if (ptr) hipFree(ptr);
    }
  }
  
  // Clean up external input buffers
  for (auto& [name, ptr] : external_input_buffers_) {
    if (ptr) hipFree(ptr);
  }
  
  // Clean up weight buffers
  for (auto& [node_id, weights] : weight_buffers_) {
    for (auto* ptr : weights) {
      if (ptr) hipFree(ptr);
    }
  }
}
```

## See Also

- [01_DESIGN.md](01_DESIGN.md) - Overall project design
- [02_LEVEL1_PASS_DESIGN.md](02_LEVEL1_PASS_DESIGN.md) - Subgraph construction
- [05_GPU_TIMEOUT_HANDLING.md](05_GPU_TIMEOUT_HANDLING.md) - Timeout mechanisms
