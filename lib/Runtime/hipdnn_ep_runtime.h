/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_EP_RUNTIME_H
#define HIP_EP_RUNTIME_H

#include "hipdnn_ep_errors.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for runtime state
typedef struct RuntimeState RuntimeState;

//==============================================================================
// RuntimeState: Opaque Execution State
//==============================================================================
//
// RuntimeState encapsulates GPU execution resources (stream, library handles,
// model constants). Generated code treats it as opaque void*, runtime library
// owns the internal structure.
//
// Design rationale: Opaque pointer pattern allows runtime to evolve internal
// layout without breaking generated code.
//
// Lifecycle: init -> use -> cleanup (must call in this order)
// Thread safety: Not thread-safe (one inference per state at a time)
//==============================================================================

// Initialize runtime state with external constant storage via FileSystem.
// Used when compiled with hip_compile_with_fs.
// Reads constants_filename and constant_sizes from the FlatBuffers blob
// (UdnaModelMetaInfo schema), opens the file via fs, reads each constant
// sequentially, uploads each to GPU via hipMalloc+hipMemcpy.
//   out_state:     Pointer to receive allocated RuntimeState
//   fs:            morphizen::FileSystem* (void* for C ABI) - must not be null
//   metadata_blob: FlatBuffers binary blob (UdnaModelMetaInfo) baked into DLL
//   blob_size:     Size of metadata_blob in bytes
// Return codes: 0=success, 1=alloc/read error, 2-9=GPU handle init error
int hipdnn_ep_state_init_with_fs(RuntimeState** out_state, void* fs,
                                 const void* metadata_blob, size_t blob_size);

// Cleanup runtime state (destroys handles, frees memory)
// Best-effort cleanup - continues even if individual operations fail
// Returns 0 always (best-effort)
int hipdnn_ep_state_cleanup(RuntimeState* state);

// Get GPU stream from state (for passing to HIP operations)
// Returns: hipStream_t cast to void* (NULL on error)
// Ownership: Caller does NOT own stream (destroyed in cleanup)
void* hipdnn_ep_state_get_stream(RuntimeState* state);

// Get MIOpen handle from state (for MIOpen operations)
// Returns: miopenHandle_t cast to void* (NULL on error)
// Ownership: Caller does NOT own handle (destroyed in cleanup)
void* hipdnn_ep_state_get_miopen_handle(RuntimeState* state);

// Get buffer from memory pool by index
// Returns: GPU pointer at pool_base + buffer_offsets[index] (NULL on error)
// Ownership: Caller does NOT own pointer (freed in cleanup)
void* hipdnn_ep_get_buffer_from_pool(RuntimeState* state, size_t index);

// Get the base pointer of the GPU memory pool
// Returns: GPU base pointer of pool (NULL if pool not initialized)
// Used by hip.get_pool lowering in generated compute kernels
void* hipdnn_ep_get_pool_base(RuntimeState* state);

// Initialize memory pool in runtime state
// Called by generated inference_init after creating RuntimeState
// Parameters:
//   state: Runtime state to initialize pool in
//   pool_size: Total size of memory pool in bytes
//   buffer_offsets: Array of offsets for each buffer
//   num_buffers: Number of buffers
// Returns: 0=success, non-zero=error
int hipdnn_ep_pool_init(RuntimeState* state, size_t pool_size,
                        const size_t* buffer_offsets, size_t num_buffers);

//==============================================================================
// Inference API Types (for generated interface)
//==============================================================================

// Represents a tensor with host data and shape information
typedef struct {
  void* data;         // Host data pointer
  int64_t* shape;     // Array of dimension sizes
  size_t rank;        // Number of dimensions
  size_t element_size; // Bytes per element (e.g. 4=float32, 2=float16, 8=int64)
} tensor_t;

// Represents a span of tensors (inputs or outputs)
typedef struct {
  tensor_t* data; // Array of tensors
  size_t count;   // Number of tensors
} span_t;

// Represents a prepared tensor with GPU buffer and metadata
// Used internally by tensor preparation helpers
typedef struct {
  void* gpu_ptr;      // GPU memory (allocated or from pool)
  void* host_ptr;     // Host memory (from tensor_t.data)
  int64_t* shape_ptr; // Shape array (from tensor_t.shape) for memref building
  size_t rank;        // Tensor rank (for validation)
  size_t size_bytes;  // Buffer size
  bool is_pooled;     // Internal: true if from pool, false if allocated
} TensorBuffer;

//==============================================================================
// Constant Access (used by generated inference code)
//==============================================================================

// Get GPU pointer for constant at index
// Returns: GPU pointer (NULL if index out of range or state invalid)
// Ownership: Caller does NOT own pointer (freed in state_cleanup)
void* hipdnn_ep_constant_get(RuntimeState* state, int64_t index);

//==============================================================================
// Tensor Preparation Helpers (allocation-strategy agnostic)
//==============================================================================
//
// These helpers abstract tensor preparation logic (parsing, validation,
// GPU allocation, H2D/D2H transfer) from the generated code.
//
// Design principle: The runtime handles allocation strategy internally.
// Generated code is allocation-strategy agnostic.
//
// Element size: Read from tensor_t.element_size, set by the EP caller.
//==============================================================================

// Prepare input tensor: parse, validate, get/allocate GPU buffer, H2D transfer
//
// Parameters:
//   state: Runtime state (provides stream, may contain pre-allocated buffers)
//   inputs: Span of input tensors
//   index: Which tensor to prepare (0-based)
//   expected_rank: Compile-time known rank (from module metadata)
//   out_buffer: Output TensorBuffer to populate
int hipdnn_ep_tensor_prepare_input(RuntimeState* state, span_t* inputs,
                                   size_t index, size_t expected_rank,
                                   TensorBuffer* out_buffer);

// Prepare output tensor: parse, validate, get/allocate GPU buffer (no H2D)
//
// Parameters: same as prepare_input
int hipdnn_ep_tensor_prepare_output(RuntimeState* state, span_t* outputs,
                                    size_t index, size_t expected_rank,
                                    TensorBuffer* out_buffer);

// Finalize output tensor: D2H transfer, sync, release buffer
//
// The runtime handles buffer release internally (free, return to pool, or keep
// if pre-allocated).
//
// Parameters:
//   state: Runtime state
//   buffer: TensorBuffer from prepare_output
//
// Return codes:
//   HIPDNN_EP_SUCCESS (0) = success
//   HIPDNN_EP_ERR_D2H_TRANSFER_FAILED = D2H transfer failed
//   HIPDNN_EP_ERR_STREAM_SYNC_FAILED = stream sync failed
//
// Note: Buffer is released even on error (best-effort cleanup)
int hipdnn_ep_tensor_finalize_output(RuntimeState* state, TensorBuffer* buffer);

// Release input tensor buffer (no D2H transfer needed)
//
// Parameters:
//   state: Runtime state
//   buffer: TensorBuffer from prepare_input
void hipdnn_ep_tensor_free_input(RuntimeState* state, TensorBuffer* buffer);

// TensorBuffer Field Accessors (Opaque Pattern)
//==============================================================================
//
// These accessors allow generated code to extract fields from TensorBuffer
// without knowing its internal layout. This maintains abstraction and allows
// the struct definition to evolve without breaking generated code.
//
// Design: TensorBuffer is opaque to generated MLIR code, accessed only via
// these functions (same pattern as RuntimeState accessors above).
//==============================================================================

// Get GPU pointer from TensorBuffer
// Returns: GPU memory pointer (NULL on error)
void* hipdnn_ep_tensor_buffer_get_gpu_ptr(TensorBuffer* buffer);

// Get host pointer from TensorBuffer
// Returns: Host memory pointer (NULL on error)
void* hipdnn_ep_tensor_buffer_get_host_ptr(TensorBuffer* buffer);

// Get shape array pointer from TensorBuffer
// Returns: Pointer to int64_t shape array (NULL on error)
int64_t* hipdnn_ep_tensor_buffer_get_shape_ptr(TensorBuffer* buffer);

// Get rank from TensorBuffer
// Returns: Tensor rank (number of dimensions)
size_t hipdnn_ep_tensor_buffer_get_rank(TensorBuffer* buffer);

// Get buffer size in bytes from TensorBuffer
// Returns: Size in bytes
size_t hipdnn_ep_tensor_buffer_get_size_bytes(TensorBuffer* buffer);

//==============================================================================
// Memory Operations
//==============================================================================

// HIP memory copy wrapper (GPU-to-GPU using hipMemcpyAsync)
// Follows opaque RuntimeState pattern - extracts stream internally
//
// Parameters:
//   state: Runtime state (provides GPU stream)
//   dst_ptr: Destination GPU buffer pointer
//   src_ptr: Source GPU buffer pointer
//   size_bytes: Number of bytes to copy
//
// Return codes:
//   0 = success
//   -1 = copy failed
int wrap_hipMemcpyAsync(RuntimeState* state, void* dst_ptr, const void* src_ptr,
                        size_t size_bytes);

//==============================================================================
// Library Operations (MIOpen, hipBLAS)
//==============================================================================

// MIOpen convolution forward operation
// Full wrapper with descriptor creation, algorithm finding, workspace
// management Follows opaque RuntimeState pattern - extracts handle/stream
// internally Parameters match generated LLVM IR from HipToLLVM pass
int wrap_miopenConvolutionForward(
    RuntimeState*
        state, // RuntimeState (opaque - extracts handle/stream internally)
    const void* input,   // Input tensor GPU pointer
    int64_t input_n,     // Input batch size
    int64_t input_c,     // Input channels
    int64_t input_h,     // Input height
    int64_t input_w,     // Input width
    const void* weights, // Weights tensor GPU pointer
    int64_t weights_k,   // Output channels (number of filters)
    const void* bias,    // Bias tensor GPU pointer (nullable)
    void* output,        // Output tensor GPU pointer (in-place)
    int64_t output_h,    // Output height
    int64_t output_w,    // Output width
    int64_t kernel_h,    // Kernel height
    int64_t kernel_w,    // Kernel width
    int64_t stride_h,    // Stride height
    int64_t stride_w,    // Stride width
    int64_t pad_top,     // Padding top
    int64_t pad_left,    // Padding left
    int64_t pad_bottom,  // Padding bottom
    int64_t pad_right,   // Padding right
    int64_t dilation_h,  // Dilation height
    int64_t dilation_w,  // Dilation width
    int64_t group);      // Number of groups

// hipBLASLt GEMM operation wrapper
// Called by generated IR for matrix multiplication operations
int wrap_hipblasLtGemm(void* handle,      // hipBLASLt handle
                       void* stream,      // HIP stream
                       int64_t m, int64_t n, int64_t k,
                       const void* alpha, // Scalar alpha
                       const void* A,     // Matrix A GPU pointer
                       const void* B,     // Matrix B GPU pointer
                       const void* beta,  // Scalar beta
                       void* C);          // Matrix C GPU pointer (in/out)

// MatMul operation wrapper (batched matrix multiplication)
// Called by generated IR for onnx.MatMul lowering
// Computes output = A @ B for each batch
// A: [batch_count x M x K], B: [K x N] (broadcast) or [batch_count x K x N]
// output: [batch_count x M x N]
int wrap_matmul(RuntimeState* state,
                const void* A,      // Matrix A GPU pointer
                const void* B,      // Matrix B GPU pointer
                void* output,       // Output GPU pointer
                int64_t M,          // Rows of A (per batch)
                int64_t N,          // Columns of B
                int64_t K,          // Columns of A / Rows of B
                int64_t batch_count, // Number of batches
                int64_t elem_size); // Element size in bytes (2=f16, 4=f32)

// GroupQueryAttention operation wrapper
// Called by generated IR for onnx.Custom(GroupQueryAttention) lowering
int wrap_group_query_attention(
    RuntimeState* state,
    void* query, void* key, void* value,
    void* past_key, void* past_value,
    void* seqlens_k, void* total_seq_len,
    void* output, void* present_key, void* present_value,
    int64_t num_heads, int64_t kv_num_heads,
    float scale, float softcap,
    int64_t do_rotary, int64_t rotary_interleaved);

// Element-wise multiplication wrapper
// Computes output = lhs * rhs element-wise
int wrap_elementwise_mul(RuntimeState* state, void* lhs, void* rhs,
                         void* output, int64_t num_elements,
                         int64_t element_size_bytes);

// Element-wise subtraction wrapper
// Computes output = lhs - rhs element-wise
int wrap_elementwise_sub(RuntimeState* state, void* lhs, void* rhs,
                         void* output, int64_t num_elements,
                         int64_t element_size_bytes);

// Gather operation wrapper
int wrap_gather(RuntimeState* state, void* data, void* indices,
                void* output, int64_t axis, int64_t data_num_elements,
                int64_t output_num_elements, int64_t element_size_bytes);

// ReduceSum operation wrapper
int wrap_reduce_sum(RuntimeState* state, void* data, void* axes,
                    void* output, int64_t data_num_elements,
                    int64_t output_num_elements, int64_t element_size_bytes,
                    int64_t keepdims);

// Cast operation wrapper (element type conversion)
int wrap_cast(RuntimeState* state, void* input, void* output,
              int64_t num_elements, int64_t input_element_size,
              int64_t output_element_size, int64_t to);

// Sigmoid activation wrapper
// Computes output = 1 / (1 + exp(-input)) element-wise
int wrap_sigmoid(RuntimeState* state, void* input, void* output,
                 int64_t num_elements, int64_t element_size_bytes);

// Rotary embedding operation wrapper
int wrap_rotary_embedding(RuntimeState* state,
                          void* input, void* position_ids,
                          void* cos_cache, void* sin_cache,
                          void* output,
                          int64_t interleaved, int64_t num_heads,
                          int64_t rotary_dim,
                          int64_t input_num_elements,
                          int64_t cos_cache_num_elements,
                          int64_t element_size_bytes);

// SimplifiedLayerNormalization operation wrapper
int wrap_simplified_layer_norm(RuntimeState* state,
                               void* input, void* scale, void* output,
                               int64_t input_num_elements,
                               int64_t scale_num_elements,
                               int64_t element_size_bytes,
                               int64_t axis, float epsilon,
                               int64_t stash_type);

// SkipSimplifiedLayerNormalization operation wrapper
int wrap_skip_simplified_layer_norm(RuntimeState* state,
                                    void* input, void* skip, void* gamma,
                                    void* output, void* skip_output,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes,
                                    float epsilon);

//==============================================================================
// Low-Level HIP Wrappers
//==============================================================================

// HIP memory allocation wrapper with error handling
int wrap_hipMalloc(void** ptr, int64_t size);

// HIP memory free wrapper with error handling
int wrap_hipFree(void* ptr);

// HIP memory copy host-to-device wrapper
int wrap_hipMemcpyH2D(void* dst, const void* src, int64_t size, void* stream);

// HIP memory copy device-to-host wrapper
int wrap_hipMemcpyD2H(void* dst, const void* src, int64_t size, void* stream);

// HIP stream synchronization wrapper
int wrap_hipStreamSynchronize(void* stream);

#ifdef __cplusplus
}
#endif

#endif // HIP_EP_RUNTIME_H
