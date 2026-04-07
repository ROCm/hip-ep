/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

//==============================================================================
// Backend-Independent Data Type Identifiers
//==============================================================================
//
// These are our own values -- do NOT assume they match MIOpen, cuDNN, or any
// other library's enum. Each backend provides an explicit mapping function
// (e.g. hipdnn_ep_to_miopen_type in real/elementwise.cpp) to convert these
// to library-specific types.
//
// To add a new type:
//   1. Add #define here
//   2. Update hipdnn_ep_datatype_size() and hipdnn_ep_datatype_name()
//   3. Update compiler mapping getHipdnnDataType() in HipToLLVM.cpp
//   4. Update each backend mapping function
//==============================================================================

#define HIPDNN_EP_DATATYPE_FLOAT 0    // f32, 4 bytes
#define HIPDNN_EP_DATATYPE_HALF 1     // f16, 2 bytes
#define HIPDNN_EP_DATATYPE_BFLOAT16 2 // bf16, 2 bytes
#define HIPDNN_EP_DATATYPE_INT32 3    // i32, 4 bytes
#define HIPDNN_EP_DATATYPE_INT64 4    // i64, 8 bytes

//==============================================================================
// Backend-Independent Tensor Operation Identifiers
//==============================================================================
//
// Same design as data types above -- our own values, mapped explicitly to
// library-specific ops in each backend (e.g. miopenTensorOpMul).
//==============================================================================

#define HIPDNN_EP_TENSOR_OP_MUL 0 // element-wise multiply
#define HIPDNN_EP_TENSOR_OP_ADD 1 // element-wise add
#define HIPDNN_EP_TENSOR_OP_MIN 2 // element-wise min
#define HIPDNN_EP_TENSOR_OP_MAX 3 // element-wise max

static inline const char *hipdnn_ep_tensor_op_name(int64_t op) {
  switch (op) {
  case HIPDNN_EP_TENSOR_OP_MUL:
    return "mul";
  case HIPDNN_EP_TENSOR_OP_ADD:
    return "add";
  case HIPDNN_EP_TENSOR_OP_MIN:
    return "min";
  case HIPDNN_EP_TENSOR_OP_MAX:
    return "max";
  default:
    return "unknown";
  }
}

static inline int64_t hipdnn_ep_datatype_size(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return 4;
  case HIPDNN_EP_DATATYPE_HALF:
    return 2;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return 2;
  case HIPDNN_EP_DATATYPE_INT32:
    return 4;
  case HIPDNN_EP_DATATYPE_INT64:
    return 8;
  default:
    return -1;
  }
}

static inline const char *hipdnn_ep_datatype_name(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return "f32";
  case HIPDNN_EP_DATATYPE_HALF:
    return "f16";
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return "bf16";
  case HIPDNN_EP_DATATYPE_INT32:
    return "i32";
  case HIPDNN_EP_DATATYPE_INT64:
    return "i64";
  default:
    return "unknown";
  }
}

//==============================================================================
// Backend-Independent Activation Mode Identifiers
//==============================================================================
//
// Same pattern as HIPDNN_EP_DATATYPE_* above. Each backend provides an explicit
// mapping function (e.g. hipdnn_ep_to_miopen_activation in
// real/activation.cpp).
//
// To add a new activation:
//   1. Add #define here
//   2. Update hipdnn_ep_activation_name()
//   3. Update each backend mapping function
//==============================================================================

#define HIPDNN_EP_ACTIVATION_SIGMOID 0
#define HIPDNN_EP_ACTIVATION_RELU 1
#define HIPDNN_EP_ACTIVATION_TANH 2

static inline const char *hipdnn_ep_activation_name(int64_t activation_mode) {
  switch (activation_mode) {
  case HIPDNN_EP_ACTIVATION_SIGMOID:
    return "sigmoid";
  case HIPDNN_EP_ACTIVATION_RELU:
    return "relu";
  case HIPDNN_EP_ACTIVATION_TANH:
    return "tanh";
  default:
    return "unknown";
  }
}

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
// (HipModelMetaInfo schema), opens the file via fs, reads each constant
// sequentially, uploads each to GPU via hipMalloc+hipMemcpy.
//   out_state:     Pointer to receive allocated RuntimeState
//   fs:            morphizen::FileSystem* (void* for C ABI) - must not be null
//   metadata_blob: FlatBuffers binary blob (HipModelMetaInfo) baked into DLL
//   blob_size:     Size of metadata_blob in bytes
// Return codes: 0=success, 1=alloc/read error, 2-9=GPU handle init error
int hipdnn_ep_state_init_with_fs(RuntimeState **out_state, void *fs,
                                 const void *metadata_blob, size_t blob_size);

// Cleanup runtime state (destroys handles, frees memory)
// Best-effort cleanup - continues even if individual operations fail
// Returns 0 always (best-effort)
int hipdnn_ep_state_cleanup(RuntimeState *state);

// Get GPU stream from state (for passing to HIP operations)
// Returns: hipStream_t cast to void* (NULL on error)
// Ownership: Caller does NOT own stream (destroyed in cleanup)
void *hipdnn_ep_state_get_stream(RuntimeState *state);

// Get MIOpen handle from state (for MIOpen operations)
// Returns: miopenHandle_t cast to void* (NULL on error)
// Ownership: Caller does NOT own handle (destroyed in cleanup)
void *hipdnn_ep_state_get_miopen_handle(RuntimeState *state);

// Get hipBLASLt handle from state (for GEMM operations)
// Returns: hipblasLtHandle_t cast to void* (NULL on error)
// Ownership: Caller does NOT own handle (destroyed in cleanup)
void *hipdnn_ep_state_get_hipblas_handle(RuntimeState *state);

// Get buffer from memory pool by index
// Returns: GPU pointer at pool_base + buffer_offsets[index] (NULL on error)
// Ownership: Caller does NOT own pointer (freed in cleanup)
void *hipdnn_ep_get_buffer_from_pool(RuntimeState *state, size_t index);

// Get the base pointer of the GPU memory pool
// Returns: GPU base pointer of pool (NULL if pool not initialized)
// Used by hip.get_pool lowering in generated compute kernels
void *hipdnn_ep_get_pool_base(RuntimeState *state);

// Shared workspace management (lazily grown, reused across MatMul/GQA/Conv)
void *hipdnn_ep_state_get_workspace(RuntimeState *state);
size_t hipdnn_ep_state_get_workspace_size(RuntimeState *state);
int hipdnn_ep_state_ensure_workspace(RuntimeState *state, size_t needed_size);

// Initialize memory pool in runtime state
// Called by generated inference_init after creating RuntimeState
// Parameters:
//   state: Runtime state to initialize pool in
//   pool_size: Total size of memory pool in bytes
//   buffer_offsets: Array of offsets for each buffer
//   num_buffers: Number of buffers
// Returns: 0=success, non-zero=error
int hipdnn_ep_pool_init(RuntimeState *state, size_t pool_size,
                        const size_t *buffer_offsets, size_t num_buffers);

//==============================================================================
// Inference API Types (for generated interface)
//==============================================================================

// Represents a tensor with host data and shape information
typedef struct {
  void *data;          // Host data pointer
  int64_t *shape;      // Array of dimension sizes
  size_t rank;         // Number of dimensions
  size_t element_size; // Bytes per element (e.g. 4=float32, 2=float16, 8=int64)
} tensor_t;

// Represents a span of tensors (inputs or outputs)
typedef struct {
  tensor_t *data; // Array of tensors
  size_t count;   // Number of tensors
} span_t;

// Represents a prepared tensor with GPU buffer and metadata
// Used internally by tensor preparation helpers
typedef struct {
  void *gpu_ptr;      // GPU memory (allocated or from pool)
  void *host_ptr;     // Host memory (from tensor_t.data)
  int64_t *shape_ptr; // Shape array (from tensor_t.shape) for memref building
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
void *hipdnn_ep_constant_get(RuntimeState *state, int64_t index);

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
int hipdnn_ep_tensor_prepare_input(RuntimeState *state, span_t *inputs,
                                   size_t index, size_t expected_rank,
                                   TensorBuffer *out_buffer);

// Prepare output tensor: parse, validate, get/allocate GPU buffer (no H2D)
//
// Parameters: same as prepare_input
int hipdnn_ep_tensor_prepare_output(RuntimeState *state, span_t *outputs,
                                    size_t index, size_t expected_rank,
                                    TensorBuffer *out_buffer);

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
int hipdnn_ep_tensor_finalize_output(RuntimeState *state, TensorBuffer *buffer);

// Release input tensor buffer (no D2H transfer needed)
//
// Parameters:
//   state: Runtime state
//   buffer: TensorBuffer from prepare_input
void hipdnn_ep_tensor_free_input(RuntimeState *state, TensorBuffer *buffer);

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
void *hipdnn_ep_tensor_buffer_get_gpu_ptr(TensorBuffer *buffer);

// Get host pointer from TensorBuffer
// Returns: Host memory pointer (NULL on error)
void *hipdnn_ep_tensor_buffer_get_host_ptr(TensorBuffer *buffer);

// Get shape array pointer from TensorBuffer
// Returns: Pointer to int64_t shape array (NULL on error)
int64_t *hipdnn_ep_tensor_buffer_get_shape_ptr(TensorBuffer *buffer);

// Get rank from TensorBuffer
// Returns: Tensor rank (number of dimensions)
size_t hipdnn_ep_tensor_buffer_get_rank(TensorBuffer *buffer);

// Get buffer size in bytes from TensorBuffer
// Returns: Size in bytes
size_t hipdnn_ep_tensor_buffer_get_size_bytes(TensorBuffer *buffer);

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
int wrap_hipMemcpyAsync(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                        size_t size_bytes);

//==============================================================================
// Library Operations (MIOpen, hipBLAS)
//==============================================================================

// MIOpen convolution forward operation
// Full wrapper with descriptor creation, algorithm finding, workspace
// management Follows opaque RuntimeState pattern - extracts handle/stream
// internally Parameters match generated LLVM IR from HipToLLVM pass
int wrap_miopenConvolutionForward(
    RuntimeState
        *state, // RuntimeState (opaque - extracts handle/stream internally)
    const void *input,   // Input tensor GPU pointer
    int64_t input_n,     // Input batch size
    int64_t input_c,     // Input channels
    int64_t input_h,     // Input height
    int64_t input_w,     // Input width
    const void *weights, // Weights tensor GPU pointer
    int64_t weights_k,   // Output channels (number of filters)
    const void *bias,    // Bias tensor GPU pointer (nullable)
    void *output,        // Output tensor GPU pointer (in-place)
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
int wrap_hipblasLtGemm(void *handle, // hipBLASLt handle
                       void *stream, // HIP stream
                       int64_t m, int64_t n, int64_t k,
                       const void *alpha, // Scalar alpha
                       const void *A,     // Matrix A GPU pointer
                       const void *B,     // Matrix B GPU pointer
                       const void *beta,  // Scalar beta
                       void *C);          // Matrix C GPU pointer (in/out)

// MatMul operation wrapper (batched matrix multiplication)
// Called by generated IR for onnx.MatMul lowering
// Computes output = A @ B for each batch
// A: [batch_count x M x K], B: [K x N] (broadcast) or [batch_count x K x N]
// output: [batch_count x M x N]
int wrap_hipblasLtMatmul(
    RuntimeState *state,
    const void *A,       // Matrix A GPU pointer
    const void *B,       // Matrix B GPU pointer
    void *output,        // Output GPU pointer
    int64_t M,           // Rows of A (per batch)
    int64_t N,           // Columns of B
    int64_t K,           // Columns of A / Rows of B
    int64_t batch_count, // Number of batches
    int64_t elem_size);  // Element size in bytes (2=f16, 4=f32)

// GroupQueryAttention operation wrapper (Full MS spec)
// Called by generated IR for onnx.Custom(GroupQueryAttention) lowering
// GQA runtime wrapper following the complete Microsoft ONNX Runtime
// specification (14 inputs + 12 attributes).  Supports separate Q/K/V and
// packed QKV paths, optional RoPE, KV cache management, local window
// attention (local_window_size), and smooth softmax (head_sink /
// smooth_softmax).
int wrap_group_query_attention(
    RuntimeState *state,
    // Inputs 1-7 (core GQA)
    void *query, void *key, void *value, void *past_key, void *past_value,
    void *seqlens_k, void *total_seq_len,
    // Inputs 8-10 (RoPE)
    void *cos_cache, void *sin_cache, void *position_ids,
    // Inputs 11-14 (advanced features)
    void *attention_bias, void *head_sink, void *k_scale, void *v_scale,
    // Outputs
    void *output, void *present_key, void *present_value, void *output_qk,
    // Attributes (12)
    int64_t num_heads, int64_t kv_num_heads, float scale, int64_t do_rotary,
    int64_t rotary_interleaved, float softcap, int64_t local_window_size,
    int64_t smooth_softmax, int64_t qk_output, int64_t k_quant_type,
    int64_t v_quant_type, int64_t kv_cache_bit_width,
    // Shape values (6)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t past_seq_len, int64_t head_dim, int64_t element_size_bytes);

// Generic MIOpen tensor operation wrapper with per-operand 4D shapes.
// Computes output = op(lhs, rhs) element-wise via miopenOpTensor.
// Each operand is described by 4D shape (N, C, H, W) to enable MIOpen-native
// broadcasting: dims of 1 are broadcast against the corresponding larger dim.
//   tensor_op: HIPDNN_EP_TENSOR_OP_* constant (mul, add, min, max)
//   data_type: HIPDNN_EP_DATATYPE_* constant identifying the element type
int wrap_miopenOpTensor(RuntimeState *state, void *lhs, void *rhs, void *output,
                        int64_t lhs_n, int64_t lhs_c, int64_t lhs_h,
                        int64_t lhs_w, int64_t rhs_n, int64_t rhs_c,
                        int64_t rhs_h, int64_t rhs_w, int64_t out_n,
                        int64_t out_c, int64_t out_h, int64_t out_w,
                        int64_t data_type, int64_t tensor_op);

// Element-wise subtraction wrapper
// Computes output = lhs - rhs element-wise
int wrap_elementwise_sub(RuntimeState *state, void *lhs, void *rhs,
                         void *output, int64_t num_elements,
                         int64_t element_size_bytes);

// Gather operation wrapper — supports multi-element indices for axis=0.
int wrap_gather(RuntimeState *state, void *data, void *indices, void *output,
                int64_t axis, int64_t data_num_elements,
                int64_t indices_num_elements, int64_t output_num_elements,
                int64_t element_size_bytes);

// ReduceSum operation wrapper
int wrap_reduce_sum(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t element_size_bytes, int64_t keepdims);

// Cast operation wrapper (element type conversion)
// src_data_type and dst_data_type are HIPDNN_EP_DATATYPE_* enum values.
int wrap_cast(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t src_data_type,
              int64_t dst_data_type);

// Generic MIOpen activation wrapper
// Applies activation_mode (HIPDNN_EP_ACTIVATION_*) element-wise
// data_type: HIPDNN_EP_DATATYPE_* constant identifying the element type
int wrap_miopenActivationForward(RuntimeState *state, void *input, void *output,
                                 int64_t num_elements, int64_t data_type,
                                 int64_t activation_mode);

// Rotary embedding operation wrapper
int wrap_rotary_embedding(RuntimeState *state, void *input, void *position_ids,
                          void *cos_cache, void *sin_cache, void *output,
                          int64_t interleaved, int64_t num_heads,
                          int64_t rotary_dim, int64_t input_num_elements,
                          int64_t cos_cache_num_elements,
                          int64_t element_size_bytes);

// SimplifiedLayerNormalization operation wrapper
int wrap_miopenT5LayerNormForward(RuntimeState *state, void *input, void *scale,
                                  void *output, int64_t input_num_elements,
                                  int64_t scale_num_elements,
                                  int64_t element_size_bytes, int64_t axis,
                                  float epsilon, int64_t stash_type);

// SkipSimplifiedLayerNormalization operation wrapper (Full MS spec)
// Computes: input_skip_bias_sum = input + skip [+ bias]
//           output = RMSNorm(input_skip_bias_sum) * gamma
// bias and input_skip_bias_sum may be nullptr (optional per MS spec)
int wrap_skip_simplified_layer_norm(RuntimeState *state, void *input,
                                    void *skip, void *gamma, void *bias,
                                    void *output, void *input_skip_bias_sum,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes, float epsilon);

// MatMulNBits operation wrapper (quantized N-bit matrix multiplication)
// Dequantizes packed int4 weights and computes Y = A @ dequant(B)^T + bias
// A: [batch_count x M x K], B: [N x k_blocks x blob_size] (packed uint8)
// scales: [N x k_blocks], output: [batch_count x M x N]
// Optional: zero_points, g_idx (deprecated), bias - pass nullptr if absent
int wrap_matmul_nbits(
    RuntimeState *state,
    const void *A,           // activation tensor
    const void *B,           // packed quantized weights
    const void *scales,      // per-block scale factors
    const void *zero_points, // per-block zero points (nullable)
    const void *g_idx,       // GPTQ group indices (nullable, deprecated)
    const void *bias,        // output bias [N] (nullable)
    void *output,            // result tensor
    int64_t M,               // rows per batch
    int64_t N,               // output columns
    int64_t K,               // inner dimension
    int64_t batch_count,     // number of batches
    int64_t bits,            // quantization bits (e.g. 4)
    int64_t block_size,      // quantization block size
    int64_t elem_size);      // element size in bytes

// QMoE operation wrapper (quantized Mixture-of-Experts)
// Routes tokens to top-k experts, performs quantized MLP per expert,
// applies activation (e.g. SwiGLU), and combines results.
// Optional pointer args: pass nullptr if the corresponding input is absent.
int wrap_qmoe(
    RuntimeState *state,
    const void *input,           // [num_tokens, hidden_size]
    const void *router_probs,    // [num_tokens, num_experts]
    const void *fc1_weights,     // [num_experts, fusion*inter, hidden/pack]
    const void *fc1_scales,      // [num_experts, fusion*inter, hidden/bs]
    const void *fc1_bias,        // (nullable) [num_experts, fusion*inter]
    const void *fc2_weights,     // [num_experts, hidden, inter/pack]
    const void *fc2_scales,      // [num_experts, hidden, inter/bs]
    const void *fc2_bias,        // (nullable) [num_experts, hidden]
    const void *fc3_weights,     // (nullable) unfused SwiGLU
    const void *fc3_scales,      // (nullable)
    const void *fc3_bias,        // (nullable)
    const void *fc1_zero_points, // (nullable) fc1 dequant zero points
    const void *fc2_zero_points, // (nullable) fc2 dequant zero points
    const void *fc3_zero_points, // (nullable) fc3 dequant zero points
    void *output,                // [num_tokens, hidden_size]
    int64_t num_tokens, int64_t hidden_size, int64_t inter_size,
    int64_t num_experts, int64_t k, int64_t expert_weight_bits,
    int64_t block_size, int64_t swiglu_fusion,
    int64_t activation_type, // 0=relu,1=gelu,2=silu,3=swiglu,4=identity
    float activation_alpha, float activation_beta, float swiglu_limit,
    int64_t normalize_routing_weights, int64_t elem_size);

//==============================================================================
// Low-Level HIP Wrappers
//==============================================================================

// HIP memory allocation wrapper with error handling
int wrap_hipMalloc(void **ptr, int64_t size);

// HIP memory free wrapper with error handling
int wrap_hipFree(void *ptr);

// HIP memory copy host-to-device wrapper
int wrap_hipMemcpyH2D(void *dst, const void *src, int64_t size, void *stream);

// HIP memory copy device-to-host wrapper
int wrap_hipMemcpyD2H(void *dst, const void *src, int64_t size, void *stream);

// HIP stream synchronization wrapper
int wrap_hipStreamSynchronize(void *stream);

#ifdef __cplusplus
}
#endif

#endif // HIP_EP_RUNTIME_H
