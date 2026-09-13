/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_DXCG_C_API_HPP
#define MIGRAPHX_GUARD_DXCG_C_API_HPP

/*
 * Flat C export interface for amdxcgc.dll — COMPILATION ONLY.
 *
 * amdxcgc.dll is responsible for:
 *   - Parsing DxCGC MLIR IR
 *   - Compiling to GPU code objects (AMD HSACO ELF)
 *   - Exposing per-kernel execution metadata (dispatch dims, resource bindings,
 *     constant weight data)
 *
 * amdxcgc.dll is NOT responsible for:
 *   - GPU memory allocation
 *   - Kernel dispatch (HIP or DX12)
 *   - Reading results back from the device
 *
 * Callers (amdxcgc-driver, C#, Python, DX12 runtime, HIP runtime) own all
 * execution.  Use dxcg_get_kernel_descriptors() to obtain everything needed
 * to drive a kernel via HIP or the AMD DX12 Cross-Compile API.
 *
 * Compilation flow:
 *
 *   [IR bytes]  ──► dxcg_compile ──► [compiled program blob]
 *   [blob]      ──► dxcg_get_kernel_descriptors ──► [dxcg_kernel_descriptor[]]
 *
 * Each dxcg_kernel_descriptor contains:
 *   - HSACO ELF binary  (pass to hipModuleLoadData or CreateComputePipelineCrossCompile)
 *   - Dispatch & group dimensions
 *   - Per-slot binding descriptors including constant weight data for constant buffers
 *
 * Memory management:
 *   All output buffers returned by this API are allocated by the DLL.
 *   The caller MUST release them with dxcg_free() (blobs, error strings) or
 *   dxcg_free_kernel_descriptors() (descriptor arrays).
 *   Never pass them to the system allocator — the DLL and caller may use
 *   different heaps (common on Windows).
 */

#include <migraphx/cgc/export.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Compile options
 * ---------------------------------------------------------------------- */

/**
 * Options passed to dxcg_compile / dxcg_compile_with_resource_buffer.
 * Zero-initialise to get safe defaults (fast_math=true, everything else off).
 */
typedef struct dxcg_compile_options
{
    /**
     * GPU target architecture string, e.g. "gfx1100". NULL = auto-detect.
     * Currently reserved for future use; MIGraphX always auto-detects from
     * the HIP runtime.  Pass NULL or an empty string for normal use.
     */
    const char* target_arch;

    /** 1 = fast math (default); 0 = strict IEEE. */
    int fast_math;

    /**
     * 1 = exhaustively search for the best kernel tuning configuration.
     * Slower first run, produces optimal kernels.  0 = use heuristic defaults.
     */
    int exhaustive_tune;

    /**
     * 1 = allocate GPU memory internally and copy parameters to/from device
     * automatically during execution.  Required when the caller does not
     * manage device memory directly.  0 (default) = caller manages device ptrs.
     */
    int offload_copy;

    /**
     * 1 = disable the Custom direct-HIP dispatch pass (amdxcgc_map_custom).
     * Affected ops fall through to MLSS (if enabled) or rocMLIR.
     * Kernel dispatch priority: Custom (1st) → MLSS (2nd) → rocMLIR (3rd).
     * 0 (default) = enable Custom dispatch.
     */
    int disable_custom_p0;

    /**
     * Comma-separated list of Custom op-family tokens to exclude from the
     * Custom lookup even when disable_custom_p0 is 0.  Instructions whose
     * fused-op name contains any token (case-insensitive substring match) are
     * skipped and fall through to MLSS or rocMLIR.
     *
     * Supported tokens:
     *   "gqa"              — Group Query Attention (gqaattn, gqaattn_simple)
     *   "moe"              — Mixture of Experts (moe)
     *   "rope"             — Rotary Position Embedding (rope)
     *   "linear_attention" — Linear / Recurrent Linear Attention
     *
     * NULL or empty string = no per-family exclusions (default).
     */
    const char* custom_p0_disabled_ops;

    /**
     * 1 = disable all MLSS-related fusion passes (fuse_cgc_amdgpu_ops and
     * amdxcgc_map_mlss).  decompose_amdgpu_ops still runs so any amdgpu_op:*
     * instructions already present in the input IR are lowered to primitives.
     * 0 (default) = run the full MLSS pipeline.
     */
    int disable_mlss;

    /**
     * Comma-separated list of MLSS op-family tokens to exclude from the MLSS
     * lookup even when disable_mlss is 0.  Instructions whose fused-op name
     * contains any token (case-insensitive substring match) are skipped and
     * fall back to the decompose_amdgpu_ops primitive lowering path.
     *
     * Supported tokens (one per MLSS kernel family):
     *   "gemm"   — all GEMM variants (gemm_add, gemm_relu, gemm_relu_add, ...)
     *   "conv"   — all convolution variants (conv_relu, conv_relu_add, ...)
     *   "qgemm"  — quantised GEMM (weight-only INT4/FP8 dequant + GEMM)
     *   "mvn"    — Mean-Variance Normalisation (3-kernel MVN split)
     *   "gqa"    — Group-Query Attention (gqaattn, gqaattn_simple)
     *   "rdb"    — Residual Dense Block (3-stage conv+relu+add)
     *   "fsr"    — FSR upscaling variants (fsr_mlfi, fsr_mlsr, fsr_nssd, fsr_rdb)
     *
     * Example: "gemm,gqa" disables MLSS for all GEMM and GQA ops while
     * conv, qgemm, mvn, rdb, and fsr still use MLSS shaders.
     * NULL or empty string = no per-family exclusions (default).
     */
    const char* mlss_disabled_ops;

    /**
     * 1 = disable the Custom P1 dispatch tier (amdxcgc_map_custom_p1).
     * P1 handles ops that MLSS doesn't cover: layer_norm, slice, scatter_nd,
     * cumsum, elementwise_binary, matmul_nbits, gemm_wmma.
     *
     * Dispatch priority (see amdxcgc_kernel_registry.json):
     *   P0 Custom (1st) → MLSS (2nd) → P1 Custom (3rd) → rocMLIR (4th).
     * 0 (default) = enable P1 Custom dispatch.
     */
    int disable_custom_p1;

    /**
     * Comma-separated list of P1 op-family tokens to exclude from the P1
     * Custom lookup even when disable_custom_p1 is 0.  Matched ops fall
     * through to rocMLIR JIT.
     *
     * Supported tokens:
     *   "layer_norm"          — LayerNormalization
     *   "slice"               — Generic multi-axis Slice
     *   "scatter_nd"          — ScatterND (with reduction)
     *   "cumsum"              — CumSum
     *   "elementwise_binary"  — Div, Equal, Less, Mod, And
     *   "matmul_nbits"        — Int4/Int8 dequant GEMM
     *   "gemm_wmma"           — WMMA fp16 small-M GEMM (M<=512)
     *
     * NULL or empty string = no per-family exclusions (default).
     */
    const char* custom_p1_disabled_ops;

    /**
     * When non-zero, set MIGRAPHX_SKIP_BENCHMARKING=1 during program compile so
     * rocMLIR uses its first Quick heuristic candidate directly instead of
     * benchmarking all candidates.  Useful when no tuning DB is present (e.g.
     * combined with disable_mlss on a new or uncovered architecture).
     * 0 (default) = normal benchmarking behaviour.
     */
    int skip_rocmlir_benchmarking;

    /**
     * When non-zero, compile without a live GPU context (offline/headless mode).
     * In offline mode the compiler skips hipModuleLoadData so the binary never
     * gets loaded onto the GPU.  The trade-off: rocMLIR cannot run its benchmark
     * loop to select the best tuning solution, so it falls back to the first
     * heuristic candidate or throws if all candidates fail to lower to LLVM IR
     * (as happens for fp16 WMMA shapes on Windows gfx1100).
     *
     * Default (0) = live mode: compile WITH GPU context.  rocMLIR benchmarks all
     * candidates on the real hardware, picks the fastest, and stores the tuned
     * HSACO in the program value.  hipModuleLoadData is called internally during
     * benchmarking but hipModuleUnload is a safe no-op on Windows so no crash.
     * The binary is extracted immediately after compile and dispatched via DX12
     * (never executed via HIP runtime).
     */
    int offline_compile;

    /**
     * Target execution runtime for this compilation.  Controls which ELF variant
     * the MLSS and Custom dispatch passes prefer:
     *   "hip"  — request non-relocatable (ET_DYN) binaries only.
     *   "dx12" — request relocatable (ET_REL) binaries only; ops with no ET_REL
     *            binary fall through to the next dispatch tier.
     *   NULL / "" (default) — compile both variants; the MLSS/Custom passes
     *            embed ET_DYN as the primary binary and ET_REL as the relocatable
     *            sidecar so the result is usable on both HIP and DX12.
     *
     * When calling dxcg_get_kernel_descriptors() with DXCG_RUNTIME_DX12, always
     * set this to "dx12" so the pass optimises for ET_REL from the start.
     */
    const char* target_runtime;

} dxcg_compile_options;

/* -------------------------------------------------------------------------
 * Primary compile entry points
 * ---------------------------------------------------------------------- */

/**
 * Compile a DxCGC IR buffer to a compiled program blob (resources from file).
 *
 * @param ir_data        Pointer to the DxCGC MLIR IR bytes.
 *                       May be UTF-8 text or UTF-16 LE binary — both are accepted.
 *                       The buffer does NOT need to be NUL-terminated.
 * @param ir_size        Number of bytes in ir_data.
 * @param resources_path Path to the companion resources file (.mlir with
 *                       dialect_resources block containing hex-encoded weights).
 *                       Pass NULL if the IR is self-contained or has no constants.
 * @param options        Compile options.  Pass NULL for defaults.
 * @param out_blob       On success, set to a DLL-allocated buffer containing
 *                       the compiled program in MIGraphX msgpack format.
 *                       Caller must release with dxcg_free().
 * @param out_blob_size  On success, set to the byte length of *out_blob.
 * @param out_error      On failure (return != 0), set to a DLL-allocated
 *                       NUL-terminated error string.
 *                       Caller must release with dxcg_free().
 *                       On success this is set to NULL.
 * @return               0 on success, non-zero on failure.
 */
AMDXCGC_EXPORT int dxcg_compile(
    const void*                  ir_data,
    size_t                       ir_size,
    const char*                  resources_path,
    const dxcg_compile_options*  options,
    void**                       out_blob,
    size_t*                      out_blob_size,
    char**                       out_error);

/**
 * Compile a DxCGC IR buffer to a compiled program blob (resources from buffer).
 *
 * Same as dxcg_compile but accepts the resources data as an in-memory buffer
 * rather than a file path.  The buffer must contain the same text format that
 * a resources .mlir file would contain (dialect_resources block with hex weights).
 *
 * @param ir_data            Pointer to the DxCGC MLIR IR bytes.
 * @param ir_size            Number of bytes in ir_data.
 * @param resources_data     Pointer to the resources buffer (may be NULL if
 *                           the IR contains inline resources or has no constants).
 * @param resources_size     Number of bytes in resources_data (0 if NULL).
 * @param options            Compile options.  Pass NULL for defaults.
 * @param out_blob           On success, compiled program blob (caller frees).
 * @param out_blob_size      On success, byte length of *out_blob.
 * @param out_error          On failure, NUL-terminated error string (caller frees).
 * @return                   0 on success, non-zero on failure.
 */
AMDXCGC_EXPORT int dxcg_compile_with_resource_buffer(
    const void*                  ir_data,
    size_t                       ir_size,
    const void*                  resources_data,
    size_t                       resources_size,
    const dxcg_compile_options*  options,
    void**                       out_blob,
    size_t*                      out_blob_size,
    char**                       out_error);

/* -------------------------------------------------------------------------
 * Generic kernel descriptor API
 *
 * Runtime-agnostic interface for extracting per-kernel dispatch information
 * from a compiled program blob.  The same HSACO code object is consumed by
 * both runtimes:
 *
 *   HIP  : hipModuleLoadData(hsaco_data)  → hipModuleLaunchKernel(...)
 *   DX12 : CreateComputePipelineCrossCompile(pBlob=hsaco_data,
 *             shType=ElfHsa, pKernelName=entry_point,
 *             threadsPerGroup={group_size_x,y,z})
 *          → cmdList->Dispatch(dispatch_x, dispatch_y, dispatch_z)
 *
 * Callers are responsible for ALL runtime operations:
 *   - GPU memory allocation for each binding slot
 *   - Uploading constant_data (when is_constant=1) and runtime inputs
 *   - Kernel dispatch and output readback
 * ---------------------------------------------------------------------- */

/** Target runtime that will consume the kernel descriptors. */
typedef enum dxcg_runtime_type
{
    DXCG_RUNTIME_HIP  = 0,  /**< AMD ROCm / HIP — load HSACO via hipModuleLoadData    */
    DXCG_RUNTIME_DX12 = 1,  /**< DirectX 12    — load HSACO via AMD CC CrossCompile   */
} dxcg_runtime_type;

/** Format of the binary embedded in dxcg_kernel_descriptor.hsaco_data. */
typedef enum dxcg_code_object_format
{
    DXCG_CODE_OBJECT_HSACO     = 0,  /**< Raw AMD HSACO (ELF) code object  */
    DXCG_CODE_OBJECT_HSACO_ELF = 0,  /**< Alias — HSACO is already an ELF  */
} dxcg_code_object_format;

/**
 * Per-binding descriptor — one entry per kernel argument (input, constant, or output).
 *
 *   HIP  : slot_index is the kernel argument position (0-based).
 *   DX12 : slot_index is the root parameter / descriptor table entry index.
 *
 * Constant buffers (is_constant=1):
 *   The DLL fills constant_data/constant_size with the raw bytes of the
 *   weight tensor.  Callers must upload this data to a GPU buffer and bind
 *   it at slot_index before dispatch.
 *
 * Runtime inputs (is_constant=0, is_output=0):
 *   Callers supply inference input data (e.g. image pixels, token IDs).
 *
 * Output buffers (is_output=1):
 *   Callers allocate an empty GPU buffer of size_bytes and read it back
 *   after dispatch.
 */
/**
 * ELF binary format encoded in dxcg_kernel_descriptor::hsaco_data.
 * Determines how the runner dispatches the kernel and binds arguments.
 */
typedef enum dxcg_elf_format
{
    /** HSA ELF relocatable (EI_OSABI=0x40, e_type=ET_REL): HIP/ROCm ABI, unrelocated.
     *  DX12: build_single_kernel_et_rel produces a single-kernel ET_REL from the fat
     *        bundle; passed to CreateComputePipelineCrossCompile(ElfHsa).
     *  HIP:  not directly loadable; requires link step (hipcc / LLVM linker). */
    DXCG_ELF_HSA_REL = 0,

    /** HSA ELF dynamic (EI_OSABI=0x40, e_type=ET_DYN): HIP/ROCm ABI, fully linked.
     *  DX12: CreateComputePipelineCrossCompile(ElfHsa) + SetKernelArguments.
     *  HIP:  hipModuleLoadData + hipModuleLaunchKernel with flat kernarg segment.
     *  Argument binding: per-slot via dxcg_binding_desc (buffer VAs + scalars).
     *  This is the standard HSACO format produced by hipcc and most tools. */
    DXCG_ELF_HSA_DYN = 1,

    /** Alias for DXCG_ELF_HSA_DYN (backward compat — pre-existing code uses DXCG_ELF_HSA). */
    DXCG_ELF_HSA     = 1,

    /** PAL ELF relocatable (EI_OSABI=0x41, e_type=ET_REL): PAL compute shader ABI.
     *  DX12: CreateComputePipelineCrossCompile(ElfPal) + 2 root constants (CbData VA).
     *  HIP:  hipModuleLoadData on HSA-converted variant (future).
     *  Argument binding: via cbdata_bytes / cbdata_slots — runner patches buffer VAs
     *  into the pre-assembled CbData block, uploads it to GPU, passes its VA as two
     *  32-bit root constants (lo/hi) → shader reads CbData via F_ADDR_INDIRECT. */
    DXCG_ELF_PAL_REL = 2,

    /** PAL ELF pre-linked dynamic (EI_OSABI=0x41, e_type=ET_DYN): same CbData dispatch
     *  as PAL_REL but already linked (no relocation step needed in the driver). */
    DXCG_ELF_PAL_DYN = 3,
} dxcg_elf_format;

/**
 * Binding descriptor for one kernel argument slot.
 *
 * For DXCG_ELF_HSA kernels:  used for every argument (buffer, scalar, constant).
 * For DXCG_ELF_PAL_REL/DYN:  only cbdata_slot entries in dxcg_kernel_descriptor
 *   are used; regular binding descriptors describe the logical IO layout for the
 *   caller to allocate correct buffer sizes.
 */
typedef struct dxcg_binding_desc
{
    unsigned int slot_index;    /**< Argument position / root param index              */
    unsigned int size_bytes;    /**< Buffer byte size (use constant_size for constants) */
    int          is_output;     /**< 1 = UAV / writable output, 0 = read-only          */
    int          is_constant;   /**< 1 = constant weight data (upload constant_data),
                                     0 = runtime input / output                         */
    int          is_scalar;     /**< 1 = scalar value (int32/int64/float) passed by
                                     value, not as a GPU buffer pointer.
                                     When 1: constant_data holds the raw bytes of the
                                     scalar (4 or 8 bytes); no GPU buffer is allocated. */
    int          is_double_ptr; /**< 1 = buffer arg uses double indirection (T**):
                                     the kernel dereferences the kernarg VA to get the
                                     actual buffer VA (MLSS GEMM / MHA / GQA kernels).
                                     When 1: the driver creates an 8-byte intermediate
                                     GPU buffer containing the data VA and passes its VA
                                     as the kernel argument.  0 = direct pointer (T*). */
    const void*  constant_data; /**< Non-NULL when is_constant=1 or is_scalar=1:
                                     raw bytes of the constant tensor / scalar value    */
    size_t       constant_size; /**< Byte count of constant_data (0 if not set)        */
    int          param_index;   /**< For runtime input slots: 0-based index into the
                                     runtime_inputs vector, matching the order of
                                     program::get_parameter_names().  Deprecated:
                                     use param_name for reliable lookup.
                                     -1 = not a model-level runtime parameter.       */
    const char*  param_name;    /**< For runtime input slots: NUL-terminated model
                                     parameter name (e.g. "arg0", "branch1.weight"),
                                     matching an entry in get_parameter_names().
                                     NULL = not a model-level runtime parameter.      */
} dxcg_binding_desc;

/**
 * PAL ELF CbData slot descriptor (used when elf_format == DXCG_ELF_PAL_REL/DYN).
 *
 * Describes one GPU buffer VA that must be patched into the pre-assembled CbData
 * block at dispatch time.  The runner writes the live GPU VA (8 bytes, little-endian)
 * at cbdata_bytes[byte_offset].
 */
typedef struct dxcg_cbdata_slot
{
    unsigned int byte_offset; /**< Byte offset within cbdata_bytes to write the VA.  */
    int          buf_index;   /**< -1 = output buffer; ≥0 = input buffer[buf_index]. */
} dxcg_cbdata_slot;

/**
 * Runtime-agnostic descriptor for one compiled GPU kernel.
 *
 * Lifetime: owned by the dxcg_kernel_descriptor* array returned by
 * dxcg_get_kernel_descriptors().  All pointers are valid until
 * dxcg_free_kernel_descriptors() is called on the array.
 */
typedef struct dxcg_kernel_descriptor
{
    /* -- Identity -------------------------------------------------------- */
    /** Kernel entry point / symbol name (NUL-terminated). */
    const char*             entry_point;

    /** Legacy format field (kept for ABI compat). Use elf_format instead. */
    dxcg_code_object_format code_format;

    /* -- ELF binary format ----------------------------------------------- */
    /**
     * Format of the kernel binary in hsaco_data.
     * Determines how the runner creates the PSO and binds arguments:
     *   DXCG_ELF_HSA:      SetKernelArguments (HSA kernarg segment, HIP-ABI)
     *   DXCG_ELF_PAL_REL/DYN: CbData-pointer dispatch via root constants
     */
    dxcg_elf_format         elf_format;

    /* -- Kernel binary --------------------------------------------------- */
    /**
     * Raw ELF bytes.  Format indicated by elf_format:
     *   HSA:     HSACO (EI_OSABI=0x40) — pass to hipModuleLoadData or
     *            CreateComputePipelineCrossCompile(ElfHsa).
     *   PAL_REL: PAL ET_REL (EI_OSABI=0x41, e_type=1) — pass to
     *            CreateComputePipelineCrossCompile(ElfPal).
     *   PAL_DYN: PAL ET_DYN (EI_OSABI=0x41, e_type=3) — same as PAL_REL.
     */
    const void*             hsaco_data;
    size_t                  hsaco_size;

    /* -- Dispatch dimensions --------------------------------------------- */
    unsigned int dispatch_x;   /**< Thread groups in X                              */
    unsigned int dispatch_y;   /**< Thread groups in Y                              */
    unsigned int dispatch_z;   /**< Thread groups in Z                              */
    unsigned int group_size_x; /**< Threads per group X (embedded in ELF)           */
    unsigned int group_size_y; /**< Threads per group Y                             */
    unsigned int group_size_z; /**< Threads per group Z                             */

    /* -- HSA resource binding (elf_format == DXCG_ELF_HSA) -------------- */
    /**
     * Array of num_bindings binding descriptors describing each kernel argument.
     * Used by DX12 runner via SetKernelArguments and HIP runner via kernarg segment.
     * For PAL kernels: bindings still describe the logical IO layout (buffer sizes)
     * so callers can allocate correct buffers, but dispatch uses cbdata_* instead.
     */
    const dxcg_binding_desc* bindings;
    size_t                   num_bindings;

    /* -- PAL CbData dispatch (elf_format == DXCG_ELF_PAL_REL/DYN) ------- */
    /**
     * Pre-assembled CbData struct bytes with all scalar fields pre-filled.
     * GPU buffer VAs (zeroed here) are patched at dispatch time from cbdata_slots.
     * The runner:
     *   1. Copies cbdata_bytes to a GPU buffer.
     *   2. Patches in live buffer VAs from cbdata_slots.
     *   3. Passes the CbData GPU VA as two 32-bit root constants (lo/hi).
     *      → DX12: SetComputeRoot32BitConstants(0, 2, {va_lo, va_hi})
     *      → PAL shader reads CbData via F_ADDR_INDIRECT from SGPRs[0:1]
     * NULL / 0 when elf_format == DXCG_ELF_HSA.
     */
    const void*              cbdata_bytes;
    size_t                   cbdata_size;

    /**
     * Array of num_cbdata_slots slot descriptors: which byte offsets in
     * cbdata_bytes hold GPU buffer VAs that need patching at dispatch time.
     */
    const dxcg_cbdata_slot*  cbdata_slots;
    size_t                   num_cbdata_slots;
} dxcg_kernel_descriptor;

/**
 * Enumerate all GPU kernel descriptors embedded in a compiled blob.
 *
 * @param blob             Compiled program blob (from dxcg_compile).
 * @param blob_size        Byte length of blob.
 * @param runtime          Target runtime (DXCG_RUNTIME_HIP or DXCG_RUNTIME_DX12).
 *                         Currently both return the same HSACO descriptors since
 *                         both runtimes consume HSACO via different loaders.
 * @param out_descriptors  On success, DLL-allocated array of out_count descriptors.
 *                         Release with dxcg_free_kernel_descriptors().
 * @param out_count        On success, number of descriptors in out_descriptors.
 * @param out_error        On failure, DLL-allocated error string (release with dxcg_free()).
 * @return                 0 on success, non-zero on failure.
 */
AMDXCGC_EXPORT int dxcg_get_kernel_descriptors(
    const void*               blob,
    size_t                    blob_size,
    dxcg_runtime_type         runtime,
    dxcg_kernel_descriptor**  out_descriptors,
    size_t*                   out_count,
    char**                    out_error);

/**
 * Release a descriptor array previously returned by dxcg_get_kernel_descriptors.
 * Safe to call with NULL / 0.
 */
AMDXCGC_EXPORT void dxcg_free_kernel_descriptors(
    dxcg_kernel_descriptor*   descriptors,
    size_t                    count);

/* -------------------------------------------------------------------------
 * Code object dump entry points (convenience wrappers over the descriptor API)
 * ---------------------------------------------------------------------- */

/** File format selector for dxcg_dump_code_objects. */
typedef enum dxcg_dump_format
{
    DXCG_DUMP_HSACO     = 0,  /**< Write <symbol>.hsaco     */
    DXCG_DUMP_HSACO_ELF = 1,  /**< Write <symbol>.hsaco.elf */
} dxcg_dump_format;

/**
 * Extract and dump all GPU code objects embedded in a compiled program blob.
 *
 * Walks every gpu::code_object instruction in the compiled program and writes
 * each code object to a file in out_dir using the naming convention determined
 * by format.  If out_dir does not exist the call fails.
 *
 * @param blob       Compiled program blob as returned by dxcg_compile.
 * @param blob_size  Byte length of blob.
 * @param out_dir    NUL-terminated path to the directory where files are written.
 * @param format     DXCG_FORMAT_HSACO or DXCG_FORMAT_HSACO_ELF.
 * @param out_count  On success, set to the number of files written.
 * @param out_error  On failure, DLL-allocated NUL-terminated error string
 *                   (release with dxcg_free()).  NULL on success.
 * @return           0 on success, non-zero on failure.
 */
AMDXCGC_EXPORT int dxcg_dump_code_objects(
    const void*        blob,
    size_t             blob_size,
    const char*        out_dir,
    dxcg_dump_format   format,
    size_t*            out_count,
    char**             out_error);

/* -------------------------------------------------------------------------
 * Subgraph transformation query / validation  (D3D12_FEATURE_MLIR_EXCHANGE)
 *
 * These entry points implement the DXCP side of the DxCGC subgraph exchange
 * protocol described in DxCgcThroughMIGraphX.md §6.
 *
 * DXCP resolves these exports alongside the compile exports.  If any
 * export is absent (older DLL), the pointer is left null and DXCP returns
 * DXGI_ERROR_UNSUPPORTED for D3D12_FEATURE_MLIR_EXCHANGE caps while the
 * compile/dispatch path (dxcg_compile + dxcg_get_kernel_descriptors) still
 * works normally.
 *
 * Function pointer typedefs (for GetProcAddress-style resolution):
 *
 *   PFN_dxcg_query_subgraph_transformations
 *   PFN_dxcg_free_subgraph_catalog
 *   PFN_dxcg_validate_subgraph_specialization
 *   PFN_dxcg_free_validation_result
 * ---------------------------------------------------------------------- */

/**
 * Subgraph transformation catalog query.
 *
 * Called by DXCP when handling D3D12_FEATURE_MLIR_EXCHANGE with
 * CGC_SUBGRAPH_DECLARATION_REQUEST.  Enumerates all PDL patterns in
 * src/amdxcgc/amdgpuops/ and returns the catalog as concatenated
 * CGC MLIR text (cgc_pattern.pattern declarations) in *out_catalog_bytes.
 *
 * The catalog is static per (DLL version, target_arch) — DXCP should cache
 * it for the lifetime of the device.  The catalog_hash can be used by
 * offline toolchains to detect driver updates.
 *
 * @param target_arch       GPU architecture string, e.g. "gfx1100". May be
 *                          NULL to return the full arch-independent catalog.
 * @param out_catalog_bytes DLL-allocated buffer holding the concatenated MLIR
 *                          text of all PDL pattern files.  Release with
 *                          dxcg_free_subgraph_catalog().
 * @param out_catalog_size  Number of bytes in *out_catalog_bytes (not NUL-
 *                          terminated; treat as raw text).
 * @param out_catalog_hash  Stable 64-bit FNV-1a hash of the catalog bytes,
 *                          suitable for cache validation.
 * @param out_error         On failure, DLL-allocated NUL-terminated error
 *                          string (release with dxcg_free()).  NULL on success.
 * @return                  0 on success, non-zero on failure.
 */
AMDXCGC_EXPORT int dxcg_query_subgraph_transformations(
    const char*  target_arch,
    const void** out_catalog_bytes,
    size_t*      out_catalog_size,
    uint64_t*    out_catalog_hash,
    char**       out_error);

/** Function-pointer typedef for GetProcAddress-style resolution by DXCP. */
typedef int (*PFN_dxcg_query_subgraph_transformations)(
    const char*  target_arch,
    const void** out_catalog_bytes,
    size_t*      out_catalog_size,
    uint64_t*    out_catalog_hash,
    char**       out_error);

/**
 * Release the catalog bytes returned by dxcg_query_subgraph_transformations.
 * Safe to call with NULL.
 */
AMDXCGC_EXPORT void dxcg_free_subgraph_catalog(const void* catalog_bytes);

/** Function-pointer typedef for GetProcAddress-style resolution by DXCP. */
typedef void (*PFN_dxcg_free_subgraph_catalog)(const void* catalog_bytes);

/**
 * Subgraph specialization / validation.
 *
 * Called by DXCP when handling D3D12_FEATURE_MLIR_EXCHANGE with
 * CGC_SUBGRAPH_SPECIALIZATION_REQUEST.  The DXCGC compiler passes in an
 * MLIR program with concrete tensor shapes; MIGraphX validates each matched
 * subgraph against shape constraints and hardware limits for target_arch.
 *
 * On success, *out_output_mlir is a (potentially corrected) copy of the
 * input MLIR confirming that the transformation is valid for this arch.
 * On failure, *out_error describes which pattern failed and why.
 *
 * If specialization succeeds, the subsequent dxcg_compile call with the
 * same MLIR is expected to succeed.  On failure DXCP returns
 * DXGI_ERROR_UNSUPPORTED and DXCGC falls back to non-transformed paths.
 *
 * @param input_mlir          MLIR bytecode or text with concrete shapes.
 * @param input_mlir_size     Byte count of input_mlir.
 * @param target_arch         GPU architecture, e.g. "gfx1100".
 * @param out_output_mlir     DLL-allocated validated MLIR on success.
 *                            Release with dxcg_free_validation_result().
 * @param out_output_mlir_size  Byte count of *out_output_mlir.
 * @param out_error           On failure, DLL-allocated error string
 *                            (release with dxcg_free()).  NULL on success.
 * @return                    0 on success, non-zero on failure.
 */
AMDXCGC_EXPORT int dxcg_validate_subgraph_specialization(
    const void*  input_mlir,
    size_t       input_mlir_size,
    const char*  target_arch,
    const void** out_output_mlir,
    size_t*      out_output_mlir_size,
    char**       out_error);

/** Function-pointer typedef for GetProcAddress-style resolution by DXCP. */
typedef int (*PFN_dxcg_validate_subgraph_specialization)(
    const void*  input_mlir,
    size_t       input_mlir_size,
    const char*  target_arch,
    const void** out_output_mlir,
    size_t*      out_output_mlir_size,
    char**       out_error);

/**
 * Release the validated MLIR buffer from dxcg_validate_subgraph_specialization.
 * Safe to call with NULL.
 */
AMDXCGC_EXPORT void dxcg_free_validation_result(const void* output_mlir);

/** Function-pointer typedef for GetProcAddress-style resolution by DXCP. */
typedef void (*PFN_dxcg_free_validation_result)(const void* output_mlir);

/* -------------------------------------------------------------------------
 * Memory management
 * ---------------------------------------------------------------------- */

/**
 * Release a buffer previously returned by any dxcg_* function
 * (compiled blob, error string, etc.).
 * It is safe to call with NULL.
 * Do NOT use this to free descriptor arrays — use dxcg_free_kernel_descriptors.
 */
AMDXCGC_EXPORT void dxcg_free(void* ptr);

/* -------------------------------------------------------------------------
 * Version query
 * ---------------------------------------------------------------------- */

/**
 * Return the MIGraphX engine version string (e.g. "2.16.0").
 * The returned pointer is a static string — do NOT free it.
 */
AMDXCGC_EXPORT const char* dxcg_version(void);

/**
 * Return the DxCGC API version string (e.g. "0.1.0").
 *
 * This version tracks the DxCGC C API contract (subgraph query,
 * kernel descriptor format, binding descriptor fields, etc.)
 * independently from the MIGraphX engine version returned by
 * dxcg_version().  DXCP uses this to validate ABI compatibility.
 *
 * Versioning scheme:
 *   MAJOR — incompatible API change (struct layout, removed export)
 *   MINOR — new exports added (backwards-compatible addition)
 *   PATCH — bug fixes, no API surface change
 *
 * Current: 0.1.0 — initial release of subgraph query/validation API
 *   (dxcg_query_subgraph_transformations,
 *    dxcg_validate_subgraph_specialization,
 *    dxcg_get_kernel_descriptors)
 *
 * The returned pointer is a static string — do NOT free it.
 */
AMDXCGC_EXPORT const char* dxcg_api_version(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MIGRAPHX_GUARD_DXCG_C_API_HPP
