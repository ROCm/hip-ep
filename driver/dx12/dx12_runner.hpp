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
#pragma once

/*
 * Dx12Runner — executes a single GPU kernel via:
 *   - D3D12 device + command infrastructure
 *   - AMD Cross-Compile API (CreateComputePipelineCrossCompile with ElfHsa)
 *   - SetKernelArguments for buffer binding
 *
 * Usage:
 *   Dx12Runner runner;                         // lazy-initialised on first use
 *   runner.execute(descriptor, input_args);    // returns float output
 *
 * Thread safety: not thread-safe; construct one instance per thread.
 */

#include "amdcc_api.hpp"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hip_ep {
namespace dx12 {

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// ElfFormat — binary format of the kernel blob in KernelDescriptor::hsaco_data.
// Controls how the runtime binds kernel arguments and dispatches.
// ---------------------------------------------------------------------------
enum class ElfFormat : uint8_t
{
    // HSA ELF relocatable (EI_OSABI=0x40, e_type=ET_REL): HSA ABI, unrelocated.
    // DX12: build_single_kernel_et_rel extracts a single-kernel ET_REL from fat bundle.
    // HIP:  not directly loadable without linking.
    HsaRel  = 0,

    // HSA ELF dynamic (EI_OSABI=0x40, e_type=ET_DYN): HSA ABI, fully linked HSACO.
    // DX12: CreateComputePipelineCrossCompile(ElfHsa) + SetKernelArguments + kernarg.
    // HIP:  hipModuleLoadData + flat kernarg segment.
    HsaDyn  = 1,
    HsaElf  = 1, // alias for backward compat

    // PAL ELF relocatable (EI_OSABI=0x41, e_type=ET_REL): PAL compute shader ABI.
    // DX12: CreateComputePipelineCrossCompile(ElfPal) + CbData-pointer dispatch.
    //       CbData GPU VA → SetComputeRoot32BitConstants(0, 2, {va_lo, va_hi}).
    //       Shader reads CbData struct from that VA via F_ADDR_INDIRECT (SGPRs[0:1]).
    PalRel  = 2,

    // PAL ELF linked dynamic (EI_OSABI=0x41, e_type=ET_DYN): same CbData dispatch.
    // Already linked so no relocation needed; same dispatch path as PalRel.
    PalDyn  = 3,
};

// ---------------------------------------------------------------------------
// ScalarSlot — one scalar kernel argument (int32/int64/float) passed by value.
// Only used for HsaElf format; PAL ELF kernels use cbdata_bytes + cbdata_slots.
// ---------------------------------------------------------------------------
struct ScalarSlot
{
    std::size_t       arg_index;   ///< Position in the full arg list (0-based)
    std::vector<char> value_bytes; ///< Raw bytes: 1/2/4/8 bytes depending on type
};

// ---------------------------------------------------------------------------
// KernelDescriptor — runtime-agnostic description of one compiled GPU kernel.
// Mirrors the public dxcg_kernel_descriptor C struct but as a C++ value type.
//
// Binary format and argument binding:
//   HsaElf  → hsaco_data + scalar_slots + double_ptr_arg_indices
//             DX12: SetKernelArguments; HIP: hipModuleLaunchKernel kernarg
//   PalRel/PalDyn → hsaco_data + cbdata_bytes + cbdata_slots
//             DX12: CreateComputePipelineCrossCompile(ElfPal) + root constants
//             HIP:  hipModuleLoadData(HSA variant) + flat arg array (future)
// ---------------------------------------------------------------------------
struct KernelDescriptor
{
    std::string            entry_point;       ///< Kernel symbol / entry point name
    std::vector<uint8_t>   hsaco_data;        ///< Raw ELF bytes (HSA or PAL)
    ElfFormat              elf_format = ElfFormat::HsaElf; ///< Binary format
    uint32_t               dispatch_x   = 1; ///< Thread groups in X
    uint32_t               dispatch_y   = 1; ///< Thread groups in Y
    uint32_t               dispatch_z   = 1; ///< Thread groups in Z
    uint32_t               group_size_x = 64;///< Threads per group X
    uint32_t               group_size_y = 1; ///< Threads per group Y
    uint32_t               group_size_z = 1; ///< Threads per group Z

    // Per-argument sizes in bytes (GPU buffer inputs, then output, then scalars).
    // Only used for HsaElf format.
    std::vector<uint64_t>  arg_sizes;

    // Index into arg_sizes of the output buffer slot (HsaElf only).
    std::size_t            output_arg_index = 0;

    // Byte width of each output element: 4 = float32, 2 = float16.
    uint32_t               output_element_bytes = 4;

    // Scalar args (HsaElf only): appended after GPU buffer slots in arg_sizes.
    std::vector<ScalarSlot> scalar_slots;

    // Double-ptr arg indices (HsaElf only): T** indirection.
    std::vector<std::size_t> double_ptr_arg_indices;

    // ---------------------------------------------------------------------------
    // PAL ELF argument binding (PalRel / PalDyn format)
    // ---------------------------------------------------------------------------
    // cbdata_bytes: pre-assembled CbData struct with all scalar fields filled.
    //   Buffer VA slots are left as zero and patched at dispatch time from cbdata_slots.
    //   The runtime uploads this to a GPU buffer, then passes its GPU VA as root constants.
    std::vector<uint8_t>   cbdata_bytes;

    // cbdata_slots: describes which byte ranges in cbdata_bytes hold GPU buffer VAs.
    //   At dispatch time the runner patches these with live VAs from input_data / output.
    struct CbDataSlot
    {
        uint32_t byte_offset;  ///< Byte offset within cbdata_bytes to write the VA
        int      buf_index;    ///< -1 = output buffer; ≥0 = input_data[buf_index]
    };
    std::vector<CbDataSlot> cbdata_slots;
};

// ---------------------------------------------------------------------------
// Dx12Runner — full D3D12 + AMD CC execution context.
// ---------------------------------------------------------------------------
class Dx12Runner
{
public:
    // adapter_selector: controls which AMD GPU is used for DX12 execution.
    //   ""        — auto: first AMD gfx1100+ adapter (default)
    //   "0","1"…  — 0-based AMD-only adapter index
    //   "0x…"     — DXGI LUID as 64-bit hex string
    //   other     — case-insensitive substring of adapter description name
    explicit Dx12Runner(std::string adapter_selector = "", bool verbose = false, bool assume_gfx1151 = false)
        : adapter_selector_(std::move(adapter_selector)),
          assume_gfx1151_(assume_gfx1151), verbose_(verbose) {}
    ~Dx12Runner() = default;

    Dx12Runner(const Dx12Runner&)            = delete;
    Dx12Runner& operator=(const Dx12Runner&) = delete;

    // Run kernel described by `kd`.
    // `input_data[i]` must contain exactly `kd.arg_sizes[i]` bytes of input data
    //   for argument slot i.  The last slot is treated as the output buffer.
    // Returns the output buffer contents as float32 elements.
    std::vector<float> execute(const KernelDescriptor&              kd,
                               const std::vector<std::vector<char>>& input_data);

    // Like execute() but returns the raw output bytes (no fp16→fp32 conversion).
    // Used by multi-kernel chains to pass intermediate results to the next kernel.
    std::vector<char> execute_raw(const KernelDescriptor&              kd,
                                  const std::vector<std::vector<char>>& input_data);

private:
    // Lazy initialisation — called once on first execute().
    void ensure_initialized();

    // Load the AMD extension DLL and obtain IAmdExtD3DFactory.
    void load_amd_ext();

    // Create PSO for `kd` via AMD CC CrossCompile API.
    ComPtr<ID3D12PipelineState> create_pso(const KernelDescriptor& kd);

    // Create a D3D12 root signature for PAL ELF dispatch:
    //   Root param 0: 2 x 32-bit root constants (GPU VA lo/hi of CbData buffer).
    // This maps to SGPRs[0:1] in the PAL compute shader (user-data slots 0:1 on GFX11+).
    ComPtr<ID3D12RootSignature> create_pal_root_signature();

    // Allocate a committed resource in an upload heap (CPU-writable).
    ComPtr<ID3D12Resource> create_upload_buffer(uint64_t size_bytes,
                                                const void* initial_data = nullptr);

    // Allocate a committed resource in a default heap (GPU-local, UAV capable).
    ComPtr<ID3D12Resource> create_gpu_buffer(uint64_t size_bytes);

    // Submit the current command list and block until GPU is idle.
    void flush();

    // ---------------------------------------------------------------------------
    // D3D12 infrastructure
    // ---------------------------------------------------------------------------
    std::string                  adapter_selector_;
    bool                         initialized_    = false;
    bool                         assume_gfx1151_ = false;
    ComPtr<ID3D12Device>         device_;
    ComPtr<ID3D12CommandQueue>   queue_;
    ComPtr<ID3D12CommandAllocator>      allocator_;
    ComPtr<ID3D12GraphicsCommandList>   cmd_list_;
    ComPtr<ID3D12Fence>          fence_;
    uint64_t                     fence_value_    = 0;
    HANDLE                       fence_event_    = nullptr;

    // AMD extension objects
    HMODULE                      amd_ext_module_  = nullptr;
    ComPtr<IAmdExtD3DFactory>    amd_ext_factory_;
    ComPtr<IAmdExtD3DDevice5>    amd_ext_device5_;  // v5: HSA ET_REL pipeline creation
    ComPtr<IAmdExtD3DDevice7>    amd_ext_device_;
    ComPtr<IAmdExtD3DDevice10>   amd_ext_device10_;  // v10: DispatchPalElf
    bool                         elf_hsa_supported_ = false;
    bool                         verbose_           = false;

public:
    bool HasPalElfDispatch() const { return amd_ext_device10_ != nullptr; }
};

} // namespace dx12
} // namespace hip_ep
