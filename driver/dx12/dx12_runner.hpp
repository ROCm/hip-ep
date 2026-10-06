/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
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
#include <unordered_map>
#include <vector>

namespace hip_ep {
namespace dx12 {

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// ElfFormat — binary format of the kernel blob in KernelDescriptor::hsaco_data.
// Controls how the runtime binds kernel arguments and dispatches.
// ---------------------------------------------------------------------------
enum class ElfFormat : uint8_t {
  // HSA ELF relocatable (EI_OSABI=0x40, e_type=ET_REL): HSA ABI, unrelocated.
  // DX12: build_single_kernel_et_rel extracts a single-kernel ET_REL from fat
  // bundle. HIP:  not directly loadable without linking.
  HsaRel = 0,

  // HSA ELF dynamic (EI_OSABI=0x40, e_type=ET_DYN): HSA ABI, fully linked
  // HSACO. DX12: CreateComputePipelineCrossCompile(ElfHsa) + SetKernelArguments
  // + kernarg. HIP:  hipModuleLoadData + flat kernarg segment.
  HsaDyn = 1,
  HsaElf = 1, // alias for backward compat

  // PAL ELF relocatable (EI_OSABI=0x41, e_type=ET_REL): PAL compute shader ABI.
  // DX12: CreateComputePipelineCrossCompile(ElfPal) + CbData-pointer dispatch.
  //       CbData GPU VA → SetComputeRoot32BitConstants(0, 2, {va_lo, va_hi}).
  //       Shader reads CbData struct from that VA via F_ADDR_INDIRECT
  //       (SGPRs[0:1]).
  PalRel = 2,

  // PAL ELF linked dynamic (EI_OSABI=0x41, e_type=ET_DYN): same CbData
  // dispatch. Already linked so no relocation needed; same dispatch path as
  // PalRel.
  PalDyn = 3,
};

// ---------------------------------------------------------------------------
// ScalarSlot — one scalar kernel argument (int32/int64/float) passed by value.
// Only used for HsaElf format; PAL ELF kernels use cbdata_bytes + cbdata_slots.
// ---------------------------------------------------------------------------
struct ScalarSlot {
  std::size_t arg_index;         ///< Position in the full arg list (0-based)
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
struct KernelDescriptor {
  std::string entry_point;         ///< Kernel symbol / entry point name
  std::vector<uint8_t> hsaco_data; ///< Raw ELF bytes (HSA or PAL)
  ElfFormat elf_format = ElfFormat::HsaElf; ///< Binary format
  uint32_t dispatch_x = 1;                  ///< Thread groups in X
  uint32_t dispatch_y = 1;                  ///< Thread groups in Y
  uint32_t dispatch_z = 1;                  ///< Thread groups in Z
  uint32_t group_size_x = 64;               ///< Threads per group X
  uint32_t group_size_y = 1;                ///< Threads per group Y
  uint32_t group_size_z = 1;                ///< Threads per group Z

  // Per-argument sizes in bytes (GPU buffer inputs, then output, then scalars).
  // Only used for HsaElf format.
  std::vector<uint64_t> arg_sizes;

  // Index into arg_sizes of the output buffer slot (HsaElf only).
  std::size_t output_arg_index = 0;

  // For in-place GEMM kernels in an execute_raw_group where the output buffer
  // is shared with a prior kernel (e.g. QKVProj_offset concat-elimination):
  // index of the group-local kernel whose GPU output buffer this kernel should
  // REUSE. -1 = allocate a fresh output buffer (default).  Must be < this
  // kernel's index.
  int output_reuses_kernel = -1;

  // Byte width of each output element: 4 = float32, 2 = float16.
  uint32_t output_element_bytes = 4;

  // Scalar args (HsaElf only): appended after GPU buffer slots in arg_sizes.
  std::vector<ScalarSlot> scalar_slots;

  // Double-ptr arg indices (HsaElf only): T** indirection.
  std::vector<std::size_t> double_ptr_arg_indices;

  // Indices of "null" input slots (constant buffers with null data, size ≤ 4
  // bytes). The driver uses these to detect sentinel/placeholder inputs and
  // handle them without uploading actual data. Added for compatibility with
  // drivers that emit null_ptr sentinel inputs (e.g. custom_prebuilt output
  // shape sentinel).
  std::vector<std::size_t> null_input_indices;

  // ---------------------------------------------------------------------------
  // PAL ELF argument binding (PalRel / PalDyn format)
  // ---------------------------------------------------------------------------
  // cbdata_bytes: pre-assembled CbData struct with all scalar fields filled.
  //   Buffer VA slots are left as zero and patched at dispatch time from
  //   cbdata_slots. The runtime uploads this to a GPU buffer, then passes its
  //   GPU VA as root constants.
  std::vector<uint8_t> cbdata_bytes;

  // cbdata_slots: describes which byte ranges in cbdata_bytes hold GPU buffer
  // VAs.
  //   At dispatch time the runner patches these with live VAs from input_data /
  //   output.
  struct CbDataSlot {
    uint32_t byte_offset; ///< Byte offset within cbdata_bytes to write the VA
    int buf_index;        ///< -1 = output buffer; ≥0 = input_data[buf_index]
  };
  std::vector<CbDataSlot> cbdata_slots;
};

// Convert a raw kernel-output byte buffer to float32. output_element_bytes
// selects the source element type: 2 = float16 (IEEE half → float), otherwise
// float32 (copied through).
std::vector<float> bytes_to_f32(const std::vector<char> &bytes,
                                uint32_t output_element_bytes);

// ---------------------------------------------------------------------------
// Dx12Runner — full D3D12 + AMD CC execution context.
// ---------------------------------------------------------------------------
class Dx12Runner {
public:
  // adapter_selector: controls which AMD GPU is used for DX12 execution.
  //   ""        — auto: first AMD adapter where D3D12CreateDevice succeeds
  //   (default) "0","1"…  — 0-based AMD-only adapter index "0x…"     — DXGI
  //   LUID as 64-bit hex string other     — case-insensitive substring of
  //   adapter description name assume_gfx1151 - skip the RDNA3 DeviceId
  //   heuristic note on APUs.
  explicit Dx12Runner(std::string adapter_selector = "", bool verbose = false,
                      bool assume_gfx1151 = false)
      : adapter_selector_(std::move(adapter_selector)), verbose_(verbose),
        assume_gfx1151_(assume_gfx1151) {}
  ~Dx12Runner() = default;

  Dx12Runner(const Dx12Runner &) = delete;
  Dx12Runner &operator=(const Dx12Runner &) = delete;

  // Force adapter initialization without running a kernel.
  // Call before ensure_initialized_probe to detect the selected GPU arch for
  // compile. Safe to call multiple times (no-op after first call).
  void ensure_initialized_probe() { ensure_initialized(); }

  // Run kernel described by `kd`.
  // `input_data[i]` must contain exactly `kd.arg_sizes[i]` bytes of input data
  //   for argument slot i.  The last slot is treated as the output buffer.
  // Returns the output buffer contents as float32 elements.
  std::vector<float> execute(const KernelDescriptor &kd,
                             const std::vector<std::vector<char>> &input_data);

  // Like execute() but returns the raw output bytes (no fp16→fp32 conversion).
  // Used by multi-kernel chains to pass intermediate results to the next
  // kernel.
  std::vector<char>
  execute_raw(const KernelDescriptor &kd,
              const std::vector<std::vector<char>> &input_data);

  // Execute a group of kernels in a single DX12 command list submission.
  // All kernels run on the GPU without CPU readback between them — only the
  // final kernel's output is read back. UAV barriers provide inter-kernel sync.
  // Returns the output of the LAST kernel in the group.
  //
  // GroupInput describes one input slot for one kernel in the group:
  //   - cpu_data: non-empty → upload from CPU (model param or initial input)
  //   - source_kernel: >= 0 → use gpu_outputs[source_kernel] directly (no CPU
  //   roundtrip)
  //   - is_null: true → bind VA=0 (null sentinel for optional params)
  struct GroupInput {
    std::vector<char>
        cpu_data; ///< CPU bytes to upload; empty if source_kernel >= 0
    int source_kernel = -1; ///< Index into kernels[] whose output to reuse
    bool is_null = false;   ///< Bind as null VA=0
  };

  // kernels: descriptors in execution order
  // inputs_per_kernel: one GroupInput per non-output, non-scalar arg slot of
  // each kernel output_kernel_idx: which kernel's output to return (-1 = last
  // kernel)
  std::vector<char> execute_raw_group(
      const std::vector<KernelDescriptor> &kernels,
      const std::vector<std::vector<GroupInput>> &inputs_per_kernel,
      int output_kernel_idx = -1);

private:
  // Lazy initialisation — called once on first execute().
  void ensure_initialized();

  // Load the AMD extension DLL and obtain IAmdExtD3DFactory.
  void load_amd_ext();

  // Create PSO for `kd` via AMD CC CrossCompile API.
  ComPtr<ID3D12PipelineState> create_pso(const KernelDescriptor &kd);

  // Create a D3D12 root signature for PAL ELF dispatch:
  //   Root param 0: 2 x 32-bit root constants (GPU VA lo/hi of CbData buffer).
  // This maps to SGPRs[0:1] in the PAL compute shader (user-data slots 0:1 on
  // GFX11+).
  ComPtr<ID3D12RootSignature> create_pal_root_signature();

  // Allocate a committed resource in an upload heap (CPU-writable).
  ComPtr<ID3D12Resource>
  create_upload_buffer(uint64_t size_bytes, const void *initial_data = nullptr);

  // Allocate a committed resource in a default heap (GPU-local, UAV capable).
  ComPtr<ID3D12Resource> create_gpu_buffer(uint64_t size_bytes);

  // Stage double-pointer (T**) kernel arguments.
  //
  // MLSS GEMM/GQA kernels declare buffer operands with m_indirectionLevel=2:
  // the kernarg slot must hold the address of a device cell that in turn holds
  // the buffer VA, and the kernel dereferences twice. Given the data VA each
  // cell must hold, this allocates one contiguous DEFAULT-heap buffer of 8-byte
  // cells (so they share a page), fills it with a single synchronous upload,
  // and returns the cell VAs in the same order. Callers pass the returned cell
  // VA as the kernel argument instead of the buffer VA.
  //
  // Must be called BEFORE the dispatch command list is recorded -- it resets
  // and flushes the command list to perform the upload, which avoids needing an
  // in-list barrier between the fill and the first dispatch.
  //
  // Returns the buffer, which the caller must keep alive until after the
  // dispatch. cell_vas is resized to data_vas.size(). An empty data_vas is a
  // no-op.
  ComPtr<ID3D12Resource>
  stage_pointer_cells(const std::vector<D3D12_GPU_VIRTUAL_ADDRESS> &data_vas,
                      std::vector<D3D12_GPU_VIRTUAL_ADDRESS> &cell_vas);

  // Submit the current command list and block until GPU is idle.
  void flush();
  // flush_n: submit N times, wait once (amortizes DX12 fence overhead for
  // benchmarks).
  void flush_n(int n_iter);

public:
  // Streaming mode (default ON): accumulate all kernel dispatches into one
  // command list per eval() call, submit + wait once at the end. Eliminates
  // per-kernel fence overhead (~30µs on gfx1100) — critical for GEMV. Disable
  // with --disable-streaming for per-kernel sync (PIX/RGP profiling).
  bool streaming_mode = true;

  // Benchmark repeats: when >1, execute_raw_group submits the same CL this many
  // times in a tight loop before the fence. Set before calling p.eval() in
  // benchmark mode to get true amortized GPU time without per-iter CPU
  // overhead.
  int n_benchmark_repeats = 1;

private:
  // ---------------------------------------------------------------------------
  // D3D12 infrastructure
  // ---------------------------------------------------------------------------
  std::string adapter_selector_;
  std::string selected_gfx_arch_;       // set during ensure_initialized()
  uint64_t selected_adapter_luid_ = 0;  // DXGI LUID of selected adapter
  uint32_t selected_pci_device_id_ = 0; // PCI DeviceId from DXGI desc
  uint32_t selected_pci_vendor_id_ = 0; // PCI VendorId from DXGI desc
  std::string selected_device_name_;    // adapter description string
  bool initialized_ = false;
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> cmd_list_;
  ComPtr<ID3D12Fence> fence_;
  uint64_t fence_value_ = 0;
  HANDLE fence_event_ = nullptr;

  // AMD extension objects
  HMODULE amd_ext_module_ = nullptr;
  ComPtr<IAmdExtD3DFactory> amd_ext_factory_;
  ComPtr<IAmdExtD3DDevice5>
      amd_ext_device5_; // v5: HSA ET_REL pipeline creation
  ComPtr<IAmdExtD3DDevice7> amd_ext_device_;
  ComPtr<IAmdExtD3DDevice10> amd_ext_device10_; // v10: DispatchPalElf
  bool elf_hsa_supported_ = false;
  bool verbose_ = false;
  bool assume_gfx1151_ = false;

  // PAL ELF fallback root signature (2 root constants = CbData GPU VA lo/hi)
  // Created once during initialization and reused across all PAL ELF
  // dispatches.
  ComPtr<ID3D12RootSignature> pal_root_sig_;

  // PSO cache: maps (entry_point_name) → compiled pipeline state.
  // Avoids re-creating identical PSOs across test runs (PSO pool exhaustion
  // causes TDR). Key: kernel_name (unique per kernel binary+symbol combination
  // in practice).
  std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> pso_cache_;

public:
  bool HasPalElfDispatch() const { return amd_ext_device10_ != nullptr; }

  // Returns the GFX arch string for the selected DX12 adapter (e.g. "gfx1100").
  // Empty until ensure_initialized() has run; callers should call execute()
  // first.
  const std::string &get_selected_gfx_arch() const {
    return selected_gfx_arch_;
  }

  // Returns the DXGI adapter LUID (lower 64 bits) of the selected adapter.
  uint64_t get_selected_adapter_luid() const { return selected_adapter_luid_; }

  // Returns the PCI DeviceId of the selected adapter (e.g. 0x744c for gfx1100).
  uint32_t get_selected_pci_device_id() const {
    return selected_pci_device_id_;
  }
  uint32_t get_selected_pci_vendor_id() const {
    return selected_pci_vendor_id_;
  }

  // Human-readable adapter name (set after initialization).
  const std::string &get_selected_device_name() const {
    return selected_device_name_;
  }
};

} // namespace dx12
} // namespace hip_ep
