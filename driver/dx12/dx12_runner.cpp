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

#include "dx12_runner.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace hip_ep {
namespace dx12 {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void throw_if_failed(HRESULT hr, const char* ctx)
{
    if(FAILED(hr))
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s failed with HRESULT 0x%08X", ctx, static_cast<unsigned>(hr));
        throw std::runtime_error(buf);
    }
}

#define DX12_CHECK(expr) throw_if_failed((expr), #expr)

// ---------------------------------------------------------------------------
// Lazy initialisation
// ---------------------------------------------------------------------------

void Dx12Runner::ensure_initialized()
{
    if(initialized_)
        return;

    // Enable D3D12 debug layer for validation
    {
        ComPtr<ID3D12Debug> debug;
        if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            debug->EnableDebugLayer();
    }

    // 1. Enumerate AMD adapters and select based on adapter_selector_.
    //
    // adapter_selector_ modes:
    //   ""        — auto: first AMD gfx1100+ adapter (DeviceId >= 0x7440)
    //   "0","1"…  — 0-based AMD-only index (counts only VendorId==0x1002)
    //   "0x…"     — 64-bit DXGI LUID hex string (e.g. "0x00000000000049E2")
    //   other     — case-insensitive substring of adapter Description
    //
    // gfx1100+ requirement (DeviceId >= 0x7440) is always enforced;
    // a pre-gfx1100 adapter is logged as [NOT SUPPORTED] and skipped.
    static constexpr UINT RDNA3_MIN_DEVICE_ID = 0x7440u;

    // Parse selector mode.
    enum class SelMode { Auto, Index, Luid, Name } sel_mode = SelMode::Auto;
    UINT        sel_index = 0;
    UINT64      sel_luid  = 0;
    std::string sel_name;

    if(!adapter_selector_.empty())
    {
        const auto& s = adapter_selector_;
        if((s[0] == '0' && s.size() > 2 && (s[1] == 'x' || s[1] == 'X')))
        {
            sel_mode = SelMode::Luid;
            sel_luid = std::stoull(s, nullptr, 16);
        }
        else if(std::all_of(s.begin(), s.end(), ::isdigit))
        {
            sel_mode  = SelMode::Index;
            sel_index = static_cast<UINT>(std::stoul(s));
        }
        else
        {
            sel_mode = SelMode::Name;
            sel_name = s;
            std::transform(sel_name.begin(), sel_name.end(), sel_name.begin(), ::tolower);
        }
    }

    ComPtr<IDXGIFactory4> factory;
    DX12_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));

    ComPtr<IDXGIAdapter1> adapter;
    UINT amd_index = 0; // 0-based index across AMD adapters only
    for(UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        char desc_str[256] = {};
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, desc_str, sizeof(desc_str), nullptr, nullptr);

        if(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter.Reset(); continue; }
        if(desc.VendorId == 0x1414)                 { adapter.Reset(); continue; }
        if(desc.VendorId != 0x1002)                 { adapter.Reset(); continue; }

        const UINT64 luid_val = (static_cast<UINT64>(desc.AdapterLuid.HighPart) << 32) |
                                 static_cast<UINT32>(desc.AdapterLuid.LowPart);
        const bool is_rdna3_plus = (desc.DeviceId >= RDNA3_MIN_DEVICE_ID);

        std::cerr << "[dx12_runner] AMD adapter[" << amd_index << "]: "
                  << desc_str
                  << " DeviceId=0x" << std::hex << desc.DeviceId
                  << " LUID=0x" << std::setw(16) << std::setfill('0') << luid_val << std::dec
                  << (is_rdna3_plus ? " [gfx1100+]" : " [below DeviceId heuristic]")
                  << "\n"; std::cerr.flush();

        // Check whether this adapter matches the selector.
        bool matches = false;
        switch(sel_mode)
        {
        case SelMode::Auto:
            matches = true; // capability decided later by CheckExtFeatureSupport
            break;
        case SelMode::Index:
            matches = (amd_index == sel_index);
            break;
        case SelMode::Luid:
            matches = (luid_val == sel_luid);
            break;
        case SelMode::Name:
        {
            std::string desc_lower(desc_str);
            std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(), ::tolower);
            matches = (desc_lower.find(sel_name) != std::string::npos);
            break;
        }
        }

        ++amd_index;

        if(!matches) { adapter.Reset(); continue; }

        if(!is_rdna3_plus && !assume_gfx1151_)
        {
            // DeviceId ordering does not track GPU generation - APUs sit far below the
            // discrete range - so this is advisory; CheckExtFeatureSupport is the real gate.
            std::cerr << "[dx12_runner] note: DeviceId=0x" << std::hex << desc.DeviceId
                      << " is below the RDNA3 discrete-range heuristic (0x"
                      << RDNA3_MIN_DEVICE_ID << std::dec
                      << "); relying on CheckExtFeatureSupport instead.\n";
        }

        if(SUCCEEDED(D3D12CreateDevice(adapter.Get(),
                                       D3D_FEATURE_LEVEL_12_0,
                                       IID_PPV_ARGS(&device_))))
        {
            std::cerr << "[dx12_runner] selected: " << desc_str
                      << " (AMD[" << (amd_index - 1) << "]"
                      << " DeviceId=0x" << std::hex << desc.DeviceId
                      << " LUID=0x" << std::setw(16) << std::setfill('0') << luid_val << std::dec
                      << ")\n"; std::cerr.flush();
            break;
        }
        adapter.Reset();
    }
    if(!device_)
    {
        std::string hint;
        if(!adapter_selector_.empty())
            hint = " Selector: \"" + adapter_selector_ + "\".";
        throw std::runtime_error(
            "Dx12Runner: no suitable AMD gfx1100+ D3D12 adapter found." + hint +
            " Use --dx12-adapter with an index (\"0\"), LUID (\"0x...\"), or name substring.");
    }

    // 2. Compute command queue with TDR disabled.
    // D3D12_COMMAND_LIST_TYPE_COMPUTE: GPU compute-only queue (no rasterization overhead).
    // D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT: suppresses the 2-second TDR watchdog
    // for work submitted to this queue. Long-running compute kernels (matrix ops, GEMM)
    // will run to completion instead of triggering a GPU reset and DEVICE_REMOVED.
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type  = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
    DX12_CHECK(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)));

    // 3. Command allocator + list (must match queue type).
    DX12_CHECK(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                               IID_PPV_ARGS(&allocator_)));
    DX12_CHECK(device_->CreateCommandList(0,
                                          D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                          allocator_.Get(),
                                          nullptr,
                                          IID_PPV_ARGS(&cmd_list_)));
    // Close immediately — we reopen before each dispatch.
    DX12_CHECK(cmd_list_->Close());

    // 4. Fence for CPU/GPU synchronisation.
    DX12_CHECK(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if(!fence_event_)
        throw std::runtime_error("Dx12Runner: CreateEvent failed");

    // 5. Load AMD extension and obtain IAmdExtD3DDevice.
    load_amd_ext();

    initialized_ = true;
}

void Dx12Runner::load_amd_ext()
{
    // The AMD cross-compile extension is exported from the user-mode driver DLL.
    // On Windows 64-bit this is typically amdxc64.dll on the driver search path.
    const wchar_t* candidates[] = {L"amdxc64.dll", L"amdvlk64.dll"};

    for(auto* dll : candidates)
    {
        amd_ext_module_ = LoadLibraryW(dll);
        if(amd_ext_module_)
            break;
    }
    if(!amd_ext_module_)
        throw std::runtime_error(
            "Dx12Runner: could not load AMD extension DLL (amdxc64.dll / amdvlk64.dll). "
            "Ensure AMD GPU drivers are installed.");

    auto* pfn = reinterpret_cast<PFN_AmdExtD3DCreateInterface>(
        GetProcAddress(amd_ext_module_, "AmdExtD3DCreateInterface"));
    if(!pfn)
        throw std::runtime_error(
            "Dx12Runner: AmdExtD3DCreateInterface not found in AMD driver DLL");

    // Obtain the factory using the D3D12 device as the outer object.
    // NOTE: Use __cdecl calling convention for pfn (defined in amdcc_api.hpp).
    DX12_CHECK(pfn(device_.Get(),
                   __uuidof(IAmdExtD3DFactory),
                   reinterpret_cast<void**>(amd_ext_factory_.ReleaseAndGetAddressOf())));

    // Create the per-device AMD extension interface (v7 adds CreateComputePipelineCrossCompile).
    DX12_CHECK(amd_ext_factory_->CreateInterface(
        device_.Get(),
        __uuidof(IAmdExtD3DDevice7),
        reinterpret_cast<void**>(amd_ext_device_.ReleaseAndGetAddressOf())));

    DX12_CHECK(amd_ext_device_->QueryInterface(__uuidof(IAmdExtD3DDevice5),
        reinterpret_cast<void**>(amd_ext_device5_.ReleaseAndGetAddressOf())));

    // Query v10 for DispatchPalElf (direct PAL ELF dispatch replicating metacommand path).
    HRESULT hr10 = amd_ext_device_->QueryInterface(__uuidof(IAmdExtD3DDevice10),
        reinterpret_cast<void**>(amd_ext_device10_.ReleaseAndGetAddressOf()));
    if(FAILED(hr10) || !amd_ext_device10_)
        std::cerr << "[dx12_runner] WARNING: IAmdExtD3DDevice10 (DispatchPalElf) not available (hr=0x"
                  << std::hex << (unsigned)hr10 << std::dec
                  << ") — PAL ELF kernels will fail. Update DXCP driver to a build that supports v10.\n";
    else if(verbose_)
        std::cout << "[dx12_runner] IAmdExtD3DDevice10 acquired — DispatchPalElf available.\n";

    // Query supported cross-compile features.
    AmdExtD3DCheckFeatureSupportFlags flags{};
    if(SUCCEEDED(amd_ext_device_->CheckExtFeatureSupport(
           0 /*AmdExtD3DCheckFeatureSupportType::Flags*/, &flags, sizeof(flags))))
    {
        elf_hsa_supported_ = flags.pipelineCrossCompileElfHsa;
        std::cerr << "[dx12_runner] pipelineCrossCompileElfHsa=" << flags.pipelineCrossCompileElfHsa
                  << " pipelinePalElf=" << flags.pipelinePalElf
                  << " pipelineHsaElf=" << flags.pipelineHsaElf << "\n";
    }
    if(!elf_hsa_supported_)
        std::cerr << "[dx12_runner] WARNING: pipelineCrossCompileElfHsa not supported.\n";
}

// ---------------------------------------------------------------------------
// PSO creation via AMD Cross-Compile API
// ---------------------------------------------------------------------------

// Detect PAL ELF: EI_OSABI byte [7] == 0x41 (AMD_AMDGPU_ELF_OSABI_AMDPAL).
// PAL ELFs use AmdShaderType::ElfPal (5) and route through PAL CodeObjectUploader
// instead of amdcc.dll, avoiding VOPD float dual-issue TDR crashes.
static bool is_pal_elf(const void* data, std::size_t size)
{
    if(!data || size < 8u) return false;
    const auto* p = static_cast<const uint8_t*>(data);
    if(p[0] != 0x7fu || p[1] != 'E' || p[2] != 'L' || p[3] != 'F') return false;
    return p[7] == 0x41u;
}

static bool is_hsa_rel_elf(const void* data, std::size_t size)
{
    if(!data || size < 20u) return false;
    const auto* p = static_cast<const uint8_t*>(data);
    if(p[0] != 0x7fu || p[1] != 'E' || p[2] != 'L' || p[3] != 'F') return false;
    if(p[7] != 0x40u) return false;
    return p[16] == 1u && p[17] == 0u;
}

// Convert an HSA ET_REL blob to PAL ELF format for ElfPal DX12 cross-compile.
// Three patches needed for PAL CodeObjectUploader compatibility:
//   1. EI_OSABI    byte[7]: 0x40 (HSA) → 0x41 (PAL)
//   2. EI_ABIVERSION byte[8]: 3 (HSA) → 0 (PAL expects 0)
//   3. NOTE section sh_flags: clear SHF_ALLOC (0x2) bit — PAL allocates GPU memory
//      for every SHF_ALLOC section; an allocatable NOTE corrupts GPU memory layout.
static std::vector<uint8_t> patch_to_pal_elf(const void* data, std::size_t size)
{
    std::vector<uint8_t> patched(static_cast<const uint8_t*>(data),
                                  static_cast<const uint8_t*>(data) + size);
    if(patched.size() < 64u) return patched;

    // Patch 1+2: EI_OSABI and EI_ABIVERSION
    patched[7] = 0x41u;
    patched[8] = 0x00u;

    // Patch 3: clear SHF_ALLOC from all SHT_NOTE sections.
    // ELF64 header fields (little-endian):
    //   e_shoff at byte 40, e_shnum at byte 60, e_shentsize=64
    uint64_t e_shoff; uint16_t e_shnum;
    std::memcpy(&e_shoff, patched.data() + 40, 8);
    std::memcpy(&e_shnum, patched.data() + 60, 2);
    for(uint16_t i = 0; i < e_shnum; ++i)
    {
        const std::size_t sh_base = static_cast<std::size_t>(e_shoff) + i * 64u;
        if(sh_base + 64u > patched.size()) break;
        uint32_t sh_type; uint64_t sh_flags;
        std::memcpy(&sh_type,  patched.data() + sh_base +  4, 4);
        std::memcpy(&sh_flags, patched.data() + sh_base + 16, 8);
        if(sh_type == 7u /*SHT_NOTE*/ && (sh_flags & 0x2u))  // SHF_ALLOC
        {
            sh_flags &= ~static_cast<uint64_t>(0x2u);
            std::memcpy(patched.data() + sh_base + 16, &sh_flags, 8);
        }
    }
    return patched;
}

// ---------------------------------------------------------------------------
// PAL ELF root signature (used for Winograd conv kernels)
// ---------------------------------------------------------------------------
// PAL ELF shaders read their argument buffer via a GPU VA passed in SGPRs[0:1]
// (user-data slots 0:1 on GFX11+).  In D3D12 terms, SGPRs are mapped from root
// constants: root parameter 0 = 2 DWORDs (lo/hi of the CbData GPU VA).
//
// The shader uses F_ADDR_INDIRECT: it reads a pointer from SGPRs[0:1] and then
// scalar-loads the entire CbData struct from that GPU address.
ComPtr<ID3D12RootSignature> Dx12Runner::create_pal_root_signature()
{
    // 2 root constants at shader register b0, space 0 — maps to SGPRs[0:1].
    D3D12_ROOT_PARAMETER root_param{};
    root_param.ParameterType                    = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_param.Constants.ShaderRegister         = 0;
    root_param.Constants.RegisterSpace          = 0;
    root_param.Constants.Num32BitValues         = 2;
    root_param.ShaderVisibility                 = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters     = 1;
    desc.pParameters       = &root_param;
    desc.NumStaticSamplers = 0;
    desc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig_blob, err_blob;
    HRESULT hr = D3D12SerializeRootSignature(
        &desc, D3D_ROOT_SIGNATURE_VERSION_1_0,
        sig_blob.GetAddressOf(), err_blob.GetAddressOf());
    if(FAILED(hr))
    {
        std::string msg = "D3D12SerializeRootSignature (PAL) failed";
        if(err_blob && err_blob->GetBufferPointer())
            msg += std::string(": ") +
                   static_cast<const char*>(err_blob->GetBufferPointer());
        throw std::runtime_error(msg);
    }

    ComPtr<ID3D12RootSignature> root_sig;
    DX12_CHECK(device_->CreateRootSignature(
        0, sig_blob->GetBufferPointer(), sig_blob->GetBufferSize(),
        IID_PPV_ARGS(&root_sig)));
    return root_sig;
}

ComPtr<ID3D12PipelineState> Dx12Runner::create_pso(const KernelDescriptor& kd)
{
    // Check pipelineCrossCompileElfHsa support — required for our HSA ELF kernels.
    // pipelinePalElf would be needed for PAL ELF blobs (not used for GEMM kernels).
    if(!elf_hsa_supported_ && !is_hsa_rel_elf(kd.hsaco_data.data(), kd.hsaco_data.size()))
        throw std::runtime_error(
            "Dx12Runner::create_pso: pipelineCrossCompileElfHsa not supported by installed driver.");

    const void*  blob_data = kd.hsaco_data.data();
    std::size_t  blob_size = kd.hsaco_data.size();
    AmdShaderType sh_type  = AmdShaderType::ElfHsa;

    // Blob type routing:
    // - HSA ET_REL (EI_OSABI=0x40, e_type=1): use CreateComputePipelineFromElf (IAmdExtD3DDevice5).
    //   dxcp's CreateComputePipelineCrossCompile rejects ET_REL HSA blobs with ErrorInvalidPipelineElf
    //   (computeGraphicsProgram.cpp:604 — intentional fix to prevent GPU TDR from unrelocated ELF).
    //   CreateComputePipelineFromElf routes through PAL's AMDHSA pipeline (same as DxGML InitFromElf)
    //   and correctly handles ET_REL with AMDHSA MsgPack metadata.
    // - HSA ET_DYN (EI_OSABI=0x40, e_type=3): use CreateComputePipelineCrossCompile ElfHsa.
    // - PAL ELF (EI_OSABI=0x41): use CreateComputePipelineCrossCompile ElfPal.

    // All ELF blobs go through CreateComputePipelineCrossCompile with ElfHsa.
    // dxcp handles routing to PAL's InitFromElf-equivalent path internally.
    const bool hsa_rel = is_hsa_rel_elf(blob_data, blob_size);
    auto do_create_pso = [&](ComPtr<ID3D12PipelineState>& out_pso) -> HRESULT
    {
        if(hsa_rel)
        {
            if(!amd_ext_device5_) return E_NOINTERFACE;
            AmdExtD3DPipelineElfInfo elf{};
            elf.type = AmdExtD3DStructPipelineElf;
            elf.pNext = nullptr;
            elf.pElfBinary = blob_data;
            elf.elfSizeInBytes = blob_size;
            elf.threadsPerGroup = {kd.group_size_x, kd.group_size_y, kd.group_size_z};
            std::cerr << "[dx12_runner] CreatePipelineFromElf HSA ET_REL blob="
                      << blob_size << " kernel=" << kd.entry_point << "\n";
            return amd_ext_device5_->CreateComputePipelineFromElf(
                &elf, IID_PPV_ARGS(&out_pso));
        }

        AmdShaderType sh = AmdShaderType::ElfHsa;
        switch(kd.elf_format)
        {
        case ElfFormat::PalRel:
        case ElfFormat::PalDyn:
            sh = AmdShaderType::ElfPal;
            break;
        case ElfFormat::HsaRel:
        case ElfFormat::HsaDyn:
            sh = AmdShaderType::ElfHsa;
            break;
        default:
            if(is_pal_elf(blob_data, blob_size)) sh = AmdShaderType::ElfPal;
            break;
        }
        AmdExtD3DPipelineCrossCompileInfo cc{};
        cc.type                  = AmdExtD3DStructPipelineCrossCompile;
        cc.pNext             = nullptr;
        cc.pBlob             = blob_data;
        cc.blobSizeInBytes   = blob_size;
        cc.shType            = sh;
        cc.pOptions          = nullptr;
        cc.optionSizeInBytes = 0;
        cc.pKernelName       = kd.entry_point.empty() ? nullptr : kd.entry_point.c_str();
        cc.threadsPerGroup   = {kd.group_size_x, kd.group_size_y, kd.group_size_z};
        std::cerr << "[dx12_runner] CrossCompile shType=" << static_cast<int>(sh)
                  << " blob=" << blob_size << " kernel=" << kd.entry_point << "\n";
        return amd_ext_device_->CreateComputePipelineCrossCompile(&cc, IID_PPV_ARGS(&out_pso));
    };

    ComPtr<ID3D12PipelineState> pso;
    HRESULT hr = do_create_pso(pso);

    // On DXGI_ERROR_DEVICE_REMOVED (0x887A0005): OS WDDM TDR recovery from a prior run
    // may still be in progress. Log the removal reason, then retry with increasing delays.
    // DISABLE_GPU_TIMEOUT on the queue prevents new TDRs; this handles pre-existing ones.
    if(hr == DXGI_ERROR_DEVICE_REMOVED)
    {
        // Log the specific removal reason to distinguish hung vs driver crash vs reset.
        if(device_)
        {
            HRESULT reason = device_->GetDeviceRemovedReason();
            const char* reason_str =
                reason == DXGI_ERROR_DEVICE_HUNG            ? "DEVICE_HUNG (timeout)"     :
                reason == DXGI_ERROR_DEVICE_RESET           ? "DEVICE_RESET"              :
                reason == DXGI_ERROR_DRIVER_INTERNAL_ERROR  ? "DRIVER_INTERNAL_ERROR"     :
                reason == DXGI_ERROR_INVALID_CALL           ? "INVALID_CALL"              :
                                                              "DEVICE_REMOVED";
            std::cerr << "[dx12_runner] GetDeviceRemovedReason: 0x"
                      << std::hex << static_cast<unsigned>(reason) << std::dec
                      << " (" << reason_str << ")\n";
        }
        std::cerr << "[dx12_runner] Waiting for OS WDDM GPU TDR recovery...\n";

        for(int attempt = 1; attempt <= 5 && FAILED(hr); ++attempt)
        {
            initialized_ = false;
            device_.Reset();
            queue_.Reset();
            cmd_list_.Reset();
            allocator_.Reset();
            amd_ext_device_.Reset();
            amd_ext_device10_.Reset();
            amd_ext_factory_.Reset();
            const DWORD wait_ms = static_cast<DWORD>(attempt * 3000);  // 3, 6, 9, 12, 15 s
            std::cerr << "[dx12_runner] Retry " << attempt << "/5 (waiting " << wait_ms/1000 << "s)...\n";
            ::Sleep(wait_ms);
            try { ensure_initialized(); }
            catch(const std::exception& e)
            {
                std::cerr << "[dx12_runner] device re-init failed: " << e.what() << "\n";
                continue;
            }
            pso.Reset();
            hr = do_create_pso(pso);
            if(SUCCEEDED(hr))
                std::cerr << "[dx12_runner] GPU recovered after attempt " << attempt << "\n";
        }
    }
    if(FAILED(hr))
    {
        std::ostringstream oss;
        oss << "create_pso failed with HRESULT 0x" << std::hex << std::uppercase << hr;
        throw std::runtime_error(oss.str());
    }
    return pso;
}

// ---------------------------------------------------------------------------
// Resource helpers
// ---------------------------------------------------------------------------

ComPtr<ID3D12Resource> Dx12Runner::create_upload_buffer(uint64_t size_bytes,
                                                        const void* initial_data)
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = size_bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> buf;
    DX12_CHECK(device_->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&buf)));

    if(initial_data && size_bytes > 0)
    {
        void* mapped = nullptr;
        DX12_CHECK(buf->Map(0, nullptr, &mapped));
        std::memcpy(mapped, initial_data, static_cast<std::size_t>(size_bytes));
        buf->Unmap(0, nullptr);
    }
    return buf;
}

ComPtr<ID3D12Resource> Dx12Runner::create_gpu_buffer(uint64_t size_bytes)
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = size_bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ComPtr<ID3D12Resource> buf;
    DX12_CHECK(device_->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&buf)));
    return buf;
}

// ---------------------------------------------------------------------------
// GPU flush — submit and wait for completion.
// ---------------------------------------------------------------------------

void Dx12Runner::flush()
{
    DX12_CHECK(cmd_list_->Close());
    ID3D12CommandList* lists[] = {cmd_list_.Get()};
    queue_->ExecuteCommandLists(1, lists);

    const uint64_t signal_val = ++fence_value_;
    DX12_CHECK(queue_->Signal(fence_.Get(), signal_val));

    if(fence_->GetCompletedValue() < signal_val)
    {
        DX12_CHECK(fence_->SetEventOnCompletion(signal_val, fence_event_));
        WaitForSingleObject(fence_event_, INFINITE);
    }
}

// ---------------------------------------------------------------------------
// execute() — full dispatch cycle
// ---------------------------------------------------------------------------

std::vector<float>
Dx12Runner::execute(const KernelDescriptor&              kd,
                    const std::vector<std::vector<char>>& input_data)
{
    ensure_initialized();

    // Validate: arg_sizes = GPU-buffer inputs + 1 output + N scalars.
    // Skip for PAL ELF dispatch — CbData replaces the arg_sizes mechanism.
    const std::size_t buffer_args = kd.arg_sizes.size();
    const std::size_t n_scalar = kd.scalar_slots.size();
    std::size_t n_args = buffer_args;
    for(const auto& slot : kd.scalar_slots)
        n_args = (std::max)(n_args, slot.arg_index + 1);
    const bool is_pal_elf = (kd.elf_format == ElfFormat::PalRel || kd.elf_format == ElfFormat::PalDyn);
    if(n_args < 1)
        throw std::runtime_error("Dx12Runner::execute: no arguments in descriptor");
    if(!is_pal_elf && input_data.size() + 1 != buffer_args)
        throw std::runtime_error(
            "Dx12Runner::execute: arg count mismatch (gpu_inputs=" +
            std::to_string(input_data.size()) + " scalars=" + std::to_string(n_scalar) +
            " output_arg_index=" + std::to_string(kd.output_arg_index) +
            " total=" + std::to_string(n_args) + ")");

    // 1. Build PSO.
    std::cerr << "[dx12_runner] execute: calling create_pso...\n"; std::cerr.flush();
    auto pso = create_pso(kd);
    std::cerr << "[dx12_runner] execute: create_pso done, pso=" << pso.Get() << "\n"; std::cerr.flush();

    // 2. Allocate GPU buffers.
    const uint64_t out_size = kd.arg_sizes[kd.output_arg_index];
    std::cerr << "[dx12_runner] execute: out_size=" << out_size << " output_arg_index=" << kd.output_arg_index
              << " n_args=" << n_args << " input_data.size()=" << input_data.size() << "\n"; std::cerr.flush();

    std::vector<ComPtr<ID3D12Resource>> upload_bufs(input_data.size());
    std::vector<ComPtr<ID3D12Resource>> gpu_input_bufs(input_data.size());
    for(std::size_t i = 0; i < input_data.size(); ++i)
    {
        // Use the actual buffer size from input_data (driver already sized it correctly).
        const uint64_t sz = input_data[i].size() > 0 ? input_data[i].size() : 4096u;
        // Upload heap: CPU uploads initial data.
        upload_bufs[i] = create_upload_buffer(sz, input_data[i].data());
        // GPU heap: kernel reads from here.
        gpu_input_bufs[i] = create_gpu_buffer(sz);
        std::cerr << "[dx12_runner] execute: input[" << i << "] size=" << sz << " VA=0x" << std::hex << gpu_input_bufs[i]->GetGPUVirtualAddress() << std::dec << "\n"; std::cerr.flush();
    }

    auto gpu_output_buf = create_gpu_buffer(out_size);
    std::cerr << "[dx12_runner] execute: output VA=0x" << std::hex << gpu_output_buf->GetGPUVirtualAddress() << std::dec << " size=" << out_size << "\n"; std::cerr.flush();

    // 2b. Pre-create double-ptr intermediate buffers and fill them synchronously
    //     (before the main command list, so the GPU sees fresh data without needing
    //     in-list barriers).  Mirrors D3D12DeviceMemory::setDoublePointer().
    std::unordered_set<std::size_t> double_ptr_slots(
        kd.double_ptr_arg_indices.begin(), kd.double_ptr_arg_indices.end());

    // We need data VAs upfront — compute them from the already-allocated buffers.
    // Slot ordering mirrors the arg-building loop below: non-scalar slots are visited
    // in order; output_arg_index is interspersed.
    // Build a map: arg_index → data_va for double-ptr slots.
    std::unordered_map<std::size_t, D3D12_GPU_VIRTUAL_ADDRESS> dbl_ptr_data_vas;
    {
        std::size_t ib = 0;
        const std::size_t total_args = n_args;
        std::unordered_map<std::size_t, bool> scalar_set;
        for(const auto& ss : kd.scalar_slots)
            scalar_set[ss.arg_index] = true;
        for(std::size_t i = 0; i < total_args; ++i)
        {
            if(scalar_set.count(i)) continue;
            D3D12_GPU_VIRTUAL_ADDRESS dva =
                (i == kd.output_arg_index)
                ? gpu_output_buf->GetGPUVirtualAddress()
                : gpu_input_bufs[ib++]->GetGPUVirtualAddress();
            if(double_ptr_slots.count(i))
                dbl_ptr_data_vas[i] = dva;
        }
    }

    // Allocate and fill ptr_bufs (DEFAULT heap, COMMON state) via synchronous flush.
    std::unordered_map<std::size_t, ComPtr<ID3D12Resource>> ptr_buf_map;
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> dbl_ptr_vas_storage; // keep VAs alive
    dbl_ptr_vas_storage.reserve(double_ptr_slots.size());
    if(!double_ptr_slots.empty())
    {
        DX12_CHECK(allocator_->Reset());
        DX12_CHECK(cmd_list_->Reset(allocator_.Get(), nullptr));
        // Keep upload buffers alive until after flush.
        std::vector<ComPtr<ID3D12Resource>> tmp_uploads;
        for(auto& [arg_idx, data_va] : dbl_ptr_data_vas)
        {
            static constexpr uint64_t kPtrBufSize = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
            D3D12_HEAP_PROPERTIES hp{};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = kPtrBufSize; rd.Height = 1; rd.DepthOrArraySize = 1;
            rd.MipLevels = 1; rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            ComPtr<ID3D12Resource> pb;
            DX12_CHECK(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE,
                &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&pb)));

            auto up = create_upload_buffer(kPtrBufSize,
                                           reinterpret_cast<const char*>(&data_va));
            cmd_list_->CopyBufferRegion(pb.Get(), 0, up.Get(), 0, kPtrBufSize);

            ptr_buf_map[arg_idx] = pb;
            tmp_uploads.push_back(std::move(up));
        }
        flush(); // Execute the ptr_buf fills synchronously before the main dispatch.
        // tmp_uploads kept alive until here, then destroyed (fine, copy already done).
    }


    D3D12_HEAP_PROPERTIES readback_hp{};
    readback_hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC readback_rd{};
    readback_rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    readback_rd.Width            = out_size;
    readback_rd.Height           = 1;
    readback_rd.DepthOrArraySize = 1;
    readback_rd.MipLevels        = 1;
    readback_rd.SampleDesc.Count = 1;
    readback_rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback_buf;
    DX12_CHECK(device_->CreateCommittedResource(
        &readback_hp, D3D12_HEAP_FLAG_NONE, &readback_rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback_buf)));

    std::cerr << "[dx12_runner] execute: readback_buf allocated, recording cmds...\n"; std::cerr.flush();
    // 3. Record commands.
    DX12_CHECK(allocator_->Reset());
    DX12_CHECK(cmd_list_->Reset(allocator_.Get(), nullptr));

    // 3a. Copy inputs from upload → GPU default heap.
    for(std::size_t i = 0; i < input_data.size(); ++i)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource   = gpu_input_bufs[i].Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd_list_->ResourceBarrier(1, &barrier);

        cmd_list_->CopyResource(gpu_input_bufs[i].Get(), upload_bufs[i].Get());

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        cmd_list_->ResourceBarrier(1, &barrier);
    }

    // 3b. Set PSO and bind arguments.
    std::cerr << "[dx12_runner] execute: SetPipelineState...\n"; std::cerr.flush();
    cmd_list_->SetPipelineState(pso.Get());
    std::cerr << "[dx12_runner] execute: SetPipelineState done\n"; std::cerr.flush();

    // ---------------------------------------------------------------------------
    // PAL ELF dispatch path (Winograd conv kernels)
    // ---------------------------------------------------------------------------
    // PAL ELF shaders use the F_ADDR_INDIRECT mechanism: the shader reads its
    // CbData struct from a GPU VA passed in SGPRs[0:1] (D3D12 root constants).
    // SetKernelArguments cannot be used — it asserts hsaAbiMode (HSA-only).
    // This path allocates a GPU buffer for CbData, patches buffer VAs into it,
    // then dispatches via SetComputeRoot32BitConstants with the CbData GPU VA.
    const bool is_pal_dispatch = (kd.elf_format == ElfFormat::PalRel  ||
                                   kd.elf_format == ElfFormat::PalDyn);
    std::cerr << "[dx12_runner] execute: elf_format=" << static_cast<int>(kd.elf_format)
              << " is_pal=" << is_pal_dispatch << " cbdata_bytes=" << kd.cbdata_bytes.size()
              << " cbdata_slots=" << kd.cbdata_slots.size() << "\n";
    for(const auto& sl : kd.cbdata_slots)
        std::cerr << "  cbdata_slot: off=0x" << std::hex << sl.byte_offset << std::dec << " buf=" << sl.buf_index << "\n";
    std::cerr.flush();
    if(is_pal_dispatch && !kd.cbdata_bytes.empty() && !amd_ext_device10_)
        throw std::runtime_error(
            "[dx12_runner] PAL ELF kernel requires IAmdExtD3DDevice10::DispatchPalElf"
            " but the interface is not available. Update DXCP to a build that supports v10.");
    bool dispatched_pal = false;
    if(is_pal_dispatch && !kd.cbdata_bytes.empty() && amd_ext_device10_)
    {
        // PAL ELF dispatch via DispatchPalElf (v10 API) — replicates metacommand exactly:
        // 1. Patches buffer VAs into CbData
        // 2. CmdAllocateEmbeddedData → embed CbData in command buffer memory
        // 3. CmdSetUserData(0, 2, {cbVaLo, cbVaHi}) → SGPRs[2:3] = CbData VA
        // 4. CmdDispatch — same as DdiMetaCmdConvFury::ExecuteConv

        // 1. Build CbData with direct buffer VAs (DispatchPalElf handles CmdAllocateEmbeddedData).
        // NOTE: The Fury shader with F_ADDR_INDIRECT may expect descriptor handles or raw VAs.
        // Test: Disable F_ADDR_INDIRECT (bit 6) and use direct buffer VAs.
        // The shader branches on bit 6 of flags64 (F_ADDR_INDIRECT):
        //   - If set: reads pointer buffers (double-indirection)
        //   - If not set: uses buffer VAs directly from CbData
        // Testing WITHOUT F_ADDR_INDIRECT to verify direct VA path works first.
        // Build CbData with direct buffer VAs.
        // F_ADDR_INDIRECT (bit 6) is NOT set in flags64 — shader uses direct VAs from CbData.
        std::vector<uint8_t> cbdata = kd.cbdata_bytes;

        std::vector<ComPtr<ID3D12Resource>> ptr_upload_bufs;  // unused (no pointer indirection)
        std::vector<ComPtr<ID3D12Resource>> ptr_gpu_bufs;      // unused

        for(const auto& slot : kd.cbdata_slots)
        {
            if(slot.byte_offset + 8u > cbdata.size())
                throw std::runtime_error("cbdata_buf_slot byte_offset out of range");
            D3D12_GPU_VIRTUAL_ADDRESS va{};
            if(slot.buf_index < 0)
                va = gpu_output_buf->GetGPUVirtualAddress();
            else
                va = gpu_input_bufs[static_cast<std::size_t>(slot.buf_index)]
                         ->GetGPUVirtualAddress();
            std::memcpy(cbdata.data() + slot.byte_offset, &va, 8);
        }

        // Log CbData for diagnosis.
        {
            auto rd_u32 = [&](size_t off) -> uint32_t { uint32_t v=0; memcpy(&v, cbdata.data()+off, 4); return v; };
            auto rd_u64 = [&](size_t off) -> uint64_t { uint64_t v=0; memcpy(&v, cbdata.data()+off, 8); return v; };
            std::cerr << "[dx12_runner] CbData: N=" << rd_u32(0x00) << " C=" << rd_u32(0x04)
                      << " H=" << rd_u32(0x08) << " W=" << rd_u32(0x0C)
                      << " K=" << rd_u32(0x10) << " nGroups=" << rd_u32(0x14)
                      << " flags=0x" << std::hex << rd_u64(0x18) << std::dec
                      << "\n  dataAddr=0x" << std::hex << rd_u64(0x20) << " filterAddr=0x" << rd_u64(0x28) << " outputAddr=0x" << rd_u64(0x30) << std::dec << "\n";
            std::cerr.flush();
        }

        amd_ext_device10_->DispatchPalElf(
            cmd_list_.Get(),
            reinterpret_cast<const uint32_t*>(cbdata.data()),
            static_cast<uint32_t>(cbdata.size() / 4),
            kd.dispatch_x, kd.dispatch_y, kd.dispatch_z);
        dispatched_pal = true;
    }

    if(!dispatched_pal)
    {
    // ---------------------------------------------------------------------------
    // HSA ELF dispatch path (GEMM, GQA, and all non-PAL kernels)
    // ---------------------------------------------------------------------------
    // Build argument value pointer array.
    // - GPU buffer slots: ppValues[i] → &gpu_va (uint64_t GPU virtual address)
    // - Scalar slots:     ppValues[i] → raw bytes of the scalar value (int32/int64/float)
    // - Double-ptr slots: ppValues[i] → &ptr_buf_va (VA of the 8-byte pointer buffer)
    // SetKernelArguments passes each pointer's pointed-to bytes as the kernel argument.

    // Build scalar slot lookup: arg_index → pointer to embedded scalar bytes.
    std::unordered_map<std::size_t, const void*> scalar_val_ptrs;
    for(const auto& ss : kd.scalar_slots)
        scalar_val_ptrs[ss.arg_index] = ss.value_bytes.data();

    // gpu_vas must stay alive until after SetKernelArguments — reserve to avoid realloc.
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> gpu_vas;
    gpu_vas.reserve(input_data.size() + 1);

    // dbl_ptr_vas holds the VA of each ptr_buf (must outlive SetKernelArguments).
    // ptr_buf_map built in section 2b; already flushed to GPU.
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> dbl_ptr_vas;
    dbl_ptr_vas.reserve(ptr_buf_map.size());

    std::vector<const void*> arg_ptrs(n_args, nullptr);
    std::size_t input_buf_idx = 0;

    for(std::size_t i = 0; i < n_args; ++i)
    {
        auto scalar_it = scalar_val_ptrs.find(i);
        if(scalar_it != scalar_val_ptrs.end())
        {
            // Scalar slot: point directly at the embedded scalar bytes.
            arg_ptrs[i] = scalar_it->second;
        }
        else
        {
            // Determine the data GPU VA for this slot.
            D3D12_GPU_VIRTUAL_ADDRESS data_va{};
            if(i == kd.output_arg_index)
                data_va = gpu_output_buf->GetGPUVirtualAddress();
            else
                data_va = gpu_input_bufs[input_buf_idx++]->GetGPUVirtualAddress();

            auto dbl_it = ptr_buf_map.find(i);
            if(dbl_it != ptr_buf_map.end())
            {
                // Double-ptr slot: pass VA of the pre-filled pointer buffer.
                dbl_ptr_vas.push_back(dbl_it->second->GetGPUVirtualAddress());
                arg_ptrs[i] = &dbl_ptr_vas.back();
            }
            else
            {
                gpu_vas.push_back(data_va);
                arg_ptrs[i] = &gpu_vas.back();
            }
        }
    }

    amd_ext_device_->SetKernelArguments(
        cmd_list_.Get(),
        0u,
        static_cast<uint32_t>(n_args),
        arg_ptrs.data());

    // 3c. Dispatch.
    cmd_list_->Dispatch(kd.dispatch_x, kd.dispatch_y, kd.dispatch_z);
    } // end if(!dispatched_pal)

    // 3d. UAV barrier on output, then copy to readback.
    {
        D3D12_RESOURCE_BARRIER uav_barrier{};
        uav_barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav_barrier.UAV.pResource = nullptr; // global UAV flush (matches MLSS reference)
        cmd_list_->ResourceBarrier(1, &uav_barrier);

        D3D12_RESOURCE_BARRIER copy_barrier{};
        copy_barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        copy_barrier.Transition.pResource   = gpu_output_buf.Get();
        copy_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        copy_barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        copy_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd_list_->ResourceBarrier(1, &copy_barrier);

        cmd_list_->CopyResource(readback_buf.Get(), gpu_output_buf.Get());
    }

    std::cerr << "[dx12_runner] execute: dispatched, calling flush()...\n"; std::cerr.flush();
    // 4. Submit and wait.
    flush();
    std::cerr << "[dx12_runner] execute: flush() done\n"; std::cerr.flush();

    // For f16 outputs (output_element_bytes == 2): convert each uint16 bit-pattern
    // to float32 so the caller receives the same element count as hip_run.
    std::vector<float> result;
    {
        D3D12_RANGE read_range{0, static_cast<SIZE_T>(out_size)};
        void* mapped = nullptr;
        DX12_CHECK(readback_buf->Map(0, &read_range, &mapped));

        if(kd.output_element_bytes == 2)
        {
            // f16 output: reinterpret bytes as uint16_t, convert to float32.
            const std::size_t n_elems = static_cast<std::size_t>(out_size) / 2;
            result.resize(n_elems);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(mapped);
            for(std::size_t i = 0; i < n_elems; ++i)
            {
                // IEEE 754 f16 → f32 bit-manipulation.
                uint32_t h = src[i];
                uint32_t sign     = (h & 0x8000u) << 16;
                uint32_t exponent = (h & 0x7C00u) >> 10;
                uint32_t mantissa = (h & 0x03FFu);
                uint32_t bits;
                if(exponent == 0)
                {
                    // Denormal or zero.
                    if(mantissa == 0)
                    {
                        bits = sign;
                    }
                    else
                    {
                        // Normalize.
                        exponent = 1;
                        while((mantissa & 0x0400u) == 0) { mantissa <<= 1; --exponent; }
                        mantissa &= ~0x0400u;
                        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
                    }
                }
                else if(exponent == 31)
                {
                    // Inf or NaN.
                    bits = sign | 0x7F800000u | (mantissa << 13);
                }
                else
                {
                    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
                }
                float f;
                std::memcpy(&f, &bits, sizeof(f));
                result[i] = f;
            }
        }
        else
        {
            // f32 output: memcpy directly.
            const std::size_t n_floats = static_cast<std::size_t>(out_size) / sizeof(float);
            result.resize(n_floats);
            std::memcpy(result.data(), mapped, static_cast<std::size_t>(out_size));
        }

        D3D12_RANGE no_write{0, 0};
        readback_buf->Unmap(0, &no_write);
    }

    return result;
}

std::vector<char> Dx12Runner::execute_raw(const KernelDescriptor&               kd,
                                          const std::vector<std::vector<char>>& input_data)
{
    // execute() converts fp16→fp32 which loses bit-exactness needed for chaining.
    // We run the same dispatch logic but map the readback as raw bytes.
    // Reuse execute() infrastructure by calling execute() and reinterpreting — but
    // that would lose fp16 precision.  Instead, duplicate the core dispatch and
    // return the raw bytes before any element conversion.

    ensure_initialized();
    const uint64_t out_size = kd.arg_sizes[kd.output_arg_index];

    // 1. Build PSO.
    auto pso = create_pso(kd);

    // 2. Allocate GPU buffers.
    std::vector<ComPtr<ID3D12Resource>> upload_bufs(input_data.size());
    std::vector<ComPtr<ID3D12Resource>> gpu_input_bufs(input_data.size());
    for(std::size_t i = 0; i < input_data.size(); ++i)
    {
        const uint64_t sz = input_data[i].size() > 0 ? input_data[i].size() : 4096u;
        upload_bufs[i]   = create_upload_buffer(sz, input_data[i].data());
        gpu_input_bufs[i] = create_gpu_buffer(sz);
    }
    auto gpu_output_buf = create_gpu_buffer(out_size);

    D3D12_HEAP_PROPERTIES readback_hp{};
    readback_hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC readback_rd{};
    readback_rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    readback_rd.Width            = out_size;
    readback_rd.Height           = 1;
    readback_rd.DepthOrArraySize = 1;
    readback_rd.MipLevels        = 1;
    readback_rd.SampleDesc.Count = 1;
    readback_rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback_buf;
    DX12_CHECK(device_->CreateCommittedResource(
        &readback_hp, D3D12_HEAP_FLAG_NONE, &readback_rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback_buf)));

    // 3. Record commands.
    DX12_CHECK(allocator_->Reset());
    DX12_CHECK(cmd_list_->Reset(allocator_.Get(), nullptr));

    for(std::size_t i = 0; i < input_data.size(); ++i)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource   = gpu_input_bufs[i].Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd_list_->ResourceBarrier(1, &barrier);
        cmd_list_->CopyResource(gpu_input_bufs[i].Get(), upload_bufs[i].Get());
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        cmd_list_->ResourceBarrier(1, &barrier);
    }

    cmd_list_->SetPipelineState(pso.Get());

    std::unordered_map<std::size_t, const void*> scalar_val_ptrs;
    for(const auto& ss : kd.scalar_slots)
        scalar_val_ptrs[ss.arg_index] = ss.value_bytes.data();

    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> gpu_vas;
    gpu_vas.reserve(input_data.size() + 1);
    const std::size_t buffer_args = kd.arg_sizes.size();
    std::size_t n_args = buffer_args;
    for(const auto& slot : kd.scalar_slots)
        n_args = (std::max)(n_args, slot.arg_index + 1);
    std::vector<const void*> arg_ptrs(n_args, nullptr);

    std::size_t input_buf_idx = 0;
    for(std::size_t i = 0; i < n_args; ++i)
    {
        auto it = scalar_val_ptrs.find(i);
        if(it != scalar_val_ptrs.end())
        {
            arg_ptrs[i] = it->second;
        }
        else if(i == kd.output_arg_index)
        {
            gpu_vas.push_back(gpu_output_buf->GetGPUVirtualAddress());
            arg_ptrs[i] = &gpu_vas.back();
        }
        else
        {
            gpu_vas.push_back(gpu_input_bufs[input_buf_idx]->GetGPUVirtualAddress());
            arg_ptrs[i] = &gpu_vas.back();
            ++input_buf_idx;
        }
    }

    amd_ext_device_->SetKernelArguments(
        cmd_list_.Get(), 0u, static_cast<uint32_t>(n_args), arg_ptrs.data());

    cmd_list_->Dispatch(kd.dispatch_x, kd.dispatch_y, kd.dispatch_z);

    D3D12_RESOURCE_BARRIER uav_barrier{};
    uav_barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_barrier.UAV.pResource = gpu_output_buf.Get();
    cmd_list_->ResourceBarrier(1, &uav_barrier);

    D3D12_RESOURCE_BARRIER copy_barrier{};
    copy_barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copy_barrier.Transition.pResource   = gpu_output_buf.Get();
    copy_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    copy_barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
    copy_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list_->ResourceBarrier(1, &copy_barrier);
    cmd_list_->CopyResource(readback_buf.Get(), gpu_output_buf.Get());

    // 4. Submit and wait.
    flush();

    // 5. Map readback and return raw bytes.
    std::vector<char> raw(static_cast<std::size_t>(out_size));
    {
        D3D12_RANGE read_range{0, static_cast<SIZE_T>(out_size)};
        void* mapped = nullptr;
        DX12_CHECK(readback_buf->Map(0, &read_range, &mapped));
        std::memcpy(raw.data(), mapped, raw.size());
        D3D12_RANGE no_write{0, 0};
        readback_buf->Unmap(0, &no_write);
    }
    return raw;
}

} // namespace dx12
} // namespace hip_ep
