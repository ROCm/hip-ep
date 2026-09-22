/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "dx12_runner.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <immintrin.h> // _mm_pause() for spin-wait fence polling
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

static void throw_if_failed(HRESULT hr, const char *ctx) {
  if (FAILED(hr)) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s failed with HRESULT 0x%08X", ctx,
             static_cast<unsigned>(hr));
    throw std::runtime_error(buf);
  }
}

#define DX12_CHECK(expr) throw_if_failed((expr), #expr)

// ---------------------------------------------------------------------------
// PCI DeviceId → GFX arch mapping for AMD GPUs on Windows.
// The DeviceId values come from DXGI_ADAPTER_DESC1 (same as PCI subsystem ID).
// These are the IDs we actually see in DX12 adapter enumeration.
// ---------------------------------------------------------------------------
static const char *pci_device_id_to_gfx_arch(uint32_t device_id) {
  // RDNA 3 / GFX11 — Navi31 (gfx1100): RX 7900 XTX/XT/GRE, W7900, Radeon Pro
  // variants PCI IDs sourced from amdgpu kernel driver
  // (drm/amd/include/asic_reg/gc/...)
  if (device_id == 0x744c || // Navi31 XTX (RX 7900 XTX)
      device_id == 0x744e || // Navi31 XT  (RX 7900 XT)
      device_id == 0x7448 || // Navi31 (W7900 workstation)
      device_id == 0x745e || // Navi31 (W7900S)
      device_id == 0x7480 || // Navi31 (RX 7900 GRE)
      device_id == 0x7483 || // Navi31 (RX 7900M)
      device_id == 0x7499)   // Navi31 (RX 7800M/Pro)
    return "gfx1100";
  if (device_id == 0x7470 || device_id == 0x7471 || device_id == 0x747e)
    return "gfx1101"; // Navi32 (RX 7800 XT, RX 7700)
  if (device_id == 0x7460 || device_id == 0x7461 || device_id == 0x7422 ||
      device_id == 0x7423 || device_id == 0x7424 || device_id == 0x7425)
    return "gfx1102"; // Navi33 (RX 7600/7700S)
  // RDNA 3.5 / GFX115x — APU (iGPU) devices; DX12 DeviceId from DXGI
  // gfx1150: Strix Halo (RX 890M / Radeon 890M) — high-end APU GPU
  if (device_id == 0x150e || // Strix Halo (Phoenix/Hawk Point family, high-end)
      device_id == 0x1586 || // Strix Halo variant
      device_id == 0x1587 || // Strix Halo variant
      device_id == 0x15bf || // Krackan Point
      device_id == 0x15c8)   // Krackan Point variant
    return "gfx1150";
  // gfx1151: Strix Point (Ryzen AI 300 series — Radeon 880M/860M)
  if (device_id == 0x150c || // Strix Point (Radeon 880M)
      device_id == 0x1585 || // Strix Point variant
      device_id == 0x1583 || // Strix Point (Radeon 860M)
      device_id == 0x1506 || // Strix Point
      device_id == 0x150f)   // Strix Point
    return "gfx1151";
  // RDNA 4 / GFX12
  if (device_id == 0x9060 || device_id == 0x9061 || device_id == 0x9062 ||
      device_id == 0x9063)
    return "gfx1200"; // Navi48 (RX 9070 XT/9070)
  if (device_id == 0x9070 || device_id == 0x9071 || device_id == 0x9072 ||
      device_id == 0x9073)
    return "gfx1201"; // Navi44 (RX 9060 XT)
  return nullptr; // unknown — caller should fall back to --arch flag or HIP
                  // detection
}

// ---------------------------------------------------------------------------
// Lazy initialisation
// ---------------------------------------------------------------------------

void Dx12Runner::ensure_initialized() {
  if (initialized_)
    return;

  // Enable D3D12 debug layer for validation
  {
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
      debug->EnableDebugLayer();
  }

  // 1. Enumerate AMD adapters and select based on adapter_selector_.
  //
  // adapter_selector_ modes:
  //   ""        — auto: first AMD adapter where D3D12CreateDevice succeeds
  //   "0","1"…  — 0-based AMD-only index (counts only VendorId==0x1002)
  //   "0x…"     — 64-bit DXGI LUID hex string (e.g. "0x00000000000049E2")
  //   other     — case-insensitive substring of adapter Description
  //
  // Capability detection is done by trying D3D12CreateDevice (feature
  // level 12.0). We no longer gate on DeviceId ranges -- those are arbitrary
  // PCI IDs that vary across product families and APU generations. Any AMD
  // adapter that successfully creates a D3D12 device is a candidate;
  // unsupported ops will fail at CreateComputePipelineCrossCompile time with a
  // meaningful error.
  //
  // The GFX IP version (gfx1100+) is detected by the caller via HIP's
  // gcnArchName and passed as --gfx-target; amdxcgc.dll uses that to select the
  // right binaries. Only gfx1100+ (RDNA3 and newer) is supported.
  // pci_device_id_to_gfx_arch() returns non-null only for known gfx1100+ device
  // IDs. Any adapter whose DeviceId maps to nullptr is either pre-gfx1100
  // (unsupported) or unknown new hardware. Unknown IDs are allowed tentatively
  // so new GPUs work with --arch; known old IDs (RDNA1/2 ranges: 0x7310-0x73ff,
  // Vega: 0x6860-0x687f, Polaris: 0x67c0-0x67ff) are explicitly rejected with a
  // clear message.
  auto is_supported_amd_adapter = [](UINT device_id) -> bool {
    const char *arch = pci_device_id_to_gfx_arch(device_id);
    if (arch != nullptr)
      return true; // known gfx1100+ device — supported
    // Reject known pre-gfx1100 device ID ranges (RDNA1/2, Vega, Polaris, etc.)
    // These ranges will never be supported regardless of driver version.
    if (device_id >= 0x7310u && device_id <= 0x73ffu)
      return false; // RDNA2 (Navi21/22/23/24)
    if (device_id >= 0x7340u && device_id <= 0x736fu)
      return false; // RDNA1 (Navi10/14)
    if (device_id >= 0x6860u && device_id <= 0x687fu)
      return false; // Vega10/12/20
    if (device_id >= 0x6940u && device_id <= 0x695fu)
      return false; // Vega20
    if (device_id >= 0x67c0u && device_id <= 0x67ffu)
      return false; // Polaris10/11
    if (device_id >= 0x6900u && device_id <= 0x6930u)
      return false; // Fiji
    // Unknown ID — may be new hardware; allow and require --arch flag for
    // kernel lookup.
    return true;
    return true;
  };

  // Parse selector mode.
  enum class SelMode { Auto, Index, Luid, Name } sel_mode = SelMode::Auto;
  UINT sel_index = 0;
  UINT64 sel_luid = 0;
  std::string sel_name;

  if (!adapter_selector_.empty()) {
    const auto &s = adapter_selector_;
    if ((s[0] == '0' && s.size() > 2 && (s[1] == 'x' || s[1] == 'X'))) {
      sel_mode = SelMode::Luid;
      sel_luid = std::stoull(s, nullptr, 16);
    } else if (std::all_of(s.begin(), s.end(), ::isdigit)) {
      sel_mode = SelMode::Index;
      sel_index = static_cast<UINT>(std::stoul(s));
    } else {
      sel_mode = SelMode::Name;
      sel_name = s;
      std::transform(sel_name.begin(), sel_name.end(), sel_name.begin(),
                     ::tolower);
    }
  }

  ComPtr<IDXGIFactory4> factory;
  DX12_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));

  // Auto-select: prefer discrete AMD GPU (DedicatedVideoMemory > 0) over
  // APU/iGPU. APUs share system memory so DedicatedVideoMemory == 0; discrete
  // GPUs have VRAM. Without this, DXGI adapter[0] is often the APU on multi-GPU
  // systems, which lacks pipelineCrossCompileElfHsa support and causes all DX12
  // compute tests to fail. If a discrete AMD adapter is found, switch Auto →
  // LUID targeting it explicitly.
  if (sel_mode == SelMode::Auto) {
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND;
         ++i) {
      DXGI_ADAPTER_DESC1 d{};
      a->GetDesc1(&d);
      a.Reset();
      if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
        continue;
      if (d.VendorId != 0x1002)
        continue;
      if (!is_supported_amd_adapter(d.DeviceId))
        continue;
      if (d.DedicatedVideoMemory > 0) {
        sel_mode = SelMode::Luid;
        sel_luid = (static_cast<UINT64>(d.AdapterLuid.HighPart) << 32) |
                   static_cast<UINT32>(d.AdapterLuid.LowPart);
        break;
      }
    }
  }

  ComPtr<IDXGIAdapter1> adapter;
  UINT amd_index = 0; // 0-based index across AMD adapters only
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    char desc_str[256] = {};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, desc_str,
                        sizeof(desc_str), nullptr, nullptr);

    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      adapter.Reset();
      continue;
    }
    if (desc.VendorId == 0x1414) {
      adapter.Reset();
      continue;
    }
    if (desc.VendorId != 0x1002) {
      adapter.Reset();
      continue;
    }

    const UINT64 luid_val =
        (static_cast<UINT64>(desc.AdapterLuid.HighPart) << 32) |
        static_cast<UINT32>(desc.AdapterLuid.LowPart);
    const bool is_supported = is_supported_amd_adapter(desc.DeviceId);

    const char *arch_str = pci_device_id_to_gfx_arch(desc.DeviceId);
    std::cerr << "[dx12_runner] AMD adapter[" << amd_index << "]: " << desc_str
              << " DeviceId=0x" << std::hex << desc.DeviceId << " LUID=0x"
              << std::setw(16) << std::setfill('0') << luid_val << std::dec;
    if (!is_supported)
      std::cerr << " [UNSUPPORTED: pre-gfx1100 GPU, requires RDNA3+]";
    else if (arch_str)
      std::cerr << " [" << arch_str << "]";
    else
      std::cerr << " [gfx unknown — use --arch]";
    std::cerr << "\n";
    std::cerr.flush();

    // Check whether this adapter matches the selector.
    bool matches = false;
    switch (sel_mode) {
    case SelMode::Auto:
      matches = is_supported;
      break;
    case SelMode::Index:
      matches = (amd_index == sel_index);
      break;
    case SelMode::Luid:
      matches = (luid_val == sel_luid);
      break;
    case SelMode::Name: {
      std::string desc_lower(desc_str);
      std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(),
                     ::tolower);
      matches = (desc_lower.find(sel_name) != std::string::npos);
      break;
    }
    }

    ++amd_index;

    if (!matches) {
      adapter.Reset();
      continue;
    }

    if (!is_supported) {
      adapter.Reset();
      continue;
    }

    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                    IID_PPV_ARGS(&device_)))) {
      // Capture adapter identity so callers can build a dxcg_device_info.
      selected_adapter_luid_ = luid_val;
      selected_pci_device_id_ = static_cast<uint32_t>(desc.DeviceId);
      selected_pci_vendor_id_ = static_cast<uint32_t>(desc.VendorId);
      selected_device_name_ = desc_str;
      // Map PCI DeviceId → GFX arch for kernel lookup (driver passes this to
      // dxcg_compile).
      if (const char *arch = pci_device_id_to_gfx_arch(selected_pci_device_id_))
        selected_gfx_arch_ = arch;
      else if (assume_gfx1151_)
        selected_gfx_arch_ = "gfx1151";
      std::cerr << "[dx12_runner] selected: " << desc_str << " (AMD["
                << (amd_index - 1) << "]"
                << " DeviceId=0x" << std::hex << desc.DeviceId << " LUID=0x"
                << std::setw(16) << std::setfill('0') << luid_val << std::dec
                << ")\n";
      std::cerr.flush();
      break;
    }
    adapter.Reset();
  }
  if (!device_) {
    std::string hint;
    if (!adapter_selector_.empty())
      hint = " Selector: \"" + adapter_selector_ + "\".";
    throw std::runtime_error(
        "Dx12Runner: no suitable AMD D3D12 adapter found." + hint +
        " Use --dx12-adapter with an index (\"0\"), LUID (\"0x...\"), or name "
        "substring.");
  }

  // 2. Compute command queue with TDR disabled.
  // D3D12_COMMAND_LIST_TYPE_COMPUTE: GPU compute-only queue (no rasterization
  // overhead). D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT: suppresses the
  // 2-second TDR watchdog for work submitted to this queue. Long-running
  // compute kernels (matrix ops, GEMM) will run to completion instead of
  // triggering a GPU reset and DEVICE_REMOVED.
  D3D12_COMMAND_QUEUE_DESC qd{};
  qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  qd.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
  DX12_CHECK(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)));

  // 3. Command allocator + list (must match queue type).
  DX12_CHECK(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(&allocator_)));
  DX12_CHECK(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        allocator_.Get(), nullptr,
                                        IID_PPV_ARGS(&cmd_list_)));
  // Close immediately — we reopen before each dispatch.
  DX12_CHECK(cmd_list_->Close());

  // 4. Fence for CPU/GPU synchronisation.
  DX12_CHECK(
      device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
  fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!fence_event_)
    throw std::runtime_error("Dx12Runner: CreateEvent failed");

  // 5. Load AMD extension and obtain IAmdExtD3DDevice.
  load_amd_ext();

  initialized_ = true;
}

void Dx12Runner::load_amd_ext() {
  // The AMD cross-compile extension is exported from the user-mode driver DLL.
  // On Windows 64-bit this is typically amdxc64.dll on the driver search path.
  const wchar_t *candidates[] = {L"amdxc64.dll", L"amdvlk64.dll"};

  for (auto *dll : candidates) {
    amd_ext_module_ = LoadLibraryW(dll);
    if (amd_ext_module_)
      break;
  }
  if (!amd_ext_module_)
    throw std::runtime_error("Dx12Runner: could not load AMD extension DLL "
                             "(amdxc64.dll / amdvlk64.dll). "
                             "Ensure AMD GPU drivers are installed.");

  auto *pfn = reinterpret_cast<PFN_AmdExtD3DCreateInterface>(
      GetProcAddress(amd_ext_module_, "AmdExtD3DCreateInterface"));
  if (!pfn)
    throw std::runtime_error(
        "Dx12Runner: AmdExtD3DCreateInterface not found in AMD driver DLL");

  // Obtain the factory using the D3D12 device as the outer object.
  // NOTE: Use __cdecl calling convention for pfn (defined in amdcc_api.hpp).
  DX12_CHECK(pfn(
      device_.Get(), __uuidof(IAmdExtD3DFactory),
      reinterpret_cast<void **>(amd_ext_factory_.ReleaseAndGetAddressOf())));

  // Create the per-device AMD extension interface (v7 adds
  // CreateComputePipelineCrossCompile).
  DX12_CHECK(amd_ext_factory_->CreateInterface(
      device_.Get(), __uuidof(IAmdExtD3DDevice7),
      reinterpret_cast<void **>(amd_ext_device_.ReleaseAndGetAddressOf())));

  DX12_CHECK(amd_ext_device_->QueryInterface(
      __uuidof(IAmdExtD3DDevice5),
      reinterpret_cast<void **>(amd_ext_device5_.ReleaseAndGetAddressOf())));

  // Query v10 for DispatchPalElf (direct PAL ELF dispatch replicating
  // metacommand path).
  HRESULT hr10 = amd_ext_device_->QueryInterface(
      __uuidof(IAmdExtD3DDevice10),
      reinterpret_cast<void **>(amd_ext_device10_.ReleaseAndGetAddressOf()));
  if (FAILED(hr10) || !amd_ext_device10_)
    std::cerr << "[dx12_runner] WARNING: IAmdExtD3DDevice10 (DispatchPalElf) "
                 "not available (hr=0x"
              << std::hex << (unsigned)hr10 << std::dec
              << ") — PAL ELF kernels will fail. Update DXCP driver to a build "
                 "that supports v10.\n";
  else if (verbose_)
    std::cout << "[dx12_runner] IAmdExtD3DDevice10 acquired — DispatchPalElf "
                 "available.\n";

  // Query supported cross-compile features.
  AmdExtD3DCheckFeatureSupportFlags flags{};
  if (SUCCEEDED(amd_ext_device_->CheckExtFeatureSupport(
          0 /*AmdExtD3DCheckFeatureSupportType::Flags*/, &flags,
          sizeof(flags)))) {
    elf_hsa_supported_ = flags.pipelineCrossCompileElfHsa;
    std::cerr << "[dx12_runner] pipelineCrossCompileElfHsa="
              << flags.pipelineCrossCompileElfHsa
              << " pipelinePalElf=" << flags.pipelinePalElf
              << " pipelineHsaElf=" << flags.pipelineHsaElf << "\n";
  }
  if (!elf_hsa_supported_)
    std::cerr
        << "[dx12_runner] WARNING: pipelineCrossCompileElfHsa not supported.\n";

  // Pre-create PAL ELF fallback root signature (2 root constants for CbData
  // VA). Used when IAmdExtD3DDevice10::DispatchPalElf is unavailable and we
  // dispatch via standard D3D12 root constants instead.
  if (!pal_root_sig_)
    pal_root_sig_ = create_pal_root_signature();
}

// ---------------------------------------------------------------------------
// PSO creation via AMD Cross-Compile API
// ---------------------------------------------------------------------------

// Detect PAL ELF: EI_OSABI byte [7] == 0x41 (AMD_AMDGPU_ELF_OSABI_AMDPAL).
// PAL ELFs use AmdShaderType::ElfPal (5) and route through PAL
// CodeObjectUploader instead of amdcc.dll, avoiding VOPD float dual-issue TDR
// crashes.
static bool is_pal_elf(const void *data, std::size_t size) {
  if (!data || size < 8u)
    return false;
  const auto *p = static_cast<const uint8_t *>(data);
  // ELF magic: 0x7F 'E' 'L' 'F'
  if (p[0] != 0x7fu || p[1] != 'E' || p[2] != 'L' || p[3] != 'F')
    return false;
  return p[7] == 0x41u; // EI_OSABI = 0x41 = AMDGPU PAL
}

// Convert an HSA ET_REL blob to PAL ELF format for ElfPal DX12 cross-compile.
// Three patches needed for PAL CodeObjectUploader compatibility:
//   1. EI_OSABI    byte[7]: 0x40 (HSA) → 0x41 (PAL)
//   2. EI_ABIVERSION byte[8]: 3 (HSA) → 0 (PAL expects 0)
//   3. NOTE section sh_flags: clear SHF_ALLOC (0x2) bit — PAL allocates GPU
//   memory
//      for every SHF_ALLOC section; an allocatable NOTE corrupts GPU memory
//      layout.
static std::vector<uint8_t> patch_to_pal_elf(const void *data,
                                             std::size_t size) {
  std::vector<uint8_t> patched(static_cast<const uint8_t *>(data),
                               static_cast<const uint8_t *>(data) + size);
  if (patched.size() < 64u)
    return patched;

  // Patch 1+2: EI_OSABI and EI_ABIVERSION
  patched[7] = 0x41u;
  patched[8] = 0x00u;

  // Patch 3: clear SHF_ALLOC from all SHT_NOTE sections.
  // ELF64 header fields (little-endian):
  //   e_shoff at byte 40, e_shnum at byte 60, e_shentsize=64
  uint64_t e_shoff;
  uint16_t e_shnum;
  std::memcpy(&e_shoff, patched.data() + 40, 8);
  std::memcpy(&e_shnum, patched.data() + 60, 2);
  for (uint16_t i = 0; i < e_shnum; ++i) {
    const std::size_t sh_base = static_cast<std::size_t>(e_shoff) + i * 64u;
    if (sh_base + 64u > patched.size())
      break;
    uint32_t sh_type;
    uint64_t sh_flags;
    std::memcpy(&sh_type, patched.data() + sh_base + 4, 4);
    std::memcpy(&sh_flags, patched.data() + sh_base + 16, 8);
    if (sh_type == 7u /*SHT_NOTE*/ && (sh_flags & 0x2u)) // SHF_ALLOC
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
ComPtr<ID3D12RootSignature> Dx12Runner::create_pal_root_signature() {
  // 2 root constants at shader register b0, space 0 — maps to SGPRs[0:1].
  D3D12_ROOT_PARAMETER root_param{};
  root_param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  root_param.Constants.ShaderRegister = 0;
  root_param.Constants.RegisterSpace = 0;
  root_param.Constants.Num32BitValues = 2;
  root_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC desc{};
  desc.NumParameters = 1;
  desc.pParameters = &root_param;
  desc.NumStaticSamplers = 0;
  desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ComPtr<ID3DBlob> sig_blob, err_blob;
  HRESULT hr = D3D12SerializeRootSignature(
      &desc, D3D_ROOT_SIGNATURE_VERSION_1_0, sig_blob.GetAddressOf(),
      err_blob.GetAddressOf());
  if (FAILED(hr)) {
    std::string msg = "D3D12SerializeRootSignature (PAL) failed";
    if (err_blob && err_blob->GetBufferPointer())
      msg += std::string(": ") +
             static_cast<const char *>(err_blob->GetBufferPointer());
    throw std::runtime_error(msg);
  }

  ComPtr<ID3D12RootSignature> root_sig;
  DX12_CHECK(device_->CreateRootSignature(0, sig_blob->GetBufferPointer(),
                                          sig_blob->GetBufferSize(),
                                          IID_PPV_ARGS(&root_sig)));
  return root_sig;
}

ComPtr<ID3D12PipelineState> Dx12Runner::create_pso(const KernelDescriptor &kd) {
  // Check pipelineCrossCompileElfHsa support — required for our HSA ELF
  // kernels. pipelinePalElf would be needed for PAL ELF blobs (not used for
  // GEMM kernels).
  if (!elf_hsa_supported_)
    throw std::runtime_error(
        "Dx12Runner::create_pso: pipelineCrossCompileElfHsa not supported by "
        "installed driver.");

  const void *blob_data = kd.hsaco_data.data();
  std::size_t blob_size = kd.hsaco_data.size();
  AmdShaderType sh_type = AmdShaderType::ElfHsa;

  // Blob type routing:
  // - HSA ET_REL (EI_OSABI=0x40, e_type=1): use CreateComputePipelineFromElf
  // (IAmdExtD3DDevice5).
  //   dxcp's CreateComputePipelineCrossCompile rejects ET_REL HSA blobs with
  //   ErrorInvalidPipelineElf (computeGraphicsProgram.cpp:604 — intentional fix
  //   to prevent GPU TDR from unrelocated ELF). CreateComputePipelineFromElf
  //   routes through PAL's AMDHSA pipeline (same as DxGML InitFromElf) and
  //   correctly handles ET_REL with AMDHSA MsgPack metadata.
  // - HSA ET_DYN (EI_OSABI=0x40, e_type=3): use
  // CreateComputePipelineCrossCompile ElfHsa.
  // - PAL ELF (EI_OSABI=0x41): use CreateComputePipelineCrossCompile ElfPal.

  // HSA ET_REL (EI_OSABI=0x40) uses CreateComputePipelineCrossCompile with
  // ElfHsa type. This routes through PAL's AMDHSA pipeline and does NOT require
  // IAmdExtD3DDevice10. The IAmdExtD3DDevice10 (DispatchPalElf) is only needed
  // for PAL ELF dispatch, not HSA. Note: CreateComputePipelineCrossCompile with
  // ElfHsa is available in IAmdExtD3DDevice7+.

  // AMDHSA ET_REL blobs explicitly tagged DXCG_ELF_HSA_REL use
  // CreateComputePipelineFromElf (IAmdExtD3DDevice5) which relocates the blob
  // via PAL's AMDHSA pipeline — identical to DxGML's InitFromElf path.
  // CreateComputePipelineCrossCompile with ElfHsa on ET_REL produces all-zeros
  // output because it doesn't perform ELF relocation. AMDHSA ET_DYN and PAL
  // blobs continue to use CreateComputePipelineCrossCompile.
  const bool is_hsa_et_rel = (kd.elf_format == ElfFormat::HsaRel);

  auto do_create_pso = [&](ComPtr<ID3D12PipelineState> &out_pso) -> HRESULT {
    // HSA ET_REL: use CreateComputePipelineFromElf for proper relocation.
    if (is_hsa_et_rel) {
      std::cerr << "[dx12_runner] CreateComputePipelineFromElf (ET_REL, AMDHSA)"
                << " blob=" << blob_size << " kernel=" << kd.entry_point
                << "\n";
      AmdExtD3DPipelineElfInfo elf_info{};
      elf_info.type = AmdExtD3DStructPipelineElf;
      elf_info.pNext = nullptr;
      elf_info.pElfBinary = blob_data;
      elf_info.elfSizeInBytes = blob_size;
      elf_info.threadsPerGroup = {kd.group_size_x, kd.group_size_y,
                                  kd.group_size_z};
      if (!amd_ext_device5_)
        return E_NOINTERFACE;
      return amd_ext_device5_->CreateComputePipelineFromElf(
          &elf_info, IID_PPV_ARGS(&out_pso));
    }

    // All other ELF blobs go through CreateComputePipelineCrossCompile.
    // Determine shader type from elf_format field.
    AmdShaderType sh = AmdShaderType::ElfHsa;
    switch (kd.elf_format) {
    case ElfFormat::PalRel:
    case ElfFormat::PalDyn:
      sh = AmdShaderType::ElfPal;
      break;
    case ElfFormat::HsaRel:
    case ElfFormat::HsaDyn:
      sh = AmdShaderType::ElfHsa;
      break;
    default:
      if (is_pal_elf(blob_data, blob_size))
        sh = AmdShaderType::ElfPal;
      break;
    }
    AmdExtD3DPipelineCrossCompileInfo cc{};
    cc.type = AmdExtD3DStructPipelineCrossCompile;
    cc.pNext = nullptr;
    cc.pBlob = blob_data;
    cc.blobSizeInBytes = blob_size;
    cc.shType = sh;
    cc.pOptions = nullptr;
    cc.optionSizeInBytes = 0;
    cc.pKernelName = kd.entry_point.empty() ? nullptr : kd.entry_point.c_str();
    cc.threadsPerGroup = {kd.group_size_x, kd.group_size_y, kd.group_size_z};
    const char *sh_name = (sh == AmdShaderType::ElfHsa)
                              ? "ElfHsa(AMDHSA,no-overlay)"
                          : (sh == AmdShaderType::ElfPal) ? "ElfPal(PAL)"
                                                          : "Unknown";
    std::cerr << "[dx12_runner] CrossCompile shType=" << sh_name
              << " blob=" << blob_size << " kernel=" << kd.entry_point << "\n";
    // DIAGNOSTIC: dump ELF binary for offline inspection (readelf -h/-S/-s/-r).
    if (blob_data && blob_size >= 4) {
      const auto *hdr = static_cast<const unsigned char *>(blob_data);
      if (hdr[0] == 0x7f && hdr[1] == 'E' && hdr[2] == 'L' && hdr[3] == 'F') {
        const unsigned char osabi = (blob_size >= 8) ? hdr[7] : 0;
        uint16_t e_type = 0, e_shnum = 0;
        if (blob_size >= 18)
          std::memcpy(&e_type, hdr + 16, 2);
        if (blob_size >= 62)
          std::memcpy(&e_shnum, hdr + 60, 2);
        std::cerr << "[dx12_runner] ELF header: EI_OSABI=0x" << std::hex
                  << (unsigned)osabi << " e_type=" << std::dec << e_type
                  << " e_shnum=" << e_shnum << "\n";
        std::string dump_path = "C:/Develop/amd-mlcompiler/dump/kernel_hsa_" +
                                kd.entry_point + ".elf";
        if (auto f = std::fopen(dump_path.c_str(), "wb")) {
          std::fwrite(blob_data, 1, blob_size, f);
          std::fclose(f);
          std::cerr << "[dx12_runner] ELF dumped: " << dump_path << "\n";
        }
      }
    }
    return amd_ext_device_->CreateComputePipelineCrossCompile(
        &cc, IID_PPV_ARGS(&out_pso));
  };

  // PSO cache lookup: same kernel name in the same session → reuse existing
  // PSO. This prevents GPU PSO pool exhaustion when running many tests
  // sequentially.
  const std::string &cache_key = kd.entry_point;
  if (!cache_key.empty()) {
    auto it = pso_cache_.find(cache_key);
    if (it != pso_cache_.end())
      return it->second;
  }

  ComPtr<ID3D12PipelineState> pso;
  HRESULT hr = do_create_pso(pso);

  // No ElfHsa fallback for PAL ELF — only PAL_ELF path for Winograd MLSS
  // kernels.

  // On DXGI_ERROR_DEVICE_REMOVED (0x887A0005): OS WDDM TDR recovery from a
  // prior run may still be in progress. Log the removal reason, then retry with
  // increasing delays. DISABLE_GPU_TIMEOUT on the queue prevents new TDRs; this
  // handles pre-existing ones.
  if (hr == DXGI_ERROR_DEVICE_REMOVED) {
    // Log the specific removal reason to distinguish hung vs driver crash vs
    // reset.
    if (device_) {
      HRESULT reason = device_->GetDeviceRemovedReason();
      const char *reason_str =
          reason == DXGI_ERROR_DEVICE_HUNG             ? "DEVICE_HUNG (timeout)"
          : reason == DXGI_ERROR_DEVICE_RESET          ? "DEVICE_RESET"
          : reason == DXGI_ERROR_DRIVER_INTERNAL_ERROR ? "DRIVER_INTERNAL_ERROR"
          : reason == DXGI_ERROR_INVALID_CALL          ? "INVALID_CALL"
                                                       : "DEVICE_REMOVED";
      std::cerr << "[dx12_runner] GetDeviceRemovedReason: 0x" << std::hex
                << static_cast<unsigned>(reason) << std::dec << " ("
                << reason_str << ")\n";
    }
    std::cerr << "[dx12_runner] Waiting for OS WDDM GPU TDR recovery...\n";

    for (int attempt = 1; attempt <= 5 && FAILED(hr); ++attempt) {
      initialized_ = false;
      device_.Reset();
      queue_.Reset();
      cmd_list_.Reset();
      allocator_.Reset();
      amd_ext_device_.Reset();
      amd_ext_device10_.Reset();
      amd_ext_factory_.Reset();
      const DWORD wait_ms =
          static_cast<DWORD>(attempt * 3000); // 3, 6, 9, 12, 15 s
      std::cerr << "[dx12_runner] Retry " << attempt << "/5 (waiting "
                << wait_ms / 1000 << "s)...\n";
      ::Sleep(wait_ms);
      try {
        ensure_initialized();
      } catch (const std::exception &e) {
        std::cerr << "[dx12_runner] device re-init failed: " << e.what()
                  << "\n";
        continue;
      }
      pso.Reset();
      hr = do_create_pso(pso);
      if (SUCCEEDED(hr))
        std::cerr << "[dx12_runner] GPU recovered after attempt " << attempt
                  << "\n";
    }
  }
  if (FAILED(hr)) {
    std::ostringstream oss;
    oss << "create_pso failed with HRESULT 0x" << std::hex << std::uppercase
        << hr;
    throw std::runtime_error(oss.str());
  }
  // Cache the PSO so repeated calls with the same kernel name skip GPU
  // compilation.
  if (!cache_key.empty() && pso)
    pso_cache_[cache_key] = pso;
  return pso;
}

// ---------------------------------------------------------------------------
// Resource helpers
// ---------------------------------------------------------------------------

ComPtr<ID3D12Resource>
Dx12Runner::create_upload_buffer(uint64_t size_bytes,
                                 const void *initial_data) {
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_UPLOAD;

  D3D12_RESOURCE_DESC rd{};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  rd.Width = size_bytes;
  rd.Height = 1;
  rd.DepthOrArraySize = 1;
  rd.MipLevels = 1;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  ComPtr<ID3D12Resource> buf;
  DX12_CHECK(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                              D3D12_RESOURCE_STATE_GENERIC_READ,
                                              nullptr, IID_PPV_ARGS(&buf)));

  if (initial_data && size_bytes > 0) {
    void *mapped = nullptr;
    DX12_CHECK(buf->Map(0, nullptr, &mapped));
    std::memcpy(mapped, initial_data, static_cast<std::size_t>(size_bytes));
    buf->Unmap(0, nullptr);
  }
  return buf;
}

ComPtr<ID3D12Resource> Dx12Runner::create_gpu_buffer(uint64_t size_bytes) {
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;

  D3D12_RESOURCE_DESC rd{};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  rd.Width = size_bytes;
  rd.Height = 1;
  rd.DepthOrArraySize = 1;
  rd.MipLevels = 1;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  ComPtr<ID3D12Resource> buf;
  DX12_CHECK(device_->CreateCommittedResource(
      &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      nullptr, IID_PPV_ARGS(&buf)));
  return buf;
}

// ---------------------------------------------------------------------------
// GPU flush — submit and wait for completion.
// ---------------------------------------------------------------------------

// See the declaration in dx12_runner.hpp for what a pointer cell is and why the
// upload happens before the dispatch list is recorded.
ComPtr<ID3D12Resource> Dx12Runner::stage_pointer_cells(
    const std::vector<D3D12_GPU_VIRTUAL_ADDRESS> &data_vas,
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> &cell_vas) {
  cell_vas.clear();
  if (data_vas.empty())
    return {};

  constexpr uint64_t kEntrySize = sizeof(D3D12_GPU_VIRTUAL_ADDRESS); // 8 bytes
  const uint64_t kBufSize = kEntrySize * data_vas.size();

  std::vector<uint8_t> init_data(static_cast<std::size_t>(kBufSize), 0u);
  for (std::size_t i = 0; i < data_vas.size(); ++i)
    std::memcpy(init_data.data() + i * kEntrySize, &data_vas[i],
                sizeof(data_vas[i]));

  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC rd{};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  rd.Width = kBufSize;
  rd.Height = 1;
  rd.DepthOrArraySize = 1;
  rd.MipLevels = 1;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ComPtr<ID3D12Resource> buf;
  DX12_CHECK(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, IID_PPV_ARGS(&buf)));

  DX12_CHECK(allocator_->Reset());
  DX12_CHECK(cmd_list_->Reset(allocator_.Get(), nullptr));
  auto up = create_upload_buffer(
      kBufSize, reinterpret_cast<const char *>(init_data.data()));
  cmd_list_->CopyBufferRegion(buf.Get(), 0, up.Get(), 0, kBufSize);
  flush(); // synchronous: up may be released after this returns

  const D3D12_GPU_VIRTUAL_ADDRESS base_va = buf->GetGPUVirtualAddress();
  cell_vas.resize(data_vas.size());
  for (std::size_t i = 0; i < data_vas.size(); ++i)
    cell_vas[i] = base_va + i * kEntrySize;

  const char *tenv = std::getenv("AMDCGC_TRACE");
  if (tenv && std::atoi(tenv) >= 2) {
    for (std::size_t i = 0; i < data_vas.size(); ++i)
      std::cerr << "  [ptr_cell] " << i << " cell_va=0x" << std::hex
                << cell_vas[i] << " data_va=0x" << data_vas[i] << std::dec
                << "\n";
    std::cerr.flush();
  }
  return buf;
}

void Dx12Runner::flush() {
  DX12_CHECK(cmd_list_->Close());
  ID3D12CommandList *lists[] = {cmd_list_.Get()};
  queue_->ExecuteCommandLists(1, lists);

  const uint64_t signal_val = ++fence_value_;
  DX12_CHECK(queue_->Signal(fence_.Get(), signal_val));

  // Spin-wait: avoids ~30-36µs OS sleep from WaitForSingleObject.
  // GPU kernels completing in <1ms benefit; long kernels fall back to event.
  constexpr int kMaxSpinIter = 1000000; // ~1ms at 1ns/pause
  bool completed = false;
  for (int spin = 0; spin < kMaxSpinIter; ++spin) {
    if (fence_->GetCompletedValue() >= signal_val) {
      completed = true;
      break;
    }
    _mm_pause();
  }
  if (!completed) {
    DX12_CHECK(fence_->SetEventOnCompletion(signal_val, fence_event_));
    WaitForSingleObject(fence_event_, INFINITE);
  }
}

// ---------------------------------------------------------------------------
// flush_n — submit the same command list N times, wait once at the end.
// Eliminates per-iteration CPU wakeup overhead (~36µs per fence wait).
// Used for benchmark mode to measure pure GPU throughput.
// The command list must already be recorded and closed by the caller (flush()
// was NOT called — the caller prepares the dispatch but does NOT close/submit).
// ---------------------------------------------------------------------------
void Dx12Runner::flush_n(int n_iter) {
  if (n_iter <= 0)
    return;

  // Close the command list once.
  // D3D12 SPEC: a closed command list can be submitted multiple times via
  // ExecuteCommandLists without Reset — the GPU executes it each time.
  // This is the key for eliminating per-iteration overhead: record once, submit
  // N.
  DX12_CHECK(cmd_list_->Close());
  ID3D12CommandList *lists[] = {cmd_list_.Get()};

  // Submit N times — NO Reset between submits. GPU processes them sequentially
  // since we use a single command queue (no parallelism needed here).
  // Each submit dispatches the same kernel with the same buffers (benchmark
  // mode).
  for (int i = 0; i < n_iter; ++i)
    queue_->ExecuteCommandLists(1, lists);

  // Single fence signal + spin-wait after ALL N submits complete.
  // Spin-wait replaces WaitForSingleObject (~30-36µs OS sleep) with a
  // CPU-polling loop that checks the fence value every ~1µs using _mm_pause().
  // This eliminates the kernel-mode scheduling overhead for fast kernels.
  const uint64_t signal_val = ++fence_value_;
  DX12_CHECK(queue_->Signal(fence_.Get(), signal_val));
  // Spin up to ~100ms (100M iterations at 1ns/pause), then fall back to event
  // wait.
  constexpr int kMaxSpinIter = 100000;
  bool completed = false;
  for (int spin = 0; spin < kMaxSpinIter; ++spin) {
    if (fence_->GetCompletedValue() >= signal_val) {
      completed = true;
      break;
    }
    _mm_pause();
  }
  if (!completed) {
    // Fallback to event-based wait for long-running kernels (>100ms)
    DX12_CHECK(fence_->SetEventOnCompletion(signal_val, fence_event_));
    WaitForSingleObject(fence_event_, INFINITE);
  }
}

std::vector<float> bytes_to_f32(const std::vector<char> &bytes,
                                uint32_t output_element_bytes) {
  std::vector<float> result;
  if (output_element_bytes == 2) {
    // f16 output: reinterpret bytes as uint16_t, convert to float32.
    const std::size_t n_elems = bytes.size() / 2;
    result.resize(n_elems);
    const uint16_t *src = reinterpret_cast<const uint16_t *>(bytes.data());
    for (std::size_t i = 0; i < n_elems; ++i) {
      // IEEE 754 f16 → f32 bit-manipulation.
      uint32_t h = src[i];
      uint32_t sign = (h & 0x8000u) << 16;
      uint32_t exponent = (h & 0x7C00u) >> 10;
      uint32_t mantissa = (h & 0x03FFu);
      uint32_t bits;
      if (exponent == 0) {
        if (mantissa == 0) {
          bits = sign; // zero
        } else {
          // Denormal: normalize.
          exponent = 1;
          while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            --exponent;
          }
          mantissa &= ~0x0400u;
          bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
      } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13); // Inf / NaN
      } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
      }
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      result[i] = f;
    }
  } else {
    // f32 output: copy through.
    const std::size_t n_floats = bytes.size() / sizeof(float);
    result.resize(n_floats);
    std::memcpy(result.data(), bytes.data(), n_floats * sizeof(float));
  }
  return result;
}

// ---------------------------------------------------------------------------
// execute() — full dispatch cycle
// ---------------------------------------------------------------------------

std::vector<float>
Dx12Runner::execute(const KernelDescriptor &kd,
                    const std::vector<std::vector<char>> &input_data) {
  ensure_initialized();

  // Validate: arg_sizes = GPU-buffer inputs + 1 output + N scalars.
  // Skip for PAL ELF dispatch — CbData replaces the arg_sizes mechanism.
  const std::size_t n_args = kd.arg_sizes.size();
  const std::size_t n_scalar = kd.scalar_slots.size();
  const bool is_pal_elf = (kd.elf_format == ElfFormat::PalRel ||
                           kd.elf_format == ElfFormat::PalDyn);
  // MLSS Winograd HSA_REL kernels use PAL CbData calling convention despite HSA
  // ELF wrapper. When cbdata_bytes is populated for a HsaRel kernel, it's a
  // Winograd conv kernel that must be dispatched via root constants (CbData VA
  // at s[0:1]), NOT via SetKernelArguments.
  const bool is_winograd_hsa_rel =
      (kd.elf_format == ElfFormat::HsaRel && !kd.cbdata_bytes.empty());
  const bool is_cbdata_dispatch =
      (is_pal_elf || is_winograd_hsa_rel) && !kd.cbdata_bytes.empty();
  if (n_args < 1)
    throw std::runtime_error("Dx12Runner::execute: no arguments in descriptor");
  if (!is_pal_elf && !is_winograd_hsa_rel &&
      input_data.size() + 1 + n_scalar != n_args)
    throw std::runtime_error(
        "Dx12Runner::execute: arg count mismatch (gpu_inputs=" +
        std::to_string(input_data.size()) +
        " scalars=" + std::to_string(n_scalar) +
        " output_arg_index=" + std::to_string(kd.output_arg_index) +
        " total=" + std::to_string(n_args) + ")");

  // 1. Build PSO.
  // PAL ELF CbData dispatch options:
  //   a) With IAmdExtD3DDevice10 (DispatchPalElf): no PSO needed — Device10
  //   handles pipeline.
  //      PSO creation skipped. DispatchPalElf does: pipeline create + CbData
  //      embed + dispatch.
  //   b) With DXCP overlay (clientInternal=0 fix):
  //   CreateComputePipelineCrossCompile(ElfPal)
  //      creates PSO without TDR. Then dispatch via
  //      SetComputeRoot32BitConstants + Dispatch.
  //   c) Without overlay or Device10: PSO creation TDRs (clientInternal=1 →
  //   GpuHeapLocal hang).
  //      Fallback to P3 (MLSS Winograd blocked in amdxcgc_mlss.cpp until
  //      overlay is installed).
  // For HSA_REL Winograd: create PSO via CreateComputePipelineFromElf (works
  // without overlay).
  //
  // skip_pso_for_pal: skip ONLY when Device10 is available (DispatchPalElf
  // handles it). When fallback active (no Device10), create PSO via
  // CrossCompile ElfPal — requires DXCP overlay.
  const bool skip_pso_for_pal =
      is_pal_elf && !kd.cbdata_bytes.empty() && amd_ext_device10_;
  ComPtr<ID3D12PipelineState> pso;
  if (!skip_pso_for_pal)
    pso = create_pso(kd);

  // 2. Allocate GPU buffers.
  const uint64_t out_size = kd.arg_sizes[kd.output_arg_index];

  std::vector<ComPtr<ID3D12Resource>> upload_bufs(input_data.size());
  std::vector<ComPtr<ID3D12Resource>> gpu_input_bufs(input_data.size());
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    const uint64_t raw_sz = input_data[i].size();
    // Scalar broadcast: if an input buffer is smaller than the output buffer
    // and its element size divides evenly into the output buffer,
    // broadcast-fill to the output size. This handles constants like 0.044715
    // in gelu approximation chains where the literal is 1 element but the
    // kernel reads n_elems elements.
    uint64_t sz = raw_sz > 0 ? raw_sz : 4096u;
    std::vector<char> broadcast_buf;
    const uint64_t elem_sz = raw_sz > 0 ? raw_sz : 1u;
    if (raw_sz > 0 && raw_sz < out_size && out_size % raw_sz == 0) {
      // Broadcast: repeat the elem_sz-byte scalar pattern to fill out_size
      // bytes.
      const uint64_t repeats = out_size / raw_sz;
      broadcast_buf.resize(static_cast<std::size_t>(out_size));
      for (uint64_t r = 0; r < repeats; ++r)
        std::memcpy(broadcast_buf.data() + r * raw_sz, input_data[i].data(),
                    raw_sz);
      sz = out_size;
    }
    const char *upload_src =
        broadcast_buf.empty() ? input_data[i].data() : broadcast_buf.data();
    // Upload heap: CPU uploads initial data.
    upload_bufs[i] = create_upload_buffer(sz, upload_src);
    // GPU heap: kernel reads from here.
    gpu_input_bufs[i] = create_gpu_buffer(sz);
  }

  auto gpu_output_buf = create_gpu_buffer(out_size);

  // 2b. Pre-create double-ptr intermediate buffers and fill them synchronously
  //     (before the main command list, so the GPU sees fresh data without
  //     needing in-list barriers).  Mirrors
  //     D3D12DeviceMemory::setDoublePointer().
  std::unordered_set<std::size_t> double_ptr_slots(
      kd.double_ptr_arg_indices.begin(), kd.double_ptr_arg_indices.end());

  // We need data VAs upfront — compute them from the already-allocated buffers.
  // Slot ordering mirrors the arg-building loop below: non-scalar slots are
  // visited in order; output_arg_index is interspersed. Build a map: arg_index
  // → data_va for double-ptr slots.
  std::unordered_map<std::size_t, D3D12_GPU_VIRTUAL_ADDRESS> dbl_ptr_data_vas;
  {
    std::size_t ib = 0;
    const std::size_t n_args = kd.arg_sizes.size();
    std::unordered_map<std::size_t, bool> scalar_set;
    for (const auto &ss : kd.scalar_slots)
      scalar_set[ss.arg_index] = true;
    for (std::size_t i = 0; i < n_args; ++i) {
      if (scalar_set.count(i))
        continue;
      D3D12_GPU_VIRTUAL_ADDRESS dva =
          (i == kd.output_arg_index)
              ? gpu_output_buf->GetGPUVirtualAddress()
              : gpu_input_bufs[ib++]->GetGPUVirtualAddress();
      if (double_ptr_slots.count(i))
        dbl_ptr_data_vas[i] = dva;
    }
  }

  // Allocate ALL ptr_bufs in ONE contiguous DEFAULT heap buffer (one entry per
  // double-ptr slot, each 8 bytes).  All data VAs live in the same page, so
  // there is no risk of inter-resource page-table aliasing.  The buffer is
  // filled via a single upload-copy and kept alive until after the dispatch.
  //
  // slot_order gives a deterministic sorted order so VA+offset is stable.
  std::vector<std::size_t> slot_order(double_ptr_slots.begin(),
                                      double_ptr_slots.end());
  std::sort(slot_order.begin(), slot_order.end());

  // ptr_buf_gpu_vas maps arg_idx → GPU VA of the 8-byte cell for that slot.
  std::unordered_map<std::size_t, D3D12_GPU_VIRTUAL_ADDRESS> ptr_buf_gpu_vas;
  ComPtr<ID3D12Resource> combined_ptr_buf; // keep alive until after dispatch
  std::vector<D3D12_GPU_VIRTUAL_ADDRESS> dbl_ptr_vas_storage; // keep VAs alive
  dbl_ptr_vas_storage.reserve(double_ptr_slots.size());
  {
    const char *tenv = std::getenv("AMDCGC_TRACE");
    if (tenv && std::atoi(tenv) >= 2) {
      std::cerr << "[dx12_runner] double_ptr_slots count="
                << double_ptr_slots.size() << "\n";
      for (auto s : double_ptr_slots)
        std::cerr << "  dbl slot=" << s << " data_va=0x" << std::hex
                  << dbl_ptr_data_vas[s] << std::dec << "\n";
      std::cerr.flush();
    }
  }
  if (!double_ptr_slots.empty()) {
    const uint64_t kEntrySize = sizeof(D3D12_GPU_VIRTUAL_ADDRESS); // 8 bytes
    const uint64_t kBufSize = kEntrySize * slot_order.size();

    // Build a flat byte array of all data VAs in slot_order order.
    std::vector<uint8_t> init_data(static_cast<std::size_t>(kBufSize), 0u);
    for (std::size_t i = 0; i < slot_order.size(); ++i) {
      const std::size_t ai = slot_order[i];
      D3D12_GPU_VIRTUAL_ADDRESS dva = dbl_ptr_data_vas.at(ai);
      std::memcpy(init_data.data() + i * kEntrySize, &dva, sizeof(dva));
    }

    // Allocate a DEFAULT heap buffer and upload the data.
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = kBufSize;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    DX12_CHECK(device_->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&combined_ptr_buf)));

    DX12_CHECK(allocator_->Reset());
    DX12_CHECK(cmd_list_->Reset(allocator_.Get(), nullptr));
    auto up = create_upload_buffer(
        kBufSize, reinterpret_cast<const char *>(init_data.data()));
    cmd_list_->CopyBufferRegion(combined_ptr_buf.Get(), 0, up.Get(), 0,
                                kBufSize);
    flush(); // execute upload synchronously; up can be destroyed after this

    // Compute per-slot GPU VAs as offsets into combined_ptr_buf.
    const D3D12_GPU_VIRTUAL_ADDRESS base_va =
        combined_ptr_buf->GetGPUVirtualAddress();
    for (std::size_t i = 0; i < slot_order.size(); ++i) {
      const std::size_t ai = slot_order[i];
      ptr_buf_gpu_vas[ai] = base_va + i * kEntrySize;
    }

    const char *tenv2 = std::getenv("AMDCGC_TRACE");
    if (tenv2 && std::atoi(tenv2) >= 2) {
      for (std::size_t i = 0; i < slot_order.size(); ++i) {
        const std::size_t ai = slot_order[i];
        std::cerr << "  [combined_ptr_buf] slot=" << ai << " buf_va=0x"
                  << std::hex << ptr_buf_gpu_vas[ai] << " data_va=0x"
                  << dbl_ptr_data_vas.at(ai) << std::dec << "\n";
      }
      std::cerr.flush();
    }
  }

  D3D12_HEAP_PROPERTIES readback_hp{};
  readback_hp.Type = D3D12_HEAP_TYPE_READBACK;
  D3D12_RESOURCE_DESC readback_rd{};
  readback_rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  readback_rd.Width = out_size;
  readback_rd.Height = 1;
  readback_rd.DepthOrArraySize = 1;
  readback_rd.MipLevels = 1;
  readback_rd.SampleDesc.Count = 1;
  readback_rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ComPtr<ID3D12Resource> readback_buf;
  DX12_CHECK(device_->CreateCommittedResource(
      &readback_hp, D3D12_HEAP_FLAG_NONE, &readback_rd,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_buf)));

  // 3. Record commands.
  DX12_CHECK(allocator_->Reset());
  DX12_CHECK(cmd_list_->Reset(allocator_.Get(), nullptr));

  // 3a. Copy inputs from upload → GPU default heap.
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = gpu_input_bufs[i].Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list_->ResourceBarrier(1, &barrier);

    cmd_list_->CopyResource(gpu_input_bufs[i].Get(), upload_bufs[i].Get());

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmd_list_->ResourceBarrier(1, &barrier);
  }

  // 3b. Set PSO and bind arguments.
  // PAL ELF with CbData: skip SetPipelineState — DispatchPalElf handles
  // pipeline internally. MLSS Winograd HSA_REL: PSO was created via
  // CreateComputePipelineFromElf, set it here.
  if (!skip_pso_for_pal)
    cmd_list_->SetPipelineState(pso.Get());

  // ---------------------------------------------------------------------------
  // PAL ELF dispatch path (Winograd conv kernels)
  // ---------------------------------------------------------------------------
  // PAL ELF shaders use the F_ADDR_INDIRECT mechanism: the shader reads its
  // CbData struct from a GPU VA passed in SGPRs[0:1] (D3D12 root constants).
  // SetKernelArguments cannot be used — it asserts hsaAbiMode (HSA-only).
  // This path allocates a GPU buffer for CbData, patches buffer VAs into it,
  // then dispatches via SetComputeRoot32BitConstants with the CbData GPU VA.
  // is_pal_dispatch: CbData dispatch required. True for:
  //   - PAL ELF (PalRel/PalDyn): uses PAL calling convention natively
  //   - MLSS Winograd HSA_REL: PAL calling convention in HSA ELF wrapper
  // Both require dispatching via root constants {CbData_VA_lo, CbData_VA_hi} at
  // s[0:1].
  const bool is_pal_dispatch = is_cbdata_dispatch;
  // CbData fallback: for PAL ELF or MLSS Winograd HSA_REL without
  // IAmdExtD3DDevice10, dispatch via standard D3D12 root constants. Uploads
  // CbData to a small upload buffer and sets root constants = {CbData_VA_lo,
  // CbData_VA_hi} → SGPRs[0:1] = CbData VA. For HSA_REL Winograd: PSO was
  // created via CreateComputePipelineFromElf (no TDR). DispatchPalElf (v10) is
  // preferred for PAL ELF but can't be used for HSA_REL.
  bool pal_fallback_active = (is_pal_dispatch && !kd.cbdata_bytes.empty() &&
                              (!amd_ext_device10_ || is_winograd_hsa_rel));
  bool dispatched_pal = false;
  if (is_pal_dispatch && !kd.cbdata_bytes.empty() &&
      (amd_ext_device10_ || pal_fallback_active)) {
    // PAL ELF dispatch via DispatchPalElf (v10 API) — replicates metacommand
    // exactly:
    // 1. Patches buffer VAs into CbData
    // 2. CmdAllocateEmbeddedData → embed CbData in command buffer memory
    // 3. CmdSetUserData(0, 2, {cbVaLo, cbVaHi}) → SGPRs[2:3] = CbData VA
    // 4. CmdDispatch — same as DdiMetaCmdConvFury::ExecuteConv

    // 1. Build CbData with direct buffer VAs (DispatchPalElf handles
    // CmdAllocateEmbeddedData). NOTE: The Fury shader with F_ADDR_INDIRECT may
    // expect descriptor handles or raw VAs. Test: Disable F_ADDR_INDIRECT (bit
    // 6) and use direct buffer VAs. The shader branches on bit 6 of flags64
    // (F_ADDR_INDIRECT):
    //   - If set: reads pointer buffers (double-indirection)
    //   - If not set: uses buffer VAs directly from CbData
    // Testing WITHOUT F_ADDR_INDIRECT to verify direct VA path works first.
    // Build CbData with direct buffer VAs.
    // F_ADDR_INDIRECT (bit 6) is NOT set in flags64 — shader uses direct VAs
    // from CbData.
    std::vector<uint8_t> cbdata = kd.cbdata_bytes;

    std::vector<ComPtr<ID3D12Resource>>
        ptr_upload_bufs; // unused (no pointer indirection)
    std::vector<ComPtr<ID3D12Resource>> ptr_gpu_bufs; // unused

    for (const auto &slot : kd.cbdata_slots) {
      if (slot.byte_offset + 8u > cbdata.size())
        throw std::runtime_error("cbdata_buf_slot byte_offset out of range");
      D3D12_GPU_VIRTUAL_ADDRESS va{};
      if (slot.buf_index < 0)
        va = gpu_output_buf->GetGPUVirtualAddress();
      else
        va = gpu_input_bufs[static_cast<std::size_t>(slot.buf_index)]
                 ->GetGPUVirtualAddress();
      std::memcpy(cbdata.data() + slot.byte_offset, &va, 8);
    }

    // Log CbData for diagnosis.
    {
      auto rd_u32 = [&](size_t off) -> uint32_t {
        uint32_t v = 0;
        memcpy(&v, cbdata.data() + off, 4);
        return v;
      };
      auto rd_u64 = [&](size_t off) -> uint64_t {
        uint64_t v = 0;
        memcpy(&v, cbdata.data() + off, 8);
        return v;
      };
      std::cerr << "[dx12_runner] CbData: N=" << rd_u32(0x00)
                << " C=" << rd_u32(0x04) << " H=" << rd_u32(0x08)
                << " W=" << rd_u32(0x0C) << " K=" << rd_u32(0x10)
                << " nGroups=" << rd_u32(0x14) << " flags=0x" << std::hex
                << rd_u64(0x18) << std::dec << "\n  dataAddr=0x" << std::hex
                << rd_u64(0x20) << " filterAddr=0x" << rd_u64(0x28)
                << " outputAddr=0x" << rd_u64(0x30) << std::dec << "\n";
      std::cerr.flush();
    }

    if (amd_ext_device10_) {
      // Fast path: CmdAllocateEmbeddedData (no extra upload buffer needed)
      amd_ext_device10_->DispatchPalElf(
          cmd_list_.Get(), reinterpret_cast<const uint32_t *>(cbdata.data()),
          static_cast<uint32_t>(cbdata.size() / 4), kd.dispatch_x,
          kd.dispatch_y, kd.dispatch_z);
    } else {
      // Fallback: standard D3D12 dispatch when IAmdExtD3DDevice10 not
      // available. Upload CbData to a small upload buffer, pass GPU VA as root
      // constants. The PAL/Winograd shader reads CbData from SGPRs[0:1] =
      // {CbData_VA_lo, CbData_VA_hi}.
      ComPtr<ID3D12Resource> cbdata_upload_buf =
          create_upload_buffer(cbdata.size(), cbdata.data());
      D3D12_GPU_VIRTUAL_ADDRESS cbdata_va =
          cbdata_upload_buf->GetGPUVirtualAddress();
      uint32_t va_lo = static_cast<uint32_t>(cbdata_va & 0xFFFFFFFF);
      uint32_t va_hi = static_cast<uint32_t>(cbdata_va >> 32);
      const uint32_t root_data[2] = {va_lo, va_hi};
      // For both PAL ELF and HSA_REL Winograd: set the PSO before dispatching.
      // PAL ELF PSO: created via CrossCompile ElfPal (requires DXCP overlay
      // with clientInternal=0). HSA_REL PSO: created via
      // CreateComputePipelineFromElf.
      if (pso) {
        cmd_list_->SetPipelineState(pso.Get());
        std::cerr << "[dx12_runner] CbData fallback: SetPipelineState (HSA_REL "
                     "Winograd)\n";
        std::cerr.flush();
      }
      // CRITICAL: Must set root signature before setting root constants.
      // Use our pal_root_sig_ (2 root constants at b0) so that
      // SetComputeRoot32BitConstants(0, ...) maps correctly to SGPRs[0:1].
      cmd_list_->SetComputeRootSignature(pal_root_sig_.Get());
      // Set SGPRs[0:1] = {CbData_VA_lo, CbData_VA_hi} via root constants (slot
      // 0, 2 DWORDs). Equivalent to metacommand's CmdSetUserData(Compute, 0, 2,
      // ...) on GFX11+.
      cmd_list_->SetComputeRoot32BitConstants(0, 2, root_data, 0);
      cmd_list_->Dispatch(kd.dispatch_x, kd.dispatch_y, kd.dispatch_z);
      flush(); // submit + wait for GPU — cbdata_upload_buf stays alive until
               // here
      std::cerr
          << "[dx12_runner] PAL ELF fallback dispatch via D3D12 root constants "
          << "cbdata_va=0x" << std::hex << cbdata_va << std::dec << " grid=("
          << kd.dispatch_x << "," << kd.dispatch_y << "," << kd.dispatch_z
          << ")\n";
    }
    dispatched_pal = true;
  }

  if (!dispatched_pal) {
    // ---------------------------------------------------------------------------
    // HSA ELF dispatch path (GEMM, GQA, and all non-PAL kernels)
    // ---------------------------------------------------------------------------
    // Build argument value pointer array.
    // - GPU buffer slots: ppValues[i] → &gpu_va (uint64_t GPU virtual address)
    // - Scalar slots:     ppValues[i] → raw bytes of the scalar value
    // (int32/int64/float)
    // - Double-ptr slots: ppValues[i] → &ptr_buf_va (VA of the 8-byte pointer
    // buffer) SetKernelArguments passes each pointer's pointed-to bytes as the
    // kernel argument.

    // Build scalar slot lookup: arg_index → pointer to embedded scalar bytes.
    std::unordered_map<std::size_t, const void *> scalar_val_ptrs;
    for (const auto &ss : kd.scalar_slots)
      scalar_val_ptrs[ss.arg_index] = ss.value_bytes.data();

    // gpu_vas must stay alive until after SetKernelArguments — reserve to avoid
    // realloc.
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> gpu_vas;
    gpu_vas.reserve(input_data.size() + 1);

    // dbl_ptr_vas holds the per-slot GPU VA from combined_ptr_buf (must outlive
    // SetKernelArguments).
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> dbl_ptr_vas;
    dbl_ptr_vas.reserve(ptr_buf_gpu_vas.size());

    std::vector<const void *> arg_ptrs(n_args, nullptr);
    std::size_t input_buf_idx = 0;

    for (std::size_t i = 0; i < n_args; ++i) {
      auto scalar_it = scalar_val_ptrs.find(i);
      if (scalar_it != scalar_val_ptrs.end()) {
        // Scalar slot: point directly at the embedded scalar bytes.
        arg_ptrs[i] = scalar_it->second;
      } else {
        // Determine the data GPU VA for this slot.
        D3D12_GPU_VIRTUAL_ADDRESS data_va{};
        if (i == kd.output_arg_index)
          data_va = gpu_output_buf->GetGPUVirtualAddress();
        else
          data_va = gpu_input_bufs[input_buf_idx++]->GetGPUVirtualAddress();

        auto dbl_it = ptr_buf_gpu_vas.find(i);
        // Debug: AMDCGC_DIRECT_PTR=1 bypasses double-pointer staging to test
        // direct VA path
        const char *direct_ptr_env = std::getenv("AMDCGC_DIRECT_PTR");
        const bool force_direct =
            direct_ptr_env && std::atoi(direct_ptr_env) != 0;
        if (dbl_it != ptr_buf_gpu_vas.end() && !force_direct) {
          // Double-ptr slot: pass GPU VA of the 8-byte cell in
          // combined_ptr_buf.
          dbl_ptr_vas.push_back(dbl_it->second);
          arg_ptrs[i] = &dbl_ptr_vas.back();
        } else {
          gpu_vas.push_back(data_va);
          arg_ptrs[i] = &gpu_vas.back();
        }
      }
    }

    // Diagnostic: print kernel argument values before dispatch.
    // Enabled at AMDCGC_TRACE >= 2 OR --verbose.
    const char *trace_env = std::getenv("AMDCGC_TRACE");
    const int trace_level = trace_env ? std::atoi(trace_env) : 0;
    if (verbose_ || trace_level >= 2) {
      std::cerr << "[dx12_runner] kernargs (" << n_args << " slots):\n";
      std::cerr << "  double_ptr_slots:";
      for (auto s : double_ptr_slots)
        std::cerr << " " << s;
      std::cerr << "\n";
      for (std::size_t i = 0; i < n_args && arg_ptrs[i]; ++i) {
        auto scalar_it = scalar_val_ptrs.find(i);
        if (scalar_it != scalar_val_ptrs.end()) {
          uint32_t u32 = 0;
          float f32 = 0.f;
          std::memcpy(&u32, scalar_it->second, 4);
          std::memcpy(&f32, scalar_it->second, 4);
          std::cerr << "  slot[" << i << "] scalar u32=" << u32
                    << " f32=" << f32 << "\n";
        } else {
          D3D12_GPU_VIRTUAL_ADDRESS va = 0;
          std::memcpy(&va, arg_ptrs[i], sizeof(va));
          bool is_dbl = double_ptr_slots.count(i) != 0;
          std::cerr << "  slot[" << i << "] " << (is_dbl ? "dbl_ptr" : "direct")
                    << " VA=0x" << std::hex << va << std::dec << "\n";
        }
      }
      std::cerr.flush();
    }

    amd_ext_device_->SetKernelArguments(
        cmd_list_.Get(), 0u, static_cast<uint32_t>(n_args), arg_ptrs.data());

    // 3c. Dispatch.
    cmd_list_->Dispatch(kd.dispatch_x, kd.dispatch_y, kd.dispatch_z);
  } // end if(!dispatched_pal)

  // 3d. UAV barrier on output, then copy to readback.
  {
    D3D12_RESOURCE_BARRIER uav_barrier{};
    uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_barrier.UAV.pResource =
        nullptr; // global UAV flush (matches MLSS reference)
    cmd_list_->ResourceBarrier(1, &uav_barrier);

    D3D12_RESOURCE_BARRIER copy_barrier{};
    copy_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copy_barrier.Transition.pResource = gpu_output_buf.Get();
    copy_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    copy_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    copy_barrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list_->ResourceBarrier(1, &copy_barrier);

    cmd_list_->CopyResource(readback_buf.Get(), gpu_output_buf.Get());
  }

  // 4. Submit and wait.
  flush();

  // Copy the output buffer to host, then convert to float32 (element count
  // matches hip_run).
  std::vector<char> raw(static_cast<std::size_t>(out_size));
  {
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(out_size)};
    void *mapped = nullptr;
    DX12_CHECK(readback_buf->Map(0, &read_range, &mapped));
    std::memcpy(raw.data(), mapped, static_cast<std::size_t>(out_size));
    D3D12_RANGE no_write{0, 0};
    readback_buf->Unmap(0, &no_write);
  }
  return bytes_to_f32(raw, kd.output_element_bytes);
}

std::vector<char>
Dx12Runner::execute_raw(const KernelDescriptor &kd,
                        const std::vector<std::vector<char>> &input_data) {
  // execute() converts fp16→fp32 which loses bit-exactness needed for chaining.
  // We run the same dispatch logic but map the readback as raw bytes.
  // Reuse execute() infrastructure by calling execute() and reinterpreting —
  // but that would lose fp16 precision.  Instead, duplicate the core dispatch
  // and return the raw bytes before any element conversion.

  ensure_initialized();
  const uint64_t out_size = kd.arg_sizes[kd.output_arg_index];

  // 1. Build PSO.
  auto pso = create_pso(kd);

  // 2. Allocate GPU buffers.
  // Inputs with ≤ 4 bytes are null_ptr sentinels (from cgc_op.null_ptr
  // literals). Pass GPU VA=0 for these so kernels see nullptr for optional
  // arguments.
  std::vector<ComPtr<ID3D12Resource>> upload_bufs(input_data.size());
  std::vector<ComPtr<ID3D12Resource>> gpu_input_bufs(input_data.size());
  std::vector<bool> is_null_sentinel(input_data.size(), false);
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    const std::size_t raw_sz = input_data[i].size();
    // Null sentinel: only treat as NULL VA if the buffer data is actually
    // all-zeros (empty literal / null_ptr sentinel). DO NOT mark buffers that
    // contain real data (e.g. pos_ids=128, seqlens_k=31) as null — that zeroes
    // the GPU VA causing the kernel to read position 0 and produce wrong/zero
    // output.
    if (raw_sz <= 4 && kd.arg_sizes.size() > i && kd.arg_sizes[i] <= 4) {
      bool all_zero = input_data[i].empty();
      if (!all_zero) {
        all_zero = true;
        for (auto b : input_data[i])
          if (b != 0) {
            all_zero = false;
            break;
          }
      }
      if (all_zero) {
        // Null sentinel — keep null GPU resource so GPU sees VA=0 (nullptr).
        is_null_sentinel[i] = true;
        continue;
      }
    }
    const uint64_t sz = raw_sz > 0 ? raw_sz : 4096u;
    upload_bufs[i] = create_upload_buffer(sz, input_data[i].data());
    gpu_input_bufs[i] = create_gpu_buffer(sz);
  }
  auto gpu_output_buf = create_gpu_buffer(out_size);

  D3D12_HEAP_PROPERTIES readback_hp{};
  readback_hp.Type = D3D12_HEAP_TYPE_READBACK;
  D3D12_RESOURCE_DESC readback_rd{};
  readback_rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  readback_rd.Width = out_size;
  readback_rd.Height = 1;
  readback_rd.DepthOrArraySize = 1;
  readback_rd.MipLevels = 1;
  readback_rd.SampleDesc.Count = 1;
  readback_rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ComPtr<ID3D12Resource> readback_buf;
  DX12_CHECK(device_->CreateCommittedResource(
      &readback_hp, D3D12_HEAP_FLAG_NONE, &readback_rd,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_buf)));

  // 2b. Stage double-pointer (T**) arguments, before the dispatch list is
  // recorded. Without this the kernel is handed a buffer VA where it expects
  // the address of a cell holding that VA; it dereferences tensor bytes as a
  // pointer and writes nothing usable -- the "dx12 output is all 0" failure
  // signature.
  std::unordered_map<std::size_t, D3D12_GPU_VIRTUAL_ADDRESS> ptr_cell_of;
  ComPtr<ID3D12Resource> ptr_cell_buf; // must outlive the dispatch below
  if (not kd.double_ptr_arg_indices.empty()) {
    const std::unordered_set<std::size_t> dbl(kd.double_ptr_arg_indices.begin(),
                                              kd.double_ptr_arg_indices.end());
    std::unordered_set<std::size_t> scalar_set;
    for (const auto &ss : kd.scalar_slots)
      scalar_set.insert(ss.arg_index);

    // This walk must mirror the argument loop below exactly -- scalars and the
    // output slot do not consume an input buffer -- or a cell is filled with
    // the wrong buffer's address and the kernel reads the wrong tensor.
    std::vector<std::size_t> cell_slots;
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> cell_data;
    std::size_t ib = 0;
    for (std::size_t i = 0; i < kd.arg_sizes.size(); ++i) {
      if (scalar_set.count(i))
        continue;
      D3D12_GPU_VIRTUAL_ADDRESS va = 0;
      if (i == kd.output_arg_index) {
        va = gpu_output_buf->GetGPUVirtualAddress();
      } else {
        // A null sentinel keeps VA=0 so the kernel still sees nullptr.
        if (ib < is_null_sentinel.size() && not is_null_sentinel[ib] &&
            gpu_input_bufs[ib])
          va = gpu_input_bufs[ib]->GetGPUVirtualAddress();
        ++ib;
      }
      if (dbl.count(i)) {
        cell_slots.push_back(i);
        cell_data.push_back(va);
      }
    }
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> cell_vas;
    ptr_cell_buf = stage_pointer_cells(cell_data, cell_vas);
    for (std::size_t i = 0; i < cell_slots.size(); ++i)
      ptr_cell_of[cell_slots[i]] = cell_vas[i];
  }

  // 3. Record commands.
  DX12_CHECK(allocator_->Reset());
  DX12_CHECK(cmd_list_->Reset(allocator_.Get(), nullptr));

  for (std::size_t i = 0; i < input_data.size(); ++i) {
    if (is_null_sentinel[i])
      continue; // skip null sentinels — no GPU resource
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = gpu_input_bufs[i].Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list_->ResourceBarrier(1, &barrier);
    cmd_list_->CopyResource(gpu_input_bufs[i].Get(), upload_bufs[i].Get());
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmd_list_->ResourceBarrier(1, &barrier);
  }

  cmd_list_->SetPipelineState(pso.Get());

  std::unordered_map<std::size_t, const void *> scalar_val_ptrs;
  for (const auto &ss : kd.scalar_slots)
    scalar_val_ptrs[ss.arg_index] = ss.value_bytes.data();

  std::vector<D3D12_GPU_VIRTUAL_ADDRESS> gpu_vas;
  gpu_vas.reserve(input_data.size() + 1);
  const std::size_t n_args = kd.arg_sizes.size();
  std::vector<const void *> arg_ptrs(n_args, nullptr);

  std::size_t input_buf_idx = 0;
  for (std::size_t i = 0; i < n_args; ++i) {
    auto it = scalar_val_ptrs.find(i);
    if (it != scalar_val_ptrs.end()) {
      arg_ptrs[i] = it->second;
    } else if (i == kd.output_arg_index) {
      // A double-ptr slot passes the VA of its 8-byte cell (staged in 2b)
      // rather than the buffer VA, so the kernel's extra dereference lands on
      // the buffer address instead of on buffer contents.
      auto c = ptr_cell_of.find(i);
      gpu_vas.push_back(c != ptr_cell_of.end()
                            ? c->second
                            : gpu_output_buf->GetGPUVirtualAddress());
      arg_ptrs[i] = &gpu_vas.back();
    } else {
      // Null sentinel (null_ptr literal ≤ 4 bytes): bind VA=0 so kernel sees
      // nullptr.
      D3D12_GPU_VIRTUAL_ADDRESS va =
          (input_buf_idx < is_null_sentinel.size() &&
           is_null_sentinel[input_buf_idx])
              ? D3D12_GPU_VIRTUAL_ADDRESS(0)
              : gpu_input_bufs[input_buf_idx]->GetGPUVirtualAddress();
      auto c = ptr_cell_of.find(i);
      if (c != ptr_cell_of.end())
        va = c->second;
      gpu_vas.push_back(va);
      arg_ptrs[i] = &gpu_vas.back();
      ++input_buf_idx;
    }
  }

  amd_ext_device_->SetKernelArguments(
      cmd_list_.Get(), 0u, static_cast<uint32_t>(n_args), arg_ptrs.data());

  cmd_list_->Dispatch(kd.dispatch_x, kd.dispatch_y, kd.dispatch_z);

  D3D12_RESOURCE_BARRIER uav_barrier{};
  uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uav_barrier.UAV.pResource = gpu_output_buf.Get();
  cmd_list_->ResourceBarrier(1, &uav_barrier);

  D3D12_RESOURCE_BARRIER copy_barrier{};
  copy_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  copy_barrier.Transition.pResource = gpu_output_buf.Get();
  copy_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  copy_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  copy_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  cmd_list_->ResourceBarrier(1, &copy_barrier);
  cmd_list_->CopyResource(readback_buf.Get(), gpu_output_buf.Get());

  // 4. Submit and wait.
  flush();

  // 5. Map readback and return raw bytes.
  std::vector<char> raw(static_cast<std::size_t>(out_size));
  {
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(out_size)};
    void *mapped = nullptr;
    DX12_CHECK(readback_buf->Map(0, &read_range, &mapped));
    std::memcpy(raw.data(), mapped, raw.size());
    D3D12_RANGE no_write{0, 0};
    readback_buf->Unmap(0, &no_write);
  }
  return raw;
}

// ---------------------------------------------------------------------------
// execute_raw_group — submit multiple kernels in a single DX12 command list.
//
// Key design: inter-kernel dependencies are expressed via
// GroupInput::source_kernel (not via CPU roundtrip). When source_kernel >= 0,
// the GPU output buffer of that kernel is bound directly as an input — no
// readback/re-upload needed. UAV barriers after each dispatch ensure ordering.
// Only the designated output kernel's result is read back.
//
// The driver reads KernelDescriptor::source_kernel_index (from the compiler) to
// fill GroupInput entries — the driver itself has no prior knowledge of the
// kernel semantics, only the descriptor structure.
// ---------------------------------------------------------------------------
std::vector<char> Dx12Runner::execute_raw_group(
    const std::vector<KernelDescriptor> &kernels,
    const std::vector<std::vector<GroupInput>> &inputs_per_kernel,
    int output_kernel_idx) {
  ensure_initialized();
  const std::size_t n_kernels = kernels.size();
  if (n_kernels == 0)
    return {};

  const std::size_t out_k =
      (output_kernel_idx >= 0 &&
       static_cast<std::size_t>(output_kernel_idx) < n_kernels)
          ? static_cast<std::size_t>(output_kernel_idx)
          : n_kernels - 1;

  // 1. Create PSOs for all kernels up front.
  std::vector<ComPtr<ID3D12PipelineState>> psos(n_kernels);
  for (std::size_t k = 0; k < n_kernels; ++k)
    psos[k] = create_pso(kernels[k]);

  // 2. Allocate GPU output buffers for all kernels (stays on GPU as
  // intermediate).
  //    Upload buffers and GPU input buffers only for CPU-sourced inputs.
  std::vector<ComPtr<ID3D12Resource>> gpu_outputs(n_kernels);
  // upload_bufs[k][i] / gpu_cpu_bufs[k][i]: only for CPU-sourced inputs
  std::vector<std::vector<ComPtr<ID3D12Resource>>> upload_bufs(n_kernels);
  std::vector<std::vector<ComPtr<ID3D12Resource>>> gpu_cpu_bufs(n_kernels);

  for (std::size_t k = 0; k < n_kernels; ++k) {
    const auto &kd = kernels[k];
    // For in-place GEMM kernels that share a buffer with a prior kernel (e.g.
    // QKVProj_offset concat-elimination), reuse the producer's GPU buffer
    // instead of allocating a fresh one.  This ensures all kernels write to the
    // same GPU buffer and the correct final state is read back after the last
    // kernel.
    if (kd.output_reuses_kernel >= 0 &&
        static_cast<std::size_t>(kd.output_reuses_kernel) < k) {
      gpu_outputs[k] =
          gpu_outputs[static_cast<std::size_t>(kd.output_reuses_kernel)];
    } else {
      const uint64_t out_sz = kd.arg_sizes[kd.output_arg_index];
      gpu_outputs[k] = create_gpu_buffer(out_sz > 0 ? out_sz : 4096u);
    }

    const auto &inputs = inputs_per_kernel[k];
    upload_bufs[k].resize(inputs.size());
    gpu_cpu_bufs[k].resize(inputs.size());

    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].is_null || inputs[i].source_kernel >= 0)
        continue; // null VA or GPU inter-kernel — no CPU upload needed
      const uint64_t sz =
          inputs[i].cpu_data.size() > 0 ? inputs[i].cpu_data.size() : 4096u;
      upload_bufs[k][i] = create_upload_buffer(sz, inputs[i].cpu_data.data());
      gpu_cpu_bufs[k][i] = create_gpu_buffer(sz);
    }
  }

  // 2b. Stage double-pointer (T**) kernel arguments for the whole group.
  //
  // MLSS GEMM/GQA kernels declare their buffer operands as T** (MLSSarg
  // m_indirectionLevel=2): the kernarg slot must hold the address of a device
  // cell that in turn holds the buffer VA. Dx12Runner::execute does this for
  // the single-kernel path; without the same staging here the kernel reads the
  // first bytes of the buffer as an address and dereferences it, removing the
  // device.
  //
  // Every buffer the group can name is already allocated above -- each kernel's
  // output, each CPU-sourced input, and each inter-kernel edge -- so all data
  // VAs are known at this point. Cells for the entire group live in ONE
  // contiguous DEFAULT heap buffer so they share a page, and are filled by a
  // single synchronous upload before the dispatch command list is recorded
  // (mirroring execute()'s approach, which avoids needing an in-list barrier
  // between the fill and the first dispatch).
  std::vector<std::unordered_map<std::size_t, D3D12_GPU_VIRTUAL_ADDRESS>>
      ptr_cell_vas(n_kernels);
  ComPtr<ID3D12Resource> combined_ptr_buf; // must outlive every dispatch below
  {
    // One entry per (kernel, arg index) needing a cell, with the VA it must
    // hold.
    std::vector<std::pair<std::size_t, std::size_t>> cell_slots;
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> cell_data_vas;
    for (std::size_t k = 0; k < n_kernels; ++k) {
      const auto &kd = kernels[k];
      if (kd.double_ptr_arg_indices.empty())
        continue;
      const std::unordered_set<std::size_t> dbl(
          kd.double_ptr_arg_indices.begin(), kd.double_ptr_arg_indices.end());
      std::unordered_set<std::size_t> scalar_set;
      for (const auto &ss : kd.scalar_slots)
        scalar_set.insert(ss.arg_index);

      const auto &inputs = inputs_per_kernel[k];
      const std::size_t n_args = kd.arg_sizes.size();
      std::size_t input_slot = 0;
      for (std::size_t i = 0; i < n_args; ++i) {
        // This walk must mirror the argument loop in the dispatch section below
        // exactly -- scalars and the output slot do not consume a GroupInput --
        // or a cell would be filled with the wrong buffer's address.
        if (scalar_set.count(i))
          continue;
        D3D12_GPU_VIRTUAL_ADDRESS va = 0;
        if (i == kd.output_arg_index) {
          va = gpu_outputs[k]->GetGPUVirtualAddress();
        } else {
          if (input_slot < inputs.size()) {
            const auto &gi = inputs[input_slot];
            if (gi.is_null)
              va = 0;
            else if (gi.source_kernel >= 0 &&
                     static_cast<std::size_t>(gi.source_kernel) < n_kernels)
              va = gpu_outputs[static_cast<std::size_t>(gi.source_kernel)]
                       ->GetGPUVirtualAddress();
            else if (gpu_cpu_bufs[k][input_slot])
              va = gpu_cpu_bufs[k][input_slot]->GetGPUVirtualAddress();
          }
          ++input_slot;
        }
        if (dbl.count(i)) {
          cell_slots.emplace_back(k, i);
          cell_data_vas.push_back(va);
        }
      }
    }

    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> cell_vas;
    combined_ptr_buf = stage_pointer_cells(cell_data_vas, cell_vas);
    for (std::size_t i = 0; i < cell_slots.size(); ++i)
      ptr_cell_vas[cell_slots[i].first][cell_slots[i].second] = cell_vas[i];
  }

  // 3. Readback buffer for the selected output kernel.
  const uint64_t final_out_sz =
      kernels[out_k].arg_sizes[kernels[out_k].output_arg_index];
  ComPtr<ID3D12Resource> readback_buf;
  {
    D3D12_HEAP_PROPERTIES rhp{};
    rhp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rrd{};
    rrd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rrd.Width = final_out_sz > 0 ? final_out_sz : 4096u;
    rrd.Height = 1;
    rrd.DepthOrArraySize = 1;
    rrd.MipLevels = 1;
    rrd.SampleDesc.Count = 1;
    rrd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    DX12_CHECK(device_->CreateCommittedResource(
        &rhp, D3D12_HEAP_FLAG_NONE, &rrd, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&readback_buf)));
  }

  // 4. Record ALL kernels into ONE command list.
  DX12_CHECK(allocator_->Reset());
  DX12_CHECK(cmd_list_->Reset(allocator_.Get(), nullptr));

  for (std::size_t k = 0; k < n_kernels; ++k) {
    const auto &kd = kernels[k];
    const auto &inputs = inputs_per_kernel[k];

    // Upload CPU-sourced inputs before this kernel's dispatch.
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (!gpu_cpu_bufs[k][i])
        continue;
      D3D12_RESOURCE_BARRIER b{};
      b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      b.Transition.pResource = gpu_cpu_bufs[k][i].Get();
      b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
      b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      cmd_list_->ResourceBarrier(1, &b);
      cmd_list_->CopyResource(gpu_cpu_bufs[k][i].Get(),
                              upload_bufs[k][i].Get());
      b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      cmd_list_->ResourceBarrier(1, &b);
    }

    // Bind PSO and build argument table.
    cmd_list_->SetPipelineState(psos[k].Get());

    std::unordered_map<std::size_t, const void *> scalar_ptrs;
    for (const auto &ss : kd.scalar_slots)
      scalar_ptrs[ss.arg_index] = ss.value_bytes.data();

    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> gpu_vas;
    gpu_vas.reserve(inputs.size() + 1);
    const std::size_t n_args = kd.arg_sizes.size();
    std::vector<const void *> arg_ptrs(n_args, nullptr);

    std::size_t input_slot = 0;
    for (std::size_t i = 0; i < n_args; ++i) {
      auto it = scalar_ptrs.find(i);
      if (it != scalar_ptrs.end()) {
        arg_ptrs[i] = it->second;
        continue;
      }
      // A double-ptr slot passes the VA of its 8-byte cell (staged in step 2b)
      // rather than the buffer VA, so the kernel's extra dereference lands on
      // the buffer address instead of on buffer contents.
      const auto cell_it = ptr_cell_vas[k].find(i);
      const bool has_cell = cell_it != ptr_cell_vas[k].end();
      if (i == kd.output_arg_index) {
        gpu_vas.push_back(has_cell ? cell_it->second
                                   : gpu_outputs[k]->GetGPUVirtualAddress());
        arg_ptrs[i] = &gpu_vas.back();
        continue;
      }
      // Regular input slot — resolve from GroupInput.
      D3D12_GPU_VIRTUAL_ADDRESS va = 0;
      if (input_slot < inputs.size()) {
        const auto &gi = inputs[input_slot];
        if (gi.is_null) {
          va = 0; // null sentinel
        } else if (gi.source_kernel >= 0 &&
                   static_cast<std::size_t>(gi.source_kernel) < n_kernels) {
          // GPU inter-kernel: use producer's output buffer directly.
          va = gpu_outputs[static_cast<std::size_t>(gi.source_kernel)]
                   ->GetGPUVirtualAddress();
        } else if (gpu_cpu_bufs[k][input_slot]) {
          va = gpu_cpu_bufs[k][input_slot]->GetGPUVirtualAddress();
        }
      }
      gpu_vas.push_back(has_cell ? cell_it->second : va);
      arg_ptrs[i] = &gpu_vas.back();
      ++input_slot;
    }

    amd_ext_device_->SetKernelArguments(
        cmd_list_.Get(), 0u, static_cast<uint32_t>(n_args), arg_ptrs.data());
    cmd_list_->Dispatch(kd.dispatch_x, kd.dispatch_y, kd.dispatch_z);

    // Global UAV barrier after each kernel to ensure writes are visible to the
    // next.
    D3D12_RESOURCE_BARRIER uav_b{};
    uav_b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_b.UAV.pResource = nullptr;
    cmd_list_->ResourceBarrier(1, &uav_b);
  }

  // Transition the selected output buffer for readback.
  {
    D3D12_RESOURCE_BARRIER copy_b{};
    copy_b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copy_b.Transition.pResource = gpu_outputs[out_k].Get();
    copy_b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    copy_b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    copy_b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list_->ResourceBarrier(1, &copy_b);
    cmd_list_->CopyResource(readback_buf.Get(), gpu_outputs[out_k].Get());
  }

  // 5. Submit: streaming_mode=true → flush_n(n_benchmark_repeats) for benchmark
  // CL reuse,
  //            streaming_mode=false → single flush() (normal execution).
  if (streaming_mode && n_benchmark_repeats > 1)
    flush_n(
        n_benchmark_repeats); // N submits, 1 fence — eliminates per-iter ~30µs
  else
    flush();

  // 6. Read back only the final output.
  const std::size_t rb_sz =
      final_out_sz > 0 ? static_cast<std::size_t>(final_out_sz) : 4096u;
  std::vector<char> raw(rb_sz);
  {
    D3D12_RANGE rr{0, rb_sz};
    void *mapped = nullptr;
    DX12_CHECK(readback_buf->Map(0, &rr, &mapped));
    std::memcpy(raw.data(), mapped, rb_sz);
    D3D12_RANGE nw{0, 0};
    readback_buf->Unmap(0, &nw);
  }
  return raw;
}

} // namespace dx12
} // namespace hip_ep
