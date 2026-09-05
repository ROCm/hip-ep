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
 * Declarations for the AMD Cross-Compile DX12 API.
 *
 * Full header: AmdExtD3DDeviceApi.h
 * Source:      https://github.amd.com/AMD-Radeon-Driver/dxcp/blob/amd/stg/dxcp/public/AmdExtD3DDeviceApi.h
 *
 * If the AMD SDK is available, prefer to include that header directly and
 * set MIGRAPHX_AMDCC_USE_SDK_HEADER.  This file declares only the subset
 * needed to drive CreateComputePipelineCrossCompile + SetKernelArguments.
 */

#ifdef MIGRAPHX_AMDCC_USE_SDK_HEADER
#include <AmdExtD3DDeviceApi.h>
#else

#include <d3d12.h>
#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// AmdExtD3DStructType — discriminator tag placed at the start of every AMD
// extension info struct.
// Ordinal values match dxcp/public/AmdExtD3DDeviceApi.h exactly.
// ---------------------------------------------------------------------------
enum AmdExtD3DStructType : uint32_t
{
    AmdExtD3DStructUnknown              = 0,
    AmdExtD3DStructPipelineState        = 1,
    AmdExtD3DStructPipelineElf          = 2,
    AmdExtD3DStructPipelineCrossCompile = 3,
};

// ---------------------------------------------------------------------------
// AmdShaderType — input blob language / format.
// ---------------------------------------------------------------------------
enum class AmdShaderType : uint32_t
{
    Hip    = 0,  ///< HIP C++ source (hiprtc compiled)
    Ocl    = 1,  ///< OpenCL C source
    Hlsl   = 2,  ///< HLSL compute shader source
    Spirv  = 3,  ///< SPIR-V binary
    ElfHsa = 4,  ///< Compiled HSACO ELF (AMD code object) — our primary path
    ElfPal = 5,  ///< PAL ELF binary
};

// ---------------------------------------------------------------------------
// Base struct — every AMD extension info struct starts with this.
// IMPORTANT: pNext must be nullptr when not chaining structures.
// Layout matches dxcp/public/AmdExtD3DDeviceApi.h exactly (16 bytes on x64).
// ---------------------------------------------------------------------------
struct AmdExtD3DCreateInfo
{
    AmdExtD3DStructType type;
    void*               pNext;  ///< Chain pointer — set to nullptr
};

// ---------------------------------------------------------------------------
// AmdExtD3DPipelineCrossCompileInfo
// Passed to CreateComputePipelineCrossCompile to describe the kernel binary
// and its dispatch configuration.
// Layout matches dxcp/public/AmdExtD3DDeviceApi.h exactly.
// ---------------------------------------------------------------------------
struct AmdExtD3DPipelineCrossCompileInfo : AmdExtD3DCreateInfo
{
    /// Pointer to the kernel blob (HSACO ELF bytes for ElfHsa, source for others).
    const void*  pBlob;

    /// Byte size of the blob.
    size_t       blobSizeInBytes;

    /// Thread group dimensions — must match what the kernel was compiled with.
    struct
    {
        uint32_t dimx;
        uint32_t dimy;
        uint32_t dimz;
    } threadsPerGroup;

    /// Input blob language / format.
    AmdShaderType shType;

    /// Optional build options string (e.g. "-D__HIPCC_RTC__" for HIP source).
    /// For pre-compiled ElfHsa blobs this is typically NULL / zero.
    const void*  pOptions;
    size_t       optionSizeInBytes;

    /// When the blob contains multiple kernels, name of the kernel entry point
    /// to compile.  NULL selects the only (or first) kernel.
    const char*  pKernelName;
};

// ELF pipeline info (used by CreateComputePipelineFromElf).
struct AmdExtD3DPipelineElfInfo : AmdExtD3DCreateInfo
{
    const void* pElfBinary;
    size_t      elfSizeInBytes;
    struct { uint32_t width; uint32_t height; uint32_t depth; } threadsPerGroup;
};

// Feature-support bitfield returned by CheckExtFeatureSupport(Flags).
struct AmdExtD3DCheckFeatureSupportFlags
{
    union
    {
        struct
        {
            uint32_t depthBoundsTest                : 1;
            uint32_t abortCreateIfPipelineNotCached : 1;
            uint32_t rectangleListPrimitive         : 1;
            uint32_t pipelinePalElf                 : 1;
            uint32_t pipelineHsaElf                 : 1;
            uint32_t workTiling                     : 1;
            uint32_t pipelineCrossCompileHip        : 1;
            uint32_t pipelineCrossCompileOcl        : 1;
            uint32_t pipelineCrossCompileHlsl       : 1;
            uint32_t pipelineCrossCompileElfHsa     : 1;
            uint32_t waveMatrixSupported            : 1;
            uint32_t reserved                       : 21;
        };
        uint32_t all;
    };
};

// ---------------------------------------------------------------------------
// IAmdExtD3DDevice — base extension interface (v0).
// GUIDs and vtable layout match dxcp/public/AmdExtD3DDeviceApi.h exactly.
// ---------------------------------------------------------------------------
MIDL_INTERFACE("8104C0FC-7413-410F-8E83-AA617E908648")
IAmdExtD3DDevice : public IUnknown
{
public:
    virtual HRESULT STDMETHODCALLTYPE CreateGraphicsPipelineState(
        const AmdExtD3DCreateInfo*                pAmdExtCreateInfo,
        const D3D12_GRAPHICS_PIPELINE_STATE_DESC* pDesc,
        REFIID                                    riid,
        void**                                    ppPipelineState) = 0;
};

// v1 — marker / debug
MIDL_INTERFACE("4BBCAF68-EAF7-4FA4-B653-CB458C334A4E")
IAmdExtD3DDevice1 : public IAmdExtD3DDevice
{
public:
    virtual void STDMETHODCALLTYPE PushMarker(ID3D12GraphicsCommandList*, const char*) = 0;
    virtual void STDMETHODCALLTYPE PopMarker(ID3D12GraphicsCommandList*) = 0;
    virtual void STDMETHODCALLTYPE SetMarker(ID3D12GraphicsCommandList*, const char*) = 0;
};

// v2 — feature query
MIDL_INTERFACE("A7BECF5D-2930-4FDA-8EEE-C797D8A52B7E")
IAmdExtD3DDevice2 : public IAmdExtD3DDevice1
{
public:
    virtual HRESULT STDMETHODCALLTYPE CheckExtFeatureSupport(
        uint32_t featureType,
        void*    pFeatureData,
        size_t   featureDataSize) const = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateComputePipelineState(
        const AmdExtD3DCreateInfo*               pAmdExtCreateInfo,
        const D3D12_COMPUTE_PIPELINE_STATE_DESC* pDesc,
        REFIID                                   refiid,
        void**                                   ppPipelineState) = 0;
};

// v3
MIDL_INTERFACE("397E3533-111E-4A9D-A171-2BAE8EF6CB24")
IAmdExtD3DDevice3 : public IAmdExtD3DDevice2
{
public:
    virtual HRESULT STDMETHODCALLTYPE CreatePipelineState(
        const AmdExtD3DCreateInfo*              pAmdExtCreateInfo,
        const D3D12_PIPELINE_STATE_STREAM_DESC* pDesc,
        REFIID                                  riid,
        void**                                  ppPipelineState) = 0;
};

// v4
MIDL_INTERFACE("BE9A8C6A-868E-490D-8FBF-29DAC2650F3B")
IAmdExtD3DDevice4 : public IAmdExtD3DDevice3
{
public:
    virtual void STDMETHODCALLTYPE SetPrimitiveTopology(
        ID3D12GraphicsCommandList* pCmdList,
        uint32_t                   topology) = 0;
};

// v5 — adds CreateComputePipelineFromElf + SetKernelArguments
MIDL_INTERFACE("BDC14598-B7D2-4A8D-9CA5-67848E2AF745")
IAmdExtD3DDevice5 : public IAmdExtD3DDevice4
{
public:
    virtual HRESULT STDMETHODCALLTYPE CreateComputePipelineFromElf(
        AmdExtD3DPipelineElfInfo* pAmdExtCreateInfo,
        REFIID                    refiid,
        void**                    ppPipelineState) = 0;

    // Bind kernel arguments on a command list before Dispatch().
    virtual void STDMETHODCALLTYPE SetKernelArguments(
        ID3D12GraphicsCommandList* pCmdList,
        uint32_t                   firstArg,
        uint32_t                   argCount,
        const void* const*         ppValues) = 0;
};

// v6 — GPURT version queries
MIDL_INTERFACE("F764A768-48B4-46A5-9779-928ED6896D2A")
IAmdExtD3DDevice6 : public IAmdExtD3DDevice5
{
public:
    virtual void STDMETHODCALLTYPE GetGpuRtInterfaceVersion(uint32_t* pVersion) = 0;
    virtual void STDMETHODCALLTYPE GetGpuRtBinaryVersion(uint32_t* pVersion) = 0;
};

// v7 — adds CreateComputePipelineCrossCompile
MIDL_INTERFACE("FEE37AFC-3C50-4ABF-86CC-1622349B29C0")
IAmdExtD3DDevice7 : public IAmdExtD3DDevice6
{
public:
    // Create a D3D12 compute PSO from a cross-compiled kernel blob.
    virtual HRESULT STDMETHODCALLTYPE CreateComputePipelineCrossCompile(
        const AmdExtD3DPipelineCrossCompileInfo* pInfo,
        REFIID                                   riid,
        void**                                   ppPipelineState) = 0;
};

// v8 — adds GetWaveMatrixProperties (matches DXCP public API IAmdExtD3DDevice8)
MIDL_INTERFACE("F714E11A-B54E-4E0F-ABC5-DF58B18133D1")
IAmdExtD3DDevice8 : public IAmdExtD3DDevice7
{
public:
    virtual HRESULT STDMETHODCALLTYPE GetWaveMatrixProperties(
        size_t*  pCount,
        void*    pProperties) = 0;  // AmdExtWaveMatrixProperties
};

// v9 — adds GetVideoProcessorInfo
MIDL_INTERFACE("E8F3A2D1-5B6C-4D9E-A7F2-3C8B1E0D4F96")
IAmdExtD3DDevice9 : public IAmdExtD3DDevice8
{
public:
    virtual HRESULT STDMETHODCALLTYPE GetVideoProcessorInfo(void* pInfo) = 0;
};

// v10 — adds SetPalElfComputeUserData for PAL ELF dispatch (CmdSetUserData bypass)
// Equivalent to metacommand CmdSetUserData(Compute, 0, 2, {cbVaLo, cbVaHi}).
MIDL_INTERFACE("B3C4D5E6-F7A8-90BC-DEF0-123456789ABC")
IAmdExtD3DDevice10 : public IAmdExtD3DDevice9
{
public:
    virtual void STDMETHODCALLTYPE SetPalElfComputeUserData(
        ID3D12GraphicsCommandList* pCmdList,
        uint32_t                   firstSlot,
        uint32_t                   slotCount,
        const uint32_t*            pData) = 0;

    // PAL ELF full dispatch: embeds CbData in command buffer (CmdAllocateEmbeddedData),
    // sets user data 0:1 = CbData VA, dispatches. Matches metacommand dispatch exactly.
    virtual void STDMETHODCALLTYPE DispatchPalElf(
        ID3D12GraphicsCommandList* pCmdList,
        const uint32_t*            pCbDataDwords,
        uint32_t                   cbDataDwords,
        uint32_t                   dispatchX,
        uint32_t                   dispatchY,
        uint32_t                   dispatchZ) = 0;
};

// ---------------------------------------------------------------------------
// IAmdExtD3DFactory — top-level factory, obtained from AmdExtD3DCreateInterface.
// ---------------------------------------------------------------------------
MIDL_INTERFACE("014937EC-9288-446F-A9AC-D75A8E3A984F")
IAmdExtD3DFactory : public IUnknown
{
public:
    virtual HRESULT STDMETHODCALLTYPE CreateInterface(
        IUnknown* pOuter,
        REFIID    riid,
        void**    ppvObject) = 0;
};

// ---------------------------------------------------------------------------
// AmdExtD3DCreateInterface — exported from the AMD driver DLL (amdxc64.dll).
// Calling convention is __cdecl, NOT __stdcall/__cdecl.
// Using WINAPI (__stdcall) would corrupt the stack (GS cookie violation).
// ---------------------------------------------------------------------------
using PFN_AmdExtD3DCreateInterface =
    HRESULT(__cdecl*)(IUnknown* pOuter, REFIID riid, void** ppvObject);

#endif // MIGRAPHX_AMDCC_USE_SDK_HEADER
