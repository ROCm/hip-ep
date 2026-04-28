/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Host-side Vulkan dispatch for MatMulNBits compute shader.
 *
 * Creates a VkComputePipeline from embedded SPIR-V, uploads buffers,
 * dispatches the shader, and downloads the result.
 *
 * This is an allocate-per-call implementation (correctness-first).
 * A production version would cache pipelines and reuse device buffers.
 */

#include "vulkan_kernels.h"
#include "matmul_nbits_spv.h"       // naive kernel SPIR-V
#include "matmul_nbits_vec_spv.h"   // optimized GEMV kernel SPIR-V

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Internal accessors from vulkan_device.cpp
extern VkDevice         vk_get_device();
extern VkQueue          vk_get_queue();
extern VkCommandPool    vk_get_command_pool();
extern VkPhysicalDevice vk_get_physical_device();
extern uint32_t         vk_find_memory_type(uint32_t filter,
                                            VkMemoryPropertyFlags props);
extern float            vk_get_timestamp_period();
extern bool             vk_has_external_memory_host();
extern VkDeviceSize     vk_get_min_import_alignment();

#include <unordered_map>

// Last GPU kernel time (set after each dispatch)
static double s_last_kernel_time_ms = 0.0;

extern "C" double vulkan_get_last_kernel_time_ms(void) {
    return s_last_kernel_time_ms;
}

#define VK_CHECK_RET(call, msg)                                                \
    do {                                                                        \
        VkResult _r = (call);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            fprintf(stderr, "[vulkan_matmul_nbits] %s: VkResult=%d\n",          \
                    (msg), (int)_r);                                            \
            result = -1; goto cleanup;                                          \
        }                                                                       \
    } while (0)

// Push-constant layout — must match the shader exactly
struct MatMulNBitsPushConstants {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t block_size;
    int32_t batch_count;
    int32_t has_zp;
    int32_t has_bias;
};

// ---------------------------------------------------------------------------
// Helper: create a device-local buffer, upload host data
// ---------------------------------------------------------------------------
struct VkBuf {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
};

static int createBufferAndUpload(VkDevice device, VkQueue queue,
                                 VkCommandPool cmdPool,
                                 const void *hostData, VkDeviceSize dataSize,
                                 VkBufferUsageFlags usage,
                                 VkBuf &out) {
    out.size = dataSize;
    if (dataSize == 0) return 0;

    // Create host-visible buffer (UMA-friendly: skip staging for simplicity)
    VkBufferCreateInfo bufCI{};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size  = dataSize;
    bufCI.usage = usage;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &bufCI, nullptr, &out.buffer);
    if (r != VK_SUCCESS) return -1;

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, out.buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReqs.size;
    allocInfo.memoryTypeIndex = vk_find_memory_type(
        memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (allocInfo.memoryTypeIndex == UINT32_MAX) return -1;

    r = vkAllocateMemory(device, &allocInfo, nullptr, &out.memory);
    if (r != VK_SUCCESS) return -1;

    vkBindBufferMemory(device, out.buffer, out.memory, 0);

    // Map and copy
    if (hostData) {
        void *mapped = nullptr;
        vkMapMemory(device, out.memory, 0, dataSize, 0, &mapped);
        memcpy(mapped, hostData, dataSize);
        vkUnmapMemory(device, out.memory);
    }

    return 0;
}

static void destroyBuffer(VkDevice device, VkBuf &buf) {
    if (buf.buffer) vkDestroyBuffer(device, buf.buffer, nullptr);
    if (buf.memory) vkFreeMemory(device, buf.memory, nullptr);
    buf = {};
}

// ---------------------------------------------------------------------------
// Cached pipeline (created once, reused across calls)
// ---------------------------------------------------------------------------
static VkPipeline            s_pipeline_naive = VK_NULL_HANDLE;
static VkPipeline            s_pipeline_gemv  = VK_NULL_HANDLE;
static VkPipelineLayout      s_pipelineLayout = VK_NULL_HANDLE;
static VkDescriptorSetLayout s_dsLayout       = VK_NULL_HANDLE;
static VkDescriptorPool      s_dsPool         = VK_NULL_HANDLE;
static VkShaderModule        s_shaderNaive    = VK_NULL_HANDLE;
static VkShaderModule        s_shaderGemv     = VK_NULL_HANDLE;
static VkQueryPool           s_queryPool      = VK_NULL_HANDLE;

static int createPipelineFromSpirv(VkDevice device, const uint32_t *code,
                                   size_t codeSize, VkShaderModule *outModule,
                                   VkPipeline *outPipeline) {
    VkShaderModuleCreateInfo smCI{};
    smCI.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = codeSize;
    smCI.pCode    = code;
    VkResult r = vkCreateShaderModule(device, &smCI, nullptr, outModule);
    if (r != VK_SUCCESS) return -1;

    VkComputePipelineCreateInfo cpCI{};
    cpCI.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpCI.layout = s_pipelineLayout;
    cpCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpCI.stage.module = *outModule;
    cpCI.stage.pName  = "main";
    r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpCI, nullptr,
                                 outPipeline);
    return (r == VK_SUCCESS) ? 0 : -1;
}

static int ensurePipeline(VkDevice device) {
    if (s_pipeline_naive != VK_NULL_HANDLE) return 0;

    VkResult r;

    // Descriptor set layout (6 storage buffers) — shared by both pipelines
    VkDescriptorSetLayoutBinding bindings[6]{};
    for (int i = 0; i < 6; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsLayoutCI{};
    dsLayoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsLayoutCI.bindingCount = 6;
    dsLayoutCI.pBindings    = bindings;
    r = vkCreateDescriptorSetLayout(device, &dsLayoutCI, nullptr, &s_dsLayout);
    if (r != VK_SUCCESS) return -1;

    // Push constant range
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(MatMulNBitsPushConstants);

    // Pipeline layout (shared)
    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount         = 1;
    plCI.pSetLayouts            = &s_dsLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges    = &pcRange;
    r = vkCreatePipelineLayout(device, &plCI, nullptr, &s_pipelineLayout);
    if (r != VK_SUCCESS) return -1;

    // Create both pipelines
    if (createPipelineFromSpirv(device, matmul_nbits_spv, matmul_nbits_spv_size,
                                &s_shaderNaive, &s_pipeline_naive) != 0)
        return -1;

    if (createPipelineFromSpirv(device, matmul_nbits_vec_spv, matmul_nbits_vec_spv_size,
                                &s_shaderGemv, &s_pipeline_gemv) != 0)
        return -1;

    // Descriptor pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 6 * 16;

    VkDescriptorPoolCreateInfo dpCI{};
    dpCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpCI.maxSets       = 16;
    dpCI.poolSizeCount = 1;
    dpCI.pPoolSizes    = &poolSize;
    dpCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    r = vkCreateDescriptorPool(device, &dpCI, nullptr, &s_dsPool);
    if (r != VK_SUCCESS) return -1;

    // Timestamp query pool
    VkQueryPoolCreateInfo qpCI{};
    qpCI.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpCI.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpCI.queryCount = 2;
    r = vkCreateQueryPool(device, &qpCI, nullptr, &s_queryPool);
    if (r != VK_SUCCESS) return -1;

    return 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" int vulkan_matmul_nbits(
    void *stream,
    const void *A, const void *B, const void *scales,
    const void *zero_points, const void *bias, void *output,
    int64_t M, int64_t N, int64_t K, int64_t batch_count,
    int64_t bits, int64_t block_size, int64_t element_size_bytes) {

    (void)stream; // Vulkan manages its own queue

    // --- Validation (same as HIP path) ---
    if (bits != 4) {
        fprintf(stderr, "[vulkan_matmul_nbits] only bits=4 supported, got %lld\n",
                (long long)bits);
        return -1;
    }
    if (element_size_bytes != 2) {
        fprintf(stderr, "[vulkan_matmul_nbits] only fp16 supported, got elem_size=%lld\n",
                (long long)element_size_bytes);
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0) return 0;

    // --- Ensure Vulkan context + pipeline ---
    if (vulkan_context_init() != 0) return -1;

    VkDevice      device  = vk_get_device();
    VkQueue       queue   = vk_get_queue();
    VkCommandPool cmdPool = vk_get_command_pool();

    if (ensurePipeline(device) != 0) return -1;

    int result = 0;

    // --- Compute buffer sizes ---
    int64_t k_blocks = (K + block_size - 1) / block_size;

    VkDeviceSize sizeA      = (VkDeviceSize)(batch_count * M * K * 2);        // FP16
    VkDeviceSize sizeB      = (VkDeviceSize)(N * (K / 2));                    // packed uint8
    VkDeviceSize sizeScales = (VkDeviceSize)(N * k_blocks * 2);               // FP16
    VkDeviceSize sizeZP     = zero_points ? (VkDeviceSize)(N * k_blocks) : 4; // uint8 or dummy
    VkDeviceSize sizeBias   = bias ? (VkDeviceSize)(N * 2) : 4;               // FP16 or dummy
    VkDeviceSize sizeOut    = (VkDeviceSize)(batch_count * M * N * 2);        // FP16

    // --- Create buffers ---
    VkBuf bufA{}, bufB{}, bufScales{}, bufZP{}, bufBias{}, bufOut{};
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (createBufferAndUpload(device, queue, cmdPool, A, sizeA, usage, bufA) != 0 ||
        createBufferAndUpload(device, queue, cmdPool, B, sizeB, usage, bufB) != 0 ||
        createBufferAndUpload(device, queue, cmdPool, scales, sizeScales, usage, bufScales) != 0 ||
        createBufferAndUpload(device, queue, cmdPool, zero_points, sizeZP, usage, bufZP) != 0 ||
        createBufferAndUpload(device, queue, cmdPool, bias, sizeBias, usage, bufBias) != 0 ||
        createBufferAndUpload(device, queue, cmdPool, nullptr, sizeOut, usage, bufOut) != 0) {
        result = -1;
        goto cleanup;
    }

    // --- Descriptor set ---
    {
        VkDescriptorSetAllocateInfo dsAI{};
        dsAI.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAI.descriptorPool     = s_dsPool;
        dsAI.descriptorSetCount = 1;
        dsAI.pSetLayouts        = &s_dsLayout;
        VK_CHECK_RET(vkAllocateDescriptorSets(device, &dsAI, &ds),
                     "allocate descriptor set");

        VkBuf *bufs[] = {&bufA, &bufB, &bufScales, &bufZP, &bufBias, &bufOut};
        VkWriteDescriptorSet writes[6]{};
        VkDescriptorBufferInfo bufInfos[6]{};

        for (int i = 0; i < 6; i++) {
            bufInfos[i].buffer = bufs[i]->buffer;
            bufInfos[i].offset = 0;
            bufInfos[i].range  = bufs[i]->size;

            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = ds;
            writes[i].dstBinding      = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &bufInfos[i];
        }
        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }

    // --- Command buffer ---
    {
        VkCommandBufferAllocateInfo cbAI{};
        cbAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbAI.commandPool        = cmdPool;
        cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAI.commandBufferCount = 1;
        VK_CHECK_RET(vkAllocateCommandBuffers(device, &cbAI, &cmd),
                     "allocate command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Select pipeline and grid dimensions based on M
        bool use_gemv = (M == 1) && (K % 2 == 0);
        VkPipeline activePipeline = use_gemv ? s_pipeline_gemv : s_pipeline_naive;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, activePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s_pipelineLayout, 0, 1, &ds, 0, nullptr);

        MatMulNBitsPushConstants pc{};
        pc.M           = (int32_t)M;
        pc.N           = (int32_t)N;
        pc.K           = (int32_t)K;
        pc.block_size  = (int32_t)block_size;
        pc.batch_count = (int32_t)batch_count;
        pc.has_zp      = zero_points ? 1 : 0;
        pc.has_bias    = bias ? 1 : 0;

        vkCmdPushConstants(cmd, s_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t gx, gy, gz;
        if (use_gemv) {
            // GEMV: each workgroup of 256 threads computes 4 output N values
            gx = ((uint32_t)N + 3) / 4;     // ceil(N / NUM_ROWS)
            gy = (uint32_t)batch_count;
            gz = 1;
        } else {
            // Naive: 16x16 thread block, 1 thread per output element
            gx = ((uint32_t)N + 15) / 16;
            gy = ((uint32_t)M + 15) / 16;
            gz = (uint32_t)batch_count;
        }

        // Reset and write timestamp BEFORE dispatch
        vkCmdResetQueryPool(cmd, s_queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            s_queryPool, 0);

        vkCmdDispatch(cmd, gx, gy, gz);

        // Write timestamp AFTER dispatch
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            s_queryPool, 1);

        vkEndCommandBuffer(cmd);
    }

    // --- Submit and wait ---
    {
        VkFenceCreateInfo fCI{};
        fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK_RET(vkCreateFence(device, &fCI, nullptr, &fence), "create fence");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        VK_CHECK_RET(vkQueueSubmit(queue, 1, &si, fence), "queue submit");
        VK_CHECK_RET(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
                     "wait for fence");
    }

    // --- Read GPU timestamps ---
    {
        uint64_t timestamps[2] = {0, 0};
        VkResult qr = vkGetQueryPoolResults(
            device, s_queryPool, 0, 2,
            sizeof(timestamps), timestamps, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (qr == VK_SUCCESS && timestamps[1] > timestamps[0]) {
            float period = vk_get_timestamp_period(); // ns per tick
            double elapsed_ns = (double)(timestamps[1] - timestamps[0]) * (double)period;
            s_last_kernel_time_ms = elapsed_ns / 1e6;
        } else {
            s_last_kernel_time_ms = 0.0;
        }
    }

    // --- Read back output ---
    {
        void *mapped = nullptr;
        vkMapMemory(device, bufOut.memory, 0, sizeOut, 0, &mapped);
        memcpy(output, mapped, sizeOut);
        vkUnmapMemory(device, bufOut.memory);
    }

cleanup:
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (cmd)   vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
    if (ds)    vkFreeDescriptorSets(device, s_dsPool, 1, &ds);
    destroyBuffer(device, bufA);
    destroyBuffer(device, bufB);
    destroyBuffer(device, bufScales);
    destroyBuffer(device, bufZP);
    destroyBuffer(device, bufBias);
    destroyBuffer(device, bufOut);

    return result;
}

// ---------------------------------------------------------------------------
// GPU-RESIDENT POINTER MODE (zero-overhead for EP integration)
// ---------------------------------------------------------------------------
// Uses VK_EXT_external_memory_host to import GPU-accessible pointers
// (from hipMalloc on UMA) directly as VkBuffers — no copies.

// Cache imported buffers by pointer address to avoid re-importing
static std::unordered_map<uintptr_t, VkBuf> s_importCache;

static int importHostPointerAsBuffer(VkDevice device, const void *hostPtr,
                                     VkDeviceSize size, VkBuf &out) {
    if (!hostPtr || size == 0) {
        out = {};
        return 0;
    }

    // Cache lookup disabled in test mode to avoid stale pointers
    // In production (EP flow), pointers are stable across inferences
    // and caching is safe. TODO: add explicit cache invalidation API.

    // Align pointer down to minImportedHostPointerAlignment
    VkDeviceSize alignment = vk_get_min_import_alignment();
    uintptr_t rawAddr = (uintptr_t)hostPtr;
    uintptr_t alignedAddr = rawAddr & ~(alignment - 1);
    VkDeviceSize offset = rawAddr - alignedAddr;
    VkDeviceSize alignedSize = (size + offset + alignment - 1) & ~(alignment - 1);

    // Create buffer
    VkBufferCreateInfo bufCI{};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size  = alignedSize;
    bufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &bufCI, nullptr, &out.buffer);
    if (r != VK_SUCCESS) return -1;

    // Get memory requirements
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, out.buffer, &memReqs);

    // Import the host pointer as external memory
    VkImportMemoryHostPointerInfoEXT importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    importInfo.pHostPointer = (void *)alignedAddr;

    // Find a memory type that supports host import
    uint32_t memType = vk_find_memory_type(
        memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    if (memType == UINT32_MAX) {
        // Try any memory type
        memType = vk_find_memory_type(memReqs.memoryTypeBits, 0);
    }
    if (memType == UINT32_MAX) {
        vkDestroyBuffer(device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return -1;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &importInfo;
    allocInfo.allocationSize = alignedSize;
    allocInfo.memoryTypeIndex = memType;

    r = vkAllocateMemory(device, &allocInfo, nullptr, &out.memory);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return -1;
    }

    // Bind buffer at offset 0 — the import already covers the aligned region
    // and the buffer was created with alignedSize (from the aligned base)
    vkBindBufferMemory(device, out.buffer, out.memory, 0);

    // Create a second buffer at the correct offset if pointer wasn't page-aligned
    if (offset > 0) {
        vkDestroyBuffer(device, out.buffer, nullptr);
        VkBufferCreateInfo bufCI2{};
        bufCI2.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufCI2.size  = size;
        bufCI2.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufCI2.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        r = vkCreateBuffer(device, &bufCI2, nullptr, &out.buffer);
        if (r != VK_SUCCESS) return -1;
        vkBindBufferMemory(device, out.buffer, out.memory, offset);
    }
    out.size = size;

    return 0;
}

#define VK_CHECK_GPU(call, msg)                                                \
    do {                                                                        \
        VkResult _r = (call);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            fprintf(stderr, "[vulkan_matmul_nbits_gpu] %s: VkResult=%d\n",      \
                    (msg), (int)_r);                                            \
            result = -1; goto cleanup_gpu;                                      \
        }                                                                       \
    } while (0)

extern "C" int vulkan_matmul_nbits_gpu(
    const void *A_gpu, const void *B_gpu, const void *scales_gpu,
    const void *zp_gpu, const void *bias_gpu, void *output_gpu,
    int64_t M, int64_t N, int64_t K, int64_t batch_count,
    int64_t bits, int64_t block_size, int64_t element_size_bytes) {

    // Validation
    if (bits != 4 || element_size_bytes != 2) return -1;
    if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0) return 0;

    if (vulkan_context_init() != 0) return -1;

    // If external memory host not available, fall back to host-pointer path
    if (!vk_has_external_memory_host()) {
        fprintf(stderr, "[vulkan_matmul_nbits_gpu] No VK_EXT_external_memory_host, "
                "falling back to copy mode\n");
        return vulkan_matmul_nbits(nullptr, A_gpu, B_gpu, scales_gpu, zp_gpu,
                                    bias_gpu, output_gpu, M, N, K, batch_count,
                                    bits, block_size, element_size_bytes);
    }

    VkDevice      device  = vk_get_device();
    VkQueue       queue   = vk_get_queue();
    VkCommandPool cmdPool = vk_get_command_pool();

    if (ensurePipeline(device) != 0) return -1;

    int result = 0;

    // Buffer sizes
    int64_t k_blocks = (K + block_size - 1) / block_size;
    VkDeviceSize sizeA      = (VkDeviceSize)(batch_count * M * K * 2);
    VkDeviceSize sizeB      = (VkDeviceSize)(N * (K / 2));
    VkDeviceSize sizeScales = (VkDeviceSize)(N * k_blocks * 2);
    VkDeviceSize sizeZP     = zp_gpu ? (VkDeviceSize)(N * k_blocks) : 4;
    VkDeviceSize sizeBias   = bias_gpu ? (VkDeviceSize)(N * 2) : 4;
    VkDeviceSize sizeOut    = (VkDeviceSize)(batch_count * M * N * 2);

    // Import GPU pointers as Vulkan buffers (cached)
    VkBuf bufA{}, bufB{}, bufScales{}, bufZP{}, bufBias{}, bufOut{};
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    if (importHostPointerAsBuffer(device, A_gpu, sizeA, bufA) != 0 ||
        importHostPointerAsBuffer(device, B_gpu, sizeB, bufB) != 0 ||
        importHostPointerAsBuffer(device, scales_gpu, sizeScales, bufScales) != 0) {
        result = -1; goto cleanup_gpu;
    }

    // Optional buffers — use dummy if NULL
    if (zp_gpu) {
        if (importHostPointerAsBuffer(device, zp_gpu, sizeZP, bufZP) != 0)
            { result = -1; goto cleanup_gpu; }
    } else {
        // Create tiny dummy buffer
        if (createBufferAndUpload(device, queue, cmdPool, nullptr, 4,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, bufZP) != 0)
            { result = -1; goto cleanup_gpu; }
    }
    if (bias_gpu) {
        if (importHostPointerAsBuffer(device, bias_gpu, sizeBias, bufBias) != 0)
            { result = -1; goto cleanup_gpu; }
    } else {
        if (createBufferAndUpload(device, queue, cmdPool, nullptr, 4,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, bufBias) != 0)
            { result = -1; goto cleanup_gpu; }
    }
    if (importHostPointerAsBuffer(device, output_gpu, sizeOut, bufOut) != 0)
        { result = -1; goto cleanup_gpu; }

    // Descriptor set
    {
        VkDescriptorSetAllocateInfo dsAI{};
        dsAI.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAI.descriptorPool     = s_dsPool;
        dsAI.descriptorSetCount = 1;
        dsAI.pSetLayouts        = &s_dsLayout;
        VK_CHECK_GPU(vkAllocateDescriptorSets(device, &dsAI, &ds),
                     "allocate descriptor set (gpu)");

        VkBuf *bufs[] = {&bufA, &bufB, &bufScales, &bufZP, &bufBias, &bufOut};
        VkWriteDescriptorSet writes[6]{};
        VkDescriptorBufferInfo bufInfos[6]{};
        for (int i = 0; i < 6; i++) {
            bufInfos[i].buffer = bufs[i]->buffer;
            bufInfos[i].offset = 0;
            bufInfos[i].range  = bufs[i]->size;
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = ds;
            writes[i].dstBinding      = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &bufInfos[i];
        }
        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }

    // Command buffer
    {
        VkCommandBufferAllocateInfo cbAI{};
        cbAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbAI.commandPool        = cmdPool;
        cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAI.commandBufferCount = 1;
        VK_CHECK_GPU(vkAllocateCommandBuffers(device, &cbAI, &cmd),
                     "allocate cmd buffer (gpu)");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        bool use_gemv = (M == 1) && (K % 2 == 0);
        VkPipeline activePipeline = use_gemv ? s_pipeline_gemv : s_pipeline_naive;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, activePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s_pipelineLayout, 0, 1, &ds, 0, nullptr);

        MatMulNBitsPushConstants pc{};
        pc.M = (int32_t)M; pc.N = (int32_t)N; pc.K = (int32_t)K;
        pc.block_size = (int32_t)block_size;
        pc.batch_count = (int32_t)batch_count;
        pc.has_zp = zp_gpu ? 1 : 0;
        pc.has_bias = bias_gpu ? 1 : 0;

        vkCmdPushConstants(cmd, s_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t gx, gy, gz;
        if (use_gemv) {
            gx = ((uint32_t)N + 3) / 4;
            gy = (uint32_t)batch_count;
            gz = 1;
        } else {
            gx = ((uint32_t)N + 15) / 16;
            gy = ((uint32_t)M + 15) / 16;
            gz = (uint32_t)batch_count;
        }

        vkCmdResetQueryPool(cmd, s_queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, s_queryPool, 0);
        vkCmdDispatch(cmd, gx, gy, gz);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s_queryPool, 1);

        vkEndCommandBuffer(cmd);
    }

    // Submit and wait
    {
        VkFenceCreateInfo fCI{};
        fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK_GPU(vkCreateFence(device, &fCI, nullptr, &fence), "create fence (gpu)");

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK_GPU(vkQueueSubmit(queue, 1, &si, fence), "queue submit (gpu)");
        VK_CHECK_GPU(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
                     "wait fence (gpu)");
    }

    // Read GPU timestamps
    {
        uint64_t timestamps[2] = {0, 0};
        VkResult qr = vkGetQueryPoolResults(
            device, s_queryPool, 0, 2,
            sizeof(timestamps), timestamps, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (qr == VK_SUCCESS && timestamps[1] > timestamps[0]) {
            float period = vk_get_timestamp_period();
            s_last_kernel_time_ms = (double)(timestamps[1] - timestamps[0]) * (double)period / 1e6;
        }
    }

cleanup_gpu:
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (cmd)   vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
    if (ds)    vkFreeDescriptorSets(device, s_dsPool, 1, &ds);
    destroyBuffer(device, bufA);
    destroyBuffer(device, bufB);
    destroyBuffer(device, bufScales);
    destroyBuffer(device, bufZP);
    destroyBuffer(device, bufBias);
    destroyBuffer(device, bufOut);

    return result;
}
