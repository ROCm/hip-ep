/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Vulkan compute context — singleton managing VkInstance, VkDevice, VkQueue.
 * Provides command buffer allocation and fence-based synchronization.
 */

#include "vulkan_kernels.h"

#define VK_API_VERSION_1_2 VK_MAKE_API_VERSION(0, 1, 2, 0)
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------
// Internal context structure
// ---------------------------------------------------------------------------

struct VulkanContext {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          queue          = VK_NULL_HANDLE;
    VkCommandPool    commandPool    = VK_NULL_HANDLE;
    uint32_t         queueFamily    = 0;

    VkPhysicalDeviceMemoryProperties memProps{};
    bool initialized = false;
    bool hasExternalMemoryHost = false;
    VkDeviceSize minImportAlignment = 0;
};

static VulkanContext g_ctx;
static std::mutex    g_mutex;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define VK_CHECK(call, msg)                                                    \
    do {                                                                        \
        VkResult _r = (call);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            fprintf(stderr, "[vulkan_kernels] %s failed: VkResult=%d\n",        \
                    (msg), (int)_r);                                            \
            return -1;                                                          \
        }                                                                       \
    } while (0)

static uint32_t findMemoryType(uint32_t typeFilter,
                               VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < g_ctx.memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) &&
            (g_ctx.memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" int vulkan_context_init(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx.initialized) return 0;

    // --- Instance ---
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "onnx-hipdnn-ep-vulkan";
    appInfo.apiVersion         = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instCI{};
    instCI.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;

    VK_CHECK(vkCreateInstance(&instCI, nullptr, &g_ctx.instance),
             "vkCreateInstance");

    // --- Physical device (pick first compute-capable GPU) ---
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(g_ctx.instance, &devCount, nullptr);
    if (devCount == 0) {
        fprintf(stderr, "[vulkan_kernels] No Vulkan devices found\n");
        return -1;
    }

    VkPhysicalDevice *devs = new VkPhysicalDevice[devCount];
    vkEnumeratePhysicalDevices(g_ctx.instance, &devCount, devs);
    g_ctx.physicalDevice = devs[0]; // pick first device
    delete[] devs;

    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(g_ctx.physicalDevice, &devProps);
    fprintf(stderr, "[vulkan_kernels] Using device: %s\n",
            devProps.deviceName);

    vkGetPhysicalDeviceMemoryProperties(g_ctx.physicalDevice, &g_ctx.memProps);

    // --- Queue family (compute) ---
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physicalDevice, &qfCount,
                                             nullptr);
    VkQueueFamilyProperties *qfProps = new VkQueueFamilyProperties[qfCount];
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physicalDevice, &qfCount,
                                             qfProps);

    g_ctx.queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g_ctx.queueFamily = i;
            break;
        }
    }
    delete[] qfProps;

    if (g_ctx.queueFamily == UINT32_MAX) {
        fprintf(stderr, "[vulkan_kernels] No compute queue family found\n");
        return -1;
    }

    // --- Check for VK_EXT_external_memory_host support ---
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(g_ctx.physicalDevice, nullptr,
                                         &extCount, nullptr);
    std::vector<VkExtensionProperties> availExts(extCount);
    vkEnumerateDeviceExtensionProperties(g_ctx.physicalDevice, nullptr,
                                         &extCount, availExts.data());

    g_ctx.hasExternalMemoryHost = false;
    for (const auto &ext : availExts) {
        if (strcmp(ext.extensionName, "VK_EXT_external_memory_host") == 0) {
            g_ctx.hasExternalMemoryHost = true;
            break;
        }
    }

    if (g_ctx.hasExternalMemoryHost) {
        // Query minimum alignment for imported host pointers
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostMemProps{};
        hostMemProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &hostMemProps;
        vkGetPhysicalDeviceProperties2(g_ctx.physicalDevice, &props2);

        g_ctx.minImportAlignment = hostMemProps.minImportedHostPointerAlignment;
        fprintf(stderr, "[vulkan_kernels] VK_EXT_external_memory_host: YES "
                "(alignment=%llu)\n", (unsigned long long)g_ctx.minImportAlignment);
    } else {
        fprintf(stderr, "[vulkan_kernels] VK_EXT_external_memory_host: NO "
                "(GPU-pointer mode will fall back to copies)\n");
    }

    // --- Logical device ---
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCI{};
    queueCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueFamilyIndex = g_ctx.queueFamily;
    queueCI.queueCount       = 1;
    queueCI.pQueuePriorities = &queuePriority;

    // Use Vulkan 1.2 features (includes 16-bit and 8-bit storage)
    VkPhysicalDeviceVulkan12Features vk12Features{};
    vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12Features.storageBuffer8BitAccess = VK_TRUE;
    vk12Features.shaderFloat16 = VK_TRUE;
    vk12Features.shaderInt8 = VK_TRUE;

    VkPhysicalDeviceVulkan11Features vk11Features{};
    vk11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vk11Features.pNext = &vk12Features;
    vk11Features.storageBuffer16BitAccess = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vk11Features;

    // Enable external memory host extension if available
    const char *enabledExts[2];
    uint32_t numExts = 0;
    if (g_ctx.hasExternalMemoryHost) {
        enabledExts[numExts++] = "VK_KHR_external_memory";
        enabledExts[numExts++] = "VK_EXT_external_memory_host";
    }

    VkDeviceCreateInfo devCI{};
    devCI.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.pNext                   = &features2;
    devCI.queueCreateInfoCount    = 1;
    devCI.pQueueCreateInfos       = &queueCI;
    devCI.enabledExtensionCount   = numExts;
    devCI.ppEnabledExtensionNames = enabledExts;

    VK_CHECK(vkCreateDevice(g_ctx.physicalDevice, &devCI, nullptr,
                            &g_ctx.device),
             "vkCreateDevice");

    vkGetDeviceQueue(g_ctx.device, g_ctx.queueFamily, 0, &g_ctx.queue);

    // --- Command pool ---
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = g_ctx.queueFamily;
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(g_ctx.device, &poolCI, nullptr,
                                 &g_ctx.commandPool),
             "vkCreateCommandPool");

    g_ctx.initialized = true;
    fprintf(stderr, "[vulkan_kernels] Vulkan context initialized\n");
    return 0;
}

extern "C" void vulkan_context_destroy(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ctx.initialized) return;

    if (g_ctx.device) {
        vkDeviceWaitIdle(g_ctx.device);
        if (g_ctx.commandPool)
            vkDestroyCommandPool(g_ctx.device, g_ctx.commandPool, nullptr);
        vkDestroyDevice(g_ctx.device, nullptr);
    }
    if (g_ctx.instance)
        vkDestroyInstance(g_ctx.instance, nullptr);

    g_ctx = VulkanContext{};
    fprintf(stderr, "[vulkan_kernels] Vulkan context destroyed\n");
}

// ---------------------------------------------------------------------------
// Internal accessors (used by vulkan_matmul_nbits.cpp)
// ---------------------------------------------------------------------------

VkDevice            vk_get_device()       { return g_ctx.device; }
VkQueue             vk_get_queue()        { return g_ctx.queue; }
VkCommandPool       vk_get_command_pool() { return g_ctx.commandPool; }
VkPhysicalDevice    vk_get_physical_device() { return g_ctx.physicalDevice; }
uint32_t            vk_find_memory_type(uint32_t filter,
                                        VkMemoryPropertyFlags props) {
    return findMemoryType(filter, props);
}

float vk_get_timestamp_period() {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_ctx.physicalDevice, &props);
    return props.limits.timestampPeriod; // nanoseconds per tick
}

bool vk_has_external_memory_host() { return g_ctx.hasExternalMemoryHost; }
VkDeviceSize vk_get_min_import_alignment() { return g_ctx.minImportAlignment; }

extern "C" size_t vulkan_get_host_pointer_alignment(void) {
    return (size_t)g_ctx.minImportAlignment;
}
