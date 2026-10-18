#define VOLK_IMPLEMENTATION
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "rhi/vk_context.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sf::rhi {

namespace {

VkImageAspectFlags infer_aspect(VkFormat format) noexcept {
    switch (format) {
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D16_UNORM:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

} // namespace

void VkContext::init(GLFWwindow* window, bool enable_validation) {
    window_ = window;
    VK_CHECK(volkInitialize());

    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "Stellar Forge";
    app_info.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
    app_info.pEngineName = "StellarForgeEngine";
    app_info.engineVersion = VK_MAKE_VERSION(2, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    std::vector<const char*> instance_exts(glfw_exts, glfw_exts + glfw_ext_count);

    std::vector<const char*> layers;
    if (enable_validation) {
        uint32_t prop_count = 0;
        vkEnumerateInstanceLayerProperties(&prop_count, nullptr);
        std::vector<VkLayerProperties> available_layers(prop_count);
        vkEnumerateInstanceLayerProperties(&prop_count, available_layers.data());
        for (const auto& lp : available_layers) {
            if (std::strcmp(lp.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                layers.push_back("VK_LAYER_KHRONOS_validation");
                break;
            }
        }
    }

    VkInstanceCreateInfo inst_ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    inst_ci.pApplicationInfo = &app_info;
    inst_ci.enabledExtensionCount = static_cast<uint32_t>(instance_exts.size());
    inst_ci.ppEnabledExtensionNames = instance_exts.data();
    inst_ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    inst_ci.ppEnabledLayerNames = layers.data();

    VK_CHECK(vkCreateInstance(&inst_ci, nullptr, &instance_));
    volkLoadInstance(instance_);

    VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));

    uint32_t dev_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &dev_count, nullptr));
    if (dev_count == 0) {
        sf::log::fatal("No Vulkan-capable physical devices found.");
    }
    std::vector<VkPhysicalDevice> devices(dev_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &dev_count, devices.data()));

    physical_device_ = devices[0];
    for (VkPhysicalDevice pd : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = pd;
            break;
        }
    }
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &driver_props_;
    vkGetPhysicalDeviceProperties2(physical_device_, &properties);
    device_props_ = properties.properties;
    sf::log::info("Selected GPU: {} (Vulkan {}.{}.{})", device_props_.deviceName,
                  VK_API_VERSION_MAJOR(device_props_.apiVersion),
                  VK_API_VERSION_MINOR(device_props_.apiVersion),
                  VK_API_VERSION_PATCH(device_props_.apiVersion));

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qf_props.data());

    bool found_queue = false;
    for (uint32_t i = 0; i < qf_count; ++i) {
        VkBool32 present_support = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_, &present_support);
        if ((qf_props[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) && present_support) {
            graphics_queue_family_ = i;
            found_queue = true;
            break;
        }
    }
    if (!found_queue) {
        sf::log::fatal("No queue family supporting graphics, compute, and presentation found.");
    }
    timestamp_valid_bits_ = qf_props[graphics_queue_family_].timestampValidBits;
    if (device_props_.limits.timestampPeriod <= 0.0f) timestamp_valid_bits_ = 0;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo q_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q_ci.queueFamilyIndex = graphics_queue_family_;
    q_ci.queueCount = 1;
    q_ci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan13Features feat13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    feat13.dynamicRendering = VK_TRUE;
    feat13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features feat12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    feat12.pNext = &feat13;
    feat12.bufferDeviceAddress = VK_TRUE;
    feat12.descriptorIndexing = VK_TRUE;
    feat12.runtimeDescriptorArray = VK_TRUE;
    feat12.scalarBlockLayout = VK_TRUE;
    feat12.hostQueryReset = VK_TRUE;

    VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat2.pNext = &feat12;
    feat2.features.shaderFloat64 = VK_TRUE;
    feat2.features.samplerAnisotropy = VK_TRUE;
    feat2.features.geometryShader = VK_TRUE;
    feat2.features.fillModeNonSolid = VK_TRUE;
    feat2.features.multiDrawIndirect = VK_TRUE;
    feat2.features.largePoints = VK_TRUE;
    VkPhysicalDeviceFeatures supported_features{};
    vkGetPhysicalDeviceFeatures(physical_device_, &supported_features);
    fragment_stores_enabled_ = supported_features.fragmentStoresAndAtomics == VK_TRUE;
    feat2.features.fragmentStoresAndAtomics = fragment_stores_enabled_ ? VK_TRUE : VK_FALSE;
    dual_source_blend_enabled_ = supported_features.dualSrcBlend == VK_TRUE && device_props_.limits.maxFragmentDualSrcAttachments >= 1;
    feat2.features.dualSrcBlend = dual_source_blend_enabled_ ? VK_TRUE : VK_FALSE;

    std::vector<const char*> dev_exts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    };

    uint32_t extension_count = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, nullptr));
    std::vector<VkExtensionProperties> extensions(extension_count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, extensions.data()));
    auto has_extension = [&](const char* name) {
        return std::any_of(extensions.begin(), extensions.end(), [&](const auto& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
    };
    const char* line_extension = has_extension(VK_KHR_LINE_RASTERIZATION_EXTENSION_NAME)
        ? VK_KHR_LINE_RASTERIZATION_EXTENSION_NAME
        : has_extension(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME) ? VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME : nullptr;
    VkPhysicalDeviceLineRasterizationFeaturesKHR line_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR};
    if (line_extension) {
        VkPhysicalDeviceLineRasterizationFeaturesKHR supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        query.pNext = &supported;
        vkGetPhysicalDeviceFeatures2(physical_device_, &query);
        bresenham_lines_enabled_ = supported.bresenhamLines == VK_TRUE;
        if (bresenham_lines_enabled_) {
            dev_exts.push_back(line_extension);
            line_features.bresenhamLines = VK_TRUE;
            feat13.pNext = &line_features;
        }
    }

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executable_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    pipeline_statistics_enabled_ = false;
    if (std::getenv("STELLAR_SHADER_STATS")) {
        if (has_extension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            query.pNext = &executable_features;
            vkGetPhysicalDeviceFeatures2(physical_device_, &query);
            pipeline_statistics_enabled_ = executable_features.pipelineExecutableInfo == VK_TRUE;
        }
        if (pipeline_statistics_enabled_) {
            dev_exts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
            executable_features.pNext = feat13.pNext;
            feat13.pNext = &executable_features;
        } else {
            sf::log::warn("Pipeline executable statistics are unavailable on this device.");
        }
    }

    VkDeviceCreateInfo dev_ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dev_ci.pNext = &feat2;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos = &q_ci;
    dev_ci.enabledExtensionCount = static_cast<uint32_t>(dev_exts.size());
    dev_ci.ppEnabledExtensionNames = dev_exts.data();

    VK_CHECK(vkCreateDevice(physical_device_, &dev_ci, nullptr, &device_));
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);

    VmaVulkanFunctions vma_funcs{};
    vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo vma_ci{};
    vma_ci.physicalDevice = physical_device_;
    vma_ci.device = device_;
    vma_ci.instance = instance_;
    vma_ci.vulkanApiVersion = VK_API_VERSION_1_3;
    vma_ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vma_ci.pVulkanFunctions = &vma_funcs;
    VK_CHECK(vmaCreateAllocator(&vma_ci, &allocator_));

    VkCommandPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_ci.queueFamilyIndex = graphics_queue_family_;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandBufferAllocateInfo cmd_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;

    VkSemaphoreCreateInfo sem_ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence_ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkQueryPoolCreateInfo qp_ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qp_ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qp_ci.queryCount = MAX_GPU_TIMERS * 2;

    for (auto& f : frames_) {
        VK_CHECK(vkCreateCommandPool(device_, &pool_ci, nullptr, &f.cmd_pool));
        cmd_ai.commandPool = f.cmd_pool;
        VK_CHECK(vkAllocateCommandBuffers(device_, &cmd_ai, &f.cmd));
        VK_CHECK(vkCreateSemaphore(device_, &sem_ci, nullptr, &f.image_acquired_sem));
        VK_CHECK(vkCreateFence(device_, &fence_ci, nullptr, &f.in_flight_fence));
        if (timestamps_supported()) {
            VK_CHECK(vkCreateQueryPool(device_, &qp_ci, nullptr, &f.timestamp_pool));
            vkResetQueryPool(device_, f.timestamp_pool, 0, MAX_GPU_TIMERS * 2);
        }
    }

    VK_CHECK(vkCreateCommandPool(device_, &pool_ci, nullptr, &imm_pool_));
    cmd_ai.commandPool = imm_pool_;
    VK_CHECK(vkAllocateCommandBuffers(device_, &cmd_ai, &imm_cmd_));
    VkFenceCreateInfo imm_fence_ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(device_, &imm_fence_ci, nullptr, &imm_fence_));

    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    };
    VkDescriptorPoolCreateInfo dp_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dp_ci.maxSets = 256;
    dp_ci.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
    dp_ci.pPoolSizes = pool_sizes;
    VK_CHECK(vkCreateDescriptorPool(device_, &dp_ci, nullptr, &imgui_pool_));

    VkSamplerCreateInfo samp_ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samp_ci.magFilter = VK_FILTER_LINEAR;
    samp_ci.minFilter = VK_FILTER_LINEAR;
    samp_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samp_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_ci.maxLod = 16.0f;
    VK_CHECK(vkCreateSampler(device_, &samp_ci, nullptr, &sampler_linear_clamp_));

    samp_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VK_CHECK(vkCreateSampler(device_, &samp_ci, nullptr, &sampler_linear_repeat_));

    samp_ci.magFilter = VK_FILTER_NEAREST;
    samp_ci.minFilter = VK_FILTER_NEAREST;
    samp_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samp_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &samp_ci, nullptr, &sampler_nearest_clamp_));

    create_swapchain();
}

void VkContext::shutdown() {
    wait_idle();
    destroy_swapchain();

    vkDestroySampler(device_, sampler_linear_clamp_, nullptr);
    vkDestroySampler(device_, sampler_linear_repeat_, nullptr);
    vkDestroySampler(device_, sampler_nearest_clamp_, nullptr);

    vkDestroyDescriptorPool(device_, imgui_pool_, nullptr);

    vkDestroyFence(device_, imm_fence_, nullptr);
    vkDestroyCommandPool(device_, imm_pool_, nullptr);

    for (auto& f : frames_) {
        vkDestroyQueryPool(device_, f.timestamp_pool, nullptr);
        vkDestroyFence(device_, f.in_flight_fence, nullptr);
        vkDestroySemaphore(device_, f.image_acquired_sem, nullptr);
        vkDestroyCommandPool(device_, f.cmd_pool, nullptr);
    }

    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyInstance(instance_, nullptr);
}

void VkContext::wait_idle() const {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
}

void VkContext::create_swapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &caps));

    uint32_t fmt_count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &fmt_count, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &fmt_count, formats.data()));

    VkSurfaceFormatKHR chosen_fmt = formats[0];
    for (const auto& f : formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen_fmt = f;
            break;
        }
    }
    swapchain_format_ = chosen_fmt.format;

    if (caps.currentExtent.width != UINT32_MAX) {
        swapchain_extent_ = caps.currentExtent;
    } else {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        swapchain_extent_.width = std::clamp(static_cast<uint32_t>(w), caps.minImageExtent.width, caps.maxImageExtent.width);
        swapchain_extent_.height = std::clamp(static_cast<uint32_t>(h), caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t img_count = std::max(caps.minImageCount + 1, 3u);
    if (caps.maxImageCount > 0 && img_count > caps.maxImageCount) {
        img_count = caps.maxImageCount;
    }

    uint32_t mode_count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &mode_count, nullptr));
    std::vector<VkPresentModeKHR> modes(mode_count);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &mode_count, modes.data()));

    VkPresentModeKHR chosen_mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync_enabled_) {
        bool has_immediate = false;
        bool has_mailbox = false;
        for (VkPresentModeKHR m : modes) {
            if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) has_immediate = true;
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) has_mailbox = true;
        }
        if (has_immediate) {
            chosen_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        } else if (has_mailbox) {
            chosen_mode = VK_PRESENT_MODE_MAILBOX_KHR;
        }
    }
    present_mode_ = chosen_mode;

    VkSwapchainCreateInfoKHR sc_ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sc_ci.surface = surface_;
    sc_ci.minImageCount = img_count;
    sc_ci.imageFormat = chosen_fmt.format;
    sc_ci.imageColorSpace = chosen_fmt.colorSpace;
    sc_ci.imageExtent = swapchain_extent_;
    sc_ci.imageArrayLayers = 1;
    sc_ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sc_ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sc_ci.preTransform = caps.currentTransform;
    sc_ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc_ci.presentMode = chosen_mode;
    sc_ci.clipped = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(device_, &sc_ci, nullptr, &swapchain_));

    uint32_t actual_count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(device_, swapchain_, &actual_count, nullptr));
    swapchain_images_.resize(actual_count);
    VK_CHECK(vkGetSwapchainImagesKHR(device_, swapchain_, &actual_count, swapchain_images_.data()));

    swapchain_views_.resize(actual_count);
    render_finished_sems_.resize(actual_count);
    VkSemaphoreCreateInfo sem_ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    for (uint32_t i = 0; i < actual_count; ++i) {
        VkImageViewCreateInfo iv_ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        iv_ci.image = swapchain_images_[i];
        iv_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv_ci.format = swapchain_format_;
        iv_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        iv_ci.subresourceRange.levelCount = 1;
        iv_ci.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device_, &iv_ci, nullptr, &swapchain_views_[i]));
        VK_CHECK(vkCreateSemaphore(device_, &sem_ci, nullptr, &render_finished_sems_[i]));
    }
}

void VkContext::destroy_swapchain() {
    for (VkSemaphore sem : render_finished_sems_) {
        vkDestroySemaphore(device_, sem, nullptr);
    }
    render_finished_sems_.clear();
    for (VkImageView view : swapchain_views_) {
        vkDestroyImageView(device_, view, nullptr);
    }
    swapchain_views_.clear();
    swapchain_images_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VkContext::recreate_swapchain() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    while (w == 0 || h == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window_, &w, &h);
    }
    wait_idle();
    destroy_swapchain();
    create_swapchain();
    swapchain_dirty_ = false;
}

void VkContext::set_vsync(bool enabled) noexcept {
    if (vsync_enabled_ != enabled) {
        vsync_enabled_ = enabled;
        swapchain_dirty_ = true;
    }
}

void VkContext::resolve_gpu_timers(FrameContext& frame) {
    if (frame.timer_count == 0) {
        return;
    }
    std::vector<uint64_t> timestamps(frame.timer_count * 2, 0);
    VkResult res = vkGetQueryPoolResults(
        device_, frame.timestamp_pool, 0, frame.timer_count * 2,
        timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);
    if (res == VK_SUCCESS) {
        GpuTimingSample sample;
        sample.frame_id = frame.serial;
        double ns_per_tick = static_cast<double>(device_props_.limits.timestampPeriod);
        uint64_t mask = timestamp_valid_bits_ >= 64 ? UINT64_MAX : (uint64_t{1} << timestamp_valid_bits_) - 1;
        for (uint32_t i = 0; i < frame.timer_count; ++i) {
            uint64_t t0 = timestamps[i * 2];
            uint64_t t1 = timestamps[i * 2 + 1];
            double ms = static_cast<double>((t1 - t0) & mask) * ns_per_tick * 1e-6;
            gpu_timings_ms_[frame.timer_names[i]] = ms;
            if (capture_gpu_timings_) sample.timings[frame.timer_names[i]] = ms;
        }
        if (capture_gpu_timings_) captured_gpu_timings_.push_back(std::move(sample));
    }
    vkResetQueryPool(device_, frame.timestamp_pool, 0, MAX_GPU_TIMERS * 2);
    frame.timer_count = 0;
    frame.timer_names.clear();
}

bool VkContext::begin_frame(VkCommandBuffer& out_cmd, uint32_t& out_swapchain_idx, FrameHostTimings* timings) {
    if (swapchain_dirty_) {
        recreate_swapchain();
    }

    FrameContext& frame = frames_[frame_index_];
    {
        ScopedTimer timer(timings ? &timings->fence_wait_ms : nullptr);
        VK_CHECK(vkWaitForFences(device_, 1, &frame.in_flight_fence, VK_TRUE, UINT64_MAX));
    }
    completed_serial_ = std::max(completed_serial_, frame.completion_serial);

    VkResult acq;
    {
        ScopedTimer timer(timings ? &timings->acquire_wait_ms : nullptr);
        acq = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, frame.image_acquired_sem,
                                   VK_NULL_HANDLE, &out_swapchain_idx);
    }
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain();
        return false;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        VK_CHECK(acq);
    }

    resolve_gpu_timers(frame);
    frame.serial = submitted_frames_;
    recording_serial_ = next_serial_++;
    frame.completion_serial = recording_serial_;

    VK_CHECK(vkResetFences(device_, 1, &frame.in_flight_fence));
    VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));

    VkCommandBufferBeginInfo beg_ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beg_ci.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(frame.cmd, &beg_ci));

    out_cmd = frame.cmd;
    return true;
}

void VkContext::end_frame(VkCommandBuffer cmd, uint32_t swapchain_idx, FrameHostTimings* timings) {
    FrameContext& frame = frames_[frame_index_];
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSemaphoreSubmitInfo wait_sem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait_sem.semaphore = frame.image_acquired_sem;
    wait_sem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo sig_sem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig_sem.semaphore = render_finished_sems_[swapchain_idx];
    sig_sem.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

    VkCommandBufferSubmitInfo cmd_si{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_si.commandBuffer = cmd;

    VkSubmitInfo2 sub_info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub_info.waitSemaphoreInfoCount = 1;
    sub_info.pWaitSemaphoreInfos = &wait_sem;
    sub_info.commandBufferInfoCount = 1;
    sub_info.pCommandBufferInfos = &cmd_si;
    sub_info.signalSemaphoreInfoCount = 1;
    sub_info.pSignalSemaphoreInfos = &sig_sem;

    {
        ScopedTimer timer(timings ? &timings->submit_ms : nullptr);
        VK_CHECK(vkQueueSubmit2(graphics_queue_, 1, &sub_info, frame.in_flight_fence));
    }

    VkPresentInfoKHR pres_info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pres_info.waitSemaphoreCount = 1;
    pres_info.pWaitSemaphores = &render_finished_sems_[swapchain_idx];
    pres_info.swapchainCount = 1;
    pres_info.pSwapchains = &swapchain_;
    pres_info.pImageIndices = &swapchain_idx;

    VkResult pres;
    {
        ScopedTimer timer(timings ? &timings->present_ms : nullptr);
        pres = vkQueuePresentKHR(graphics_queue_, &pres_info);
    }
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
        swapchain_dirty_ = true;
    } else if (pres != VK_SUCCESS) {
        VK_CHECK(pres);
    }

    frame_index_ = (frame_index_ + 1) % FRAMES_IN_FLIGHT;
    ++submitted_frames_;
    recording_serial_ = 0;
}

Buffer VkContext::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage,
                                VmaAllocationCreateFlags vma_flags, const void* initial_data) {
    Buffer buf{};
    buf.size = size;

    VkBufferCreateInfo b_ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    b_ci.size = size;
    b_ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    b_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo a_ci{};
    a_ci.usage = memory_usage;
    a_ci.flags = vma_flags;

    VmaAllocationInfo alloc_info{};
    VK_CHECK(vmaCreateBuffer(allocator_, &b_ci, &a_ci, &buf.handle, &buf.allocation, &alloc_info));
    buf.mapped = alloc_info.pMappedData;

    VkBufferDeviceAddressInfo bda_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bda_info.buffer = buf.handle;
    buf.address = vkGetBufferDeviceAddress(device_, &bda_info);

    if (initial_data != nullptr && size > 0) {
        if (buf.mapped != nullptr) {
            std::memcpy(buf.mapped, initial_data, static_cast<size_t>(size));
        } else {
            Buffer staging = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                                           VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT, initial_data);
            immediate_submit([&](VkCommandBuffer cmd) {
                VkBufferCopy copy{0, 0, size};
                vkCmdCopyBuffer(cmd, staging.handle, buf.handle, 1, &copy);
            });
            destroy_buffer(staging);
        }
    }
    return buf;
}

Buffer VkContext::create_host_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* initial_data) {
    return create_buffer(size, usage, VMA_MEMORY_USAGE_AUTO,
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                         initial_data);
}

Buffer VkContext::create_device_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* initial_data) {
    VkBufferUsageFlags flags = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return create_buffer(size, flags, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, initial_data);
}

void VkContext::destroy_buffer(Buffer& buffer) {
    if (buffer.handle != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer.handle, buffer.allocation);
        buffer = {};
    }
}

void VkContext::flush_buffer(const Buffer& buffer, VkDeviceSize offset, VkDeviceSize size) {
    if (buffer.mapped != nullptr) VK_CHECK(vmaFlushAllocation(allocator_, buffer.allocation, offset, size));
}

void VkContext::copy_buffer(VkCommandBuffer cmd, const Buffer& source, const Buffer& destination, VkDeviceSize size,
                            VkDeviceSize source_offset, VkDeviceSize destination_offset) {
    VkBufferCopy region{source_offset, destination_offset, size};
    vkCmdCopyBuffer(cmd, source.handle, destination.handle, 1, &region);
}

Image VkContext::create_image_2d(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                                 uint32_t mip_levels, uint32_t array_layers) {
    Image img{};
    img.format = format;
    img.extent = {width, height, 1};
    img.mip_levels = mip_levels;
    img.layers = array_layers;
    img.aspect = infer_aspect(format);

    VkImageCreateInfo i_ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    i_ci.imageType = VK_IMAGE_TYPE_2D;
    i_ci.format = format;
    i_ci.extent = img.extent;
    i_ci.mipLevels = mip_levels;
    i_ci.arrayLayers = array_layers;
    i_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    i_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    i_ci.usage = usage;
    i_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo a_ci{};
    a_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(allocator_, &i_ci, &a_ci, &img.handle, &img.allocation, nullptr));

    VkImageViewCreateInfo iv_ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv_ci.image = img.handle;
    iv_ci.viewType = array_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    iv_ci.format = format;
    iv_ci.subresourceRange.aspectMask = img.aspect;
    iv_ci.subresourceRange.baseMipLevel = 0;
    iv_ci.subresourceRange.levelCount = mip_levels;
    iv_ci.subresourceRange.baseArrayLayer = 0;
    iv_ci.subresourceRange.layerCount = array_layers;
    VK_CHECK(vkCreateImageView(device_, &iv_ci, nullptr, &img.view));

    return img;
}

Image VkContext::create_image_3d(uint32_t width, uint32_t height, uint32_t depth, VkFormat format,
                                 VkImageUsageFlags usage, uint32_t mip_levels) {
    Image img{};
    img.format = format;
    img.extent = {width, height, depth};
    img.mip_levels = mip_levels;
    img.layers = 1;
    img.aspect = infer_aspect(format);

    VkImageCreateInfo i_ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    i_ci.imageType = VK_IMAGE_TYPE_3D;
    i_ci.format = format;
    i_ci.extent = img.extent;
    i_ci.mipLevels = mip_levels;
    i_ci.arrayLayers = 1;
    i_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    i_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    i_ci.usage = usage;
    i_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo a_ci{};
    a_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(allocator_, &i_ci, &a_ci, &img.handle, &img.allocation, nullptr));

    VkImageViewCreateInfo iv_ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv_ci.image = img.handle;
    iv_ci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    iv_ci.format = format;
    iv_ci.subresourceRange.aspectMask = img.aspect;
    iv_ci.subresourceRange.levelCount = mip_levels;
    iv_ci.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_, &iv_ci, nullptr, &img.view));

    return img;
}

void VkContext::upload_image_2d(Image& image, const void* pixels, VkDeviceSize byte_size, VkImageLayout final_layout) {
    Buffer staging = create_host_buffer(byte_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, pixels);
    immediate_submit([&](VkCommandBuffer cmd) {
        transition_image(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = image.aspect;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = image.layers;
        region.imageExtent = image.extent;
        vkCmdCopyBufferToImage(cmd, staging.handle, image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        transition_image(cmd, image, final_layout,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    });
    destroy_buffer(staging);
}

void VkContext::destroy_image(Image& image) {
    if (image.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, image.view, nullptr);
    }
    if (image.handle != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image.handle, image.allocation);
    }
    image = {};
}

void VkContext::immediate_submit(const std::function<void(VkCommandBuffer)>& fn) {
    if (recording_serial_ != 0) throw std::logic_error("Immediate submission during frame recording is unsupported");
    recording_serial_ = next_serial_++;
    VK_CHECK(vkResetFences(device_, 1, &imm_fence_));
    VK_CHECK(vkResetCommandBuffer(imm_cmd_, 0));

    VkCommandBufferBeginInfo beg_ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beg_ci.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(imm_cmd_, &beg_ci));

    fn(imm_cmd_);

    VK_CHECK(vkEndCommandBuffer(imm_cmd_));

    VkCommandBufferSubmitInfo cmd_si{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_si.commandBuffer = imm_cmd_;

    VkSubmitInfo2 sub_info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub_info.commandBufferInfoCount = 1;
    sub_info.pCommandBufferInfos = &cmd_si;

    VK_CHECK(vkQueueSubmit2(graphics_queue_, 1, &sub_info, imm_fence_));
    VK_CHECK(vkWaitForFences(device_, 1, &imm_fence_, VK_TRUE, UINT64_MAX));
    completed_serial_ = recording_serial_;
    recording_serial_ = 0;
}

void VkContext::transition_image_range(VkCommandBuffer cmd, const Image& image, VkImageSubresourceRange range,
                                       VkImageLayout old_layout, VkImageLayout new_layout,
                                       VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                       VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.handle;
    barrier.subresourceRange = range;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

void VkContext::transition_image(VkCommandBuffer cmd, Image& image, VkImageLayout new_layout,
                                 VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                 VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    if (image.layout == new_layout) {
        return;
    }
    transition_raw_image(cmd, image.handle, image.aspect, image.layout, new_layout,
                         src_stage, src_access, dst_stage, dst_access);
    image.layout = new_layout;
}

void VkContext::transition_raw_image(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                                     VkImageLayout old_layout, VkImageLayout new_layout,
                                     VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                     VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void VkContext::memory_barrier(VkCommandBuffer cmd,
                               VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                               VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void VkContext::begin_rendering(VkCommandBuffer cmd, VkExtent2D extent,
                                std::span<const RenderingAttachment> color_attachments,
                                const RenderingAttachment* depth_attachment) {
    std::vector<VkRenderingAttachmentInfo> colors(color_attachments.size());
    for (size_t i = 0; i < color_attachments.size(); ++i) {
        const auto& src = color_attachments[i];
        VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        info.imageView = src.view;
        info.imageLayout = src.layout;
        info.loadOp = src.load_op;
        info.storeOp = src.store_op;
        info.clearValue = src.clear;
        colors[i] = info;
    }

    VkRenderingAttachmentInfo depth_info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (depth_attachment != nullptr) {
        depth_info.imageView = depth_attachment->view;
        depth_info.imageLayout = depth_attachment->layout;
        depth_info.loadOp = depth_attachment->load_op;
        depth_info.storeOp = depth_attachment->store_op;
        depth_info.clearValue = depth_attachment->clear;
    }

    VkRenderingInfo render_info{VK_STRUCTURE_TYPE_RENDERING_INFO};
    render_info.renderArea = {{0, 0}, extent};
    render_info.layerCount = 1;
    render_info.colorAttachmentCount = static_cast<uint32_t>(colors.size());
    render_info.pColorAttachments = colors.data();
    render_info.pDepthAttachment = depth_attachment ? &depth_info : nullptr;

    vkCmdBeginRendering(cmd, &render_info);
    set_viewport_and_scissor(cmd, extent);
}

void VkContext::end_rendering(VkCommandBuffer cmd) {
    vkCmdEndRendering(cmd);
}

void VkContext::set_viewport_and_scissor(VkCommandBuffer cmd, VkExtent2D extent) {
    VkViewport vp{};
    vp.x = 0.0f;
    vp.y = 0.0f;
    vp.width = static_cast<float>(extent.width);
    vp.height = static_cast<float>(extent.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void VkContext::push_descriptors(VkCommandBuffer cmd, const Pipeline& pipeline,
                                 std::span<const PushDescriptorWrite> writes) {
    std::vector<VkWriteDescriptorSet> vk_writes(writes.size());
    for (size_t i = 0; i < writes.size(); ++i) {
        const auto& src = writes[i];
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstBinding = src.binding;
        w.descriptorCount = 1;
        w.descriptorType = src.type;
        if (src.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || src.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            w.pBufferInfo = &src.buffer_info;
        } else {
            w.pImageInfo = &src.image_info;
        }
        vk_writes[i] = w;
    }
    vkCmdPushDescriptorSetKHR(cmd, pipeline.bind_point, pipeline.layout, 0,
                              static_cast<uint32_t>(vk_writes.size()), vk_writes.data());
}

uint32_t VkContext::begin_gpu_timer(VkCommandBuffer cmd, std::string_view name) {
    FrameContext& frame = frames_[frame_index_];
    if (!timestamps_supported() || frame.timer_count >= MAX_GPU_TIMERS) {
        return UINT32_MAX;
    }
    uint32_t id = frame.timer_count++;
    frame.timer_names.emplace_back(name);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame.timestamp_pool, id * 2);
    return id;
}

void VkContext::end_gpu_timer(VkCommandBuffer cmd, uint32_t timer_id) {
    if (timer_id >= MAX_GPU_TIMERS) {
        return;
    }
    FrameContext& frame = frames_[frame_index_];
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame.timestamp_pool, timer_id * 2 + 1);
}

double VkContext::gpu_time_ms(std::string_view name) const {
    auto it = gpu_timings_ms_.find(std::string(name));
    return it != gpu_timings_ms_.end() ? it->second : 0.0;
}

std::vector<GpuTimingSample> VkContext::drain_gpu_timings() {
    wait_idle();
    for (auto& frame : frames_) resolve_gpu_timers(frame);
    std::vector<GpuTimingSample> result;
    result.swap(captured_gpu_timings_);
    return result;
}

void VkContext::log_pipeline_statistics(VkPipeline pipeline, std::string_view name) const {
    if (!pipeline_statistics_enabled_) return;
    VkPipelineInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    info.pipeline = pipeline;
    uint32_t count = 0;
    VK_CHECK(vkGetPipelineExecutablePropertiesKHR(device_, &info, &count, nullptr));
    std::vector<VkPipelineExecutablePropertiesKHR> executables(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    VK_CHECK(vkGetPipelineExecutablePropertiesKHR(device_, &info, &count, executables.data()));
    for (uint32_t i = 0; i < count; ++i) {
        VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        executable.pipeline = pipeline;
        executable.executableIndex = i;
        uint32_t statistic_count = 0;
        VK_CHECK(vkGetPipelineExecutableStatisticsKHR(device_, &executable, &statistic_count, nullptr));
        std::vector<VkPipelineExecutableStatisticKHR> statistics(statistic_count);
        std::memset(statistics.data(), 0, statistics.size() * sizeof(VkPipelineExecutableStatisticKHR));
        for (auto& stat : statistics) stat.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        VK_CHECK(vkGetPipelineExecutableStatisticsKHR(device_, &executable, &statistic_count, statistics.data()));
        for (const auto& stat : statistics) {
            std::string value;
            switch (stat.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: value = stat.value.b32 ? "true" : "false"; break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: value = std::format("{}", stat.value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: value = std::format("{}", stat.value.u64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: value = std::format("{}", stat.value.f64); break;
            default: value = "unknown"; break;
            }
            sf::log::info("Shader statistics {} [{}]: {} = {} ({})", name, executables[i].name, stat.name, value, stat.description);
        }
    }
}

} // namespace sf::rhi
