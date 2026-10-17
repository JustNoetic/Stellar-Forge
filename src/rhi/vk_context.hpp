#pragma once

#include "rhi/vk_types.hpp"
#include "core/timing.hpp"
#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct GLFWwindow;

namespace sf::rhi {

inline constexpr uint32_t FRAMES_IN_FLIGHT = 2;
inline constexpr uint32_t MAX_GPU_TIMERS = 64;

struct FrameContext {
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore image_acquired_sem = VK_NULL_HANDLE;
    VkFence in_flight_fence = VK_NULL_HANDLE;
    VkQueryPool timestamp_pool = VK_NULL_HANDLE;
    uint32_t timer_count = 0;
    std::vector<std::string> timer_names;
    uint64_t serial = 0;
    uint64_t completion_serial = 0;
};

struct GpuTimingSample {
    uint64_t frame_id = 0;
    std::unordered_map<std::string, double> timings;
};

class VkContext {
public:
    void init(GLFWwindow* window, bool enable_validation = false);
    void shutdown();

    void wait_idle() const;

    bool begin_frame(VkCommandBuffer& out_cmd, uint32_t& out_swapchain_idx, FrameHostTimings* timings = nullptr);
    void end_frame(VkCommandBuffer cmd, uint32_t swapchain_idx, FrameHostTimings* timings = nullptr);
    void request_swapchain_resize() noexcept { swapchain_dirty_ = true; }

    Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage,
                         VmaAllocationCreateFlags vma_flags = 0, const void* initial_data = nullptr);
    Buffer create_host_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* initial_data = nullptr);
    Buffer create_device_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* initial_data = nullptr);
    void destroy_buffer(Buffer& buffer);
    void flush_buffer(const Buffer& buffer, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

    Image create_image_2d(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                          uint32_t mip_levels = 1, uint32_t array_layers = 1);
    Image create_image_3d(uint32_t width, uint32_t height, uint32_t depth, VkFormat format,
                          VkImageUsageFlags usage, uint32_t mip_levels = 1);
    void upload_image_2d(Image& image, const void* pixels, VkDeviceSize byte_size,
                         VkImageLayout final_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void destroy_image(Image& image);

    void immediate_submit(const std::function<void(VkCommandBuffer)>& fn);

    void transition_image(VkCommandBuffer cmd, Image& image, VkImageLayout new_layout,
                          VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VkAccessFlags2 src_access = VK_ACCESS_2_MEMORY_WRITE_BIT,
                          VkPipelineStageFlags2 dst_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VkAccessFlags2 dst_access = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT);

    void transition_raw_image(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                              VkImageLayout old_layout, VkImageLayout new_layout,
                              VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                              VkAccessFlags2 src_access = VK_ACCESS_2_MEMORY_WRITE_BIT,
                              VkPipelineStageFlags2 dst_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                              VkAccessFlags2 dst_access = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT);
    void transition_image_range(VkCommandBuffer cmd, const Image& image, VkImageSubresourceRange range,
                                VkImageLayout old_layout, VkImageLayout new_layout,
                                VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);

    void memory_barrier(VkCommandBuffer cmd,
                        VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VkAccessFlags2 src_access = VK_ACCESS_2_MEMORY_WRITE_BIT,
                        VkPipelineStageFlags2 dst_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VkAccessFlags2 dst_access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

    void begin_rendering(VkCommandBuffer cmd, VkExtent2D extent,
                         std::span<const RenderingAttachment> color_attachments,
                         const RenderingAttachment* depth_attachment = nullptr);
    void end_rendering(VkCommandBuffer cmd);

    void set_viewport_and_scissor(VkCommandBuffer cmd, VkExtent2D extent);

    void push_descriptors(VkCommandBuffer cmd, const Pipeline& pipeline,
                          std::span<const PushDescriptorWrite> writes);

    template <typename T>
    void push_constants(VkCommandBuffer cmd, const Pipeline& pipeline, const T& data) {
        vkCmdPushConstants(cmd, pipeline.layout, pipeline.push_constant_stages, 0, sizeof(T), &data);
    }

    uint32_t begin_gpu_timer(VkCommandBuffer cmd, std::string_view name);
    void end_gpu_timer(VkCommandBuffer cmd, uint32_t timer_id);
    double gpu_time_ms(std::string_view name) const;
    const std::unordered_map<std::string, double>& gpu_timings() const noexcept { return gpu_timings_ms_; }
    void capture_gpu_timings(bool enabled, size_t capacity = 0) {
        capture_gpu_timings_ = enabled;
        if (enabled) captured_gpu_timings_.reserve(capacity);
    }
    std::vector<GpuTimingSample> drain_gpu_timings();
    uint64_t submitted_frames() const noexcept { return submitted_frames_; }
    uint64_t recording_serial() const noexcept { return recording_serial_; }
    uint64_t completed_serial() const noexcept { return completed_serial_; }
    bool timestamps_supported() const noexcept { return timestamp_valid_bits_ > 0; }
    const VkPhysicalDeviceProperties& device_properties() const noexcept { return device_props_; }
    const VkPhysicalDeviceDriverProperties& driver_properties() const noexcept { return driver_props_; }

    VkInstance instance() const noexcept { return instance_; }
    VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
    VkDevice device() const noexcept { return device_; }
    VkQueue graphics_queue() const noexcept { return graphics_queue_; }
    uint32_t graphics_queue_family() const noexcept { return graphics_queue_family_; }
    VmaAllocator allocator() const noexcept { return allocator_; }
    VkDescriptorPool imgui_descriptor_pool() const noexcept { return imgui_pool_; }

    void set_vsync(bool enabled) noexcept;
    bool vsync_enabled() const noexcept { return vsync_enabled_; }
    VkPresentModeKHR present_mode() const noexcept { return present_mode_; }

    VkFormat swapchain_format() const noexcept { return swapchain_format_; }
    VkExtent2D swapchain_extent() const noexcept { return swapchain_extent_; }
    VkImage swapchain_image(uint32_t idx) const noexcept { return swapchain_images_[idx]; }
    VkImageView swapchain_view(uint32_t idx) const noexcept { return swapchain_views_[idx]; }
    uint32_t swapchain_image_count() const noexcept { return static_cast<uint32_t>(swapchain_images_.size()); }
    uint32_t frame_index() const noexcept { return frame_index_; }
    const char* device_name() const noexcept { return device_props_.deviceName; }
    bool bresenham_lines_enabled() const noexcept { return bresenham_lines_enabled_; }
    bool pipeline_statistics_enabled() const noexcept { return pipeline_statistics_enabled_; }
    void log_pipeline_statistics(VkPipeline pipeline, std::string_view name) const;

    VkSampler sampler_linear_clamp() const noexcept { return sampler_linear_clamp_; }
    VkSampler sampler_linear_repeat() const noexcept { return sampler_linear_repeat_; }
    VkSampler sampler_nearest_clamp() const noexcept { return sampler_nearest_clamp_; }

private:
    void create_swapchain();
    void destroy_swapchain();
    void recreate_swapchain();
    void resolve_gpu_timers(FrameContext& frame);

    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties device_props_{};
    VkPhysicalDeviceDriverProperties driver_props_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    uint32_t timestamp_valid_bits_ = 0;
    bool bresenham_lines_enabled_ = false;
    bool pipeline_statistics_enabled_ = false;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_ = 0;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchain_format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapchain_extent_{0, 0};
    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_views_;
    std::vector<VkSemaphore> render_finished_sems_;
    bool swapchain_dirty_ = false;
    bool vsync_enabled_ = false;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;

    std::array<FrameContext, FRAMES_IN_FLIGHT> frames_{};
    uint32_t frame_index_ = 0;

    VkCommandPool imm_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer imm_cmd_ = VK_NULL_HANDLE;
    VkFence imm_fence_ = VK_NULL_HANDLE;

    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;

    VkSampler sampler_linear_clamp_ = VK_NULL_HANDLE;
    VkSampler sampler_linear_repeat_ = VK_NULL_HANDLE;
    VkSampler sampler_nearest_clamp_ = VK_NULL_HANDLE;

    std::unordered_map<std::string, double> gpu_timings_ms_;
    bool capture_gpu_timings_ = false;
    uint64_t submitted_frames_ = 0;
    uint64_t next_serial_ = 1;
    uint64_t recording_serial_ = 0;
    uint64_t completed_serial_ = 0;
    std::vector<GpuTimingSample> captured_gpu_timings_;
};

} // namespace sf::rhi
