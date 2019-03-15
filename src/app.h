#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>

namespace graphics {

class SoftImage;

enum ShaderType { kVertexShader, kFragmentShader };

struct VulkanDeviceInfo {
  std::mutex mutex;
  int counter{0};
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  uint32_t queue_family_index;
  VkQueue _queue;
};

class App final {
 public:
  using CreateSurface = std::function<VkSurfaceKHR(VkInstance &instance)>;
  using DestroySurface = std::function<void(VkSurfaceKHR *)>;
  using GetShaderCode = std::function<std::vector<char>(ShaderType)>;

  App(const std::vector<const char *> &instance_extensions,
      const CreateSurface &create_surface,
      const DestroySurface &destroy_surface,
      const GetShaderCode &get_shader_code);
  ~App();

  void init();

 private:
  void initVulkan();
  void initView();
  void cleanupView();
  void cleanupVulkan();

  void cleanupSwapChain();
  void recreateSwapChain();

 private:
  void createInstance();
  void pickPhysicalDevice();
  void createLogicalDevice();

  void createSwapChain();
  void getSwapChainImages();
  void createImageViews();
  void createRenderPass();
  void createGraphicsPipeline(const VkDescriptorSetLayout &layout);
  void createFramebuffers();
  void createCommandPool();
  void createDepthResources();
  void createDescriptorPool();
  void createCommandBuffers();

  void createSyncObjects();

 private:
  uint32_t findQueueFamily();

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(VkPhysicalDevice device);
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) const;

  VkShaderModule createShaderModule(const std::vector<char> &code);

  uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags properties);
  VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features);
  VkFormat findDepthFormat();
  bool hasStencilComponent(VkFormat format);

  void createImage(uint32_t width,
                   uint32_t height,
                   VkFormat format,
                   VkImageTiling tiling,
                   VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties,
                   VkImage &image,
                   VkDeviceMemory &imageMemory);

  void createImageView(VkImage image,
                       VkFormat format,
                       VkImageAspectFlags aspectFlags,
                       VkImageView &imageView);

  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer commandBuffer);

  void transitionImageLayout(VkImage image,
                             VkFormat format,
                             VkImageLayout oldLayout,
                             VkImageLayout newLayout);

 public:
  void initUpdate();

  void onDragStart(float x, float y, float aspect);
  void onDragEnd();
  void onDragMove(float x, float y);

  void onResize();

  void drawFrame();

 private:
  const std::vector<const char *> _instance_extensions;
  const CreateSurface _create_surface;
  const DestroySurface _destroy_surface;
  const GetShaderCode _get_shader_code;

  static VulkanDeviceInfo _device_info;

  VkSurfaceKHR _surface;
  VkSwapchainKHR _swapchain;
  std::vector<VkImage> _swapchain_images;
  VkFormat _swapchain_image_format;
  VkExtent2D _swapchain_extent;
  std::vector<VkImageView> _swapchain_image_views;
  std::vector<VkFramebuffer> _swapchain_framebuffers;

  VkRenderPass _render_pass;

  VkPipelineLayout _pipeline_layout;
  VkPipeline _graphics_pipeline;

  VkCommandPool _command_pool;

  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;

  VkDescriptorPool _descriptor_pool;

  std::vector<VkCommandBuffer> _command_buffers;

  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  size_t _current_frame{0};

  std::atomic<bool> _framebuffer_resized{false};
  std::unique_ptr<SoftImage> _soft_image;
};

}  // namespace graphics
