#include "app.h"

#include <algorithm>
#include <array>
#include <limits>
#include "soft_image.h"

namespace graphics {
namespace {

constexpr auto kMaxFramesInFlight = 2;
const std::vector<const char *> kDeviceExtensions = {"VK_KHR_swapchain"};

}  // namespace

VulkanDeviceInfo App::_device_info;

App::App(const std::vector<const char *> &instance_extensions,
         const CreateSurface &create_surface,
         const DestroySurface &destroy_surface,
         const GetShaderCode &get_shader_code)
    : _instance_extensions{instance_extensions},
      _create_surface{create_surface},
      _destroy_surface{destroy_surface},
      _get_shader_code{get_shader_code} {}

App::~App() {
  cleanupView();
  {
    std::scoped_lock lock(_device_info.mutex);
    _device_info.counter--;
    if (_device_info.counter == 0) {
      // last surface released, destroy instance
      cleanupVulkan();
    }
  }
}

void App::init() {
  {
    std::scoped_lock lock(_device_info.mutex);
    if (_device_info.counter == 0) {
      // Need new instance
      initVulkan();
    }
    _device_info.counter++;
  }
  initView();
}

void App::initVulkan() {
  createInstance();
  pickPhysicalDevice();
  createLogicalDevice();
}

void App::initView() {
  _surface = _create_surface(_device_info.instance);

  createSwapChain();
  getSwapChainImages();
  createImageViews();
  createRenderPass();

  auto descriptor_set_layout = SoftImage::createDescriptorSetLayout(_device_info.device);

  createGraphicsPipeline(descriptor_set_layout);
  createCommandPool();
  createDepthResources();
  createFramebuffers();
  createDescriptorPool();

  _soft_image = std::make_unique<SoftImage>(_device_info.physical_device,
                                            _device_info.device,
                                            _device_info._queue,
                                            _pipeline_layout,
                                            _graphics_pipeline,
                                            _command_pool,
                                            _descriptor_pool,
                                            std::move(descriptor_set_layout),
                                            static_cast<uint32_t>(_swapchain_images.size()),
                                            kMaxFramesInFlight,
                                            18);

  createCommandBuffers();
  createSyncObjects();
}

void App::cleanupView() {
  vkWaitForFences(_device_info.device,
                  inFlightFences.size(),
                  inFlightFences.data(),
                  VK_TRUE,
                  std::numeric_limits<uint64_t>::max());

  cleanupSwapChain();

  _soft_image.reset();

  vkDestroyDescriptorPool(_device_info.device, _descriptor_pool, nullptr);

  for (size_t i = 0; i < kMaxFramesInFlight; i++) {
    vkDestroySemaphore(_device_info.device, renderFinishedSemaphores[i], nullptr);
    vkDestroySemaphore(_device_info.device, imageAvailableSemaphores[i], nullptr);
    vkDestroyFence(_device_info.device, inFlightFences[i], nullptr);
  }

  vkDestroyCommandPool(_device_info.device, _command_pool, nullptr);
  vkDestroySurfaceKHR(_device_info.instance, _surface, nullptr);
}

void App::cleanupVulkan() {
  vkDeviceWaitIdle(_device_info.device);
  vkDestroyDevice(_device_info.device, nullptr);
  vkDestroyInstance(_device_info.instance, nullptr);
}

void App::cleanupSwapChain() {
  vkDestroyImageView(_device_info.device, depthImageView, nullptr);
  vkDestroyImage(_device_info.device, depthImage, nullptr);
  vkFreeMemory(_device_info.device, depthImageMemory, nullptr);

  for (auto framebuffer : _swapchain_framebuffers) {
    vkDestroyFramebuffer(_device_info.device, framebuffer, nullptr);
  }

  vkFreeCommandBuffers(_device_info.device,
                       _command_pool,
                       static_cast<uint32_t>(_command_buffers.size()),
                       _command_buffers.data());

  vkDestroyPipeline(_device_info.device, _graphics_pipeline, nullptr);
  vkDestroyPipelineLayout(_device_info.device, _pipeline_layout, nullptr);
  vkDestroyRenderPass(_device_info.device, _render_pass, nullptr);

  for (auto image_view : _swapchain_image_views) {
    vkDestroyImageView(_device_info.device, image_view, nullptr);
  }

  vkDestroySwapchainKHR(_device_info.device, _swapchain, nullptr);
}

void App::recreateSwapChain() {
  vkDeviceWaitIdle(_device_info.device);

  cleanupSwapChain();

  createSwapChain();
  getSwapChainImages();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline(_soft_image->descriptor_set_layout());
  createDepthResources();
  createFramebuffers();
  createCommandBuffers();
}

void App::createInstance() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "";
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pEngineName = "none";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  create_info.enabledExtensionCount = static_cast<uint32_t>(_instance_extensions.size());
  create_info.ppEnabledExtensionNames = _instance_extensions.data();
  create_info.enabledLayerCount = 0;

  if (vkCreateInstance(&create_info, nullptr, &_device_info.instance) != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  }
}

void App::pickPhysicalDevice() {
  uint32_t size{};
  vkEnumeratePhysicalDevices(_device_info.instance, &size, nullptr);
  if (!size) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }
  std::vector<VkPhysicalDevice> devices(size);
  vkEnumeratePhysicalDevices(_device_info.instance, &size, devices.data());
  _device_info.physical_device = devices[0];
}

void App::createLogicalDevice() {
  _device_info.queue_family_index = findQueueFamily();

  const auto queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_create_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = _device_info.queue_family_index,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
  };

  VkPhysicalDeviceFeatures device_features{};
  device_features.samplerAnisotropy = VK_TRUE;

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount = 1;
  create_info.pQueueCreateInfos = &queue_create_info;
  create_info.pEnabledFeatures = &device_features;
  create_info.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
  create_info.ppEnabledExtensionNames = kDeviceExtensions.data();
  create_info.enabledLayerCount = 0;

  if (vkCreateDevice(_device_info.physical_device, &create_info, nullptr, &_device_info.device) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }
  vkGetDeviceQueue(_device_info.device, _device_info.queue_family_index, 0, &_device_info._queue);
}

void App::createSwapChain() {
  VkSurfaceCapabilitiesKHR surface_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      _device_info.physical_device, _surface, &surface_capabilities);

  const auto surface_format = chooseSwapSurfaceFormat(_device_info.physical_device);
  const auto extent = chooseSwapExtent(surface_capabilities);

  _swapchain_image_format = surface_format.format;
  _swapchain_extent = extent;

  auto image_count = surface_capabilities.minImageCount + 1;
  if (auto max_count = surface_capabilities.maxImageCount) {
    image_count = std::min(image_count, max_count);
  }

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = _surface;

  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  create_info.queueFamilyIndexCount = 1;
  create_info.pQueueFamilyIndices = &_device_info.queue_family_index;

  create_info.preTransform = surface_capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(_device_info.device, &create_info, nullptr, &_swapchain) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }
}

void App::getSwapChainImages() {
  uint32_t count;
  vkGetSwapchainImagesKHR(_device_info.device, _swapchain, &count, nullptr);
  _swapchain_images.resize(count);
  vkGetSwapchainImagesKHR(_device_info.device, _swapchain, &count, _swapchain_images.data());
}

void App::createImageViews() {
  _swapchain_image_views.resize(_swapchain_images.size());
  for (auto i = 0; i < _swapchain_images.size(); ++i) {
    createImageView(_swapchain_images[i],
                    _swapchain_image_format,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    _swapchain_image_views[i]);
  }
}

void App::createRenderPass() {
  VkAttachmentDescription color_attachment{};
  color_attachment.format = _swapchain_image_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref{};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentDescription depth_attachment{};
  depth_attachment.format = findDepthFormat();
  depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_attachment_ref{};
  depth_attachment_ref.attachment = 1;
  depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;
  subpass.pDepthStencilAttachment = &depth_attachment_ref;

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  std::array<VkAttachmentDescription, 2> attachments{color_attachment, depth_attachment};
  VkRenderPassCreateInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  render_pass_info.pAttachments = attachments.data();
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  if (vkCreateRenderPass(_device_info.device, &render_pass_info, nullptr, &_render_pass) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
}

void App::createGraphicsPipeline(const VkDescriptorSetLayout &layout) {
  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &layout;
  pipeline_layout_info.pushConstantRangeCount = 0;

  if (vkCreatePipelineLayout(
          _device_info.device, &pipeline_layout_info, nullptr, &_pipeline_layout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  auto vs_module = createShaderModule(_get_shader_code(kVertexShader));
  auto fs_module = createShaderModule(_get_shader_code(kFragmentShader));

  VkPipelineShaderStageCreateInfo vs_stage{};
  vs_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vs_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vs_stage.module = vs_module;
  vs_stage.pName = "main";

  VkPipelineShaderStageCreateInfo fs_stage{};
  fs_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fs_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fs_stage.module = fs_module;
  fs_stage.pName = "main";

  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

  VkPipelineShaderStageCreateInfo stages[] = {vs_stage, fs_stage};

  VkPipelineVertexInputStateCreateInfo vertex_input_info{};
  vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();

  vertex_input_info.vertexBindingDescriptionCount = 1;
  vertex_input_info.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributeDescriptions.size());
  vertex_input_info.pVertexBindingDescriptions = &bindingDescription;
  vertex_input_info.pVertexAttributeDescriptions = attributeDescriptions.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(_swapchain_extent.width);
  viewport.height = static_cast<float>(_swapchain_extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = _swapchain_extent;

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;

  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = 2;
  pipeline_info.pStages = stages;

  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.layout = _pipeline_layout;
  pipeline_info.renderPass = _render_pass;
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(
          _device_info.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_graphics_pipeline) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }
  vkDestroyShaderModule(_device_info.device, fs_module, nullptr);
  vkDestroyShaderModule(_device_info.device, vs_module, nullptr);
}

void App::createFramebuffers() {
  _swapchain_framebuffers.resize(_swapchain_image_views.size());
  for (auto i = 0; i < _swapchain_image_views.size(); ++i) {
    std::array<VkImageView, 2> attachments = {_swapchain_image_views[i], depthImageView};

    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = _render_pass;
    framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = _swapchain_extent.width;
    framebuffer_info.height = _swapchain_extent.height;
    framebuffer_info.layers = 1;

    if (vkCreateFramebuffer(
            _device_info.device, &framebuffer_info, nullptr, &_swapchain_framebuffers[i]) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

void App::createCommandPool() {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = _device_info.queue_family_index;

  if (vkCreateCommandPool(_device_info.device, &pool_info, nullptr, &_command_pool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
}

void App::createDepthResources() {
  auto depthFormat = findDepthFormat();

  createImage(_swapchain_extent.width,
              _swapchain_extent.height,
              depthFormat,
              VK_IMAGE_TILING_OPTIMAL,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
              depthImage,
              depthImageMemory);
  createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, depthImageView);
  transitionImageLayout(depthImage,
                        depthFormat,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void App::createDescriptorPool() {
  std::array<VkDescriptorPoolSize, 2> poolSizes{};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = static_cast<uint32_t>(_swapchain_images.size());
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = static_cast<uint32_t>(_swapchain_images.size());

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = static_cast<uint32_t>(_swapchain_images.size());

  if (vkCreateDescriptorPool(_device_info.device, &poolInfo, nullptr, &_descriptor_pool) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void App::createCommandBuffers() {
  _command_buffers.resize(_swapchain_framebuffers.size());

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = _command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = static_cast<uint32_t>(_command_buffers.size());

  if (vkAllocateCommandBuffers(_device_info.device, &alloc_info, _command_buffers.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }

  for (auto i = 0; i < _command_buffers.size(); ++i) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(_command_buffers[i], &begin_info) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = _render_pass;
    render_pass_info.framebuffer = _swapchain_framebuffers[i];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = _swapchain_extent;

    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clear_values[1].depthStencil = {1.0f, 0};
    render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    render_pass_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(_command_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    _soft_image->recordDrawCommands(_command_buffers[i], i);
    vkCmdEndRenderPass(_command_buffers[i]);

    if (vkEndCommandBuffer(_command_buffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }
}

void App::createSyncObjects() {
  imageAvailableSemaphores.resize(kMaxFramesInFlight);
  renderFinishedSemaphores.resize(kMaxFramesInFlight);
  inFlightFences.resize(kMaxFramesInFlight);

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (auto i = 0; i < kMaxFramesInFlight; ++i) {
    if (vkCreateSemaphore(
            _device_info.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) !=
            VK_SUCCESS ||
        vkCreateSemaphore(
            _device_info.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) !=
            VK_SUCCESS ||
        vkCreateFence(_device_info.device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create semaphores for a frame!");
    }
  }
}

uint32_t App::findQueueFamily() {
  uint32_t size{};
  vkGetPhysicalDeviceQueueFamilyProperties(_device_info.physical_device, &size, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(size);
  vkGetPhysicalDeviceQueueFamilyProperties(
      _device_info.physical_device, &size, queue_families.data());

  for (uint32_t index = 0; index < size; ++index) {
    if (queue_families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      return index;
    }
  }
  throw std::runtime_error("failed to find queue family!");
}

VkSurfaceFormatKHR App::chooseSwapSurfaceFormat(VkPhysicalDevice device) {
  uint32_t num_formats{};
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &num_formats, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(num_formats);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &num_formats, formats.data());

  for (auto &surface_format : formats) {
    if (surface_format.format == VK_FORMAT_B8G8R8A8_UNORM) {
      return surface_format;
    }
  }
  throw std::runtime_error("failed to find surface format!");
}

VkExtent2D App::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) const {
  if (capabilities.currentExtent.width == 0xFFFFFFFF) {
    throw std::runtime_error("freeform swapchain extent not supported!");
  }
  return capabilities.currentExtent;
}

VkShaderModule App::createShaderModule(const std::vector<char> &code) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module;
  if (vkCreateShaderModule(_device_info.device, &create_info, nullptr, &module) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }
  return module;
}

uint32_t App::findMemoryType(uint32_t filter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(_device_info.physical_device, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((filter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

VkFormat App::findSupportedFormat(const std::vector<VkFormat> &candidates,
                                  VkImageTiling tiling,
                                  VkFormatFeatureFlags features) {
  for (auto format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(_device_info.physical_device, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }
  throw std::runtime_error("failed to find supported format!");
}

VkFormat App::findDepthFormat() {
  return findSupportedFormat(
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL,
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

bool App::hasStencilComponent(VkFormat format) {
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void App::createImage(uint32_t width,
                      uint32_t height,
                      VkFormat format,
                      VkImageTiling tiling,
                      VkImageUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkImage &image,
                      VkDeviceMemory &imageMemory) {
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(_device_info.device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(_device_info.device, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(_device_info.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }

  vkBindImageMemory(_device_info.device, image, imageMemory, 0);
}

void App::createImageView(VkImage image,
                          VkFormat format,
                          VkImageAspectFlags aspectFlags,
                          VkImageView &imageView) {
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(_device_info.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image view!");
  }
}

VkCommandBuffer App::beginSingleTimeCommands() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = _command_pool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(_device_info.device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

void App::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(_device_info._queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(_device_info._queue);

  vkFreeCommandBuffers(_device_info.device, _command_pool, 1, &commandBuffer);
}

void App::transitionImageLayout(VkImage image,
                                VkFormat format,
                                VkImageLayout oldLayout,
                                VkImageLayout newLayout) {
  auto commandBuffer = beginSingleTimeCommands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (hasStencilComponent(format)) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  } else {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  }
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else {
    throw std::invalid_argument("unsupported layout transition!");
  }

  vkCmdPipelineBarrier(
      commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  endSingleTimeCommands(commandBuffer);
}

void App::initUpdate() {
  _soft_image->start();
}

void App::onDragStart(float x, float y, float aspect) {
  _soft_image->onDragStart(x, y, aspect);
}
void App::onDragEnd() {
  _soft_image->onDragEnd();
}
void App::onDragMove(float x, float y) {
  _soft_image->onDragMove(x, y);
}

void App::onResize() {
  _framebuffer_resized.store(true);
}

void App::drawFrame() {
  vkWaitForFences(_device_info.device,
                  1,
                  &inFlightFences[_current_frame],
                  VK_TRUE,
                  std::numeric_limits<uint64_t>::max());

  uint32_t image_index;
  auto result = vkAcquireNextImageKHR(_device_info.device,
                                      _swapchain,
                                      std::numeric_limits<uint64_t>::max(),
                                      imageAvailableSemaphores[_current_frame],
                                      VK_NULL_HANDLE,
                                      &image_index);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  auto update_semaphore = _soft_image->updateVertices(_current_frame, image_index);

  const auto aspect = _swapchain_extent.width / static_cast<float>(_swapchain_extent.height);
  _soft_image->updateUniformBuffer(image_index, aspect);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore wait_semaphores[] = {imageAvailableSemaphores[_current_frame],
                                   update_semaphore ? *update_semaphore : VK_NULL_HANDLE};
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};
  submit_info.waitSemaphoreCount = update_semaphore ? 2 : 1;

  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = waitStages;

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &_command_buffers[image_index];

  VkSemaphore signal_semaphores[] = {renderFinishedSemaphores[_current_frame]};
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;

  vkResetFences(_device_info.device, 1, &inFlightFences[_current_frame]);

  if (vkQueueSubmit(_device_info._queue, 1, &submit_info, inFlightFences[_current_frame]) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  VkSwapchainKHR swapchains[] = {_swapchain};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &image_index;

  result = vkQueuePresentKHR(_device_info._queue, &present_info);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      _framebuffer_resized.load()) {
    _framebuffer_resized.store(false);
    recreateSwapChain();
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  _current_frame = (_current_frame + 1) % kMaxFramesInFlight;
}

}  // namespace graphics
