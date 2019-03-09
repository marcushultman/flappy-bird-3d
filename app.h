#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "game.h"
#include "src/shader_reader.h"
#include "src/soft_image.h"

namespace flappy {
namespace {

const auto kWidth = 480;
const auto kHeight = 640;

constexpr auto MAX_FRAMES_IN_FLIGHT = 2;

const auto kValidationLayerStandard = "VK_LAYER_LUNARG_standard_validation";

const std::vector<const char *> kValidationLayers = {kValidationLayerStandard};
const std::vector<const char *> kDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool kEnableValidationLayers = false;
#else
const bool kEnableValidationLayers = true;
#endif

std::vector<const char *> getRequiredExtensions() {
  uint32_t size{};
  const auto extensions = glfwGetRequiredInstanceExtensions(&size);
  return {extensions, extensions + size};
}

std::vector<VkExtensionProperties> getInstalledExtensions() {
  uint32_t size{};
  vkEnumerateInstanceExtensionProperties(nullptr, &size, nullptr);
  std::vector<VkExtensionProperties> extensions(size);
  vkEnumerateInstanceExtensionProperties(nullptr, &size, extensions.data());

  std::cout << "available extensions:" << std::endl;
  for (const auto &extension : extensions) {
    std::cout << "\t" << extension.extensionName << std::endl;
  }
  return extensions;
}

bool checkExtensions(const std::vector<const char *> &extensions) {
  const auto required_extensions = getInstalledExtensions();

  std::cout << "required extensions:" << std::endl;
  for (auto &extension : extensions) {
    auto equals = [&extension](auto e) { return strcmp(e.extensionName, extension) == 0; };
    if (!std::any_of(required_extensions.begin(), required_extensions.end(), equals)) {
      std::cout << "\t" << extension << ": NO" << std::endl;
      return false;
    }
    std::cout << "\t" << extension << ": YES" << std::endl;
  }
  return true;
}

std::vector<VkLayerProperties> getValidationLayers() {
  uint32_t size{};
  vkEnumerateInstanceLayerProperties(&size, nullptr);
  std::vector<VkLayerProperties> layers(size);
  vkEnumerateInstanceLayerProperties(&size, layers.data());

  std::cout << "available validation layers:" << std::endl;
  for (const auto &layer : layers) {
    std::cout << "\t" << layer.layerName << std::endl;
  }
  return layers;
}

bool checkValidationLayers(const std::vector<const char *> &layers) {
  const auto available_layers = getValidationLayers();

  std::cout << "requested layers:" << std::endl;
  for (auto &layer : layers) {
    auto equals = [&layer](auto l) { return strcmp(l.layerName, layer) == 0; };
    if (!std::any_of(available_layers.begin(), available_layers.end(), equals)) {
      std::cout << "\t" << layer << ": NO" << std::endl;
      return false;
    }
    std::cout << "\t" << layer << ": YES" << std::endl;
  }
  return true;
}

std::vector<VkExtensionProperties> getInstalledDeviceExtensions(VkPhysicalDevice device) {
  uint32_t size{};
  vkEnumerateDeviceExtensionProperties(device, nullptr, &size, nullptr);
  std::vector<VkExtensionProperties> extensions(size);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &size, extensions.data());

  std::cout << "available device extensions:" << std::endl;
  for (const auto &extension : extensions) {
    std::cout << "\t" << extension.extensionName << std::endl;
  }
  return extensions;
}

VkResult createDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT *create_info,
                                      const VkAllocationCallbacks *allocator,
                                      VkDebugUtilsMessengerEXT *debug_messenger) {
  if (auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkCreateDebugUtilsMessengerEXT")) {
    return func(instance, create_info, allocator, debug_messenger);
  }
  return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void destroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debug_messenger,
                                   const VkAllocationCallbacks *allocator) {
  if (auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          instance, "vkDestroyDebugUtilsMessengerEXT")) {
    func(instance, debug_messenger, allocator);
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

}  // namespace

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;

  operator bool() const { return graphics_family && present_family; }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;

  operator bool() const { return !formats.empty() && !present_modes.empty(); }
};

// struct Vertex {
//   glm::vec3 pos;
//   glm::vec3 color;
//   glm::vec2 texCoord;

//   static VkVertexInputBindingDescription getBindingDescription() {
//     VkVertexInputBindingDescription binding{};
//     binding.binding = 0;
//     binding.stride = sizeof(Vertex);
//     binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

//     return binding;
//   }
//   static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
//     std::array<VkVertexInputAttributeDescription, 3> attrs{};

//     attrs[0].binding = 0;
//     attrs[0].location = 0;
//     attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
//     attrs[0].offset = offsetof(Vertex, pos);

//     attrs[1].binding = 0;
//     attrs[1].location = 1;
//     attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
//     attrs[1].offset = offsetof(Vertex, color);

//     attrs[2].binding = 0;
//     attrs[2].location = 2;
//     attrs[2].format = VK_FORMAT_R32G32_SFLOAT;
//     attrs[2].offset = offsetof(Vertex, texCoord);

//     return attrs;
//   }
// };

// struct UniformBufferObject {
//   glm::mat4 model;
//   glm::mat4 view;
//   glm::mat4 proj;
// };

// vec2 positions[3] = vec2[](
//   vec2(-0.2, -0.7),
//   vec2(0.4, 0.6),
//   vec2(-0.8, 0.7)
// );

// vec3 colors[3] = vec3[](
//   vec3(1.0, 0.0, 0.0),
//   vec3(0.0, 1.0, 0.0),
//   vec3(0.0, 0.0, 1.0)
// );

// const std::vector<Vertex> kVertices{{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
//                                     {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
//                                     {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
//                                     {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

//                                     {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
//                                     {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
//                                     {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
//                                     {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}};

// const std::vector<uint16_t> kIndices{0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};

class App {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    _window = glfwCreateWindow(kWidth, kHeight, "flappy bird", nullptr, nullptr);
    glfwSetWindowUserPointer(_window, this);
    glfwSetFramebufferSizeCallback(_window, framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
    auto &app = *static_cast<App *>(glfwGetWindowUserPointer(window));
    app._framebuffer_resized = true;
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();

    _soft_image = std::make_unique<SoftImage>(_physical_device,
                                              _device,
                                              _graphics_queue,
                                              _pipeline_layout,
                                              _graphics_pipeline,
                                              _command_pool,
                                              _descriptor_pool,
                                              static_cast<uint32_t>(_swapchain_images.size()));
  }

  void mainLoop() {
    _game = std::make_unique<Game>([this] { glfwSetWindowShouldClose(_window, true); });

    const auto on_key = [](auto window, int key, int scancode, int action, int mods) {
      auto &app = *static_cast<App *>(glfwGetWindowUserPointer(window));
      app._game->onKey(key, scancode, action, mods);
    };
    glfwSetKeyCallback(_window, on_key);
    while (!glfwWindowShouldClose(_window)) {
      glfwPollEvents();
      drawFrame();
      // _game->draw();
    }
    glfwSetKeyCallback(_window, nullptr);

    vkDeviceWaitIdle(_device);
  }

  void cleanupSwapChain() {
    vkDestroyImageView(_device, depthImageView, nullptr);
    vkDestroyImage(_device, depthImage, nullptr);
    vkFreeMemory(_device, depthImageMemory, nullptr);

    for (auto framebuffer : _swapchain_framebuffers) {
      vkDestroyFramebuffer(_device, framebuffer, nullptr);
    }

    vkFreeCommandBuffers(_device,
                         _command_pool,
                         static_cast<uint32_t>(_command_buffers.size()),
                         _command_buffers.data());

    vkDestroyPipeline(_device, _graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
    vkDestroyRenderPass(_device, _render_pass, nullptr);

    for (auto image_view : _swapchain_image_views) {
      vkDestroyImageView(_device, image_view, nullptr);
    }

    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
  }

  void cleanup() {
    cleanupSwapChain();

    vkDestroySampler(_device, _texture_sampler, nullptr);
    vkDestroyImageView(_device, _texture_image_view, nullptr);

    vkDestroyImage(_device, _texture_image, nullptr);
    vkFreeMemory(_device, _texture_image_memory, nullptr);

    vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(_device, _descriptor_set_layout, nullptr);

    for (size_t i = 0; i < _swapchain_images.size(); i++) {
      vkDestroyBuffer(_device, _uniform_buffers[i], nullptr);
      vkFreeMemory(_device, _uniform_buffers_memory[i], nullptr);
    }

    vkDestroyBuffer(_device, _index_buffer, nullptr);
    vkFreeMemory(_device, _index_buffer_memory, nullptr);

    vkDestroyBuffer(_device, _vertex_buffer, nullptr);
    vkFreeMemory(_device, _vertex_buffer_memory, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(_device, renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(_device, imageAvailableSemaphores[i], nullptr);
      vkDestroyFence(_device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(_device, _command_pool, nullptr);

    vkDestroyDevice(_device, nullptr);

    if (kEnableValidationLayers) {
      destroyDebugUtilsMessengerEXT(_instance, _debug_messenger, nullptr);
    }

    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkDestroyInstance(_instance, nullptr);

    glfwDestroyWindow(_window);
    glfwTerminate();
  }

  void recreateSwapChain() {
    int width{0}, height{0};
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(_window, &width, &height);
      glfwWaitEvents();
    }
    vkDeviceWaitIdle(_device);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createDepthResources();
    createFramebuffers();
    createCommandBuffers();
  }

  void createInstance() {
    auto extensions = getRequiredExtensions();

    if (!checkExtensions(extensions)) {
      throw std::runtime_error("required extensions not found");
    }

    if (kEnableValidationLayers) {
      if (!checkValidationLayers(kValidationLayers)) {
        throw std::runtime_error("required validation layers not found");
      }
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "flappy bird";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    if (kEnableValidationLayers) {
      create_info.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
      create_info.ppEnabledLayerNames = kValidationLayers.data();
    } else {
      create_info.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&create_info, nullptr, &_instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
  }

  void setupDebugMessenger() {
    if (!kEnableValidationLayers) {
      return;
    }
    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debugCallback;

    if (createDebugUtilsMessengerEXT(_instance, &create_info, nullptr, &_debug_messenger) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }

  void createSurface() {
    if (glfwCreateWindowSurface(_instance, _window, nullptr, &_surface) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  void pickPhysicalDevice() {
    uint32_t size{};
    vkEnumeratePhysicalDevices(_instance, &size, nullptr);
    if (!size) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(size);
    vkEnumeratePhysicalDevices(_instance, &size, devices.data());
    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        _physical_device = device;
        return;
      }
    }
    throw std::runtime_error("failed to find a suitable GPU!");
  }

  void createLogicalDevice() {
    const auto indices = findQueueFamilies(_physical_device);

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    const std::set<uint32_t> families{*indices.graphics_family, *indices.present_family};

    const auto queue_priority = 1.0f;
    for (auto family : families) {
      VkDeviceQueueCreateInfo queue_create_info{};
      queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queue_create_info.queueFamilyIndex = family;
      queue_create_info.queueCount = 1;
      queue_create_info.pQueuePriorities = &queue_priority;
      queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures device_features{};
    device_features.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
    create_info.ppEnabledExtensionNames = kDeviceExtensions.data();

    if (kEnableValidationLayers) {
      create_info.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
      create_info.ppEnabledLayerNames = kValidationLayers.data();
    } else {
      create_info.enabledLayerCount = 0;
    }

    if (vkCreateDevice(_physical_device, &create_info, nullptr, &_device) != VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }
    vkGetDeviceQueue(_device, *indices.graphics_family, 0, &_graphics_queue);
    vkGetDeviceQueue(_device, *indices.present_family, 0, &_present_queue);
  }

  void createSwapChain() {
    const auto swapchain_support = querySwapChainSupport(_physical_device);

    const auto surface_format = chooseSwapSurfaceFormat(swapchain_support.formats);
    const auto present_mode = chooseSwapPresentMode(swapchain_support.present_modes);
    const auto extent = chooseSwapExtent(swapchain_support.capabilities);

    auto image_count = swapchain_support.capabilities.minImageCount + 1;
    if (auto max_count = swapchain_support.capabilities.maxImageCount) {
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

    const auto indices = findQueueFamilies(_physical_device);
    const uint32_t queue_family_indices[] = {*indices.graphics_family, *indices.present_family};

    if (indices.graphics_family != indices.present_family) {
      create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      create_info.queueFamilyIndexCount = 2;
      create_info.pQueueFamilyIndices = queue_family_indices;
    } else {
      create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    create_info.preTransform = swapchain_support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(_device, &create_info, nullptr, &_swapchain) != VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }
    vkGetSwapchainImagesKHR(_device, _swapchain, &image_count, nullptr);
    _swapchain_images.resize(image_count);
    vkGetSwapchainImagesKHR(_device, _swapchain, &image_count, _swapchain_images.data());

    _swapchain_image_format = surface_format.format;
    _swapchain_extent = extent;
  }

  void createImageViews() {
    _swapchain_image_views.resize(_swapchain_images.size());
    for (auto i = 0; i < _swapchain_images.size(); ++i) {
      createImageView(_swapchain_images[i],
                      _swapchain_image_format,
                      VK_IMAGE_ASPECT_COLOR_BIT,
                      _swapchain_image_views[i]);
    }
  }

  void createRenderPass() {
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

    if (vkCreateRenderPass(_device, &render_pass_info, nullptr, &_render_pass) != VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{uboLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(_device, &layoutInfo, nullptr, &_descriptor_set_layout) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }
  }

  void createGraphicsPipeline() {
    auto vs_code = readFile("shaders/vert.spv");
    auto fs_code = readFile("shaders/frag.spv");
    auto vs_module = createShaderModule(std::move(vs_code));
    auto fs_module = createShaderModule(std::move(fs_code));

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
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
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

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &_descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_pipeline_layout) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

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
            _device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_graphics_pipeline) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }
    vkDestroyShaderModule(_device, fs_module, nullptr);
    vkDestroyShaderModule(_device, vs_module, nullptr);
  }

  void createFramebuffers() {
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

      if (vkCreateFramebuffer(_device, &framebuffer_info, nullptr, &_swapchain_framebuffers[i]) !=
          VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  }

  void createCommandPool() {
    const auto indices = findQueueFamilies(_physical_device);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = *indices.graphics_family;

    if (vkCreateCommandPool(_device, &pool_info, nullptr, &_command_pool) != VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  void createDepthResources() {
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

  void createBuffer(VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties,
                    VkBuffer &buffer,
                    VkDeviceMemory &buffer_memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(_device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(_device, &allocInfo, nullptr, &buffer_memory) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(_device, buffer, buffer_memory, 0);
  }

  void createTextureImage() {
    int width, height, channels;
    const auto path = "../src/assets/textures/c8a60bd180d5efe02e44cb44634802fbc95f8e49.jpeg";
    auto *pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
    VkDeviceSize image_size(width * height * 4);

    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(image_size,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,
                 stagingBufferMemory);
    void *data;
    vkMapMemory(_device, stagingBufferMemory, 0, image_size, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(image_size));
    vkUnmapMemory(_device, stagingBufferMemory);

    stbi_image_free(pixels);

    createImage(width,
                height,
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                _texture_image,
                _texture_image_memory);

    transitionImageLayout(_texture_image,
                          VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(
        stagingBuffer, _texture_image, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    transitionImageLayout(_texture_image,
                          VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(_device, stagingBuffer, nullptr);
    vkFreeMemory(_device, stagingBufferMemory, nullptr);
  }

  void createTextureImageView() {
    createImageView(
        _texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, _texture_image_view);
  }

  void createTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(_device, &samplerInfo, nullptr, &_texture_sampler) != VK_SUCCESS) {
      throw std::runtime_error("failed to create texture sampler!");
    }
  }

  void createVertexBuffer() {
    const auto size = sizeof(kVertices[0]) * kVertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,
                 stagingBufferMemory);

    void *data;
    vkMapMemory(_device, stagingBufferMemory, 0, size, 0, &data);
    memcpy(data, kVertices.data(), static_cast<size_t>(size));
    vkUnmapMemory(_device, stagingBufferMemory);

    createBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 _vertex_buffer,
                 _vertex_buffer_memory);

    copyBuffer(stagingBuffer, _vertex_buffer, size);

    vkDestroyBuffer(_device, stagingBuffer, nullptr);
    vkFreeMemory(_device, stagingBufferMemory, nullptr);
  }

  void createIndexBuffer() {
    const auto size = sizeof(kIndices[0]) * kIndices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,
                 stagingBufferMemory);

    void *data;
    vkMapMemory(_device, stagingBufferMemory, 0, size, 0, &data);
    memcpy(data, kIndices.data(), (size_t)size);
    vkUnmapMemory(_device, stagingBufferMemory);

    createBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 _index_buffer,
                 _index_buffer_memory);

    copyBuffer(stagingBuffer, _index_buffer, size);

    vkDestroyBuffer(_device, stagingBuffer, nullptr);
    vkFreeMemory(_device, stagingBufferMemory, nullptr);
  }

  void createUniformBuffers() {
    const auto size = sizeof(UniformBufferObject);

    _uniform_buffers.resize(_swapchain_images.size());
    _uniform_buffers_memory.resize(_swapchain_images.size());

    for (size_t i = 0; i < _swapchain_images.size(); i++) {
      createBuffer(size,
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   _uniform_buffers[i],
                   _uniform_buffers_memory[i]);
    }
  }

  void createDescriptorPool() {
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

    if (vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptor_pool) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
  }

  void createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(_swapchain_images.size(), _descriptor_set_layout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _descriptor_pool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(_swapchain_images.size());
    allocInfo.pSetLayouts = layouts.data();

    _descriptor_sets.resize(_swapchain_images.size());
    if (vkAllocateDescriptorSets(_device, &allocInfo, _descriptor_sets.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < _swapchain_images.size(); i++) {
      VkDescriptorBufferInfo bufferInfo{};
      bufferInfo.buffer = _uniform_buffers[i];
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject);

      VkDescriptorImageInfo imageInfo{};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = _texture_image_view;
      imageInfo.sampler = _texture_sampler;

      std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[0].dstSet = _descriptor_sets[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrites[0].descriptorCount = 1;
      descriptorWrites[0].pBufferInfo = &bufferInfo;

      descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[1].dstSet = _descriptor_sets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].pImageInfo = &imageInfo;

      vkUpdateDescriptorSets(_device,
                             static_cast<uint32_t>(descriptorWrites.size()),
                             descriptorWrites.data(),
                             0,
                             nullptr);
    }
  }

  void createCommandBuffers() {
    _command_buffers.resize(_swapchain_framebuffers.size());
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = _command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(_command_buffers.size());

    if (vkAllocateCommandBuffers(_device, &alloc_info, _command_buffers.data()) != VK_SUCCESS) {
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
      vkCmdBindPipeline(_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _graphics_pipeline);

      VkBuffer vertex_buffers[] = {_vertex_buffer};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(_command_buffers[i], 0, 1, vertex_buffers, offsets);
      vkCmdBindIndexBuffer(_command_buffers[i], _index_buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdBindDescriptorSets(_command_buffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              _pipeline_layout,
                              0,
                              1,
                              &_descriptor_sets[i],
                              0,
                              nullptr);

      vkCmdDrawIndexed(_command_buffers[i], static_cast<uint32_t>(kIndices.size()), 1, 0, 0, 0);

      vkCmdEndRenderPass(_command_buffers[i]);

      if (vkEndCommandBuffer(_command_buffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  void createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (auto i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      if (vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) !=
              VK_SUCCESS ||
          vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) !=
              VK_SUCCESS ||
          vkCreateFence(_device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create semaphores for a frame!");
      }
    }
  }

 private:
  bool isDeviceSuitable(VkPhysicalDevice device) {
    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return findQueueFamilies(device) && checkDeviceExtensionSupport(device) &&
           querySwapChainSupport(device) && supportedFeatures.samplerAnisotropy;
  }

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t size{};
    vkGetPhysicalDeviceQueueFamilyProperties(device, &size, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(size);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &size, queue_families.data());

    uint32_t i = 0;
    VkBool32 present_support{};
    for (const auto &queue_family : queue_families) {
      if (!queue_family.queueCount) {
        ++i;
        continue;
      }
      if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphics_family = i;
      }
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface, &present_support);
      if (present_support) {
        indices.present_family = i;
      }
      if (indices) {
        break;
      }
      ++i;
    }
    return indices;
  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    std::set<std::string> required_extensions(kDeviceExtensions.begin(), kDeviceExtensions.end());
    for (const auto &extension : getInstalledDeviceExtensions(device)) {
      required_extensions.erase(extension.extensionName);
    }
    return required_extensions.empty();
  }

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _surface, &details.capabilities);

    uint32_t num_formats{};
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &num_formats, nullptr);
    if (num_formats) {
      details.formats.resize(num_formats);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &num_formats, details.formats.data());
    }

    uint32_t num_modes{};
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &num_modes, nullptr);
    if (num_modes) {
      details.present_modes.resize(num_modes);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, _surface, &num_modes, details.present_modes.data());
    }

    return details;
  }

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) {
    if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
      return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }
    for (const auto &format : formats) {
      if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
          format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return format;
      }
    }
    return formats[0];
  }

  VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> present_modes) {
    auto best_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (const auto &present_mode : present_modes) {
      if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return present_mode;
      } else if (present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        best_mode = present_mode;
      }
    }
    return best_mode;
  }

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
    if (capabilities.currentExtent.width != 0xFFFFFFFF) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(_window, &width, &height);

    VkExtent2D extent{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    return {std::max(capabilities.minImageExtent.width,
                     std::min(capabilities.maxImageExtent.width, extent.width)),
            std::max(capabilities.minImageExtent.height,
                     std::min(capabilities.maxImageExtent.height, extent.height))};
  }

  VkShaderModule createShaderModule(std::vector<char> code) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule module;
    if (vkCreateShaderModule(_device, &create_info, nullptr, &module) != VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }
    return module;
  }

  uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(_physical_device, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((filter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }
    throw std::runtime_error("failed to find suitable memory type!");
  }

  VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features) {
    for (auto format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(_physical_device, format, &props);
      if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                 (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }
    throw std::runtime_error("failed to find supported format!");
  }

  VkFormat findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
  }

  void createImage(uint32_t width,
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

    if (vkCreateImage(_device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(_device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(_device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(_device, image, imageMemory, 0);
  }

  void createImageView(VkImage image,
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

    if (vkCreateImageView(_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image view!");
    }
  }

  VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = _command_pool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(_graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_graphics_queue);

    vkFreeCommandBuffers(_device, _command_pool, 1, &commandBuffer);
  }

  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    auto commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
  }

  void transitionImageLayout(VkImage image,
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

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
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
      barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
  }

  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    auto commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(
        commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
  }

 private:
  void drawFrame() {
    vkWaitForFences(
        _device, 1, &inFlightFences[_current_frame], VK_TRUE, std::numeric_limits<uint64_t>::max());

    uint32_t image_index;
    auto result = vkAcquireNextImageKHR(_device,
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

    updateUniformBuffer(image_index);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore wait_semaphores[] = {imageAvailableSemaphores[_current_frame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = waitStages;

    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_command_buffers[image_index];

    VkSemaphore signal_semaphores[] = {renderFinishedSemaphores[_current_frame]};
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    vkResetFences(_device, 1, &inFlightFences[_current_frame]);

    if (vkQueueSubmit(_graphics_queue, 1, &submit_info, inFlightFences[_current_frame]) !=
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

    result = vkQueuePresentKHR(_present_queue, &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || _framebuffer_resized) {
      _framebuffer_resized = false;
      recreateSwapChain();
    } else if (result != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    _current_frame = (_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void updateUniformBuffer(uint32_t current_image) {
    using namespace std::literals::chrono_literals;
    static const auto start_time = std::chrono::high_resolution_clock::now();
    const auto now = std::chrono::high_resolution_clock::now();
    const auto up = glm::vec3(0.0f, 0.0f, 1.0f);
    const auto angle = (3.141592f / 2) * (now - start_time) / 1s;
    const auto aspect = _swapchain_extent.width / static_cast<float>(_swapchain_extent.height);

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), angle, up);
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), up);
    ubo.proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10.0f);

    // OpenGL -> Vulkan
    ubo.proj[1][1] *= -1;

    void *data;
    vkMapMemory(_device, _uniform_buffers_memory[current_image], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(_device, _uniform_buffers_memory[current_image]);
  }

 private:
  GLFWwindow *_window;

  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;
  VkSurfaceKHR _surface;

  VkPhysicalDevice _physical_device;
  VkDevice _device;

  VkQueue _graphics_queue;
  VkQueue _present_queue;

  VkSwapchainKHR _swapchain;
  std::vector<VkImage> _swapchain_images;
  VkFormat _swapchain_image_format;
  VkExtent2D _swapchain_extent;
  std::vector<VkImageView> _swapchain_image_views;
  std::vector<VkFramebuffer> _swapchain_framebuffers;

  VkRenderPass _render_pass;

  VkDescriptorSetLayout _descriptor_set_layout;

  VkPipelineLayout _pipeline_layout;
  VkPipeline _graphics_pipeline;

  VkCommandPool _command_pool;

  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;

  VkBuffer _vertex_buffer;
  VkDeviceMemory _vertex_buffer_memory;
  VkBuffer _index_buffer;
  VkDeviceMemory _index_buffer_memory;
  VkImage _texture_image;
  VkDeviceMemory _texture_image_memory;

  VkImageView _texture_image_view;
  VkSampler _texture_sampler;

  std::vector<VkBuffer> _uniform_buffers;
  std::vector<VkDeviceMemory> _uniform_buffers_memory;

  VkDescriptorPool _descriptor_pool;
  std::vector<VkDescriptorSet> _descriptor_sets;

  std::vector<VkCommandBuffer> _command_buffers;

  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  size_t _current_frame{0};

  bool _framebuffer_resized{false};

 private:
  std::unique_ptr<Game> _game;
  std::unique_ptr<SoftImage> _soft_image;
};

}  // namespace flappy
