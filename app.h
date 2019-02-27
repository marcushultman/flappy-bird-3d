#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "game.h"

namespace flappy {
namespace {

const auto kWidth = 480;
const auto kHeight = 640;

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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    _window = glfwCreateWindow(kWidth, kHeight, "flappy bird", nullptr, nullptr);
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
  }

  void mainLoop() {
    auto game = std::make_unique<Game>([this] { glfwSetWindowShouldClose(_window, true); });
    glfwSetWindowUserPointer(_window, game.get());

    const auto on_key = [](auto window, int key, int scancode, int action, int mods) {
      auto &game = *static_cast<Game *>(glfwGetWindowUserPointer(window));
      game.onKey(key, scancode, action, mods);
    };
    glfwSetKeyCallback(_window, on_key);
    while (!glfwWindowShouldClose(_window)) {
      // glClear(GL_COLOR_BUFFER_BIT);

      game->draw();

      // glfwSwapBuffers(window);
      glfwPollEvents();
    }
    glfwSetKeyCallback(_window, nullptr);
  }

  void cleanup() {
    for (auto image_view : _swapchain_image_views) {
      vkDestroyImageView(_device, image_view, nullptr);
    }

    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);

    if (kEnableValidationLayers) {
      destroyDebugUtilsMessengerEXT(_instance, _debug_messenger, nullptr);
    }

    vkDestroyInstance(_instance, nullptr);

    glfwDestroyWindow(_window);
    glfwTerminate();
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
    const auto swap_chain_support = querySwapChainSupport(_physical_device);

    const auto surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
    const auto present_mode = chooseSwapPresentMode(swap_chain_support.present_modes);
    const auto extent = chooseSwapExtent(swap_chain_support.capabilities);

    auto image_count = swap_chain_support.capabilities.minImageCount + 1;
    if (auto max_count = swap_chain_support.capabilities.maxImageCount) {
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

    create_info.preTransform = swap_chain_support.capabilities.currentTransform;
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
      VkImageViewCreateInfo create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      create_info.image = _swapchain_images[i];
      create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      create_info.format = _swapchain_image_format;
      create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      create_info.subresourceRange.baseMipLevel = 0;
      create_info.subresourceRange.levelCount = 1;
      create_info.subresourceRange.baseArrayLayer = 0;
      create_info.subresourceRange.layerCount = 1;
      if (vkCreateImageView(_device, &create_info, nullptr, &_swapchain_image_views[i]) !=
          VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");
      }
    }
  }

 private:
  bool isDeviceSuitable(VkPhysicalDevice device) {
    return findQueueFamilies(device) && checkDeviceExtensionSupport(device) &&
           querySwapChainSupport(device);
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
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    VkExtent2D extent{kWidth, kHeight};
    return {std::max(capabilities.minImageExtent.width,
                     std::min(capabilities.maxImageExtent.width, extent.width)),
            std::max(capabilities.minImageExtent.height,
                     std::min(capabilities.maxImageExtent.height, extent.height))};
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
  }

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
};

}  // namespace flappy
