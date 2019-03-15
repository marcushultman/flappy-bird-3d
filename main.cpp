#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "src/app.h"
#include "src/shader_reader.h"

using namespace graphics;

namespace {

const auto kWidth = 480;
const auto kHeight = 640;

const auto kValidationLayerStandard = "VK_LAYER_LUNARG_standard_validation";

const std::vector<const char *> kValidationLayers = {kValidationLayerStandard};

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

GLFWwindow *createWindow() {
  return glfwCreateWindow(kWidth, kHeight, "flappy bird", nullptr, nullptr);
}

}  // namespace

int main() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  auto *window = createWindow();
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

  auto create_surface = [window](auto &instance) {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
    return surface;
  };
  auto get_shader_code = [](auto type) {
    if (type == kVertexShader) {
      return readFile("shaders/vert.spv");
    } else {
      return readFile("shaders/frag.spv");
    }
  };
  auto on_key = [](auto window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
      glfwSetWindowShouldClose(window, true);
    }
    if (action != GLFW_PRESS && key == GLFW_KEY_L) {
      printf("Load image!\n");
      // curl_easy
    }
  };
  auto on_mouse_down = [](auto window, int button, int action, int mods) {
    if (button != GLFW_MOUSE_BUTTON_1) {
      return;
    }
    auto &app = *static_cast<App *>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS) {
      double screen_x, screen_y;
      int width, height;
      glfwGetCursorPos(window, &screen_x, &screen_y);
      glfwGetWindowSize(window, &width, &height);
      const auto x = 1.0f - (2.0f * screen_x) / static_cast<float>(width);
      const auto y = (2.0f * screen_y) / static_cast<float>(height) - 1.0f;
      const auto aspect = width / static_cast<float>(height);
      app.onDragStart(x, y, aspect);
    } else if (action == GLFW_RELEASE) {
      app.onDragEnd();
    }
  };
  auto on_mouse_move = [](auto window, double xpos, double ypos) {
    auto &app = *static_cast<App *>(glfwGetWindowUserPointer(window));
    app.onDragMove(static_cast<float>(xpos), static_cast<float>(ypos));
  };
  auto on_resize = [](auto window, int width, int height) {
    auto &app = *static_cast<App *>(glfwGetWindowUserPointer(window));
    app.onResize();
  };

  App app(extensions, std::move(create_surface), {}, std::move(get_shader_code));

  glfwSetWindowUserPointer(window, &app);
  glfwSetKeyCallback(window, std::move(on_key));
  glfwSetMouseButtonCallback(window, std::move(on_mouse_down));
  glfwSetCursorPosCallback(window, std::move(on_mouse_move));
  glfwSetFramebufferSizeCallback(window, std::move(on_resize));

  try {
    app.init();

    printf("starting...\n");
    std::atomic<bool> running{true};
    std::thread runner_thread([&app, &running] {
      app.initUpdate();
      while (running.load()) {
        app.drawFrame();
      }
    });
    printf("running!\n");

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
    printf("stopping...\n");

    running.store(false);
    printf("joining thread...\n");
    runner_thread.join();
    printf("done!\n");

    glfwSetKeyCallback(window, nullptr);
    glfwSetMouseButtonCallback(window, nullptr);
    glfwSetCursorPosCallback(window, nullptr);
    glfwSetFramebufferSizeCallback(window, nullptr);

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
