#define GLFW_INCLUDE_VULKAN

#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION

#include <GLFW/glfw3.h>
#include <curl/curl.h>
#include <frag.h>
#include <stb_image.h>
#include <vert.h>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "app.h"
#include "shader_reader.h"

using namespace graphics;

namespace {

const auto kWidth = 512;
const auto kHeight = 512;

const auto kValidationLayerStandard = "VK_LAYER_LUNARG_standard_validation";

const std::vector<const char *> kValidationLayers = {kValidationLayerStandard};

const bool kEnableValidationLayers = true;

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

bool curl_perform_and_check(CURL *curl, long &status) {
  return curl_easy_perform(curl) || curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status) ||
         status >= 400;
}

bool curl_perform_and_check(CURL *curl) {
  long status{0};
  return curl_perform_and_check(curl, status);
}

size_t bufferArray(char *ptr, size_t size, size_t nmemb, void *obj) {
  size *= nmemb;
  auto &encoded = *static_cast<std::vector<unsigned char> *>(obj);
  encoded.insert(encoded.end(), ptr, ptr + size);
  return size;
}

const std::string &img() {
  static const std::array<std::string, 6> files = {
      "../src/assets/textures/c8a60bd180d5efe02e44cb44634802fbc95f8e49.jpeg",
      "../src/assets/textures/4a1ea5b936aa01c45dc021d74d6bf0b3a6b8afae.jpeg",
      "../src/assets/textures/8deed63f89e0a215e90de1ee5809780921a47747.jpeg",
      "../src/assets/textures/c8a60bd180d5efe02e44cb44634802fbc95f8e49.jpeg",
      "../src/assets/textures/dfa9264c5427a0dfcfdf99a6592d608b42420e84.jpeg",
      "../src/assets/textures/e62440784009e6696cbecef6b78120e9292e63b0.jpeg",
  };
  static size_t index{0};
  index = (index + 1) % files.size();
  return files[index];
}

struct Context {
  App *app;
  CURL *curl;

  void updateImage() {
    printf("Load image!\n");
    std::async(
        std::launch::async,
        [](auto *curl, auto *app) {
          curl_easy_reset(curl);
          curl_easy_setopt(curl, CURLOPT_URL, "https://cataas.com/cat");
          std::vector<unsigned char> encoded;
          curl_easy_setopt(curl, CURLOPT_WRITEDATA, &encoded);
          curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, bufferArray);
          if (curl_perform_and_check(curl)) {
            throw std::runtime_error("failed to fetch cat image");
          }
          int width, height, channels;
          auto *pixels = stbi_load_from_memory(
              encoded.data(), encoded.size(), &width, &height, &channels, STBI_rgb_alpha);
          app->updateImageFromMemory(pixels, width, height);
          stbi_image_free(pixels);
        },
        curl,
        app);
  }
};

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
    return type == kVertexShader ? readFile("shaders/vert.spv") : readFile("shaders/frag.spv");
  };
  auto log = [](auto &s) { printf("%s\n", s.c_str()); };

  auto curl = curl_easy_init();

  if (!curl) {
    throw std::runtime_error("curl not found");
  }

  auto on_key = [](auto window, int key, int scancode, int action, int mods) {
    auto &context = *static_cast<Context *>(glfwGetWindowUserPointer(window));
    if (action != GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
      glfwSetWindowShouldClose(window, true);
    }
    if (action != GLFW_PRESS && key == GLFW_KEY_L) {
      context.updateImage();
    }
  };
  auto on_mouse_down = [](auto window, int button, int action, int mods) {
    if (button != GLFW_MOUSE_BUTTON_1) {
      return;
    }
    auto &context = *static_cast<Context *>(glfwGetWindowUserPointer(window));
    auto &app = *context.app;
    if (action == GLFW_PRESS) {
      double screen_x, screen_y;
      glfwGetCursorPos(window, &screen_x, &screen_y);
      int width, height;
      glfwGetWindowSize(window, &width, &height);
      const auto x = (2.0f * screen_x) / static_cast<float>(width) - 1.0f;
      const auto y = 1.0f - (2.0f * screen_y) / static_cast<float>(height);
      const auto aspect = width / static_cast<float>(height);
      app.onDragStart(x, y, aspect);
    } else if (action == GLFW_RELEASE) {
      app.onDragEnd();
    }
  };
  auto on_mouse_move = [](auto window, double screen_x, double screen_y) {
    auto &context = *static_cast<Context *>(glfwGetWindowUserPointer(window));
    auto &app = *context.app;
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    const auto x = (2.0f * screen_x) / static_cast<float>(width) - 1.0f;
    const auto y = 1.0f - (2.0f * screen_y) / static_cast<float>(height);
    const auto aspect = width / static_cast<float>(height);
    app.onDragMove(x, y, aspect);
  };
  auto on_resize = [](auto window, int width, int height) {
    auto &context = *static_cast<Context *>(glfwGetWindowUserPointer(window));
    auto &app = *context.app;
    app.onResize();
  };

  auto app = App{extensions,
                 std::move(create_surface),
                 App::DestroySurface{},
                 std::move(get_shader_code),
                 log};

  auto context = Context{&app, curl};

  glfwSetWindowUserPointer(window, &context);
  glfwSetKeyCallback(window, std::move(on_key));
  glfwSetMouseButtonCallback(window, std::move(on_mouse_down));
  glfwSetCursorPosCallback(window, std::move(on_mouse_move));
  glfwSetFramebufferSizeCallback(window, std::move(on_resize));

  try {
    app.init();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  app.initUpdate();

  context.updateImage();

  printf("running!\n");
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    app.drawFrame();
  }

  printf("stopping...\n");

  glfwSetKeyCallback(window, nullptr);
  glfwSetMouseButtonCallback(window, nullptr);
  glfwSetCursorPosCallback(window, nullptr);
  glfwSetFramebufferSizeCallback(window, nullptr);

  curl_easy_cleanup(curl);

  return 0;
}
