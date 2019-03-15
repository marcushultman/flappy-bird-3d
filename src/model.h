#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/norm.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "constraint.h"

namespace graphics {

const std::string MODEL_PATH = "../src/assets/models/chalet.obj";
const std::string TEXTURE_PATH = "../src/assets/models/chalet.jpg";

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  bool operator==(const Vertex &other) const {
    return pos == other.pos && color == other.color && texCoord == other.texCoord;
  }
  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return binding;
  }
  static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 3> attrs{};

    attrs[0].binding = 0;
    attrs[0].location = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = offsetof(Vertex, pos);

    attrs[1].binding = 0;
    attrs[1].location = 1;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = offsetof(Vertex, color);

    attrs[2].binding = 0;
    attrs[2].location = 2;
    attrs[2].format = VK_FORMAT_R32G32_SFLOAT;
    attrs[2].offset = offsetof(Vertex, texCoord);

    return attrs;
  }
};

}  // namespace

namespace std {
template <>
struct hash<graphics::Vertex> {
  size_t operator()(graphics::Vertex const &vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
           (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};

}  // namespace std

namespace graphics {

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class SoftImage {
 public:
  static VkDescriptorSetLayout createDescriptorSetLayout(VkDevice &device) {
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

    VkDescriptorSetLayoutBinding bindings[2] = {uboLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    create_info.bindingCount = 2;
    create_info.pBindings = bindings;

    VkDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(device, &create_info, nullptr, &layout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }
    return layout;
  }

  SoftImage(VkPhysicalDevice &physical_device,
            VkDevice &device,
            VkQueue &graphics_queue,
            VkPipelineLayout &pipeline_layout,
            VkPipeline &graphics_pipeline,
            VkCommandPool &command_pool,
            VkDescriptorPool &descriptor_pool,
            VkDescriptorSetLayout descriptor_set_layout,
            uint32_t swapchain_images_size,
            uint32_t frames_in_flight,
            uint32_t subdivisions)
      : _physical_device(physical_device),
        _device(device),
        _graphics_queue(graphics_queue),
        _pipeline_layout(pipeline_layout),
        _graphics_pipeline(graphics_pipeline),
        _command_pool(command_pool),
        _descriptor_pool(descriptor_pool),
        _descriptor_set_layout(std::move(descriptor_set_layout)),
        _swapchain_images_size(swapchain_images_size),
        _frames_in_flight(frames_in_flight) {
    createTextureImage();
    createTextureImageView();
    createTextureSampler();

    loadGrid(subdivisions);
    // loadModel();

    createConstraints();
    printf("vertices: %lu constraints: %lu\n", _vertices.size(), _constraints.size());

    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorSets();
    createCopyCommandBuffers();
    createSyncObjects();
  }

  ~SoftImage() {
    for (auto &semaphore : _vertex_copy_semaphores) {
      vkDestroySemaphore(_device, semaphore, nullptr);
    }

    vkDestroySampler(_device, _texture_sampler, nullptr);
    vkDestroyImageView(_device, _texture_image_view, nullptr);
    vkDestroyImage(_device, _texture_image, nullptr);
    vkFreeMemory(_device, _texture_image_memory, nullptr);

    vkDestroyDescriptorSetLayout(_device, _descriptor_set_layout, nullptr);

    for (size_t i = 0; i < _swapchain_images_size; i++) {
      vkDestroyBuffer(_device, _uniform_buffers[i], nullptr);
      vkFreeMemory(_device, _uniform_buffers_memory[i], nullptr);
      vkDestroyBuffer(_device, _staging_buffers[i], nullptr);
      vkFreeMemory(_device, _staging_buffer_memory[i], nullptr);
      vkDestroyBuffer(_device, _vertex_buffers[i], nullptr);
      vkFreeMemory(_device, _vertex_buffer_memory[i], nullptr);
    }
    vkDestroyBuffer(_device, _index_buffer, nullptr);
    vkFreeMemory(_device, _index_buffer_memory, nullptr);
  }

  const VkDescriptorSetLayout &descriptor_set_layout() const { return _descriptor_set_layout; }

  void recordDrawCommands(VkCommandBuffer &command_buffer, uint32_t image_index) {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphics_pipeline);

    VkBuffer vertex_buffers[] = {_vertex_buffers[image_index]};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);
    vkCmdBindIndexBuffer(command_buffer, _index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            _pipeline_layout,
                            0,
                            1,
                            &_descriptor_sets[image_index],
                            0,
                            nullptr);

    vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(_indices.size()), 1, 0, 0, 0);
  }

  void start() { _previous_update = std::chrono::high_resolution_clock::now(); }

  VkSemaphore updateVertices(uint32_t current_frame, uint32_t current_image) {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_vertex_copy_command_buffers[current_image];

    const auto size = sizeof(Vertex) * _vertices.size();

    using namespace std::literals::chrono_literals;
    // static const auto start_time = std::chrono::high_resolution_clock::now();
    const auto now = std::chrono::high_resolution_clock::now();
    const auto elapsed_time = static_cast<float>((now - _previous_update) / 1ms) / 1000;
    _previous_update = now;

    // for (auto i = 0; i < _vertices.size(); ++i) {
    //   _vertices[i].pos =
    //       _rest_positions[i] + glm::vec3(0.025f * sin(i + 1.0f * (now - start_time) / 123ms),
    //                                      0.050f * sin(i + 1.0f * (now - start_time) / 234ms),
    //                                      0.075f * sin(i + 1.0f * (now - start_time) / 345ms));
    // }

    // symplectic Euler
    for (auto i = 0; i < _vertices.size(); ++i) {
      // _pbd_velocities[i] += elapsed_time * glm::vec3(0, 0, -9.82f);
      _pbd_positions[i] = _vertices[i].pos + elapsed_time * _pbd_velocities[i];
    }

    std::map<Indices, Solve> constraints;

    // manual drag
    // if (_drag) {
    //   auto &[index, position] = *_drag;
    //   printf("drag %d\n", index);
    // }

    if (_drag && _drag->first < _vertices.size()) {
      auto &[index, position] = *_drag;
      constraints.insert(makeCollisionConstraint(index, position));
    }

    for (auto i = 0; i < _solver_iterations; ++i) {
      // projectConstraints(C1,...,CM+Mcoll, p1,...,pN, s1,...,sN)
      for (auto &[i, solve] : _constraints) {
        solve(_solver_iterations, _pbd_positions, _rest_positions);
      }
      for (auto &[i, solve] : constraints) {
        solve(_solver_iterations, _pbd_positions, _rest_positions);
      }
    }

    constraints.clear();

    // Layer Blending Function
    for (auto i = 0; i < _vertices.size(); ++i) {
      // const auto weight = _pbd_weights[i];
      const auto weight = _weight;
      _pbd_positions[i] = (1 - weight) * _rest_positions[i] + weight * _pbd_positions[i];
    }

    // Calculate velocities from updated positions
    for (auto i = 0; i < _vertices.size(); ++i) {
      _pbd_velocities[i] = (_pbd_positions[i] - _vertices[i].pos) / elapsed_time;
      _vertices[i].pos = _pbd_positions[i];
    }

    void *data;
    vkMapMemory(_device, _staging_buffer_memory[current_image], 0, size, 0, &data);
    memcpy(data, _vertices.data(), static_cast<size_t>(size));
    vkUnmapMemory(_device, _staging_buffer_memory[current_image]);

    VkSemaphore signal_semaphores[] = {_vertex_copy_semaphores[current_frame]};
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    if (vkQueueSubmit(_graphics_queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit copy vertices command buffer!");
    }

    return _vertex_copy_semaphores[current_frame];
  }

  void updateUniformBuffer(uint32_t current_image, float aspect) {
    using namespace std::literals::chrono_literals;
    static const auto start_time = std::chrono::high_resolution_clock::now();
    const auto now = std::chrono::high_resolution_clock::now();
    const auto angle = (3.141592f / 2) * (now - start_time) / 1s;

    UniformBufferObject ubo{};
    // ubo.model = glm::rotate(glm::mat4(1.0f), angle, up);
    // ubo.view = glm::lookAt(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), up);
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(_camera_eye, _camera_center, _camera_up);
    ubo.proj = glm::perspective(glm::radians(45.0f), aspect, 0.2f, 20.0f);

    // OpenGL -> Vulkan
    ubo.proj[1][1] *= -1;

    void *data;
    vkMapMemory(_device, _uniform_buffers_memory[current_image], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(_device, _uniform_buffers_memory[current_image]);
  }

  void onDragStart(float x, float y, float aspect) {
    const auto view = glm::lookAt(_camera_eye, _camera_center, _camera_up);
    const auto proj = glm::perspective(glm::radians(45.0f), aspect, 0.2f, 20.0f);

    auto ray_eye = glm::inverse(proj) * glm::vec4(x, y, 0.0, 1.0);
    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, 1.0f, 0.0f);
    auto ray_world = glm::normalize(glm::vec3(glm::inverse(view) * ray_eye));

    if (auto index = tryRaycastVertex(_camera_eye, ray_world)) {
      _drag = std::pair{*index, _vertices[*index].pos};
    }
  }

  void onDragEnd() {
    _drag = {};
    _prev = {};
  }

  void onDragMove(float xpos, float ypos) {
    // Drag mesh
    if (_drag && _prev) {
      glm::vec2 dpos(_prev->x - xpos, ypos - _prev->y);
      _drag->second += 0.01f * glm::vec3(dpos, 0);
      _drag->second.z = 0.5f;
    }
    _prev = {xpos, ypos};
  }

  std::optional<uint32_t> tryRaycastVertex(glm::vec3 origin, glm::vec3 dir) const {
    std::optional<uint32_t> index;
    float minDist = FLT_MAX;
    for (uint32_t i = 0; i < _vertices.size(); i++) {
      glm::vec3 a = _vertices[i].pos - origin;
      glm::vec3 b = glm::dot(a, dir) * dir;
      if (glm::length2(a - b) < 0.02f) {
        float d = glm::length2(b);
        if (d < minDist) {
          minDist = d;
          index = i;
        }
      }
    }
    return index;
  }

 private:
  void createTextureImage() {
    int width, height, channels;
    const auto path = "../src/assets/textures/c8a60bd180d5efe02e44cb44634802fbc95f8e49.jpeg";
    auto *pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
    // auto *pixels = stbi_load(TEXTURE_PATH.c_str(), &width, &height, &channels, STBI_rgb_alpha);
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
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    // samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    // samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    // samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
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

  void loadGrid(uint32_t subdivisions) {
    const auto dp = 2.0f / subdivisions;
    const auto dt = 1.0f / subdivisions;
    const auto start = -dp * static_cast<float>(subdivisions) / 2;
    auto x = start;
    auto u = 1.0f;
    for (auto i = 0; i <= subdivisions; ++i, x += dp, u -= dt) {
      auto y = start;
      auto v = 0.0f;
      for (auto j = 0; j <= subdivisions; ++j, y += dp, v += dt) {
        glm::vec3 pos = {x, y, 0};
        _vertices.push_back({pos, {}, {u, v}});
        _rest_positions.push_back(pos);
        _pbd_positions.push_back(pos);
        _pbd_velocities.push_back({0, 0, 0});
        _pbd_weights.push_back(0.25f);
      }
    }
    for (auto i = 0; i < subdivisions; ++i) {
      auto a = i * (subdivisions + 1);
      auto b = (i + 1) * (subdivisions + 1);
      for (auto j = 0; j < subdivisions; ++j) {
        _indices.push_back(a + j);
        _indices.push_back(b + j);
        _indices.push_back(b + j + 1);
        _indices.push_back(b + j + 1);
        _indices.push_back(a + j + 1);
        _indices.push_back(a + j);
      }
    }
  }

  void loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
      throw std::runtime_error(warn + err);
    }
    std::unordered_map<Vertex, uint32_t> unique_vertices;
    std::unordered_map<int, uint32_t> vertex_index_map;

    std::vector<float> z;

    for (const auto &shape : shapes) {
      for (const auto &index : shape.mesh.indices) {
        glm::vec3 pos = {attrib.vertices[3 * index.vertex_index + 0],
                         attrib.vertices[3 * index.vertex_index + 1],
                         attrib.vertices[3 * index.vertex_index + 2]};
        glm::vec2 texcoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                              1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};
        Vertex vertex{pos, {1.0f, 1.0f, 1.0f}, texcoord};
        if (unique_vertices.count(vertex) == 0) {
          const auto i = static_cast<uint32_t>(_vertices.size());
          unique_vertices[vertex] = vertex_index_map[index.vertex_index] = i;
          _vertices.push_back(vertex);
          _rest_positions.push_back(pos);
          _pbd_positions.push_back(pos);
          _pbd_velocities.push_back({0, 0, 0});
          _pbd_weights.push_back(0.25f);
          // z.push_back(pos.z);
        }
        _indices.push_back(unique_vertices[vertex]);
      }
    }

    std::sort(z.begin(), z.end());
    printf("p0: %f p10: %f p25: %f\n", z[0], z[z.size() / 10], z[z.size() / 4]);
    z.clear();
  }

  void createConstraints() {
    assert(_indices.size() % 3 == 0);
    std::map<std::array<uint32_t, 2>, std::vector<uint32_t>> adjacent_faces;

    // Add a distance constraint for each edge in a triangle
    for (auto index = _indices.begin(); index != _indices.end(); index += 3) {
      auto c1 = makeDistanceConstraint({*index, *(index + 1)}, _pbd_weights);
      auto c2 = makeDistanceConstraint({*index, *(index + 2)}, _pbd_weights);
      auto c3 = makeDistanceConstraint({*(index + 1), *(index + 2)}, _pbd_weights);
      _constraints.insert(c1);
      _constraints.insert(c2);
      _constraints.insert(c3);
      adjacent_faces[{c1.first[0], c1.first[1]}].push_back(*(index + 2));
      adjacent_faces[{c2.first[0], c2.first[1]}].push_back(*(index + 1));
      adjacent_faces[{c3.first[0], c3.first[1]}].push_back(*index);
      // for (auto o = 0; o < 3; ++o) {
      //   auto i = *(index + o);
      //   auto pos = _vertices[i].pos;
      //   if (pos.z < 0.15f) {
      //     _constraints.insert(makeCollisionConstraint(i, pos));
      //   }
      // }
    }
    for (auto &[edge, corners] : adjacent_faces) {
      if (corners.size() > 1) {
        auto &[i0, i1] = edge;
        _constraints.insert(makeBendingConstraint({i0, i1, corners[0], corners[1]}, _pbd_weights));
      }
    }
  }

  void createVertexBuffer() {
    _staging_buffers.resize(_swapchain_images_size);
    _staging_buffer_memory.resize(_swapchain_images_size);
    _vertex_buffers.resize(_swapchain_images_size);
    _vertex_buffer_memory.resize(_swapchain_images_size);

    const auto size = sizeof(Vertex) * _vertices.size();

    for (auto i = 0; i < _swapchain_images_size; ++i) {
      createBuffer(size,
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   _staging_buffers[i],
                   _staging_buffer_memory[i]);
      createBuffer(size,
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   _vertex_buffers[i],
                   _vertex_buffer_memory[i]);

      // todo: don't do this when we copy every frame
      void *data;
      vkMapMemory(_device, _staging_buffer_memory[i], 0, size, 0, &data);
      memcpy(data, _vertices.data(), static_cast<size_t>(size));
      vkUnmapMemory(_device, _staging_buffer_memory[i]);

      copyBuffer(_staging_buffers[i], _vertex_buffers[i], size);
    }
  }

  void createIndexBuffer() {
    const auto size = sizeof(uint32_t) * _indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,
                 stagingBufferMemory);

    void *data;
    vkMapMemory(_device, stagingBufferMemory, 0, size, 0, &data);
    memcpy(data, _indices.data(), static_cast<size_t>(size));
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

    _uniform_buffers.resize(_swapchain_images_size);
    _uniform_buffers_memory.resize(_swapchain_images_size);

    for (size_t i = 0; i < _swapchain_images_size; i++) {
      createBuffer(size,
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   _uniform_buffers[i],
                   _uniform_buffers_memory[i]);
    }
  }

  void createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(_swapchain_images_size, _descriptor_set_layout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _descriptor_pool;
    allocInfo.descriptorSetCount = _swapchain_images_size;
    allocInfo.pSetLayouts = layouts.data();

    _descriptor_sets.resize(_swapchain_images_size);
    if (vkAllocateDescriptorSets(_device, &allocInfo, _descriptor_sets.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < _swapchain_images_size; i++) {
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

  void createCopyCommandBuffers() {
    _vertex_copy_command_buffers.resize(_swapchain_images_size);

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = _command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(_vertex_copy_command_buffers.size());

    if (vkAllocateCommandBuffers(_device, &alloc_info, _vertex_copy_command_buffers.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }

    const auto size = sizeof(Vertex) * _vertices.size();

    for (auto i = 0; i < _vertex_copy_command_buffers.size(); ++i) {
      auto &command_buffer = _vertex_copy_command_buffers[i];
      VkCommandBufferBeginInfo begin_info{};
      begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

      if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
      }

      VkBufferCopy region{};
      region.size = size;
      vkCmdCopyBuffer(command_buffer, _staging_buffers[i], _vertex_buffers[i], 1, &region);

      if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  void createSyncObjects() {
    _vertex_copy_semaphores.resize(_frames_in_flight);

    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (auto &semaphore : _vertex_copy_semaphores) {
      if (vkCreateSemaphore(_device, &info, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("failed to create semaphores for soft image");
      }
    }
  }

 private:
  void createBuffer(VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties,
                    VkBuffer &buffer,
                    VkDeviceMemory &buffer_memory) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(_device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(_device, buffer, &requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = requirements.size;
    alloc_info.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(_device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(_device, buffer, buffer_memory, 0);
  }

  void createImage(uint32_t width,
                   uint32_t height,
                   VkFormat format,
                   VkImageTiling tiling,
                   VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties,
                   VkImage &image,
                   VkDeviceMemory &image_memory) {
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = tiling;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = usage;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(_device, &image_info, nullptr, &image) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(_device, image, &requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = requirements.size;
    alloc_info.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(_device, &alloc_info, nullptr, &image_memory) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(_device, image, image_memory, 0);
  }

  uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(_physical_device, &memory_properties);

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
      if ((filter & (1 << i)) &&
          (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }
    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createImageView(VkImage image,
                       VkFormat format,
                       VkImageAspectFlags aspect_flags,
                       VkImageView &image_view) {
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = aspect_flags;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(_device, &view_info, nullptr, &image_view) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image view!");
    }
  }

  VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = _command_pool;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(_device, &alloc_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(command_buffer, &begin_info);

    return command_buffer;
  }

  void endSingleTimeCommands(VkCommandBuffer command_buffer) {
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer;

    vkQueueSubmit(_graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_graphics_queue);

    vkFreeCommandBuffers(_device, _command_pool, 1, &command_buffer);
  }

  void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    auto commandBuffer = beginSingleTimeCommands();

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &region);

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
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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
  VkPhysicalDevice &_physical_device;
  VkDevice &_device;
  VkQueue &_graphics_queue;
  VkPipelineLayout &_pipeline_layout;
  VkPipeline &_graphics_pipeline;

  VkCommandPool &_command_pool;
  VkDescriptorPool &_descriptor_pool;

  uint32_t _swapchain_images_size{0};
  uint32_t _frames_in_flight{0};

  const glm::vec3 _camera_eye{0.0f, 0.5f, 4.5f};
  const glm::vec3 _camera_center{0.0f, 0.0f, 0.0f};
  const glm::vec3 _camera_up{0.0f, 0.0f, 1.0f};

  std::vector<Vertex> _vertices;
  std::vector<uint32_t> _indices;

  std::vector<glm::vec3> _rest_positions;
  std::vector<glm::vec3> _pbd_positions;
  std::vector<glm::vec3> _pbd_velocities;
  std::vector<float> _pbd_weights;

  VkDescriptorSetLayout _descriptor_set_layout;

  std::vector<VkBuffer> _vertex_buffers;
  std::vector<VkDeviceMemory> _vertex_buffer_memory;
  std::vector<VkBuffer> _staging_buffers;
  std::vector<VkDeviceMemory> _staging_buffer_memory;

  VkBuffer _index_buffer;
  VkDeviceMemory _index_buffer_memory;
  VkImage _texture_image;
  VkDeviceMemory _texture_image_memory;

  VkImageView _texture_image_view;
  VkSampler _texture_sampler;

  std::vector<VkBuffer> _uniform_buffers;
  std::vector<VkDeviceMemory> _uniform_buffers_memory;

  std::vector<VkDescriptorSet> _descriptor_sets;

  std::vector<VkCommandBuffer> _vertex_copy_command_buffers;
  std::vector<VkSemaphore> _vertex_copy_semaphores;

  std::map<Indices, Solve> _constraints;

  size_t _solver_iterations{6};
  float _weight{0.8f};

  std::chrono::high_resolution_clock::time_point _previous_update;

  std::optional<std::pair<uint32_t, glm::vec3>> _drag;
  std::optional<glm::vec2> _prev;
};

}  // namespace graphics
