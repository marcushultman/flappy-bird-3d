#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <utility>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
// #include <glm/gtc/matrix_transform.hpp>

namespace flappy {

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

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

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

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

const std::vector<Vertex> kVertices{{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                                    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                                    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
                                    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

                                    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                                    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                                    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
                                    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}};

const std::vector<uint16_t> kIndices{0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};

class SoftImage {
 public:
  SoftImage(VkPhysicalDevice &physical_device,
            VkDevice &device,
            VkQueue &graphics_queue,
            VkPipelineLayout &pipeline_layout,
            VkPipeline &graphics_pipeline,
            VkCommandPool &command_pool,
            VkDescriptorPool &descriptor_pool,
            uint32_t swapchain_images_size)
      : _physical_device(physical_device),
        _device(device),
        _graphics_queue(graphics_queue),
        _pipeline_layout(pipeline_layout),
        _graphics_pipeline(graphics_pipeline),
        _command_pool(command_pool),
        _descriptor_pool(descriptor_pool),
        _swapchain_images_size(swapchain_images_size) {}

  void init() {
    createDescriptorSetLayout();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    // createUniformBuffers();
    createDescriptorSets();
  }

  ~SoftImage() {}

  void cleanup() {
    vkDestroySampler(_device, _texture_sampler, nullptr);
    vkDestroyImageView(_device, _texture_image_view, nullptr);
    vkDestroyImage(_device, _texture_image, nullptr);
    vkFreeMemory(_device, _texture_image_memory, nullptr);

    vkDestroyDescriptorSetLayout(_device, _descriptor_set_layout, nullptr);

    // todo: do we need uniform data?
    // for (size_t i = 0; i < images.size(); i++) {
    //   vkDestroyBuffer(_device, _uniform_buffers[i], nullptr);
    //   vkFreeMemory(_device, _uniform_buffers_memory[i], nullptr);
    // }

    vkDestroyBuffer(_device, _index_buffer, nullptr);
    vkFreeMemory(_device, _index_buffer_memory, nullptr);
    vkDestroyBuffer(_device, _staging_buffer, nullptr);
    vkFreeMemory(_device, _staging_buffer_memory, nullptr);
    vkDestroyBuffer(_device, _vertex_buffer, nullptr);
    vkFreeMemory(_device, _vertex_buffer_memory, nullptr);
  }

  void recordDrawCommands(VkCommandBuffer &command_buffer, uint32_t image_index) {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphics_pipeline);

    VkBuffer vertex_buffers[] = {_vertex_buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);
    vkCmdBindIndexBuffer(command_buffer, _index_buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            _pipeline_layout,
                            0,
                            1,
                            &_descriptor_sets[image_index],
                            0,
                            nullptr);

    vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(kIndices.size()), 1, 0, 0, 0);
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

    createBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 _staging_buffer,
                 _staging_buffer_memory);

    createBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 _vertex_buffer,
                 _vertex_buffer_memory);

    // todo: move to update()
    void *data;
    vkMapMemory(_device, _staging_buffer_memory, 0, size, 0, &data);
    memcpy(data, kVertices.data(), static_cast<size_t>(size));
    vkUnmapMemory(_device, _staging_buffer_memory);

    copyBuffer(_staging_buffer, _vertex_buffer, size);
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

  // todo: if we need uniform
  // void createUniformBuffers() {
  //   const auto size = sizeof(UniformBufferObject);

  //   _uniform_buffers.resize(_swapchain_images_size;
  //   _uniform_buffers_memory.resize(_swapchain_images_size);

  //   for (size_t i = 0; i < _swapchain_images_size; i++) {
  //     createBuffer(size,
  //                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
  //                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
  //                  _uniform_buffers[i],
  //                  _uniform_buffers_memory[i]);
  //   }
  // }

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
      // bufferInfo.buffer = _uniform_buffers[i];
      // bufferInfo.offset = 0;
      // bufferInfo.range = sizeof(UniformBufferObject);

      // todo: 0 uniform
      bufferInfo.buffer = nullptr;
      bufferInfo.offset = 0;
      bufferInfo.range = 0;

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
  // void updateUniformBuffer(uint32_t current_image) {
  //   using namespace std::literals::chrono_literals;
  //   static const auto start_time = std::chrono::high_resolution_clock::now();
  //   const auto now = std::chrono::high_resolution_clock::now();
  //   const auto up = glm::vec3(0.0f, 0.0f, 1.0f);
  //   const auto angle = (3.141592f / 2) * (now - start_time) / 1s;
  //   const auto aspect = _swapchain_extent.width / static_cast<float>(_swapchain_extent.height);

  //   UniformBufferObject ubo{};
  //   ubo.model = glm::rotate(glm::mat4(1.0f), angle, up);
  //   ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), up);
  //   ubo.proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10.0f);

  //   // OpenGL -> Vulkan
  //   ubo.proj[1][1] *= -1;

  //   void *data;
  //   vkMapMemory(_device, _uniform_buffers_memory[current_image], 0, sizeof(ubo), 0, &data);
  //   memcpy(data, &ubo, sizeof(ubo));
  //   vkUnmapMemory(_device, _uniform_buffers_memory[current_image]);
  // }

 private:
  VkPhysicalDevice &_physical_device;
  VkDevice &_device;
  VkQueue &_graphics_queue;
  VkPipelineLayout &_pipeline_layout;
  VkPipeline &_graphics_pipeline;

  VkCommandPool &_command_pool;
  VkDescriptorPool &_descriptor_pool;

  uint32_t _swapchain_images_size{0};

  VkDescriptorSetLayout _descriptor_set_layout;

  VkBuffer _vertex_buffer;
  VkDeviceMemory _vertex_buffer_memory;
  VkBuffer _staging_buffer;
  VkDeviceMemory _staging_buffer_memory;
  VkBuffer _index_buffer;
  VkDeviceMemory _index_buffer_memory;
  VkImage _texture_image;
  VkDeviceMemory _texture_image_memory;

  VkImageView _texture_image_view;
  VkSampler _texture_sampler;

  // std::vector<VkBuffer> _uniform_buffers;
  // std::vector<VkDeviceMemory> _uniform_buffers_memory;

  std::vector<VkDescriptorSet> _descriptor_sets;
};

}  // namespace flappy
