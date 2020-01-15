#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<unsigned char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }
  const auto file_size = file.tellg();
  std::vector<unsigned char> buffer(file_size);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(buffer.data()), file_size);
  return buffer;
}
