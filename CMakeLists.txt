cmake_minimum_required(VERSION 3.12)

project(flappy-bird)

set(GLFW_VULKAN_STATIC 0)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(VULKAN_SDK $ENV{VULKAN_SDK})

add_subdirectory(vendor/glfw)

include_directories(vendor/glfw/include)
include_directories(${VULKAN_SDK}/include)
include_directories(/usr/local/include)

link_directories(${VULKAN_SDK}/lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo")

add_executable(flappy-bird main.cpp)

target_sources(flappy-bird PUBLIC
  app.h
  bird.h
  game.h
)

target_link_libraries(flappy-bird glfw vulkan)
