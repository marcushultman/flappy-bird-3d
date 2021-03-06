cmake_minimum_required(VERSION 3.12)

project(flappy-bird)

set(GLFW_VULKAN_STATIC 0)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(VULKAN_SDK $ENV{VULKAN_SDK})

add_subdirectory(vendor/glfw)

set(NOWPLAYING_HACK /Users/hultman/spotify/client-android/native/src/core/bindings/android/nowplaying-hack/jni)

include_directories(src/vulkan)
include_directories(${NOWPLAYING_HACK}/src)
include_directories(${NOWPLAYING_HACK}/shaders)
include_directories(vendor/glfw/include)
include_directories(vendor/tinyobjloader)
include_directories(${VULKAN_SDK}/include)
include_directories(/usr/local/include)

include_directories(/Users/hultman/spotify/client-android/native/src/core/vendor/stb)

link_directories(${VULKAN_SDK}/lib)

find_package(curl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo")

add_executable(flappy-bird src/main2.cpp)

target_sources(flappy-bird PUBLIC
  ${NOWPLAYING_HACK}/src/app.cpp
  ${NOWPLAYING_HACK}/src/constraint.cpp
  ${NOWPLAYING_HACK}/src/soft_image.cpp
)

set(GLSLANG ${VULKAN_SDK}/bin/glslangValidator)
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/src/assets/shaders)

add_custom_command(OUTPUT shaders/vert.spv
  DEPENDS ${SHADER_DIR}/main.vert
  COMMAND ${GLSLANG} -V ${SHADER_DIR}/main.vert
  WORKING_DIRECTORY shaders
  COMMENT "Compiling shader...")

add_custom_command(OUTPUT shaders/frag.spv
  DEPENDS ${SHADER_DIR}/main.frag
  COMMAND ${GLSLANG} -V ${SHADER_DIR}/main.frag
  WORKING_DIRECTORY shaders
  COMMENT "Compiling shader...")
  
add_custom_target(shaders DEPENDS shaders/vert.spv shaders/frag.spv)

add_dependencies(flappy-bird shaders)

target_link_libraries(flappy-bird glfw vulkan curl)
