cmake_minimum_required(VERSION 3.6.0)

project (flappy-bird)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo")

add_executable(flappy-bird main.cpp)

target_link_libraries(flappy-bird glfw3)
