# HCV project
cmake_minimum_required(VERSION 3.16)
project(HCV CXX)

# C++ compiler standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add 3rd party tools
include(cmake/args.cmake)
include(cmake/tflite.cmake)
include(cmake/spdlog.cmake)

# Application's executable
add_executable(hcvmain src/hcvmain.cpp)
target_include_directories(hcvmain PRIVATE ${args_SOURCE_DIR})
target_link_libraries(hcvmain PRIVATE tflite spdlog)
