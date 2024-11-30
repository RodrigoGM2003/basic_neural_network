cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER g++)

project(neural_network LANGUAGES CXX)

# Project name
project(neural_network)

# Set the C standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin)

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files (recursively add all .cpp files within /src and its subdirectories)
file(GLOB_RECURSE SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# Add executable
add_executable(main WIN32 ${SOURCES})

target_link_options(main PRIVATE -Wl,-subsystem,console)

# Set the output directory
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)