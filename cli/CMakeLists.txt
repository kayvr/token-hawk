cmake_minimum_required(VERSION 3.24)
project(tokenhawk VERSION 0.1.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DAWN_ENABLE_D3D12 OFF)
set(DAWN_ENABLE_METAL OFF)
set(DAWN_ENABLE_DESKTOP_GL OFF)
set(DAWN_ENABLE_OPENGLES OFF)
set(DAWN_BUILD_SAMPLES OFF)
set(DAWN_BUILD_NODE_BINDINGS OFF)
set(TINT_BUILD_TESTS OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dawn dawn)

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../*.cpp)
file(GLOB headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../*.hpp)

file(GLOB sources_llama_cpp CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../llama-cpp/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../llama-cpp/*.c)
file(GLOB headers_llama_cpp CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../llama-cpp/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/../llama-cpp/*.h)

file(GLOB sources_cli CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB headers_cli CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

set(sources ${sources} ${sources_cli} ${sources_llama_cpp})
set(headers ${headers} ${headers_cli} ${headers_llama_cpp})

add_executable(th ${sources} ${headers})
add_compile_options(th PRIVATE
        -Werror
        -Wall
        -Wextra
        -Wmissing-field-initializers)

target_include_directories(th PRIVATE ../)

target_link_libraries(th PRIVATE
        dawn_public_config
        dawn_native
        dawn_proc)
