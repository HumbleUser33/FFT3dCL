cmake_minimum_required(VERSION 3.20)
project(fft3dgpu_vs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(fft3dgpu SHARED
  src/fft3dgpu.cpp
  src/cpu/fft.cpp
  src/cpu/tables.cpp
  src/cpu/reference.cpp
  src/cl/cl.cpp
  src/cl/gpu.cpp
)

target_include_directories(fft3dgpu PRIVATE src src/cl src/cpu third_party/vapoursynth)

if (MSVC)
  # /fp:precise = IEEE-compliant, no fast-math reassociation (reproducibility).
  target_compile_options(fft3dgpu PRIVATE /W4 /fp:precise /EHsc /utf-8)
  target_compile_definitions(fft3dgpu PRIVATE _CRT_SECURE_NO_WARNINGS)
  # Static CRT so the DLL can be dropped into StaxRip's plugin folder.
  set_property(TARGET fft3dgpu PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# VapourSynth is not linked: the plugin receives the API through VapourSynthPluginInit2.

add_executable(fft_test tools/fft_test.cpp src/cpu/fft.cpp)
target_include_directories(fft_test PRIVATE src/cpu)
if (MSVC)
  target_compile_options(fft_test PRIVATE /W4 /fp:precise /EHsc /utf-8)
  set_property(TARGET fft_test PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_executable(clinfo tools/clinfo.cpp)
if (MSVC)
  target_compile_options(clinfo PRIVATE /W4 /utf-8)
  set_property(TARGET clinfo PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_executable(gpu_test tools/gpu_test.cpp src/cl/cl.cpp src/cpu/reference.cpp
                        src/cpu/fft.cpp src/cpu/tables.cpp)
target_include_directories(gpu_test PRIVATE src src/cl src/cpu)
if (MSVC)
  target_compile_options(gpu_test PRIVATE /W4 /fp:precise /EHsc /utf-8)
  set_property(TARGET gpu_test PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
