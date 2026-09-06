cmake_minimum_required(VERSION 3.24)
project(sm90_cute_gemm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 直接指定 CUTLASS 源码目录，无需 DeepGEMM。接口参照 CUTLASS v3.9.2。
set(CUTLASS_ROOT "" CACHE PATH "Path to a CUTLASS source tree")
set(CUTLASS_INCLUDE "${CUTLASS_ROOT}/include")
if(NOT EXISTS "${CUTLASS_INCLUDE}/cute/tensor.hpp")
    message(FATAL_ERROR
        "CUTLASS headers not found. Configure with -DCUTLASS_ROOT=/path/to/cutlass")
endif()

add_library(sm90_fp8_persistent_1d2d OBJECT sm90_fp8_persistent_1d2d.cu)
target_include_directories(sm90_fp8_persistent_1d2d PRIVATE
    "${CUTLASS_INCLUDE}")
target_link_libraries(sm90_fp8_persistent_1d2d PRIVATE CUDA::cudart CUDA::cuda_driver)
target_compile_features(sm90_fp8_persistent_1d2d PRIVATE cxx_std_20 cuda_std_20)

# WGMMA 依赖 Hopper architecture-specific target，必须使用 sm_90a 而不是 sm_90。
set_target_properties(sm90_fp8_persistent_1d2d PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_options(sm90_fp8_persistent_1d2d PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-arch=sm_90a;--expt-relaxed-constexpr>")
