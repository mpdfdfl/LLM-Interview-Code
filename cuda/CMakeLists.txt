cmake_minimum_required(VERSION 3.24)
project(sm90_cute_gemm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 默认假设本仓库与 DeepGEMM 位于同一个父目录；也可用
# cmake -DDEEP_GEMM_ROOT=/path/to/DeepGEMM 覆盖。
set(DEEP_GEMM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../DeepGEMM"
    CACHE PATH "Path to a DeepGEMM source tree")

set(DEEP_GEMM_INCLUDE "${DEEP_GEMM_ROOT}/deep_gemm/include")
set(CUTLASS_INCLUDE "${DEEP_GEMM_ROOT}/third-party/cutlass/include")

if(NOT EXISTS "${DEEP_GEMM_INCLUDE}/deep_gemm/mma/sm90.cuh")
    message(FATAL_ERROR "DeepGEMM headers not found: ${DEEP_GEMM_ROOT}")
endif()
if(NOT EXISTS "${CUTLASS_INCLUDE}/cute/tensor.hpp")
    message(FATAL_ERROR
        "CUTLASS submodule not found. Run: "
        "git -C ${DEEP_GEMM_ROOT} submodule update --init third-party/cutlass")
endif()

add_library(sm90_fp8_persistent_1d2d OBJECT sm90_fp8_persistent_1d2d.cu)
target_include_directories(sm90_fp8_persistent_1d2d PRIVATE
    "${DEEP_GEMM_INCLUDE}"
    "${CUTLASS_INCLUDE}")
target_compile_features(sm90_fp8_persistent_1d2d PRIVATE cxx_std_20 cuda_std_20)

# WGMMA 依赖 Hopper architecture-specific target，必须使用 sm_90a 而不是 sm_90。
set_target_properties(sm90_fp8_persistent_1d2d PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_options(sm90_fp8_persistent_1d2d PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-arch=sm_90a;--expt-relaxed-constexpr>")
