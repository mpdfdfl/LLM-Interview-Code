# SM90 FP8 1D2D Persistent GEMM：CUTLASS CuTe C++

[`sm90_fp8_persistent_1d2d.cu`](sm90_fp8_persistent_1d2d.cu) 用 CuTe 的 tensor、TiledCopy、TiledMMA 和 CUTLASS 的 `PipelineTmaAsync` 实现 kernel，并提供一个最小 host 入口。固定 `BM=64, BN=128, BK=128, Stages=4`，每个 CTA 包含一个 producer warp-group 和一个 consumer warp-group，共 256 threads。代码不依赖 DeepGEMM。

## 1. 计算和输入

计算 `A @ B^T`，在每个 K block 上恢复量化 scale：

```text
D[m,n] = sum_kb SFA[m,kb] * SFB[n/128,kb]
                 * dot(A[m, kb*128:(kb+1)*128],
                       B[n, kb*128:(kb+1)*128])
```

| 参数 | dtype | 逻辑形状 | CuTe stride |
|---|---|---|---|
| `A` | `cutlass::float_e4m3_t` | `[M,K]` | `(K,1)` |
| `B` | `cutlass::float_e4m3_t` | `[N,K]` | `(K,1)` |
| `SFA` | FP32 | `[M,K/128]` | `(1,M)` |
| `SFB` | FP32 | `[N/128,K/128]` | `(K/128,1)` |
| `D` | `cutlass::bfloat16_t` | `[M,N]` | `(N,1)` |

“1D2D”指 A 每 `1x128` 个元素共享 scale，B 每 `128x128` 个元素共享 scale。SFA 保留原来的 MN-major 存储，M 维连续，且 leading dimension 固定为 M。这里的 scale 是乘回反量化的系数。

## 2. CuTe 的实现主线

1. **Layout**：`tile_to_shape(GMMA::Layout_K_SW128_Atom<FP8>{}, Shape<MN,BK,Stages>{})` 把 stage 也作为 shared tensor 的一维。
2. **TMA**：host 用 `make_tma_copy` 创建 typed TiledCopy；device 用 `get_tma_tensor`、`local_tile`、`partition_S/D` 获取搬运视图，再执行 `copy(tma.with(barrier), src, dst)`。
3. **Pipeline**：`producer_acquire` 等待 stage 可写并设置 transaction bytes，TMA 硬件完成 full barrier；`consumer_wait` 等待数据，`consumer_release` 交还 stage。单 CTA cluster 下，全部 128 个 consumer 线程都调用 release。
4. **MMA**：`make_tiled_mma(SM90_64x128x32_F32E4M3E4M3_SS_TN<>{})` 定义一个 warp-group 的 MMA；`partition_A/B` 和 `make_fragment_A/B` 自动生成 shared-memory descriptor tensor。
5. **Scale**：每个 `BK=128` 清零 partial fragment，`cute::gemm` 遍历 4 个 K=32 atom；等待 WGMMA 完成后，乘当前 `SFA*SFB` 累加到最终 FP32 fragment。
6. **Epilogue**：对 identity tensor 做 `partition_C` 得到 accumulator 的行坐标，索引 SFA；对输出 tile 做同样的 `partition_C`，转换为 BF16 后直接写回。

producer 和 consumer 各自从 `blockIdx.x` 开始，按 `tile += gridDim.x` 的顺序遍历输出 tile。pipeline 的 index/phase 跨输出 tile 连续推进，不能每个 tile 重新初始化。producer 最后调用 `producer_tail` 等待消费完毕。

没有手写 WGMMA descriptor 位字段、PTX MMA 调用或 lane/register 到输出坐标的映射。SFB 每个 K block 直接从 global memory 读取，不再缓存整个 N tile 的 scale，因此 shared-memory 大小不随 K 增长。

## 3. Host 入口与约束

```cpp
cudaError_t launch_fp8_gemm_1d2d_persistent(
    FP8 const* a, FP8 const* b, float const* sfa, float const* sfb,
    BF16* d, int m, int n, int k, cudaStream_t stream = nullptr);
```

函数位于 `interview_gemm` namespace，负责创建三份 TMA copy、设置动态 shared memory、按 SM 数启动 persistent CTA。kernel 模板由这个入口实际实例化；原先直接接收裸 `TmaDescriptor` 的 kernel 参数已替换为 CuTe TiledCopy 类型。FP8 的 C++ 类型改为 `cutlass::float_e4m3_t`，存储仍为每元素一字节的 E4M3。

要求 Hopper SM90、`M,N,K > 0`、`M%64 == 0`、`N%128 == 0`、`K%128 == 0`，以及 A/B/SFA 起始地址 16B 对齐。输入输出缓冲区须在传入 stream 上保持有效直到执行结束。返回值检查提交错误；调用者还需检查 stream 同步时的执行错误。

这是教学内核：省略边界 predicate、cluster multicast、tile swizzle heuristic 和 TMA store epilogue；没有独立 benchmark 或数值测试程序。

## 4. 构建

需要支持 `sm_90a` 的 CUDA 12.x 工具链、CMake 3.24+ 和 CUTLASS 源码。CuTe 接口对照 [NVIDIA CUTLASS v3.9.2 Hopper 教程](https://github.com/NVIDIA/cutlass/blob/v3.9.2/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu) 与该版本头文件；建议使用这个版本复现。

```sh
cmake -S . -B build -DCUTLASS_ROOT=/path/to/cutlass
cmake --build build --target sm90_fp8_persistent_1d2d -j
```

[`CMakeLists.txt`](CMakeLists.txt) 生成 object library，编译目标为 `sm_90a`。集成可执行程序时需要链接 CUDA runtime 和 driver（host TMA descriptor 构造会使用 driver API）。
