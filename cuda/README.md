# SM90 FP8 1D2D Persistent GEMM 手撕版

[`sm90_fp8_persistent_1d2d.cu`](sm90_fp8_persistent_1d2d.cu) 写的是 **kernel 本体**，不是 DeepGEMM launcher，也不是为了复制一整套可运行框架。它固定 `BM=64, BN=128, BK=128, Stages=4`，把面试时最需要讲清的部分留在一个文件里：

- kernel 传什么数据；
- global/shared memory 的 CUTE layout；
- persistent CTA 怎么取 tile；
- TMA producer 和 WGMMA consumer 怎么用 barrier 交接；
- 1D2D scale 怎么乘回 FP32 accumulator。

## 1. 计算和输入

kernel 计算 NT GEMM：

```text
D[M, N] = A[M, K] @ B[N, K]^T
```

| 参数 | dtype | 逻辑形状 | 含义 |
|---|---|---|---|
| `A` | FP8 E4M3 | `[M, K]` | row-major，K 连续 |
| `B` | FP8 E4M3 | `[N, K]` | row-major，K 连续；计算时视为 `B^T` |
| `SFA` | FP32 | `[M, K/128]` | 每行、每 128 个 K 共享一个 scale |
| `SFB` | FP32 | `[N/128, K/128]` | 每个 `128 x 128` B block 共享一个 scale |
| `D` | BF16 | `[M, N]` | row-major 输出 |

对于一个 `(m, n)` 元素，教学版的数学形式是：

```text
D[m,n] = sum_kb SFA[m,kb] * SFB[n/128,kb]
                 * dot(A[m, kb*128:(kb+1)*128],
                       B[n, kb*128:(kb+1)*128])
```

“1D2D”说的是量化 block：A 是 `1 x 128`，B 是 `128 x 128`。它不是说 A/B 矩阵的维数。

## 2. 必须记住的 CUTE layout

global-memory 逻辑 layout：

```cpp
auto layout_A   = make_layout(make_shape(M, K),   make_stride(K, _1{}));
auto layout_B   = make_layout(make_shape(N, K),   make_stride(K, _1{}));
auto layout_D   = make_layout(make_shape(M, N),   make_stride(N, _1{}));

// DeepGEMM 会先把 SFA 转成 MN-major，使 M 维连续：
auto layout_SFA = make_layout(make_shape(M, Kb),  make_stride(_1{}, aligned_M));

// 普通 row-major [N/128, K/128]：
auto layout_SFB = make_layout(make_shape(Nb, Kb), make_stride(Kb, _1{}));
```

shared-memory A/B tile 要给 WGMMA 读，所以使用 Hopper 128B swizzle：

```cpp
using SmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<FP8>{}, Shape<Int<64>, Int<128>>{}));
using SmemLayoutB = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<FP8>{}, Shape<Int<128>, Int<128>>{}));
```

`TmaDescriptor` 在 host 端由 `A/B/SFA` 的地址、global shape/stride、shared tile shape 和 swizzle 生成。descriptor 已经嵌入 global pointer；手撕 kernel 里仍显式传 raw pointer，只是为了把 CUTE global tensor 和简化写回完整展示出来。

## 3. kernel 内部主线

1. 启动约 `num_sms` 个 CTA，每个 CTA 256 threads。
2. CTA 中前 128 threads 是 math warp-group，后 128 threads 是 TMA warp-group。
3. persistent scheduler 从 `blockIdx.x` 开始，每算完一个 tile 就 `tile += gridDim.x`。
4. producer 等 `empty[stage]`，用 DeepGEMM `tma::copy` 搬 `A[64,128]`、`B[128,128]` 和 `SFA[64]`，然后完成 `full[stage]`。
5. consumer 等 `full[stage]`，从 shared memory 建 WGMMA descriptor。
6. FP8 atom 是 `m64n128k32`，所以一个 `BK=128` tile 发射 4 次 WGMMA。
7. WGMMA 结果是 FP32 register accumulator，再乘 `SFA * SFB` 加入 `final_accum`。
8. consumer 完成该 stage 后通知 `empty[stage]`，producer 才能复用这块 shared memory。

## 4. 面试时可以主动说的简化

这份代码假设 `M%64 == 0, N%128 == 0, K%128 == 0`，且省略 cluster multicast、tile swizzle heuristic 和边界 predicate。写回也直接按 WGMMA register mapping 写 BF16 global memory；正式 DeepGEMM 会先用 STSM 写 shared memory，再用 TMA store 做 epilogue。

这些是有意的面试简化，不影响 kernel 的主干：`persistent scheduler -> TMA pipeline -> WGMMA -> scale promotion -> epilogue`。

## 5. 依赖

代码使用 CUTE/CUTLASS 和 DeepGEMM 的 `tma::copy`、SM90 MMA descriptor、PTX barrier/WGMMA utilities。[`CMakeLists.txt`](CMakeLists.txt) 只提供一个可选的 `sm_90a` 编译目标，不生成 launcher 或可执行文件。
