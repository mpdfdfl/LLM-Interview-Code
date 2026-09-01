/*
 * 面试手撕版：SM90 FP8 1D2D persistent GEMM kernel 本体
 *
 *   D[M, N] = A[M, K] @ B[N, K]^T
 *
 * 数据约定
 * --------------------------------------------------------------------------
 * A    : FP8 E4M3 [M, K]，row-major / K-major
 * B    : FP8 E4M3 [N, K]，row-major / K-major
 * SFA  : 逻辑形状 FP32 [M, K / 128]，DeepGEMM 预处理成 MN-major：
 *        offset(m, k_block) = m + k_block * aligned_M
 *        SFA[m, k_block] 是 A[m, k_block * 128 : (k_block + 1) * 128] 的 scale
 * SFB  : FP32 [N / 128, K / 128]，row-major / K-block contiguous
 *        SFB[n_block, k_block] 是 B[n_block * 128 : (n_block + 1) * 128,
 *                                      k_block * 128 : (k_block + 1) * 128] 的 scale
 * D    : BF16 [M, N]，row-major
 *
 * 为把主线讲清楚，假设 M % 64 == 0、N % 128 == 0、K % 128 == 0。
 * 真实 DeepGEMM 还会处理边界、cluster multicast、tile swizzle 和动态 heuristic。
 */

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>

#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace interview_gemm {

using namespace cute;
using FP8 = __nv_fp8_e4m3;
using BF16 = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 128;       // FP8 scale granularity
constexpr int Stages = 4;
constexpr int MathThreads = 128;
constexpr int TmaThreads = 128;
constexpr int Threads = MathThreads + TmaThreads;

// -----------------------------------------------------------------------------
// 1. CUTE layouts
// -----------------------------------------------------------------------------
// GMEM 的动态 layout 在 kernel 中创建：
//   A(M,K):   make_layout(make_shape(M,K), make_stride(K,_1{}))
//   B(N,K):   make_layout(make_shape(N,K), make_stride(K,_1{}))
//   D(M,N):   make_layout(make_shape(M,N), make_stride(N,_1{}))
//   SFA(M,Kb): make_layout(make_shape(M,Kb), make_stride(_1{},aligned_M))
//   SFB(Nb,Kb): make_layout(make_shape(Nb,Kb), make_stride(Kb,_1{}))
// SFA 之所以要 MN-major，是为了让一次 TMA 沿 M 连续搬 BM 个 scale。
// SFB 按 (n_block, k_block) 存一个 128x128 B block 共享的 scale。

// WGMMA 从 shared memory 读 A/B。K-major FP8 tile 的连续 128B 正好是 BK=128，
// 因此使用 Hopper 128B XOR-swizzle，避免 shared-memory bank conflict。
using SmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<FP8>{}, Shape<Int<BM>, Int<BK>>{}));
using SmemLayoutB = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<FP8>{}, Shape<Int<BN>, Int<BK>>{}));

// SFA 不参与 WGMMA，只需普通连续 layout：每个 stage 保存 BM 个 float scale。
using SmemLayoutSFA = Layout<Shape<Int<BM>>, Stride<_1>>;

// TMA descriptor 由 host 端根据下面这组 logical layout 创建，然后传入 kernel：
//   tma_a   : GMEM A[M,K]   -> SMEM SmemLayoutA[BM,BK]
//   tma_b   : GMEM B[N,K]   -> SMEM SmemLayoutB[BN,BK]
//   tma_sfa : GMEM logical SFA[M,Kb], stride=(1,aligned_M) -> SMEM contiguous[BM]
// 注意：descriptor 自己已经包含 A/B/SFA 的 global address。这里仍显式传 raw
// pointer，是为了能在代码中写出完整的 CUTE GMEM tensor，方便面试讲 layout。

// -----------------------------------------------------------------------------
// 2. Persistent scheduler
// -----------------------------------------------------------------------------
// 启动时 gridDim.x 通常等于 SM 数。一个 CTA 完成 tile 后继续领取下一个 tile，
// 所以 CTA 常驻 SM，而不是一个输出 tile 启动一个 CTA。
struct PersistentScheduler {
    int next_tile;
    int num_m_tiles;
    int num_tiles;

    CUTE_DEVICE PersistentScheduler(int m, int n)
        : next_tile(static_cast<int>(blockIdx.x)),
          num_m_tiles(m / BM),
          num_tiles((m / BM) * (n / BN)) {}

    CUTE_DEVICE bool next(int& m_tile, int& n_tile) {
        if (next_tile >= num_tiles)
            return false;
        m_tile = next_tile % num_m_tiles;
        n_tile = next_tile / num_m_tiles;
        next_tile += static_cast<int>(gridDim.x);
        return true;
    }
};

// -----------------------------------------------------------------------------
// 3. Kernel body
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(Threads, 1)
void fp8_gemm_1d2d_persistent(
    FP8 const* a, FP8 const* b,
    float const* sfa, float const* sfb,
    BF16* d, int m, int n, int k,
    const __grid_constant__ TmaDescriptor tma_a,
    const __grid_constant__ TmaDescriptor tma_b,
    const __grid_constant__ TmaDescriptor tma_sfa) {

    using WGMMA = typename deep_gemm::mma::sm90::FP8MMASelector<BN>::type;
    static_assert(WGMMA::M == 64 && WGMMA::N == BN && WGMMA::K == 32);

    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp_group = static_cast<int>(threadIdx.x) / 128;
    const int k_blocks = k / BK;
    const int n_blocks = n / BN;

    // Dynamic shared-memory partition：A/B/SFA 都是多 stage；SFB 属于当前 N tile，
    // 一次性载入后由整个 K loop 复用。
    extern __shared__ __align__(1024) uint8_t smem[];
    constexpr int AStageElems = cosize_v<SmemLayoutA>;
    constexpr int BStageElems = cosize_v<SmemLayoutB>;

    auto smem_a = deep_gemm::utils::PatternVisitor([&](uint32_t stage) {
        return reinterpret_cast<FP8*>(smem) + stage * AStageElems;
    });
    auto smem_b = deep_gemm::utils::PatternVisitor([&](uint32_t stage) {
        return reinterpret_cast<FP8*>(smem) + Stages * AStageElems
             + stage * BStageElems;
    });
    auto smem_sfa = deep_gemm::utils::PatternVisitor([&](uint32_t stage) {
        auto offset = Stages * (AStageElems + BStageElems) * sizeof(FP8);
        return reinterpret_cast<float*>(smem + offset) + stage * BM;
    });

    const int sfb_byte_offset =
        Stages * (AStageElems + BStageElems) * sizeof(FP8) +
        Stages * BM * sizeof(float);
    float* smem_sfb = reinterpret_cast<float*>(smem + sfb_byte_offset);

    const int barrier_byte_offset =
        (sfb_byte_offset + k_blocks * static_cast<int>(sizeof(float)) + 7) & ~7;
    Barrier* full = reinterpret_cast<Barrier*>(smem + barrier_byte_offset);
    Barrier* empty = full + Stages;

    // 每个 full barrier 等一个 TMA producer；每个 empty barrier 等 math WG 的 4 个 warp。
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < Stages; ++s) {
            full[s].init(1);
            empty[s].init(MathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // 这些 tensor 主要展示输入的 CUTE layout；A/B/SFA 的实际搬运由 TMA descriptor 完成。
    // 教学版假设 aligned_M == M；正式 DeepGEMM 会将 M 补到 TMA 对齐。
    auto gA = make_tensor(make_gmem_ptr(a),
                          make_shape(m, k),
                          make_stride(k, _1{}));
    auto gB = make_tensor(make_gmem_ptr(b),
                          make_shape(n, k),
                          make_stride(k, _1{}));
    auto gSFA = make_tensor(make_gmem_ptr(sfa),
                            make_shape(m, k_blocks),
                            make_stride(_1{}, m));
    auto gD = make_tensor(make_gmem_ptr(d),
                          make_shape(m, n),
                          make_stride(n, _1{}));
    auto gSFB = make_tensor(make_gmem_ptr(sfb),
                            make_shape(n_blocks, k_blocks),
                            make_stride(k_blocks, _1{}));
    (void)gA;
    (void)gB;
    (void)gSFA;

    // producer 和 consumer 各自维护相同的 scheduler/pipeline 状态。
    PersistentScheduler scheduler(m, n);
    uint32_t iteration = 0;

    if (warp_group == 1) {
        // ---------------------------------------------------------------------
        // Producer WG：一个 elected thread 发射 TMA，其他线程只用于低寄存器占用。
        // ---------------------------------------------------------------------
        cutlass::arch::warpgroup_reg_dealloc<40>();

        if (threadIdx.x == MathThreads) {
            int mt, nt;
            while (scheduler.next(mt, nt)) {
                for (int kb = 0; kb < k_blocks; ++kb, ++iteration) {
                    const int stage = iteration % Stages;
                    const int phase = (iteration / Stages) & 1;
                    empty[stage].wait(phase ^ 1);

                    auto& ready = full[stage];
                    deep_gemm::tma::copy<BK, BM, 128>(
                        &tma_a, &ready, smem_a[stage],
                        kb * BK, mt * BM);
                    deep_gemm::tma::copy<BK, BN, 128>(
                        &tma_b, &ready, smem_b[stage],
                        kb * BK, nt * BN);
                    deep_gemm::tma::copy<BM, BK, 0>(
                        &tma_sfa, &ready, smem_sfa[stage],
                        mt * BM, kb);

                    ready.arrive_and_expect_tx(
                        BM * BK * sizeof(FP8) +
                        BN * BK * sizeof(FP8) +
                        BM * sizeof(float));
                }
            }
        }
    } else {
        // ---------------------------------------------------------------------
        // Consumer WG：等待 TMA -> WGMMA -> 乘 scale -> 累加 -> 写 D。
        // ---------------------------------------------------------------------
        cutlass::arch::warpgroup_reg_alloc<248>();

        int mt, nt;
        while (scheduler.next(mt, nt)) {
            // 1D2D 的“2D”就在这里：B 的一个 scale 同时覆盖当前 128x128 (N,K) tile。
            for (int kb = static_cast<int>(threadIdx.x); kb < k_blocks; kb += MathThreads)
                smem_sfb[kb] = gSFB(nt, kb);
            cutlass::arch::NamedBarrier::sync(MathThreads, 0);

            float final_accum[WGMMA::kNumAccum] = {0.0f};

            for (int kb = 0; kb < k_blocks; ++kb, ++iteration) {
                const int stage = iteration % Stages;
                const int phase = (iteration / Stages) & 1;
                full[stage].wait(phase);

                float accum[WGMMA::kNumAccum];
                const int row0 = warp * 16 + lane / 4;
                const int row1 = row0 + 8;
                const float scale_a0 = smem_sfa[stage][row0];
                const float scale_a1 = smem_sfa[stage][row1];
                const float scale_b = smem_sfb[kb];

                auto desc_a = deep_gemm::mma::sm90::make_smem_desc(smem_a[stage], 1);
                auto desc_b = deep_gemm::mma::sm90::make_smem_desc(smem_b[stage], 1);

#pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum; ++i)
                    deep_gemm::ptx::warpgroup_fence_operand(accum[i]);
                deep_gemm::ptx::warpgroup_arrive();

                // FP8 WGMMA atom 是 m64n128k32，所以一个 BK=128 tile 发射 4 次。
#pragma unroll
                for (int mma_k = 0; mma_k < BK / WGMMA::K; ++mma_k) {
                    desc_a.reg32_[0] += mma_k == 0 ? 0 : WGMMA::K / 16;
                    desc_b.reg32_[0] += mma_k == 0 ? 0 : WGMMA::K / 16;
                    WGMMA::wgmma(desc_a, desc_b, accum, mma_k != 0);
                }
                deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum; ++i)
                    deep_gemm::ptx::warpgroup_fence_operand(accum[i]);
                deep_gemm::ptx::warpgroup_wait<0>();

                // 每个 warp 的 lane0 通知 producer：本 stage 可以复用了。
                if (lane == 0)
                    empty[stage].arrive();

                // WGMMA accumulator 每 4 个值对应两行 x 两列；A scale 随行变化，
                // B scale 对当前 128-wide N tile 统一。
#pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
                    final_accum[i * 4 + 0] += scale_a0 * scale_b * accum[i * 4 + 0];
                    final_accum[i * 4 + 1] += scale_a0 * scale_b * accum[i * 4 + 1];
                    final_accum[i * 4 + 2] += scale_a1 * scale_b * accum[i * 4 + 2];
                    final_accum[i * 4 + 3] += scale_a1 * scale_b * accum[i * 4 + 3];
                }
            }

            // 教学版直接按 WGMMA register mapping 写回；DeepGEMM 正式版会先 STSM，
            // 再用 TMA store 完成向量化 BF16 epilogue。
            const int row0 = mt * BM + warp * 16 + lane / 4;
            const int row1 = row0 + 8;
            const int col0 = nt * BN + (lane % 4) * 2;
#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
                const int col = col0 + i * 8;
                gD(row0, col + 0) = BF16(final_accum[i * 4 + 0]);
                gD(row0, col + 1) = BF16(final_accum[i * 4 + 1]);
                gD(row1, col + 0) = BF16(final_accum[i * 4 + 2]);
                gD(row1, col + 1) = BF16(final_accum[i * 4 + 3]);
            }
        }
    }
#endif
}

}  // namespace interview_gemm
