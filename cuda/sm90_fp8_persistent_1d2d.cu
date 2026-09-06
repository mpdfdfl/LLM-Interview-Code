/*
 * 面试手撕版：CUTLASS CuTe C++ / SM90 FP8 1D2D persistent GEMM
 *
 * D[m,n] = sum_kb SFA[m,kb] * SFB[n/128,kb]
 *                  * dot(A[m,kb*128:(kb+1)*128], B[n,kb*128:(kb+1)*128])
 *
 * A   : E4M3 [M,K]，K-major，stride=(K,1)
 * B   : E4M3 [N,K]，K-major，stride=(K,1)，计算 A @ B^T
 * SFA : FP32 [M,K/128]，MN-major，stride=(1,M)，每行每 128 个 K 一个 scale
 * SFB : FP32 [N/128,K/128]，row-major，每个 128x128 B block 一个 scale
 * D   : BF16 [M,N]，row-major
 *
 * 假设 M,N,K > 0，M%64 == 0、N%128 == 0、K%128 == 0。
 * 保留教学用的固定 tile、单 CTA cluster、直接写回；不处理边界或 multicast。
 * 仅依赖 CUDA 和 CUTLASS/CuTe，不依赖 DeepGEMM。
 */

#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

namespace interview_gemm {

using namespace cute;
using FP8 = cutlass::float_e4m3_t;
using BF16 = cutlass::bfloat16_t;

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 128;
constexpr int Stages = 4;
constexpr int MathThreads = 128;
constexpr int TmaThreads = 128;
constexpr int Threads = MathThreads + TmaThreads;

// -----------------------------------------------------------------------------
// 1. CuTe：shared-memory layout / TiledMMA / pipeline
// -----------------------------------------------------------------------------
// 把 PIPE 作为 tensor 的第三维，CuTe 负责 stage offset 和 WGMMA descriptor。
// K-major FP8 的 128 个元素占 128B，使用 SW128 layout atom。
using SmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<FP8>{}, Shape<Int<BM>, Int<BK>, Int<Stages>>{}));
using SmemLayoutB = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<FP8>{}, Shape<Int<BN>, Int<BK>, Int<Stages>>{}));

// 保留长度为 1 的 K-block 维度，使 SFA 与 GMEM [M,Kb] 的 TMA partition 对应。
using SmemLayoutSFA = Layout<Shape<Int<BM>, _1, Int<Stages>>>;

// 一个 warp-group，FP32 累加；一个 atom 是 m64n128k32。
// 对 [64,128] x [128,128] 的 shared tile，cute::gemm 自动遍历 4 个 MMA_K。
using TiledMma = decltype(make_tiled_mma(
    SM90_64x128x32_F32E4M3E4M3_SS_TN<>{}));
static_assert(size(TiledMma{}) == MathThreads);

using Pipeline = cutlass::PipelineTmaAsync<Stages>;
using PipelineState = cutlass::PipelineState<Stages>;
constexpr int TmaTransactionBytes =
    (BM + BN) * BK * sizeof(FP8) + BM * sizeof(float);

struct SharedStorage {
    alignas(128) ArrayEngine<FP8, cosize_v<SmemLayoutA>> a;
    alignas(128) ArrayEngine<FP8, cosize_v<SmemLayoutB>> b;
    alignas(128) ArrayEngine<float, cosize_v<SmemLayoutSFA>> sfa;
    Pipeline::SharedStorage pipeline;
};

// -----------------------------------------------------------------------------
// 2. Persistent scheduler：producer / consumer 各自按相同顺序遍历输出 tile
// -----------------------------------------------------------------------------
struct PersistentScheduler {
    int64_t next_tile;
    int num_m_tiles;
    int64_t num_tiles;

    CUTE_DEVICE PersistentScheduler(int m, int n)
        : next_tile(blockIdx.x),
          num_m_tiles(m / BM),
          num_tiles(int64_t(m / BM) * (n / BN)) {}

    CUTE_DEVICE bool next(int& m_tile, int& n_tile) {
        if (next_tile >= num_tiles)
            return false;
        m_tile = static_cast<int>(next_tile % num_m_tiles);
        n_tile = static_cast<int>(next_tile / num_m_tiles);
        next_tile += gridDim.x;
        return true;
    }
};

// -----------------------------------------------------------------------------
// 3. Kernel：typed TMA copy + PipelineTmaAsync + CuTe fragments
// -----------------------------------------------------------------------------
// TmaA/B/SFA 是 host make_tma_copy 返回的 TiledCopy，内含 tensor-map descriptor。
// 不再额外传 A/B/SFA 裸指针；通过 get_tma_tensor 获取真正用于 TMA 的坐标 tensor。
template <class TmaA, class TmaB, class TmaSFA>
__global__ __launch_bounds__(Threads, 1)
void fp8_gemm_1d2d_persistent(
    CUTE_GRID_CONSTANT TmaA const tma_a,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    CUTE_GRID_CONSTANT TmaSFA const tma_sfa,
    float const* sfb, BF16* d, int m, int n, int k) {

    const int tid = static_cast<int>(threadIdx.x);
    const bool is_producer = tid >= MathThreads;
    const int k_blocks = k / BK;

    extern __shared__ __align__(128) unsigned char shared_memory[];
    auto& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    auto sA = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(storage.b.begin()), SmemLayoutB{});
    auto sSFA = make_tensor(make_smem_ptr(storage.sfa.begin()), SmemLayoutSFA{});

    Pipeline::Params params;
    params.role = is_producer ? Pipeline::ThreadCategory::Producer
                              : Pipeline::ThreadCategory::Consumer;
    params.is_leader = (tid == MathThreads);
    params.num_consumers = MathThreads;
    params.transaction_bytes = TmaTransactionBytes;
    Pipeline pipeline(storage.pipeline, params, Shape<_1, _1, _1>{});
    // Pipeline 构造函数初始化 barrier 并执行 fence；CTA 同步后才允许使用。
    __syncthreads();

    PersistentScheduler scheduler(m, n);

    if (is_producer) {
        cutlass::arch::warpgroup_reg_dealloc<40>();

        // 一个固定的 leader 发射全部 TMA，其余 producer 线程不参与 acquire。
        if (tid == MathThreads) {
            auto mA = tma_a.get_tma_tensor(make_shape(m, k));
            auto mB = tma_b.get_tma_tensor(make_shape(n, k));
            auto mSFA = tma_sfa.get_tma_tensor(make_shape(m, k_blocks));

            // 单 CTA cluster，取 TiledCopy 的第 0 个 slice。
            auto cta_tma_a = tma_a.get_slice(_0{});
            auto cta_tma_b = tma_b.get_slice(_0{});
            auto cta_tma_sfa = tma_sfa.get_slice(_0{});
            auto tAsA = cta_tma_a.partition_D(sA);       // (TMA,TMA_M,TMA_K,PIPE)
            auto tBsB = cta_tma_b.partition_D(sB);       // (TMA,TMA_N,TMA_K,PIPE)
            auto tSsS = cta_tma_sfa.partition_D(sSFA);   // (TMA,TMA_M,TMA_Kb,PIPE)

            // 空 stage 的初始 phase 与 consumer 相反，不需要手工预填 empty barrier。
            auto write_state = cutlass::make_producer_start_state<Pipeline>();
            int mt, nt;
            while (scheduler.next(mt, nt)) {
                auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(mt, _));
                auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(nt, _));
                auto gSFA = local_tile(mSFA, Shape<Int<BM>, _1>{}, make_coord(mt, _));
                auto tAgA = cta_tma_a.partition_S(gA);   // (TMA,TMA_M,TMA_K,Kb)
                auto tBgB = cta_tma_b.partition_S(gB);   // (TMA,TMA_N,TMA_K,Kb)
                auto tSgS = cta_tma_sfa.partition_S(gSFA);

                for (int kb = 0; kb < k_blocks; ++kb) {
                    // 等待 empty，并由 leader 设置本 stage 的 expected transaction bytes。
                    pipeline.producer_acquire(write_state);
                    const int stage = write_state.index();
                    auto* barrier = pipeline.producer_get_barrier(write_state);
                    copy(tma_a.with(*barrier), tAgA(_, _, _, kb), tAsA(_, _, _, stage));
                    copy(tma_b.with(*barrier), tBgB(_, _, _, kb), tBsB(_, _, _, stage));
                    copy(tma_sfa.with(*barrier), tSgS(_, _, _, kb), tSsS(_, _, _, stage));
                    // TMA 硬件完成 transaction，毋须手动 arrive/producer_commit。
                    ++write_state;
                }
                // persistent 切换输出 tile 时不能重置 write_state / phase。
            }
            pipeline.producer_tail(write_state);
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<248>();

        auto mD = make_tensor(make_gmem_ptr(d), make_shape(m, n),
                              make_stride(int64_t(n), _1{}));
        auto mSFB = make_tensor(make_gmem_ptr(sfb), make_shape(n / BN, k_blocks),
                                make_stride(int64_t(k_blocks), _1{}));

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(tid);          // consumer tid 为 [0,128)
        auto tCsA = thr_mma.partition_A(sA);              // (MMA,MMA_M,MMA_K,PIPE)
        auto tCsB = thr_mma.partition_B(sB);              // (MMA,MMA_N,MMA_K,PIPE)
        auto tCrA = thr_mma.make_fragment_A(tCsA);        // shared-memory descriptor tensor
        auto tCrB = thr_mma.make_fragment_B(tCsB);        // shared-memory descriptor tensor
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == Int<BK / 32>{});
        CUTE_STATIC_ASSERT_V(size<2>(tCrB) == Int<BK / 32>{});

        // 对 identity tensor 做相同 partition，得到每个 accumulator 的逻辑 (row,col)。
        // scale 取行坐标，写回取 partition_C；都不需要知道 WGMMA 的 lane/register mapping。
        auto cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
        auto tCcC = thr_mma.partition_C(cC);
        PipelineState read_state;

        int mt, nt;
        while (scheduler.next(mt, nt)) {
            auto gD = local_tile(mD, Shape<Int<BM>, Int<BN>>{}, make_coord(mt, nt));
            auto tCgD = thr_mma.partition_C(gD);          // (MMA,MMA_M,MMA_N)
            auto tCrAccum = thr_mma.make_fragment_C(tCgD); // 全 K 的 scaled FP32 累加值
            auto tCrPartial = thr_mma.make_fragment_C(tCgD); // 当前 BK 的未缩放结果
            clear(tCrAccum);

            for (int kb = 0; kb < k_blocks; ++kb) {
                pipeline.consumer_wait(read_state);
                const int stage = read_state.index();

                // 不同 kb 的 scale 不同，必须先单独算完 BK=128，再做 scale promotion。
                clear(tCrPartial);
                warpgroup_fence_operand(tCrPartial);
                warpgroup_arrive();
                gemm(tiled_mma, tCrA(_, _, _, stage), tCrB(_, _, _, stage), tCrPartial);
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                warpgroup_fence_operand(tCrPartial);

                // SFB 是整个 [128,128] block 共享的标量，直接从 GMEM 读取。
                const float scale_b = mSFB(nt, kb);
                CUTE_UNROLL
                for (int i = 0; i < size(tCrAccum); ++i) {
                    const int row = get<0>(tCcC(i));
                    const float scale = sSFA(row, 0, stage) * scale_b;
                    tCrAccum(i) += scale * tCrPartial(i);
                }

                // WGMMA 和 SFA 读取都结束后才能释放 stage。
                // 单 CTA cluster 下，128 个 consumer 都必须调用 release。
                pipeline.consumer_release(read_state);
                ++read_state;
            }

            // CuTe partition 对齐寄存器和 GMEM 元素，教学版直接转换并写回 BF16。
            CUTE_UNROLL
            for (int i = 0; i < size(tCrAccum); ++i)
                tCgD(i) = BF16(tCrAccum(i));
        }
    }
}

// -----------------------------------------------------------------------------
// 4. 最小 host 入口：展示 make_tma_copy，并实例化上面的模板 kernel
// -----------------------------------------------------------------------------
// 保留原有数据布局；SFA 的 leading dimension 固定为 M（没有额外 padding）。
// A/B/SFA 的起始地址要求 16B 对齐，CUDA 分配的未偏移指针满足此要求。
// 异步提交到 stream；执行错误由调用者在 stream 同步时检查。
cudaError_t launch_fp8_gemm_1d2d_persistent(
    FP8 const* a, FP8 const* b, float const* sfa, float const* sfb,
    BF16* d, int m, int n, int k, cudaStream_t stream = nullptr) {

    if (!a || !b || !sfa || !sfb || !d || m <= 0 || n <= 0 || k <= 0 ||
        m % BM != 0 || n % BN != 0 || k % BK != 0 ||
        reinterpret_cast<uintptr_t>(a) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(b) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(sfa) % 16 != 0)
        return cudaErrorInvalidValue;

    int device;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess)
        return status;
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess)
        return status;
    if (properties.major != 9)
        return cudaErrorNotSupported;

    const int k_blocks = k / BK;
    auto mA = make_tensor(make_gmem_ptr(a), make_shape(m, k),
                          make_stride(int64_t(k), _1{}));
    auto mB = make_tensor(make_gmem_ptr(b), make_shape(n, k),
                          make_stride(int64_t(k), _1{}));
    auto mSFA = make_tensor(make_gmem_ptr(sfa), make_shape(m, k_blocks),
                            make_stride(_1{}, int64_t(m)));

    // descriptor 描述一个 stage；PIPE 维不编码进 global tensor map。
    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{}(_, _, _0{}),
                               Shape<Int<BM>, Int<BK>>{}, _1{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{}(_, _, _0{}),
                               Shape<Int<BN>, Int<BK>>{}, _1{});
    auto tma_sfa = make_tma_copy(SM90_TMA_LOAD{}, mSFA, SmemLayoutSFA{}(_, _, _0{}),
                                 Shape<Int<BM>, _1>{}, _1{});

    auto* kernel = &fp8_gemm_1d2d_persistent<decltype(tma_a), decltype(tma_b), decltype(tma_sfa)>;
    constexpr int smem_bytes = sizeof(SharedStorage);
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (status != cudaSuccess)
        return status;

    // 最多一个 CTA/SM；tile 不足时避免启动无任务的 CTA。
    const int64_t tiles = int64_t(m / BM) * (n / BN);
    const int grid = static_cast<int>(tiles < properties.multiProcessorCount
                                         ? tiles : properties.multiProcessorCount);
    fp8_gemm_1d2d_persistent<<<grid, Threads, smem_bytes, stream>>>(
        tma_a, tma_b, tma_sfa, sfb, d, m, n, k);
    return cudaGetLastError();
}

}  // namespace interview_gemm
