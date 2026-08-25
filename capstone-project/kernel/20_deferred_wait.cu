// 20 final-polish step:
//   kernel 19 x16 score acquisition + four rowmax chains
//   + one deferred tcgen05.wait::ld before the first overlapping P store.
//
// This combines x16 score loads, four rowmax chains, and deferred TMEM-load
// waits on the kernel-17 epilogue-overlap carrier.
//
// Inherited kernel-17 context:
// Kernel 17 adds independent epilogue/non-epilogue overlap on top of role-local loops.
// This intentionally adds only:
//   1. a non-epilogue completion baton so load/MMA/softmax/correction can advance
//      to the next work item while epilogue drains the previous sO
//   2. per-q-stage epilogue TMA commit so q0's store can drain before q1 is ready
// It inherits kernel 14's launch_bounds(,1) and role-local persistent loops.

#include "common.cuh"
#include "approximations.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// tcgen05 + TMA + TMEM attention with q_stage=2 + early-PV + explicit sO[2].
//
// Assumptions:
//   - head_dim == 128
//   - BLOCK_M == 128 queries per Q stage
//   - BLOCK_N == 128 keys per KV tile
//   - q_stage == 2, so each CTA covers 256 query rows
//   - len_q % 256 == 0
//   - len_kv % 128 == 0
//   - dense, non-causal, no masking
//   - inputs/outputs are bf16
//
// Steady-state dataflow:
//   1) load warp loads Q0/Q1 for each work item and drives the unified KV[3] token stream
//   2) MMA warp seeds S0/S1, then keeps the slot-owned schedule:
//      after PVq(i), immediately write QKq(i+1) into that freed S/P slot
//   3) softmax warps for stage0/stage1 independently consume S0/S1 and write packed P0/P1,
//      producing high96 first, then low32
//   4) correction warps rescale O0/O1 using the per-row rescale factors from softmax
//   5) MMA warp waits until Pq high96 is ready and Oq is safe, starts PV on high96,
//      then waits for full Pq and finishes PV on low32
//   6) after the KV loop: softmax publishes final inverse denominators; correction
//      writes normalized O into explicit sO[2]; the epilogue warp drains sO to global memory
//
// MMA steady-state schedule:
//   - seed tile 0 by computing QK0(0), QK1(0)
//   - then for each KV tile i:
//       * consume PV0(i), then immediately produce QK0(i+1)
//       * consume PV1(i), then immediately produce QK1(i+1)
//
// The O dependency has two distinct handoffs:
// correction needs the previous PV to finish writing O; the next PV needs correction
// to finish rescaling O into the current softmax basis
//
// Correction consumes two barrier streams before rescaling O:
//   1) stats_ready(q,i): current rescale_smem values are ready
//   2) pv_done(q,i-1): direct O-ordering handoff and required observation
//      of pv_done's reused phase

namespace {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int HEAD_DIM = 128;
constexpr int MMA_K = 16;

namespace q_stage {
constexpr int STAGES = 2;
constexpr int ROWS_PER_WORK_ITEM = STAGES * BLOCK_M;  // Q0 + Q1
}  // namespace q_stage
// S register caching: one tcgen05 x8 load returns 8 scores per softmax-warp lane
// cache the high 12 x8 fragments (high96) and reread the low 4 (low32)
constexpr int FIRST_CACHED_FRAGMENT = 4;   // fragments 0..3 are reread
constexpr int NUM_CACHED_FRAGMENTS = 12;   // fragments 4..15 are cached
// P is produced right-to-left to shorten cached-score register live ranges
// so high96 P[:, 32:128] becomes ready before low32 P[:, 0:32]
// early PV starts after high96 is stored
constexpr int EARLY_PV_START_MICROSTEP = 2;           // 32/96 boundary: issue k=2..7, then k=0..1

// rebase once the candidate rowmax is at least 8 log2 units above the old reference
constexpr float ROWMAX_REBASE_THRESHOLD_LOG2 = 8.0f;
// all 32 bits of UINT32_MAX are 1, so this mask later selects all 32 threads in a warp
constexpr unsigned FULL_WARP_MASK = UINT32_MAX;

// Warp roles:
//   0..3   softmax q0, 32 rows per warp
//   4..7   softmax q1, 32 rows per warp
//   8..11  correction, 32 rows per warp; each handles q0 then q1
//   12     TMA load
//   13     tcgen05 MMA
//   14     epilogue
constexpr int SOFTMAX_WARPS_PER_STAGE = 4;  // fixed 128 rows, 32 rows per warp
constexpr int SOFTMAX_WARP_COUNT = q_stage::STAGES * SOFTMAX_WARPS_PER_STAGE;
constexpr int CORR_WARP_BASE = SOFTMAX_WARP_COUNT;  // correction starts after all softmax warps
constexpr int CORR_WARP_COUNT = 4;  // fixed 128 rows, 32 rows per warp; shared across q0 and q1 (unlike softmax warps)
constexpr int LOAD_WARP_ID = CORR_WARP_BASE + CORR_WARP_COUNT;
constexpr int MMA_WARP_ID = LOAD_WARP_ID + 1;
constexpr int EPILOGUE_WARP_ID = MMA_WARP_ID + 1;
constexpr int NUM_WARPS = EPILOGUE_WARP_ID + 1;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
constexpr int PERSISTENT_CTAS_PER_SM = 1;
constexpr int NON_EPI_ACTIVE_WARPS =
    SOFTMAX_WARP_COUNT + CORR_WARP_COUNT + 1 /*load*/ + 1 /*mma*/;
static_assert(NON_EPI_ACTIVE_WARPS == 14, "persistent baton expects 14 non-epilogue warps");

constexpr int BF16_BYTES = int(sizeof(nv_bfloat16));
constexpr int MMA_K_BYTES = MMA_K * BF16_BYTES;         // 32
// In this kernel's matrix-coordinate view, SW128 operand panels are 8 rows x 64 bf16 cols.
// K-major and MN-major descriptors organize/interpret those panels differently.
constexpr int SW128_BYTES = 128;
constexpr int SW128_ATOM_ROWS = 8;
constexpr int SW128_ATOM_COLS = SW128_BYTES / BF16_BYTES;  // 64 bf16
#include "smem_layout_swizzle128.cuh"

constexpr int Q_TILE_BYTES = BLOCK_M * HEAD_DIM * BF16_BYTES;
constexpr int O_TILE_BYTES = BLOCK_M * HEAD_DIM * BF16_BYTES;
constexpr int KV_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;

constexpr int TMEM_S_COLS = BLOCK_N;
constexpr int TMEM_O_COLS = HEAD_DIM;
constexpr int TMEM_ALLOC_COLS = q_stage::STAGES * TMEM_S_COLS + q_stage::STAGES * TMEM_O_COLS;
// Full 512-column TMEM budget is consumed by S0/S1 + O0/O1.
// That is why this q_stage=2 branch cannot also keep an extra S/P time-stage.
static_assert(TMEM_ALLOC_COLS == 512, "q_stage=2 should pack S0/S1 + O0/O1 into 512 TMEM cols");

// Carrier helpers below name two independent index spaces:
//   q: Qq, Sq/Pq, Oq, and q-stage barriers.
//   KV token/stage: logical K/V stream mapped onto physical KV[3] slots.
// The kernel body owns scheduling, phases, waits/fences, and MMA order.

// Local lightweight helpers for the two-Q-stage carrier.
// They name q-stage address math only.
namespace q_stage {

__device__ __forceinline__ int row0(int q_row0_base, int q) {
  return q_row0_base + q * BLOCK_M;
}

__device__ __forceinline__ int q_smem(int Q_smem, int q) {
  return Q_smem + q * Q_TILE_BYTES;
}

__device__ __forceinline__ int tmem_s(int taddr_s_base, int q) {
  return taddr_s_base + q * TMEM_S_COLS;
}

__device__ __forceinline__ int tmem_o(int taddr_o_base, int q) {
  return taddr_o_base + q * TMEM_O_COLS;
}

__device__ __forceinline__ int mbar_addr(int base, int q) {
  return base + q * int(sizeof(uint64_t));
}

}  // namespace q_stage

// Local lightweight helpers for the unified KV[3] ring.
// Most helpers are stateless token/stage address math. load_token_and_arrive() is the
// side-effecting producer primitive: it issues TMA and arrives on kv_ready[stage].
namespace kv_ring {

constexpr int STAGES = 3;
constexpr int TOKENS_PER_KV_TILE = 2;
constexpr int K_KIND = 0;
constexpr int V_KIND = 1;

__device__ __forceinline__ int logical_token(int kv_tile, int kind) {
  return TOKENS_PER_KV_TILE * kv_tile + kind;
}

__device__ __forceinline__ int physical_stage(int token) {
  return token % STAGES;
}

__device__ __forceinline__ int token_tile(int token) {
  return token / TOKENS_PER_KV_TILE;
}

__device__ __forceinline__ int stage_smem_addr(int KV_smem, int stage) {
  return KV_smem + stage * KV_TILE_BYTES;
}

__device__ __forceinline__ int stage_mbar_addr(int mbar_kv_ready_base, int stage) {
  return mbar_kv_ready_base + stage * int(sizeof(uint64_t));
}

__device__ __forceinline__ bool is_k_token(int token) {
  return (token % TOKENS_PER_KV_TILE) == K_KIND;
}

__device__ __forceinline__ void load_token_and_arrive(
    int token, int batch_id, int len_kv, int KV_smem, int mbar_kv_ready_base,
    const CUtensorMap* K_tmap, const CUtensorMap* V_tmap) {
  const int stage = physical_stage(token);
  const int tile = token_tile(token);
  const int kv_row0 = batch_id * len_kv + tile * BLOCK_N;
  const int mbar_kv_ready_addr = stage_mbar_addr(mbar_kv_ready_base, stage);
  const int smem_addr = stage_smem_addr(KV_smem, stage);

  // The same physical stage is interpreted as K-major for QK or MN-major for PV.
  if (is_k_token(token)) {
    swizzle128::tma_load_kmajor(smem_addr, K_tmap, BLOCK_N, kv_row0, mbar_kv_ready_addr);
  } else {
    swizzle128::tma_load_mnmajor(smem_addr, V_tmap, kv_row0, mbar_kv_ready_addr);
  }
  mbarrier::arrive_expect_tx(mbar_kv_ready_addr, KV_TILE_BYTES);
}

}  // namespace kv_ring

__device__ __forceinline__ void issue_qk_mma(int taddr_s_stage, int Q_stage_smem,
                                             int K_stage_smem, uint32_t i_desc_qk) {
  // outer loop selects SW128 64-wide block
  constexpr uint64_t qk_desc_micro_step = MMA_K_BYTES / 16;
  constexpr uint64_t qk_a_desc_panel_step = (BLOCK_M * SW128_BYTES) / 16;
  constexpr uint64_t qk_b_desc_panel_step = (BLOCK_N * SW128_BYTES) / 16;
  uint64_t a_desc_panel = swizzle128::desc_kmajor(Q_stage_smem);
  uint64_t b_desc_panel = swizzle128::desc_kmajor(K_stage_smem);
  for (int k1 = 0; k1 < HEAD_DIM / SW128_ATOM_COLS; ++k1) {
    uint64_t a_desc = a_desc_panel;
    uint64_t b_desc = b_desc_panel;
    // inner loop selects MMA_K=16 chunk inside that block
    for (int k2 = 0; k2 < SW128_ATOM_COLS / MMA_K; ++k2) {
      const int enable_input_d = (k1 == 0 && k2 == 0) ? 0 : 1;
      tcgen05::mma_f16(taddr_s_stage, a_desc, b_desc, i_desc_qk, enable_input_d);
      a_desc += qk_desc_micro_step;
      b_desc += qk_desc_micro_step;
    }
    a_desc_panel += qk_a_desc_panel_step;
    b_desc_panel += qk_b_desc_panel_step;
  }
}

__device__ __forceinline__ void issue_pv_mma_range(
    int taddr_o_stage, int taddr_p_stage, int V_stage_smem,
    uint32_t i_desc_pv, int first_microstep, int end_microstep,
    bool initialize_o) {
  constexpr int pv_v_chunk_bytes = HEAD_DIM * MMA_K_BYTES;
  constexpr uint64_t pv_desc_chunk_step = pv_v_chunk_bytes / 16;
  // Offset V by the range's first P microstep so P columns match the corresponding V rows
  const int v_range_offset_bytes = first_microstep * pv_v_chunk_bytes;
  uint64_t b_desc = swizzle128::desc_mnmajor(
      V_stage_smem + v_range_offset_bytes, HEAD_DIM);
  for (int k = first_microstep; k < end_microstep; ++k) {
    // Advance by 16 V rows. In the SW128 MN-major layout this is
    // equivalent to two 8-token row-block strides in SMEM.
    // Inherited from kernel 2: each PV microstep advances 8 packed-P TMEM columns.
    const int taddr_a = taddr_p_stage + k * 8;
    // Disabling input D makes the first issued microstep initialize O
    const int enable_input_d =
        (initialize_o && k == first_microstep) ? 0 : 1;
    tcgen05::mma_f16_a_tmem(
        taddr_o_stage, taddr_a, b_desc, i_desc_pv, enable_input_d);
    b_desc += pv_desc_chunk_step;
  }
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x16_nowait(int taddr, float (&out)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
        "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]),
        "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]),
        "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
      : "r"(taddr));
}

__device__ __forceinline__ void tma_2d_smem2gmem(const void* tmap_ptr, int src, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
      :
      : "l"(tmap_ptr), "r"(x), "r"(y), "r"(src)
      : "memory");
}

__device__ __forceinline__ void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" : : : "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_bulk_wait_group_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" : : "n"(N) : "memory");
}

}  // namespace

// Kernel 14 introduced the persistent launch of up to one CTA per SM. This
// compiler annotation fits that design but is not explicitly required.
__global__ __launch_bounds__(TB_SIZE, 1) void attention_tcgen05_v20_kernel(
    const __grid_constant__ CUtensorMap Q_tmap, const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap, const __grid_constant__ CUtensorMap O_tmap,
    int len_q, int len_kv, int total_work_items) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const bool is_softmax_warp = (warp_id < SOFTMAX_WARP_COUNT);
  const bool is_corr_warp = (warp_id >= CORR_WARP_BASE && warp_id < CORR_WARP_BASE + CORR_WARP_COUNT);
  const bool is_epi_warp = (warp_id == EPILOGUE_WARP_ID);
  const bool is_load_warp = (warp_id == LOAD_WARP_ID);
  const bool is_mma_warp = (warp_id == MMA_WARP_ID);

  extern __shared__ __align__(1024) char smem_raw[];

  const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
  const int Q_smem = smem_base;
  const int O_smem = Q_smem + q_stage::STAGES * Q_TILE_BYTES;
  const int KV_smem = O_smem + q_stage::STAGES * O_TILE_BYTES;

  // Barriers used within one output work item:
  //   q-ready:            load warp -> mma warp (Q0/Q1 loaded for one work item)
  //   kv-ready[s]:        load warp -> mma warp (per K/V stage)
  //   qk-done[q]:         mma warp -> softmax warps for Q stage q
  //   qk-done[last_q]:    load warp waits on this before recycling an old K token
  //   stats-ready[q]:     softmax -> correction; rescale_smem for Q stage q is ready
  //   p-high96-ready-o-safe[q]: softmax+correction warps -> mma warp for high96 Pq and O-safe
  //   p-full-ready[q]:    softmax warps -> mma warp after full Pq is stored
  //   pv-done[q]:         MMA -> correction;
  //                       previous phase before next-tile O rescale,
  //                       final phase before final O staging;
  //                       last-q stream also lets load recycle old V tokens
  //   final-inv-ready[q]: softmax warps -> correction warps for final O normalization of Q stage q
  //
  // Barriers used for persistent epilogue overlap:
  //   epi-ready[q]:       correction warps -> epilogue warp for Q stage q
  //   epi-free[q]:        epilogue warp -> correction warps; sO[q] is safe to overwrite
  //
  // Barriers used between persistent non-epilogue work items:
  //   setup-done:         load warp -> MMA/softmax/correction; this output work item's barrier
  //                       setup is done, so those role-local loops can begin
  //   non-epi-done:       non-epilogue warp roles -> load/MMA; their current work is done, so load
  //                       resets work-local barriers, or MMA deallocates TMEM after final work
  __shared__ uint64_t mbar_q_ready[1];
  // Kernel 14 moved work-local barrier initialization and reinitialization inside
  // the output-work-item loop. Kernel 17 changes the boundary to non_epi_done so
  // epilogue can overlap; these two barriers still signal setup and safe teardown.
  __shared__ uint64_t mbar_setup_done[1];
  __shared__ uint64_t mbar_non_epi_done[1];
  __shared__ uint64_t mbar_qk_done[q_stage::STAGES];
  __shared__ uint64_t mbar_stats_ready[q_stage::STAGES];
  // Kernel 12's p_ready_o_safe becomes high96-specific here
  // wait for O-safe once with high96; by the time low32 PV runs, O has already
  // been rescaled, so p_full_ready only needs to track full-P readiness
  __shared__ uint64_t mbar_p_high96_ready_o_safe[q_stage::STAGES];
  __shared__ uint64_t mbar_p_full_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_pv_done[q_stage::STAGES];
  __shared__ uint64_t mbar_final_inv_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_epi_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_epi_free[q_stage::STAGES];
  __shared__ uint64_t mbar_kv_ready[kv_ring::STAGES];
  // one additional shared-memory buffer is used to pass per-row rescale values
  // from softmax warps to correction warps
  // no compute-stage dimension: each Q stage has only one live rescale handoff at a time
  // because there is no temporal compute overlap (across KV-tile iterations)
  // within the same Q tile
  __shared__ float rescale_smem[q_stage::STAGES][BLOCK_M];
  __shared__ float inv_denom_smem[q_stage::STAGES][BLOCK_M];
  __shared__ int tmem_addr[1];

  const int mbar_q_ready_addr = static_cast<int>(__cvta_generic_to_shared(mbar_q_ready));
  const int mbar_setup_done_addr = static_cast<int>(__cvta_generic_to_shared(mbar_setup_done));
  const int mbar_non_epi_done_addr =
      static_cast<int>(__cvta_generic_to_shared(mbar_non_epi_done));
  const int mbar_qk_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_qk_done));
  const int mbar_stats_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_stats_ready));
  const int mbar_p_high96_ready_o_safe_base = static_cast<int>(__cvta_generic_to_shared(mbar_p_high96_ready_o_safe));
  const int mbar_p_full_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_p_full_ready));
  const int mbar_pv_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_pv_done));
  const int mbar_final_inv_ready_base =
      static_cast<int>(__cvta_generic_to_shared(mbar_final_inv_ready));
  const int mbar_epi_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_epi_ready));
  const int mbar_epi_free_base = static_cast<int>(__cvta_generic_to_shared(mbar_epi_free));
  const int mbar_kv_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_kv_ready));

  if (is_load_warp && elect_sync()) {
    // Kernel 13.5 initialized all work-local barriers once because each CTA processed only one
    // output work item (one Q0/Q1 output-tile pair). Kernel 14 reuses the CTA for multiple work
    // items, so those barriers move inside the loop over output work items: they are initialized
    // for the first item, then invalidated and reinitialized for every later item. Kernel 17
    // replaces the all-role work boundary with non_epi_done. setup_done, non_epi_done,
    // epi_ready, and epi_free persist across work items and are initialized once here.
    //
    // setup_done completes after the coordinator's one arrival. non_epi_done treats
    // NON_EPI_ACTIVE_WARPS arrivals as all non-epilogue warp roles finished, so one
    // designated lane from each such warp contributes exactly one arrival. If every
    // lane arrived, one warp could satisfy the count and advance the phase early.
    mbarrier::init(mbar_setup_done_addr, 1);
    mbarrier::init(mbar_non_epi_done_addr, NON_EPI_ACTIVE_WARPS);
    for (int q = 0; q < q_stage::STAGES; ++q) {
      const int mbar_epi_ready_addr = q_stage::mbar_addr(mbar_epi_ready_base, q);
      const int mbar_epi_free_addr = q_stage::mbar_addr(mbar_epi_free_base, q);
      mbarrier::init(mbar_epi_ready_addr, CORR_WARP_COUNT);
      mbarrier::init(mbar_epi_free_addr, 1);
      mbarrier::arrive(mbar_epi_free_addr);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (is_mma_warp) {
    // TMEM layout:
    //   S0 at [0..127], S1 at [128..255], O0 at [256..383], O1 at [384..511]
    //   P overlays S at S + 64 b32 columns
    //   low32 P:  S+[64..79]   logical P[:, 0:32]
    //   high96 P: S+[80..127]  logical P[:, 32:128]
    //   early PV reads high96 while softmax rereads S+[0..31] and writes low32 P
    const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :
                 : "r"(tmem_smem_addr), "r"(TMEM_ALLOC_COLS)
                 : "memory");
  }
  __syncthreads();

  const int taddr_base = tmem_addr[0];
  const int taddr_s_base = taddr_base;
  // Swizzle applies to SMEM operand descriptors (Q/K/V), not the TMEM O accumulator.
  const int taddr_o_base = taddr_s_base + q_stage::STAGES * TMEM_S_COLS;

  // QK MMA descriptor:
  //   dtype=fp32 (bit4), atype=bf16 (bit7), btype=bf16 (bit10)
  //   MMA_N encoded at bit17, MMA_M at bit24.
  constexpr uint32_t i_desc_qk =
      (1U << 4) |                    // accumulator dtype fp32
      (1U << 7) |                    // A dtype bf16
      (1U << 10) |                   // B dtype bf16
      ((BLOCK_N >> 3) << 17) |       // MMA_N / 8
      ((BLOCK_M >> 4) << 24);        // MMA_M / 16

  // PV uses MN-major B (bit 16).
  constexpr uint32_t i_desc_pv = i_desc_qk | (1U << 16);

  const int kv_tiles = len_kv / BLOCK_N;

  // Persistent CTA work assignment
  //
  // Role-local persistent loops avoid presenting ptxas with one giant loop body
  // containing every role. The non-epilogue roles finish one output work item
  // before starting the next, while epilogue may still drain the previous sO.
  //
  // work_id identifies one batch and one Q0/Q1 output-tile pair within that batch.
  // Earlier kernels launched one CTA per work item, so blockIdx.x was also work_id.
  // Starting with kernel 14, the stub caps the number of CTAs (gridDim.x) at the number of SMs.
  // Each CTA starts with work_id = blockIdx.x, then advances it by the number of CTAs
  // (gridDim.x) to process the remaining work items assigned to it. Together, the
  // CTAs cover all output work items. Each work item covers two consecutive Q tiles
  // and their matching O tiles, spanning q_stage::ROWS_PER_WORK_ITEM Q/O rows.

  // Load-warp K/V pipeline
  //
  // The load warp drives one generic K/V token stream through KV[3] per work item.
  // Even tokens are K(tile), odd tokens are V(tile). A physical stage is reusable
  // after the last Q stage consumes a K token for QK or a V token for PV.
  if (is_load_warp && elect_sync()) {

    int phase_non_epi_done = 0;

    // Compute this CTA's work IDs inside the kernel: start at blockIdx.x, step
    // by the number of CTAs, and stop when the next ID is >= total_work_items.
    for (int work_id = blockIdx.x, work_iter = 0; work_id < total_work_items;
         work_id += gridDim.x, ++work_iter) {

      // Persistent work-item setup (steps 1-2)
      //
      // A persistent CTA processes multiple output work items while reusing the same work-local
      // mbarrier objects. The first item initializes them. On every later item, calling
      // mbarrier.init while an object is still valid would be undefined PTX behavior, so each
      // object is invalidated here before it is reinitialized.
      //
      // After setup, the coordinator signals setup_done; the compute roles then begin this output
      // work item. This kernel assigns coordination to the elected load lane, but coordination
      // does not depend on that warp: any designated lane works if it preserves the same
      // non_epi_done -> setup_done ordering. The load lane owns setup so it can start Q/KV loads
      // and publish q_ready/kv_ready without waiting on a setup handoff from another warp.

      // 1) Barrier teardown (later work items only)
      //
      // Wait until every non-epilogue role finishes the previous work item before
      // resetting its work-local compute barriers. The epilogue can still drain sO
      // through the persistent epi_ready/epi_free streams.
      if (work_iter > 0) {
        mbarrier::wait(mbar_non_epi_done_addr, phase_non_epi_done);
        phase_non_epi_done ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

        // Keep this list in sync with the work-local initialization below.
        mbarrier::inval(mbar_q_ready_addr);
        for (int q = 0; q < q_stage::STAGES; ++q) {
          mbarrier::inval(q_stage::mbar_addr(mbar_qk_done_base, q));
          mbarrier::inval(q_stage::mbar_addr(mbar_stats_ready_base, q));
          mbarrier::inval(q_stage::mbar_addr(mbar_p_high96_ready_o_safe_base, q));
          mbarrier::inval(q_stage::mbar_addr(mbar_p_full_ready_base, q));
          mbarrier::inval(q_stage::mbar_addr(mbar_pv_done_base, q));
          mbarrier::inval(q_stage::mbar_addr(mbar_final_inv_ready_base, q));
        }
        for (int st = 0; st < kv_ring::STAGES; ++st) {
          mbarrier::inval(kv_ring::stage_mbar_addr(mbar_kv_ready_base, st));
        }
      }

      // 2) Barrier initialization

      mbarrier::init(mbar_q_ready_addr, 1);
      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
        const int mbar_stats_ready_addr = q_stage::mbar_addr(mbar_stats_ready_base, q);
        const int mbar_p_high96_ready_o_safe_addr = q_stage::mbar_addr(mbar_p_high96_ready_o_safe_base, q);
        const int mbar_p_full_ready_addr = q_stage::mbar_addr(mbar_p_full_ready_base, q);
        const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
        const int mbar_final_inv_ready_addr =
            q_stage::mbar_addr(mbar_final_inv_ready_base, q);
        mbarrier::init(mbar_qk_done_addr, 1);
        mbarrier::init(mbar_stats_ready_addr, SOFTMAX_WARPS_PER_STAGE);
        // 4 softmax arrivals publish high96 P and 4 correction arrivals publish O-safe
        // completion means both conditions hold across all 128 rows
        mbarrier::init(mbar_p_high96_ready_o_safe_addr, SOFTMAX_WARPS_PER_STAGE + CORR_WARP_COUNT);
        mbarrier::init(mbar_p_full_ready_addr, SOFTMAX_WARPS_PER_STAGE);
        mbarrier::init(mbar_pv_done_addr, 1);
        mbarrier::init(mbar_final_inv_ready_addr, SOFTMAX_WARPS_PER_STAGE);
      }
      for (int st = 0; st < kv_ring::STAGES; ++st) {
        const int mbar_kv_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, st);
        mbarrier::init(mbar_kv_ready_addr, 1);
      }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");

      // Publish that all work-local barriers are initialized and safe to use.
      // MMA, softmax, and correction warps wait on setup_done before touching
      // their reset-per-work barriers. Without this handoff, a warp could use a
      // barrier while it is invalid or being reinitialized, or observe the
      // completed phase left by the previous work item.
      // This replaces the __syncthreads() after one-time initialization in kernel 13.5.
      mbarrier::arrive(mbar_setup_done_addr);

      // Input loading (steps 3-5)
      //
      // 3) Load Q once
      //
      // Load Q0 and Q1 once for this work item.
      const int q_row0_base = work_id * q_stage::ROWS_PER_WORK_ITEM;
      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
        const int q_row0_stage = q_stage::row0(q_row0_base, q);
        swizzle128::tma_load_kmajor(Q_stage_smem, &Q_tmap, BLOCK_M, q_row0_stage,
                                          mbar_q_ready_addr);
      }
      mbarrier::arrive_expect_tx(mbar_q_ready_addr, q_stage::STAGES * Q_TILE_BYTES);

      // Load warp recycles K/V only after the last q-stage consumer stream.
      int phase_k_recycle = 0;
      int phase_v_recycle = 0;
      const int last_q = q_stage::STAGES - 1;
      const int mbar_last_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, last_q);
      const int mbar_last_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, last_q);
      const int kv_tokens = kv_ring::TOKENS_PER_KV_TILE * kv_tiles;
      const int prefill_tokens = (kv_tokens < kv_ring::STAGES) ? kv_tokens : kv_ring::STAGES;

      // 4) Prefill the K/V ring
      //
      // Prefill: the first KV[3] tokens fill one token per physical KV stage;
      // no old token exists yet, so there is no recycle wait.
      // work_id flattens (batch_id, Q-tile-pair index). Divide by the number
      // of work items per batch to recover the batch used for K/V loads.
      const int work_items_per_batch = len_q / q_stage::ROWS_PER_WORK_ITEM;
      const int batch_id = work_id / work_items_per_batch;
      for (int token = 0; token < prefill_tokens; ++token) {
        kv_ring::load_token_and_arrive(
            token, batch_id, len_kv, KV_smem, mbar_kv_ready_base, &K_tmap, &V_tmap);
      }

      // 5) Refill the K/V ring
      //
      // Refill: token T reuses the stage occupied by token T-KV[3], so wait on
      // the last Q stage's consumer before overwriting that physical stage.
      for (int token = prefill_tokens; token < kv_tokens; ++token) {
        const int recycle_token = token - kv_ring::STAGES;
        if (kv_ring::is_k_token(recycle_token)) {
          mbarrier::wait(mbar_last_qk_done_addr, phase_k_recycle);
          phase_k_recycle ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        } else {
          mbarrier::wait(mbar_last_pv_done_addr, phase_v_recycle);
          phase_v_recycle ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        }

        kv_ring::load_token_and_arrive(
            token, batch_id, len_kv, KV_smem, mbar_kv_ready_base, &K_tmap, &V_tmap);
      }

      // Signal load role done.
      // This is one non_epi_done arrival; it does not advance the barrier by itself.
      mbarrier::arrive(mbar_non_epi_done_addr);
    }
  }

  // MMA warp: after PVq(i), immediately produce QKq(i+1) into the freed q-stage slot.
  if (is_mma_warp && elect_sync()) {
    for (int work_id = blockIdx.x, work_iter = 0; work_id < total_work_items;
         work_id += gridDim.x, ++work_iter) {
      mbarrier::wait(mbar_setup_done_addr, work_iter & 1);
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      mbarrier::wait(mbar_q_ready_addr, 0);
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      int phase_kv_ready[kv_ring::STAGES] = {0, 0, 0};
      int phase_p_high96_ready_o_safe[q_stage::STAGES] = {0, 0};
      int phase_p_full_ready[q_stage::STAGES] = {0, 0};

      // Compute prologue: seed both Q stages for tile 0;
      // creates S0 and S1, so the softmax warps have S0->P0 and S1->P1 work.
      const int k_token0 = kv_ring::logical_token(0, kv_ring::K_KIND);
      const int k_stage0 = kv_ring::physical_stage(k_token0);
      const int mbar_kv_ready_addr0 = kv_ring::stage_mbar_addr(mbar_kv_ready_base, k_stage0);
      const int K_stage_smem0 = kv_ring::stage_smem_addr(KV_smem, k_stage0);

      mbarrier::wait(mbar_kv_ready_addr0, phase_kv_ready[k_stage0]);
      phase_kv_ready[k_stage0] ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
        const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
        const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, q);

        issue_qk_mma(taddr_s_stage, Q_stage_smem, K_stage_smem0, i_desc_qk);

        // Track QK completion; the mbarrier arrives when this Q stage's S is ready.
        tcgen05::commit_arrive(mbar_qk_done_addr);
      }

      // Compute main loop: for kv tile: for q_idx in num_Q_stages: PV[q_idx](i), QK[q_idx](i+1)
      for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        const int v_token = kv_ring::logical_token(kv_tile, kv_ring::V_KIND);
        const int v_stage = kv_ring::physical_stage(v_token);
        const bool first_kv = (kv_tile == 0);
        const int mbar_v_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, v_stage);
        const int V_stage_smem = kv_ring::stage_smem_addr(KV_smem, v_stage);

        const int next_tile = kv_tile + 1;
        const bool have_next = (next_tile < kv_tiles);
        int next_k_stage = 0;
        int next_mbar_kv_ready_addr = 0;
        int next_K_stage_smem = 0;
        if (have_next) {
          const int next_k_token = kv_ring::logical_token(next_tile, kv_ring::K_KIND);
          next_k_stage = kv_ring::physical_stage(next_k_token);
          next_mbar_kv_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, next_k_stage);
          next_K_stage_smem = kv_ring::stage_smem_addr(KV_smem, next_k_stage);
        }

        mbarrier::wait(mbar_v_ready_addr, phase_kv_ready[v_stage]);
        phase_kv_ready[v_stage] ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

        for (int q = 0; q < q_stage::STAGES; ++q) {
          const int mbar_p_high96_ready_o_safe_addr =
              q_stage::mbar_addr(mbar_p_high96_ready_o_safe_base, q);
          const int mbar_p_full_ready_addr = q_stage::mbar_addr(mbar_p_full_ready_base, q);
          const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
          const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, q);
          const int taddr_p_stage = taddr_s_stage + BLOCK_N / 2;  // preserve low32 S for reread
          const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, q);

          mbarrier::wait(mbar_p_high96_ready_o_safe_addr,
                         phase_p_high96_ready_o_safe[q]);
          phase_p_high96_ready_o_safe[q] ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

          // early PV starts from P[:, 32:128], not the beginning of P; offset B by the
          // same two K=16 chunks to select the corresponding V[32:128, :] rows
          // Issue the ready high96 first (k=2..7); its first microstep initializes O on KV tile 0
          issue_pv_mma_range(
              taddr_o_stage, taddr_p_stage, V_stage_smem, i_desc_pv,
              /*first_microstep=*/EARLY_PV_START_MICROSTEP,
              /*end_microstep=*/BLOCK_N / MMA_K,
              /*initialize_o=*/first_kv);

          mbarrier::wait(mbar_p_full_ready_addr, phase_p_full_ready[q]);
          phase_p_full_ready[q] ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

          // Then issue the remaining low32 (k=0..1), accumulating into the high96 result
          issue_pv_mma_range(
              taddr_o_stage, taddr_p_stage, V_stage_smem, i_desc_pv,
              /*first_microstep=*/0,
              /*end_microstep=*/EARLY_PV_START_MICROSTEP,
              /*initialize_o=*/false);

          // Track PV completion; the mbarrier arrives when this Q stage's O is ready.
          tcgen05::commit_arrive(mbar_pv_done_addr);

          if (have_next) {
            // The next K token is shared by q0 and q1, so consume its ready phase once.
            if (q == 0) {
              mbarrier::wait(next_mbar_kv_ready_addr, phase_kv_ready[next_k_stage]);
              phase_kv_ready[next_k_stage] ^= 1;
              asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
            }

            const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
            const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
            const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, q);

            issue_qk_mma(taddr_s_stage, Q_stage_smem, next_K_stage_smem, i_desc_qk);

            // Track QK completion; the mbarrier arrives when this Q stage's S is ready.
            tcgen05::commit_arrive(mbar_qk_done_addr);
          }
        }
      }

      // Signal MMA role done.
      // This is one non_epi_done arrival; it does not advance the barrier by itself.
      mbarrier::arrive(mbar_non_epi_done_addr);
    }
  }

  // Softmax warps: one 4-warp group per Q stage.
  if (is_softmax_warp) {

    // Each Q stage has 4 softmax warps, each covering 32 of its 128 rows
    const int softmax_stage = warp_id / SOFTMAX_WARPS_PER_STAGE;
    const int local_warp_idx = warp_id % SOFTMAX_WARPS_PER_STAGE;
    const int row_base = local_warp_idx * WARP_SIZE;  // first of 32 rows owned by this warp
    const int row = row_base + lane_id;               // specific row owned by this lane
    const int trow = row_base << 16;                  // encode row_base for TMEM addressing

    for (int work_id = blockIdx.x, work_iter = 0; work_id < total_work_items;
         work_id += gridDim.x, ++work_iter) {
      mbarrier::wait(mbar_setup_done_addr, work_iter & 1);
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      // This softmax warp belongs to one fixed q-stage, so it only needs one
      // work-local cursor for that qk_done stream.
      int phase_qk_done = 0;

      // q-stage-local resources for this softmax warp; invariant across all KV tiles.
      const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, softmax_stage);
      const int mbar_stats_ready_addr = q_stage::mbar_addr(mbar_stats_ready_base, softmax_stage);
      const int mbar_p_high96_ready_o_safe_addr = q_stage::mbar_addr(mbar_p_high96_ready_o_safe_base, softmax_stage);
      const int mbar_p_full_ready_addr = q_stage::mbar_addr(mbar_p_full_ready_base, softmax_stage);
      const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, softmax_stage);
      const int taddr_p_stage = taddr_s_stage + BLOCK_N / 2;  // preserve low32 S for reread

      // Online softmax state per row and per Q stage.
      // active softmax reference in exp2-input units; may lag the true running max while
      // the increase remains below ROWMAX_REBASE_THRESHOLD_LOG2
      float rowmax = -FLT_MAX;
      float rowsum = 0.0f;
      constexpr float LOG2_E = 1.4426950408889634f;
      const float softmax_scale = rsqrtf(float(HEAD_DIM));
      // Keep exp arguments in exp2 units instead of letting each __expf site
      // locally form (natural_exp_arg * log2(e)).
      const float softmax_scale_log2 = softmax_scale * LOG2_E;

      for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        const bool first_kv = (kv_tile == 0);

        mbarrier::wait(mbar_qk_done_addr, phase_qk_done);
        phase_qk_done ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

        // Retain high96 S during rowmax and consume it first to publish 3/4 of P early
        // then reread the uncached low32 S tail from TMEM
        float cached_scores[NUM_CACHED_FRAGMENTS][8];
        float rowmax0 = -FLT_MAX;
        float rowmax1 = -FLT_MAX;
        float rowmax2 = -FLT_MAX;
        float rowmax3 = -FLT_MAX;
        {
#pragma unroll
          for (int score_n16 = 0; score_n16 < BLOCK_N / 16; ++score_n16) {
            float s16[16];
            const int taddr = taddr_s_stage + trow + score_n16 * 16;
            tcgen05_ld_32x32b_x16_nowait(taddr, s16);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              rowmax0 = fmaxf(rowmax0, s16[4 * i + 0]);
              rowmax1 = fmaxf(rowmax1, s16[4 * i + 1]);
              rowmax2 = fmaxf(rowmax2, s16[4 * i + 2]);
              rowmax3 = fmaxf(rowmax3, s16[4 * i + 3]);
            }
            if (score_n16 >= FIRST_CACHED_FRAGMENT / 2) {
              // This pass visits eight x16 fragments and caches the final six
              // (high96), mapping each x16 result into two x8 cache slots.
              const int cache_fragment = (score_n16 - FIRST_CACHED_FRAGMENT / 2) * 2;
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                cached_scores[cache_fragment][i] = s16[i];
                cached_scores[cache_fragment + 1][i] = s16[8 + i];
              }
            }
          }
        }

        // We need to multiply both scores and rowmax into the same log2/exp2-input
        // units before subtracting rowmax from the scores.
        const float tile_rowmax =
            fmaxf(fmaxf(rowmax0, rowmax1), fmaxf(rowmax2, rowmax3)) * softmax_scale_log2;
        const float candidate_rowmax = fmaxf(rowmax, tile_rowmax);
        const float rowmax_increase = candidate_rowmax - rowmax;

        // below the threshold, leave rowmax and rowsum unchanged and keep rescale at 1
        const bool row_needs_o_rescale =
            !first_kv && rowmax_increase >= ROWMAX_REBASE_THRESHOLD_LOG2;

        float rescale = 1.0f;
        if (first_kv) {
          // no previous O or rowsum exists yet, so the first tile only initializes the reference
          rowmax = candidate_rowmax;
        } else if (row_needs_o_rescale) {
          // only evaluate exp2 when rebasing to a higher softmax reference
          rescale = approx::fast_exp2(rowmax - candidate_rowmax);
          rowmax = candidate_rowmax;
          rowsum *= rescale;
        }
        rescale_smem[softmax_stage][row] = rescale;

        __syncwarp();
        if (lane_id == 0) {
          mbarrier::arrive(mbar_stats_ready_addr);
        }

        float2 tile_rowsum_pair = make_float2(0.0f, 0.0f);
        {
          const int taddr_s_row = taddr_s_stage + trow;
          const int taddr_p_row = taddr_p_stage + trow;
          // Cache96/reread32 gives this P pass two sources: cached high-96 S and reread low-32 S
          // This schedule signals p_high96_ready_o_safe after high96 is stored
          // and correction has made O safe; low32 is stored before p_full_ready
          // The P pass walks high->low so the most recently cached high96 values are reused first
#pragma unroll
          for (int score_n16 = BLOCK_N / 16 - 1; score_n16 >= 0; --score_n16) {
            uint32_t p8_packed[8];
#pragma unroll
            for (int half8 = 0; half8 < 2; ++half8) {
              const int score_n8 = score_n16 * 2 + half8;
              const bool score_is_cached = score_n8 >= FIRST_CACHED_FRAGMENT;
              float s8[8];
              if (score_is_cached) {
                // reuse high96 scores retained in registers during the rowmax pass
                const int cache_fragment = score_n8 - FIRST_CACHED_FRAGMENT;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                  s8[i] = cached_scores[cache_fragment][i];
                }
              } else {
                // low32 was not cached, so reread it from TMEM
                tcgen05::ld_32x32b_x8(taddr_s_row + score_n8 * 8, s8);
              }
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                // packed FMA as one fused operation: score * scale + (-rowmax)
                const float2 shifted_score_pair = f32x2::fma_rn_ftz(
                    make_float2(s8[2 * j], s8[2 * j + 1]), softmax_scale_log2, -rowmax);
                const float shifted_score0 = shifted_score_pair.x;
                const float shifted_score1 = shifted_score_pair.y;

                // pair position within the current 16-column block
                const int pair_start_in_block = half8 * 8 + 2 * j;
                // use hardware exp2 for the first 12 values and software exp2 for the final 4
                const bool use_software_exp2 = pair_start_in_block >= 12;
                float p0;
                float p1;
                if (use_software_exp2) {
                  approx::e2e_exp2_pair(shifted_score0, shifted_score1, p0, p1);
                } else {
                  p0 = approx::fast_exp2(shifted_score0);
                  p1 = approx::fast_exp2(shifted_score1);
                }
                tile_rowsum_pair =
                    f32x2::add_rn_ftz(tile_rowsum_pair, make_float2(p0, p1));
                p8_packed[half8 * 4 + j] = bf16::pack2_to_u32(p0, p1);
              }
            }
            if (score_n16 == BLOCK_N / 16 - 1) {
              // Complete all deferred score loads before this first P store
              // overwrites the overlapping S/P TMEM region.
              asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
            }
            tcgen05::st_32x32b_x8_u32(taddr_p_row + score_n16 * 8, p8_packed);
            // after traversing 96 columns right-to-left, we reach the boundary before the
            // leftmost 32 and signal that the rightmost 96 columns are ready
            if (score_n16 == EARLY_PV_START_MICROSTEP) {
              asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
              asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
              __syncwarp();
              if (lane_id == 0) {
                mbarrier::arrive(mbar_p_high96_ready_o_safe_addr);
              }
            }
          }
          asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
        }

        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        __syncwarp();
        if (lane_id == 0) {
          mbarrier::arrive(mbar_p_full_ready_addr);
        }

        rowsum += tile_rowsum_pair.x + tile_rowsum_pair.y;
      }

      inv_denom_smem[softmax_stage][row] = 1.0f / rowsum;
      // This covers both final_inv_ready and the softmax-role completion below.
      __syncwarp();
      if (lane_id == 0) {
        const int mbar_final_inv_ready_addr =
            q_stage::mbar_addr(mbar_final_inv_ready_base, softmax_stage);
        mbarrier::arrive(mbar_final_inv_ready_addr);
      }

      // non_epi_done counts completed non-epilogue warp roles, not threads: one
      // arrival represents this whole warp finishing its work. It expects
      // NON_EPI_ACTIVE_WARPS arrivals, so only lane 0 arrives. If every lane
      // arrived, one warp could advance the phase before the other roles finished.
      if (lane_id == 0) {
        mbarrier::arrive(mbar_non_epi_done_addr);
      }
    }
  }

  // Correction warps: shared across both Q stages.
  if (is_corr_warp) {

    // Each correction warp reads the rescale factors written by the corresponding
    // softmax warp for the same 32 rows, servicing q0 then q1
    const int local_warp_idx = warp_id - CORR_WARP_BASE;
    const int row_base = local_warp_idx * WARP_SIZE;  // first of 32 rows owned by this warp
    const int row = row_base + lane_id;               // specific row owned by this lane
    const int trow = row_base << 16;                  // encode row_base for TMEM addressing
    int phase_epi_free[q_stage::STAGES] = {0, 0};

    for (int work_id = blockIdx.x, work_iter = 0; work_id < total_work_items;
         work_id += gridDim.x, ++work_iter) {
      mbarrier::wait(mbar_setup_done_addr, work_iter & 1);
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      int phase_stats_ready[q_stage::STAGES] = {0, 0};
      int phase_pv_done[q_stage::STAGES] = {0, 0};

      // Match MMA's tile-major q0->q1 order. Processing every q0 tile first
      // would withhold q1's O-safe arrival and deadlock MMA.
      for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        const bool first_kv = (kv_tile == 0);

        for (int q = 0; q < q_stage::STAGES; ++q) {
          const int mbar_stats_ready_addr = q_stage::mbar_addr(mbar_stats_ready_base, q);
          const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
          const int mbar_p_high96_ready_o_safe_addr =
              q_stage::mbar_addr(mbar_p_high96_ready_o_safe_base, q);
          const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, q);

          mbarrier::wait(mbar_stats_ready_addr, phase_stats_ready[q]);
          phase_stats_ready[q] ^= 1;
          if (!first_kv) {
            // Softmax publishes stats_ready(q,i) only after waiting on qk_done(q,i).
            // MMA issues PV(q,i-1) before QK(q,i), so stats_ready already proves
            // that the previous PV operation completed (so a separate pv_done wait
            // is not needed for the data dependency). But PTX requires at least one
            // successful wait on each mbarrier phase before the next phase receives
            // an arrival. pv_done is a different mbarrier (not the stats_ready we
            // just waited on), so we additionally observe pv_done below.
            // This wait should not expose the preceding PV's compute latency because
            // stats_ready already established completion of that computation.
            mbarrier::wait(mbar_pv_done_addr, phase_pv_done[q]);
            phase_pv_done[q] ^= 1;
          }
          // Order correction's O access after stats_ready and, after tile 0, the previous PV.
          asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
          // TODO: redundant?; current understanding, using to reconverge all lanes
          // after their per-lane barrier waits
          __syncwarp();

          if (!first_kv) {
            const float row_rescale = rescale_smem[q][row];
            // TMEM O loads/stores are warp-wide collectives, so if even one of these 32 rows
            // needs rescaling, all 32 lanes run the pass; unchanged rows multiply O by 1
            const bool any_row_needs_o_rescale =
                __any_sync(FULL_WARP_MASK, row_rescale != 1.0f);
            if (any_row_needs_o_rescale) {
              float o8[8];
#pragma unroll
              for (int out_n8 = 0; out_n8 < HEAD_DIM / 8; ++out_n8) {
                const int taddr = taddr_o_stage + trow + out_n8 * 8;
                tcgen05::ld_32x32b_x8(taddr, o8);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                  o8[i] *= row_rescale;
                }
                tcgen05::st_32x32b_x8(taddr, o8);
              }
              asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
              asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
            }
          }

          // wait until all lanes finish their part of O rescaling before lane 0 publishes O-safe
          __syncwarp();
          if (lane_id == 0) {
            mbarrier::arrive(mbar_p_high96_ready_o_safe_addr);
          }
        }
      }

      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int mbar_final_inv_ready_addr = q_stage::mbar_addr(mbar_final_inv_ready_base, q);
        const int mbar_epi_ready_addr = q_stage::mbar_addr(mbar_epi_ready_base, q);
        const int mbar_epi_free_addr = q_stage::mbar_addr(mbar_epi_free_base, q);
        const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
        const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, q);
        nv_bfloat16* sO_stage =
            reinterpret_cast<nv_bfloat16*>(smem_raw + q_stage::STAGES * Q_TILE_BYTES + q * O_TILE_BYTES);
        // Softmax publishes inv_denom from rowsum only; it no longer reads O.
        // Loop iterations 1..K-1 observed PV(q,0)..PV(q,K-2).
        // The cursor now names final PV(q,K-1), which must complete before
        // correction reads the final O accumulator.
        mbarrier::wait(mbar_pv_done_addr, phase_pv_done[q]);
        phase_pv_done[q] ^= 1;
        // final_inv_ready is reinitialized per work item and completes one phase,
        // so this single wait always uses phase 0; no local phase cursor is needed.
        mbarrier::wait(mbar_final_inv_ready_addr, 0);
        // In a persistent CTA shell, the next work item reuses this same SMEM sO stage.
        // Do not overwrite sO[q] until the previous epilogue has finished reading/storing it.
        mbarrier::wait(mbar_epi_free_addr, phase_epi_free[q]);
        phase_epi_free[q] ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        float o8[8];
#pragma unroll
        for (int out_n8 = 0; out_n8 < HEAD_DIM / 8; ++out_n8) {
          const int taddr = taddr_o_stage + trow + out_n8 * 8;
          tcgen05::ld_32x32b_x8(taddr, o8);

          nv_bfloat162 out_bf16x2[4];
          const float inv_denom = inv_denom_smem[q][row];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 v = make_float2(o8[2 * i] * inv_denom, o8[2 * i + 1] * inv_denom);
            out_bf16x2[i] = __float22bfloat162_rn(v);
          }

          nv_bfloat16* smem_ptr = sO_stage + row * HEAD_DIM + out_n8 * 8;
          reinterpret_cast<int4*>(smem_ptr)[0] = reinterpret_cast<int4*>(out_bf16x2)[0];
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncwarp();
        if (lane_id == 0) {
          mbarrier::arrive(mbar_epi_ready_addr);
        }
      }

      // Each final O load completed through tcgen05.wait::ld in the load helper.
      // Export that tcgen ordering into non_epi_done so its final wait can order
      // MMA-warp TMEM deallocation after every correction warp's O reads.
      asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
      __syncwarp();
      // non_epi_done counts completed non-epilogue warp roles, not threads: one
      // arrival represents this whole warp finishing its work. It expects
      // NON_EPI_ACTIVE_WARPS arrivals, so only lane 0 arrives. If every lane
      // arrived, one warp could advance the phase before the other roles finished.
      if (lane_id == 0) {
        mbarrier::arrive(mbar_non_epi_done_addr);
      }
    }
  }

  if (is_epi_warp) {
    int phase_epi_ready[q_stage::STAGES] = {0, 0};
    for (int work_id = blockIdx.x; work_id < total_work_items; work_id += gridDim.x) {
      const int q_row0_base = work_id * q_stage::ROWS_PER_WORK_ITEM;

      if (elect_sync()) {
        for (int q = 0; q < q_stage::STAGES; ++q) {
          const int mbar_epi_ready_addr = q_stage::mbar_addr(mbar_epi_ready_base, q);
          const int q_row0_stage = q_stage::row0(q_row0_base, q);
          const int sO_stage = O_smem + q * O_TILE_BYTES;
          mbarrier::wait(mbar_epi_ready_addr, phase_epi_ready[q]);
          phase_epi_ready[q] ^= 1;
          tma_2d_smem2gmem(&O_tmap, sO_stage, 0, q_row0_stage);
          // Launch the two TMA output copies separately: commit each q-stage store
          // as soon as it is issued instead of waiting until both q stores are
          // issued. This lets q0's global store drain while the epilogue warp waits
          // for q1's sO handoff.
          // Per-stage read-side waits below also let q0's sO be reused before
          // q1's store has drained.
          cp_async_bulk_commit_group();
        }
        cp_async_bulk_wait_group_read<1>();
        mbarrier::arrive(q_stage::mbar_addr(mbar_epi_free_base, 0));
        cp_async_bulk_wait_group_read<0>();
        mbarrier::arrive(q_stage::mbar_addr(mbar_epi_free_base, 1));
      }
    }
  }

  if (is_mma_warp) {
    // (work_count - 1) & 1 does not by itself mean "the final work iteration."
    // It is just a bit that flips every iteration, so the same value appeared on
    // earlier work items too. It means "final" here only because we check it at a
    // point where the surrounding loop and dependencies
    // already prove we are at the last relevant work item. The surrounding context
    // makes it final, not the bit value itself.
    //
    // Only the elected lane runs the persistent MMA loop. It cannot enter the next
    // work item until setup_done is published after the preceding non_epi_done phase
    // completes. Therefore, when the elected lane exits the loop, every earlier
    // non_epi_done generation is complete and the final generation is either current
    // or immediately preceding.
    //
    // The other 31 lanes skip that loop. Without reconvergence, they could reach
    // the wait below while non_epi_done is on an earlier generation with the same
    // parity, causing their waits to return before the final non-epilogue roles finish.
    //
    // Even without this __syncwarp(), the mandatory warp-collective dealloc.sync
    // would keep TMEM safe by holding those lanes until the elected lane arrived
    // after final non-epilogue completion. However, their earlier mbarrier waits
    // would not actually mean "wait for final non_epi_done."
    //
    // Reconverge here so every lane reaches the parity wait only after the elected
    // lane exits the loop. final_non_epi_phase then refers to a generation that
    // mbarrier::wait may validly observe: the current or immediately preceding one.
    __syncwarp();

    // Number of work items assigned to this persistent CTA; used to determine
    // the parity of the final non_epi_done phase before TMEM deallocation.
    const int work_count = (total_work_items - 1 - blockIdx.x) / gridDim.x + 1;
    const int final_non_epi_phase = (work_count - 1) & 1;
    mbarrier::wait(mbar_non_epi_done_addr, final_non_epi_phase);
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(taddr_base), "r"(TMEM_ALLOC_COLS)
                 : "memory");
  }
}

void attention_tcgen05_v20_launch(const nv_bfloat16* Q_ptr, const nv_bfloat16* K_ptr,
                                  const nv_bfloat16* V_ptr, nv_bfloat16* O_ptr, int bs,
                                  int len_q, int len_kv, cudaStream_t stream) {
  TORCH_CHECK(bs > 0, "bs must be positive");
  TORCH_CHECK(len_q >= q_stage::ROWS_PER_WORK_ITEM, "len_q must be at least 256");
  TORCH_CHECK(len_q % q_stage::ROWS_PER_WORK_ITEM == 0, "len_q must be a multiple of 256");
  TORCH_CHECK(len_kv >= BLOCK_N, "len_kv must be at least 128");
  TORCH_CHECK(len_kv % BLOCK_N == 0, "len_kv must be a multiple of 128");

  const int Q_height = bs * len_q;
  const int KV_height = bs * len_kv;

  CUtensorMap Q_tmap{};
  CUtensorMap K_tmap{};
  CUtensorMap V_tmap{};
  CUtensorMap O_tmap{};

  // Inherited from kernel 3: Q/K/V tensor maps use SW128 boxes matching the
  // SMEM operand layout. O_tmap uses no swizzle.
  init_tmap_2d_simple(&Q_tmap, const_cast<nv_bfloat16*>(Q_ptr), /*global_height=*/Q_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/BLOCK_M, /*shared_width=*/SW128_ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_128B);

  init_tmap_2d_simple(&K_tmap, const_cast<nv_bfloat16*>(K_ptr), /*global_height=*/KV_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/BLOCK_N, /*shared_width=*/SW128_ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_128B);

  init_tmap_2d_simple(&V_tmap, const_cast<nv_bfloat16*>(V_ptr), /*global_height=*/KV_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/SW128_ATOM_ROWS, /*shared_width=*/SW128_ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_128B);

  init_tmap_2d_simple(&O_tmap, O_ptr, /*global_height=*/Q_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/BLOCK_M, /*shared_width=*/HEAD_DIM,
                      CU_TENSOR_MAP_SWIZZLE_NONE);

  const int total_work_items = bs * (len_q / q_stage::ROWS_PER_WORK_ITEM);

  int device = 0;
  int sm_count = 0;
  check_cuda(cudaGetDevice(&device));
  check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  const int persistent_grid = sm_count * PERSISTENT_CTAS_PER_SM;
  const int grid = (total_work_items < persistent_grid) ? total_work_items : persistent_grid;
  const size_t smem_bytes =
      size_t(q_stage::STAGES * Q_TILE_BYTES + q_stage::STAGES * O_TILE_BYTES +
             kv_ring::STAGES * KV_TILE_BYTES);

  if (smem_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(attention_tcgen05_v20_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smem_bytes)));
  }

  attention_tcgen05_v20_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
      Q_tmap, K_tmap, V_tmap, O_tmap, len_q, len_kv, total_work_items);
  check_cuda(cudaGetLastError());
}
