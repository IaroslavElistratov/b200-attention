// Kernel 6 focus: add KV[3] load pipelining on top of kernel 5.
//   1. Keep q_stage=2 from kernel 5.
//   2. Replace the single K/V slot with the unified KV[3] token load pipeline.
//   3. Reuse each loaded K/V tile for Q0 and Q1; row warps still do O-rescale inline.

#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// tcgen05 + TMA + TMEM attention with q_stage=2.
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
//   1) load warp loads Q0/Q1 once and drives the unified KV[3] token stream
//   2) MMA warp computes S0 = Q0 @ K^T, then S1 = Q1 @ K^T using the same K tile
//   3) row warps for stage0/stage1 independently consume S0/S1, update online softmax,
//      rescale O0/O1, and write packed P0/P1
//   4) MMA warp waits for each Q stage's row work, then accumulates PVq into Oq for q in {0,1}
//   5) after the KV loop: normalize O0/O1 by rowsum0/rowsum1 and store bf16 output
//
namespace {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int HEAD_DIM = 128;
constexpr int MMA_K = 16;

namespace q_stage {
constexpr int STAGES = 2;
constexpr int ROWS_PER_WORK_ITEM = STAGES * BLOCK_M;  // Q0 + Q1
}  // namespace q_stage

constexpr int ROW_WARPS_PER_STAGE = 4;
constexpr int ROW_WARP_COUNT = q_stage::STAGES * ROW_WARPS_PER_STAGE;
constexpr int LOAD_WARP_ID = ROW_WARP_COUNT;
constexpr int MMA_WARP_ID = LOAD_WARP_ID + 1;
constexpr int NUM_WARPS = MMA_WARP_ID + 1;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BF16_BYTES = int(sizeof(nv_bfloat16));
constexpr int MMA_K_BYTES = MMA_K * BF16_BYTES;         // 32
// In this kernel's matrix-coordinate view, SW128 operand panels are 8 rows x 64 bf16 cols.
// K-major and MN-major descriptors organize/interpret those panels differently.
constexpr int SW128_BYTES = 128;
constexpr int SW128_ATOM_ROWS = 8;
constexpr int SW128_ATOM_COLS = SW128_BYTES / BF16_BYTES;  // 64 bf16
#include "smem_layout_swizzle128.cuh"

constexpr int Q_TILE_BYTES = BLOCK_M * HEAD_DIM * BF16_BYTES;
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
// It does not own scheduling, phases, or wait order. The kernel body defines the
// load-pipeline structure; these are just stateless ring helpers, not a pipeline object.
namespace kv_ring {

constexpr int STAGES = 3;
constexpr int TOKENS_PER_KV_TILE = 2;
constexpr int K_KIND = 0;
constexpr int V_KIND = 1;

// position in the infinite logical K0, V0, K1, V1, ... buffer stream (token)
__device__ __forceinline__ int logical_token(int kv_tile, int kind) {
  return TOKENS_PER_KV_TILE * kv_tile + kind;
}

// collapse the logical token stream onto the physical KV[3] ring (stage)
__device__ __forceinline__ int physical_stage(int token) {
  return token % STAGES;
}

__device__ __forceinline__ int token_tile(int token) {
  return token / TOKENS_PER_KV_TILE;
}

// Using the collapsed stage indexes into the circular buffer (stage),
// not logical index into the infinite logical buffer array (token).
// Select the physical SMEM slot or physical barrier for that KV stage.
__device__ __forceinline__ int stage_smem_addr(int KV_smem, int stage) {
  return KV_smem + stage * KV_TILE_BYTES;
}

__device__ __forceinline__ int stage_mbar_addr(int mbar_kv_ready_base, int stage) {
  return mbar_kv_ready_base + stage * int(sizeof(uint64_t));
}

__device__ __forceinline__ bool is_k_token(int token) {
  // K/V tokens alternate, so parity identifies the token kind.
  return (token % TOKENS_PER_KV_TILE) == K_KIND;
}

__device__ __forceinline__ void load_token_and_arrive(
    int token, int batch_id, int len_kv, int KV_smem, int mbar_kv_ready_base,
    const CUtensorMap* K_tmap, const CUtensorMap* V_tmap) {
  const int stage = physical_stage(token);
  // The load loop is token-indexed (K0, V0, K1, V1, ...), not tile-indexed.
  // Recover the KV tile to find the global K/V row range for this TMA load.
  const int tile = token_tile(token);
  const int kv_row0 = batch_id * len_kv + tile * BLOCK_N;
  // stage indexes the physical KV[3] ring slot and its matching ready barrier.
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

}  // namespace

__global__ __launch_bounds__(TB_SIZE) void attention_tcgen05_v6_kernel(
    const __grid_constant__ CUtensorMap Q_tmap, const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap, nv_bfloat16* __restrict__ O_ptr, int len_q,
    int len_kv) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const bool is_row_warp = (warp_id < ROW_WARP_COUNT);
  const bool is_load_warp = (warp_id == LOAD_WARP_ID);
  const bool is_mma_warp = (warp_id == MMA_WARP_ID);

  // Q/O is indexed by (batch_id, q_tile_pair_id). The 1D launch flattens
  // that pair in batch-major order into blockIdx.x, so work_id directly
  // selects its consecutive Q/O row range.
  const int work_id = blockIdx.x;
  const int q_row0_base =
      work_id * q_stage::ROWS_PER_WORK_ITEM;  // first flattened Q/O row

  // K/V is indexed by (batch_id, kv_tile_id), not by q_tile_pair_id.
  // Decode batch_id here; the K/V loop or token stream determines kv_tile_id later.
  const int work_items_per_batch = len_q / q_stage::ROWS_PER_WORK_ITEM;
  const int batch_id = work_id / work_items_per_batch;

  extern __shared__ __align__(1024) char smem_raw[];

  const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
  const int Q_smem = smem_base;
  const int KV_smem = Q_smem + q_stage::STAGES * Q_TILE_BYTES;

  // Stage barriers:
  //   q-ready:        load warp -> mma warp (Q0/Q1 loaded once)
  //   kv-ready[s]:    load warp -> mma warp (stage s currently holds the next K or V token)
  //   qk-done[q]:     mma warp -> row warps for Q stage q; qk-done[last_q] also recycles old K tokens
  //   softmax-done[q]: row warps -> mma warp (full Pq ready + old Oq safe)
  //   pv-done[q]:     mma warp -> row warps for O visibility; pv-done[last_q] also recycles old V tokens
  __shared__ uint64_t mbar_q_ready[1];
  __shared__ uint64_t mbar_qk_done[q_stage::STAGES];
  __shared__ uint64_t mbar_softmax_done[q_stage::STAGES];
  __shared__ uint64_t mbar_pv_done[q_stage::STAGES];
  __shared__ uint64_t mbar_kv_ready[kv_ring::STAGES];
  __shared__ int tmem_addr[1];

  const int mbar_q_ready_addr = static_cast<int>(__cvta_generic_to_shared(mbar_q_ready));
  const int mbar_qk_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_qk_done));
  const int mbar_softmax_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_softmax_done));
  const int mbar_pv_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_pv_done));
  const int mbar_kv_ready_base = static_cast<int>(__cvta_generic_to_shared(mbar_kv_ready));

  if (is_load_warp && elect_sync()) {
    mbarrier::init(mbar_q_ready_addr, 1);
    for (int q = 0; q < q_stage::STAGES; ++q) {
      const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
      const int mbar_softmax_done_addr = q_stage::mbar_addr(mbar_softmax_done_base, q);
      const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
      mbarrier::init(mbar_qk_done_addr, 1);
      mbarrier::init(mbar_softmax_done_addr, ROW_WARPS_PER_STAGE);
      mbarrier::init(mbar_pv_done_addr, 1);
    }
    for (int s = 0; s < kv_ring::STAGES; ++s) {
      const int mbar_kv_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, s);
      mbarrier::init(mbar_kv_ready_addr, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (is_mma_warp) {
    // TMEM layout:
    //   S0 at [0..127], S1 at [128..255], O0 at [256..383], O1 at [384..511]
    //   Pq overlays the lower half of Sq.
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

  // Producer loop (load warp): drive one generic K/V token stream through KV[3].
  // Even tokens are K(tile), odd tokens are V(tile). A physical stage is reusable
  // after the last Q stage consumes a K token for QK or a V token for PV.
  if (is_load_warp && elect_sync()) {
    // Load Q0 and Q1 once into SMEM.
    for (int q = 0; q < q_stage::STAGES; ++q) {
      const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
      const int q_row0_stage = q_stage::row0(q_row0_base, q);
      swizzle128::tma_load_kmajor(Q_stage_smem, &Q_tmap, BLOCK_M, q_row0_stage,
                                        mbar_q_ready_addr);
    }
    mbarrier::arrive_expect_tx(mbar_q_ready_addr, q_stage::STAGES * Q_TILE_BYTES);

    // So basically we're waiting on qk_done or pv_done barrers, but what
    // we actaully want form it -- is to know when we can re-use
    // preivous k, or v. Thus the name of the local cursors below,
    // and that's why they don't don't match names of the barriers
    // we're waiting on (qk_done, pv_done)
    int phase_k_recycle = 0;
    int phase_v_recycle = 0;
    const int last_q = q_stage::STAGES - 1;
    const int mbar_last_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, last_q);
    const int mbar_last_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, last_q);
    const int kv_tokens = kv_ring::TOKENS_PER_KV_TILE * kv_tiles;
    const int prefill_tokens = (kv_tokens < kv_ring::STAGES) ? kv_tokens : kv_ring::STAGES;

    // Prefill: the first KV[3] tokens fill one token per physical KV stage;
    // no old token exists yet, so there is no recycle wait.
    for (int token = 0; token < prefill_tokens; ++token) {
      kv_ring::load_token_and_arrive(
          token, batch_id, len_kv, KV_smem, mbar_kv_ready_base, &K_tmap, &V_tmap);
    }

    // Refill:
    // Starts after the already-prefilled tokens.
    // Refill: token T reuses the stage occupied by token T-KV[3], so wait on
    // the last Q stage's consumer before overwriting that physical stage.

    // Semantically, load warp does not advance in whole K/V tile pairs; it advances through the logical token stream.
    // So the loop over flattened tokens (kv_tile*2, ie loop over tokens) is used here, mostly as readability
    // and to emphasise that producer (load warp) semantically does not operate on tiles-granularity,
    // but rather on invidiaul tokens/buffers (and entire pipile lengh is 1.5 tiles, so the
    // window can cut through the middle of a logical kv_tile).
    // A loop over kv_tile could be equivalent if it still produced K and V
    // as separate tokens. The flat loop (token granularity) is used because it
    // explictly shows the loader state machine: per-token readiness/recycle over the KV[3] ring

    // The flat token loop is not required for correctness.
    // A nested kv_tile loop could be equivalent if it still produces K and V as separate tokens.
    //
    // But the flat loop better matches the producer state machine:
    //   producer owns KV[3] physical slot reuse,
    //   and those slots hold individual K-or-V tokens, not whole K/V tile pairs.
    for (int token = prefill_tokens; token < kv_tokens; ++token) {

      // 1) wait on the old token ocuppying the buffer to be freed
      //
      // token variable already basically means +3 (pipeline length) into the future (ie
      // next token to load after the prefilled window);
      // where does that token go? stage = token % 3; and to get the (old) token which currently
      // occupies that possition in the circular buffer, we do -3. This is needed so that before
      // overwritting the old token stored there, we wait for the operation (which is consuming that
      // token) to complete.
      const int recycle_token = token - kv_ring::STAGES;
      if (kv_ring::is_k_token(recycle_token)) {
        // wait specifically on the **last Q stage** which consumes respective buffer-resident k token
        mbarrier::wait(mbar_last_qk_done_addr, phase_k_recycle);
        phase_k_recycle ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
      } else {
        // wait specifically on the **last Q stage** which consumes respective buffer-resident v token
        mbarrier::wait(mbar_last_pv_done_addr, phase_v_recycle);
        phase_v_recycle ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
      }

      // 2) write the new token there
      kv_ring::load_token_and_arrive(
          token, batch_id, len_kv, KV_smem, mbar_kv_ready_base, &K_tmap, &V_tmap);
    }
  }

  // MMA warp: use the same K/V tile for both Q stages.
  if (is_mma_warp && elect_sync()) {
    // warp-local parity cursor for that physical barrier;
    // basically, for this physical mbarrier, wait for
    // next time this bairer flips -- same as in kernel 4;
    // phase variable is local parity/bookmark -- not a tile id,
    // not a full generation counter, not shared across warps
    int phase_kv_ready[kv_ring::STAGES] = {0, 0, 0};
    int phase_softmax_done[q_stage::STAGES] = {0, 0};

    mbarrier::wait(mbar_q_ready_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // for MMA warp we keep the loop over KV tiles (tile-granular) because semantically it
    // owns the QK -> P -> PV update. But we could have equivalently
    // represented the MMA warp loop differently (as a loop over flat tokens) -- it's just
    // the matter of semantic emphasis. A flat token-indexed loop would
    // obscure the QK_i -> P_i -> PV_i dependency.
    // Kernel 7 is where we actually use the q-stage scheduling freedom:
    // PVq(i) -> QKq(i+1)
    //
    // K/V readiness is token-based, but the compute schedule still owns one
    // logical attention update per KV tile. The next kernel uses the per-q S/P
    // slots to start QKq(i+1) earlier
    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
      const int k_token = kv_ring::logical_token(kv_tile, kv_ring::K_KIND);
      const int k_stage = kv_ring::physical_stage(k_token);
      // stage indexes the physical KV[3] ring slot and its matching ready barrier
      const int mbar_k_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, k_stage);
      const int K_stage_smem = kv_ring::stage_smem_addr(KV_smem, k_stage);

      mbarrier::wait(mbar_k_ready_addr, phase_kv_ready[k_stage]);
      phase_kv_ready[k_stage] ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
        const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
        const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, q);

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

        // Track QK completion; the mbarrier arrives when this Q stage's S is ready.
        tcgen05::commit_arrive(mbar_qk_done_addr);
      }

      const int v_token = kv_ring::logical_token(kv_tile, kv_ring::V_KIND);
      const int v_stage = kv_ring::physical_stage(v_token);
      const int mbar_v_ready_addr = kv_ring::stage_mbar_addr(mbar_kv_ready_base, v_stage);
      const int V_stage_smem = kv_ring::stage_smem_addr(KV_smem, v_stage);

      // In kernel 5, we weren't waiting on a separate V-ready barrier here.
      // Kernel 5 always loads one K tile and one V tile before even starting the
      // MMA warp's work for that KV iteration. The single kv_ready barrier is only
      // signaled after both K and V are ready. So by the time MMA starts QK, both
      // operands are already guaranteed ready, and later PV does not need a second
      // V-ready wait.
      // Here, we preload each operand separately as tokens: K0, V0, K1, V1, ...
      // Waiting for the K token before QK only proves K is ready; it does not prove
      // the matching V token is ready. Thus the additional wait below on V-ready before PV.
      mbarrier::wait(mbar_v_ready_addr, phase_kv_ready[v_stage]);
      phase_kv_ready[v_stage] ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      const bool first_kv = (kv_tile == 0);
      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int mbar_softmax_done_addr = q_stage::mbar_addr(mbar_softmax_done_base, q);
        const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
        const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, q);
        const int taddr_p_stage = taddr_s_stage;  // P overlays S in this q-stage.
        const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, q);

        mbarrier::wait(mbar_softmax_done_addr, phase_softmax_done[q]);
        phase_softmax_done[q] ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        constexpr int pv_v_chunk_bytes = HEAD_DIM * MMA_K_BYTES;
        constexpr uint64_t pv_desc_chunk_step = pv_v_chunk_bytes / 16;
        uint64_t b_desc = swizzle128::desc_mnmajor(V_stage_smem, HEAD_DIM);
        for (int k = 0; k < BLOCK_N / MMA_K; ++k) {
          // Advance by 16 V rows. In the SW128 MN-major layout this is
          // equivalent to two 8-token row-block strides in SMEM.
          // Inherited from kernel 2: each PV microstep advances 8 packed-P TMEM columns.
          const int taddr_a = taddr_p_stage + k * 8;
          const int enable_input_d = (first_kv && k == 0) ? 0 : 1;
          tcgen05::mma_f16_a_tmem(taddr_o_stage, taddr_a, b_desc, i_desc_pv, enable_input_d);
          b_desc += pv_desc_chunk_step;
        }

        // Track PV completion; the mbarrier arrives when this Q stage's O is ready.
        tcgen05::commit_arrive(mbar_pv_done_addr);
      }
    }
  }

  // Row warps: one 4-warp group per Q stage. They still own both row-side jobs:
  // online softmax/P publication and O rescale before the next PV.
  if (is_row_warp) {
    const int row_q_stage = warp_id / ROW_WARPS_PER_STAGE;
    const int local_warp_idx = warp_id % ROW_WARPS_PER_STAGE;
    const int row_base = local_warp_idx * WARP_SIZE;  // first of 32 rows owned by this warp
    const int row = row_base + lane_id;               // specific row owned by this lane
    const int trow = row_base << 16;                  // encode row_base for TMEM addressing
    int phase_qk_done = 0;
    int phase_pv_done = 0;

    // q-stage-local resources for this row warp; invariant across all KV tiles.
    const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, row_q_stage);
    const int mbar_softmax_done_addr = q_stage::mbar_addr(mbar_softmax_done_base, row_q_stage);
    const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, row_q_stage);
    const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, row_q_stage);
    const int taddr_p_stage = taddr_s_stage;  // P overlays S in this q-stage.
    const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, row_q_stage);

    // Online softmax state per row and per Q stage.
    float rowmax = -FLT_MAX;
    float rowsum = 0.0f;
    // Keep as a runtime constant (device rsqrtf) to avoid constexpr sqrt issues.
    const float softmax_scale = rsqrtf(float(HEAD_DIM));

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
      const bool first_kv = (kv_tile == 0);

      mbarrier::wait(mbar_qk_done_addr, phase_qk_done);
      phase_qk_done ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      // Pass 1: compute max for this KV tile's scores.
      float tile_rowmax = -FLT_MAX;
      {
#pragma unroll
        for (int score_n8 = 0; score_n8 < BLOCK_N / 8; ++score_n8) {
          float s8[8];
          const int taddr = taddr_s_stage + trow + score_n8 * 8;
          tcgen05::ld_32x32b_x8(taddr, s8);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            tile_rowmax = fmaxf(tile_rowmax, s8[i]);
          }
        }
      }

      // Online update: new max, rescale factor.
      tile_rowmax *= softmax_scale;
      const float new_rowmax = fmaxf(rowmax, tile_rowmax);
      const float rescale = __expf(rowmax - new_rowmax);
      rowmax = new_rowmax;
      rowsum *= rescale;

      // Rescale accumulated O before this tile's PV update.
      if (!first_kv) {
        float o8[8];
#pragma unroll
        for (int out_n8 = 0; out_n8 < HEAD_DIM / 8; ++out_n8) {
          const int taddr = taddr_o_stage + trow + out_n8 * 8;
          tcgen05::ld_32x32b_x8(taddr, o8);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            o8[i] *= rescale;
          }
          tcgen05::st_32x32b_x8(taddr, o8);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
      }

      // Pass 2: read S from TMEM, write packed P to TMEM (overlay), accumulate rowsum.
      // Lower-half overlay lets us scan low->high: each packed P write only overwrites
      // S columns that were already consumed by this or an earlier chunk.
      float tile_rowsum = 0.0f;
      {
        const int taddr_s_row = taddr_s_stage + trow;
        const int taddr_p_row = taddr_p_stage + trow;
#pragma unroll
        for (int score_n16 = 0; score_n16 < BLOCK_N / 16; ++score_n16) {
          uint32_t p8_packed[8];
#pragma unroll
          for (int load8 = 0; load8 < 2; ++load8) {
            float s8[8];
            const int col_s = score_n16 * 16 + load8 * 8;
            tcgen05::ld_32x32b_x8(taddr_s_row + col_s, s8);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float score0 = s8[2 * j] * softmax_scale;
              const float score1 = s8[2 * j + 1] * softmax_scale;
              const float p0 = __expf(score0 - rowmax);
              const float p1 = __expf(score1 - rowmax);
              tile_rowsum += p0 + p1;
              p8_packed[load8 * 4 + j] = bf16::pack2_to_u32(p0, p1);
            }
          }
          tcgen05::st_32x32b_x8_u32(taddr_p_row + score_n16 * 8, p8_packed);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
      }
      rowsum += tile_rowsum;

      // Publish row-warp TMEM updates (O rescale + P write) before signaling MMA warp.
      asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
      __syncwarp();
      if (lane_id == 0) {
        mbarrier::arrive(mbar_softmax_done_addr);
      }

      // Wait for PV completion before reading/rescaling O for the next tile.
      mbarrier::wait(mbar_pv_done_addr, phase_pv_done);
      phase_pv_done ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    }

    // Epilogue: normalize accumulated O by rowsum and store bf16 output.
    const float inv_denom = 1.0f / rowsum;
    const int q_row0_stage = q_stage::row0(q_row0_base, row_q_stage);
    float o8[8];
#pragma unroll
    // This 8 is the tcgen05.ld/st x8 TMEM column width, not a SMEM atom/slice size.
    for (int out_n8 = 0; out_n8 < HEAD_DIM / 8; ++out_n8) {
      const int taddr = taddr_o_stage + trow + out_n8 * 8;
      tcgen05::ld_32x32b_x8(taddr, o8);

      nv_bfloat162 out_bf16x2[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 v = make_float2(o8[2 * i] * inv_denom, o8[2 * i + 1] * inv_denom);
        out_bf16x2[i] = __float22bfloat162_rn(v);
      }

      nv_bfloat16* out_ptr = O_ptr + (q_row0_stage + row) * HEAD_DIM + out_n8 * 8;
      reinterpret_cast<int4*>(out_ptr)[0] = reinterpret_cast<int4*>(out_bf16x2)[0];
    }
  }

  __syncthreads();

  if (is_mma_warp) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(taddr_base), "r"(TMEM_ALLOC_COLS)
                 : "memory");
  }
}

void attention_tcgen05_v6_launch(const nv_bfloat16* Q_ptr, const nv_bfloat16* K_ptr,
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

  // Inherited from kernel 3: tensor maps use SW128 boxes matching the SMEM operand layout.
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

  const int grid = bs * (len_q / q_stage::ROWS_PER_WORK_ITEM);
  const size_t smem_bytes =
      size_t(q_stage::STAGES * Q_TILE_BYTES + kv_ring::STAGES * KV_TILE_BYTES);

  if (smem_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(attention_tcgen05_v6_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smem_bytes)));
  }

  attention_tcgen05_v6_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
      Q_tmap, K_tmap, V_tmap, O_ptr, len_q, len_kv);
  check_cuda(cudaGetLastError());
}
