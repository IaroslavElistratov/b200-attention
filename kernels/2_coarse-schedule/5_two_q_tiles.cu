// Kernel 5 focus: q_stage=2 on top of kernel 4 warp specialization.
//   1. Keep kernel 4's single-stage K/V storage and no load pipelining.
//   2. Switch from one Q tile to two Q tiles per CTA.
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
//   1) load warp loads Q0/Q1 once, then loads one K/V tile at a time
//   2) MMA warp computes S0 = Q0 @ K^T, then S1 = Q1 @ K^T using the same K tile
//   3) row warps for stage0/stage1 independently consume S0/S1, update online softmax,
//      rescale O0/O1, and write packed P0/P1
//   4) MMA warp waits for each Q stage's row work, then accumulates PVq into Oq for q in {0,1}
//   5) after the KV loop: normalize O0/O1 by rowsum0/rowsum1 and store bf16 output

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
constexpr int K_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;
constexpr int V_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;

constexpr int TMEM_S_COLS = BLOCK_N;
constexpr int TMEM_O_COLS = HEAD_DIM;
constexpr int TMEM_ALLOC_COLS = q_stage::STAGES * TMEM_S_COLS + q_stage::STAGES * TMEM_O_COLS;
// Full 512-column TMEM budget is consumed by S0/S1 + O0/O1.
// That is why this q_stage=2 branch cannot also keep an extra S/P time-stage.
static_assert(TMEM_ALLOC_COLS == 512, "q_stage=2 should pack S0/S1 + O0/O1 into 512 TMEM cols");

// Local lightweight helpers for the two-Q-stage carrier.
// They name q-stage address math only; the kernel body still owns scheduling,
// phases, waits/fences, and MMA order.
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

}  // namespace

__global__ __launch_bounds__(TB_SIZE) void attention_tcgen05_v5_kernel(
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
  const int K_smem = Q_smem + q_stage::STAGES * Q_TILE_BYTES;
  const int V_smem = K_smem + K_TILE_BYTES;

  // Stage barriers:
  //   q-ready:        load warp -> mma warp (Q0/Q1 loaded once)
  //   kv-ready:       load warp -> mma warp (single K/V slot for the current tile)
  //   qk-done[q]:     mma warp -> row warps for Q stage q
  //   softmax-done[q]: row warps -> mma warp (full Pq ready + old Oq safe)
  //   pv-done[q]:     mma warp -> row warps for O visibility; pv-done[last_q] also gates load-warp K/V-slot reuse
  // Use [1] barriers for events that are not per Q stage, and q_stage::STAGES barriers for events used per Q stage.
  __shared__ uint64_t mbar_q_ready[1];
  __shared__ uint64_t mbar_kv_ready[1];
  __shared__ uint64_t mbar_qk_done[q_stage::STAGES];
  __shared__ uint64_t mbar_softmax_done[q_stage::STAGES];
  __shared__ uint64_t mbar_pv_done[q_stage::STAGES];
  __shared__ int tmem_addr[1];

  const int mbar_q_ready_addr = static_cast<int>(__cvta_generic_to_shared(mbar_q_ready));
  const int mbar_kv_ready_addr = static_cast<int>(__cvta_generic_to_shared(mbar_kv_ready));
  const int mbar_qk_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_qk_done));
  const int mbar_softmax_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_softmax_done));
  const int mbar_pv_done_base = static_cast<int>(__cvta_generic_to_shared(mbar_pv_done));

  if (is_load_warp && elect_sync()) {
    mbarrier::init(mbar_q_ready_addr, 1);
    mbarrier::init(mbar_kv_ready_addr, 1);
    for (int q = 0; q < q_stage::STAGES; ++q) {
      const int mbar_qk_done_addr = q_stage::mbar_addr(mbar_qk_done_base, q);
      const int mbar_softmax_done_addr = q_stage::mbar_addr(mbar_softmax_done_base, q);
      const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
      mbarrier::init(mbar_qk_done_addr, 1);
      mbarrier::init(mbar_softmax_done_addr, ROW_WARPS_PER_STAGE);
      mbarrier::init(mbar_pv_done_addr, 1);
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

  // Producer loop (load warp): still single-stage K/V storage.
  // The next K/V tile is not loaded until the last Q stage's PV has consumed
  // the current V and produced visible O.
  if (is_load_warp && elect_sync()) {
    // Load Q0 and Q1 once into SMEM.
    for (int q = 0; q < q_stage::STAGES; ++q) {
      const int Q_stage_smem = q_stage::q_smem(Q_smem, q);
      const int q_row0_stage = q_stage::row0(q_row0_base, q);
      swizzle128::tma_load_kmajor(Q_stage_smem, &Q_tmap, BLOCK_M, q_row0_stage,
                                        mbar_q_ready_addr);
    }
    mbarrier::arrive_expect_tx(mbar_q_ready_addr, q_stage::STAGES * Q_TILE_BYTES);

    // phase_kv_reuse is just the local cursor, that only tracks parity
    // (ie last time I waited e.g. on phase 0, now im waiting on 1),
    // and it doens't track more info than that (ie i'm waiting on
    // pv_done of the last q stage, in the 56'th kv_tile);
    // We wait on pv_done, but the load-warp question is whether the previous
    // K/V slot can be reused -- thus the variable name
    int phase_kv_reuse = 0;
    // pv_done barier of the last Q stage
    const int last_q = q_stage::STAGES - 1;
    const int mbar_last_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, last_q);

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
      const int kv_row0 = batch_id * len_kv + kv_tile * BLOCK_N;

      swizzle128::tma_load_kmajor(K_smem, &K_tmap, BLOCK_N, kv_row0, mbar_kv_ready_addr);
      swizzle128::tma_load_mnmajor(V_smem, &V_tmap, kv_row0, mbar_kv_ready_addr);
      mbarrier::arrive_expect_tx(mbar_kv_ready_addr, K_TILE_BYTES + V_TILE_BYTES);

      // waits for the mma warp pv_done of the last q-stage (cos we specifically selected smem address of the last q's barrier)
      // to flip that barrier, and without knowing what work of which kv_tile flipped it
      // (barrier does not encode the tile id, the tile order is enforced by the surrounding code)
      // but just detecting the flip (done by the mma warp) lets this wait advance
      mbarrier::wait(mbar_last_pv_done_addr, phase_kv_reuse);
      phase_kv_reuse ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    }
  }

  // MMA warp: use the same K/V tile for both Q stages.
  if (is_mma_warp && elect_sync()) {
    // kv_ready is one shared barrier and does not change from one q
    // stage to another -- so 1 barrier is warrranted;
    // another thing is qk_done -- which we do have 2 barriers in
    // total (different per q stage)
    int phase_kv_ready = 0;
    // MMA consumes both q-stage softmax_done streams, so it needs one local phase cursor per q.
    int phase_softmax_done[q_stage::STAGES] = {0, 0};

    mbarrier::wait(mbar_q_ready_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
      const bool first_kv = (kv_tile == 0);

      mbarrier::wait(mbar_kv_ready_addr, phase_kv_ready);
      phase_kv_ready ^= 1;
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
        uint64_t b_desc_panel = swizzle128::desc_kmajor(K_smem);
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

        // Track QK completion; the mbarrier arrives when this Q stage's S is ready;
        // that mbar_qk_done_addr is waited on by the softmax warps, not the load warp
        // (load warp only waits on last-q-stage pv_done)
        tcgen05::commit_arrive(mbar_qk_done_addr);
      }

      for (int q = 0; q < q_stage::STAGES; ++q) {
        const int mbar_softmax_done_addr = q_stage::mbar_addr(mbar_softmax_done_base, q);
        const int mbar_pv_done_addr = q_stage::mbar_addr(mbar_pv_done_base, q);
        const int taddr_s_stage = q_stage::tmem_s(taddr_s_base, q);
        const int taddr_p_stage = taddr_s_stage;  // P overlays S in this q-stage.
        const int taddr_o_stage = q_stage::tmem_o(taddr_o_base, q);

        // form the prective of the mma warp, these two: mbar_qk_done_addr, mbar_softmax_done_addr
        // (which both the mma waprp uses) these both are per q_stage glboal barriers.
        // we're using phase_softmax_done[2] as one of our two **local** parity cursors,
        // and for mbar_qk_done_addr we don't need a local curosr cos we just arrive, and not wait
        mbarrier::wait(mbar_softmax_done_addr, phase_softmax_done[q]);
        phase_softmax_done[q] ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        constexpr int pv_v_chunk_bytes = HEAD_DIM * MMA_K_BYTES;
        constexpr uint64_t pv_desc_chunk_step = pv_v_chunk_bytes / 16;
        uint64_t b_desc = swizzle128::desc_mnmajor(V_smem, HEAD_DIM);
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

  // we allocate groups of softmax warps, so each group processes its own stage
  // without needing the for loop over q stages (since have 2 q stages and 2 sets of softmax warps).
  // the only thing that changes in the softmax warps from kernel 4 is now
  // we compute per-q-stage pointer offsets into TMEM (per stage/warpgroup-- S, O, P)

  // Row warps: one 4-warp group per Q stage. They still own both row-side jobs:
  // online softmax/P publication and O rescale before the next PV.
  if (is_row_warp) {
    const int row_q_stage = warp_id / ROW_WARPS_PER_STAGE;
    const int local_warp_idx = warp_id % ROW_WARPS_PER_STAGE;
    const int row_base = local_warp_idx * WARP_SIZE;  // first of 32 rows owned by this warp
    const int row = row_base + lane_id;               // specific row owned by this lane
    const int trow = row_base << 16;                  // encode row_base for TMEM addressing
    // Row warps are already partitioned by q-stage: q0 has one row-warp group,
    // q1 has another. qk_done is shared per q-stage group, not per row warp, so
    // this warp only needs one local phase cursor for its group's qk_done stream
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

      tile_rowmax *= softmax_scale;
      const float new_rowmax = fmaxf(rowmax, tile_rowmax);
      const float rescale = __expf(rowmax - new_rowmax);
      rowmax = new_rowmax;
      rowsum *= rescale;

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

    const float inv_denom = 1.0f / rowsum;
    const int q_row0_stage = q_stage::row0(q_row0_base, row_q_stage);
    float o8[8];
#pragma unroll
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

void attention_tcgen05_v5_launch(const nv_bfloat16* Q_ptr, const nv_bfloat16* K_ptr,
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
      size_t(q_stage::STAGES * Q_TILE_BYTES + K_TILE_BYTES + V_TILE_BYTES);

  if (smem_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(attention_tcgen05_v5_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smem_bytes)));
  }

  attention_tcgen05_v5_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
      Q_tmap, K_tmap, V_tmap, O_ptr, len_q, len_kv);
  check_cuda(cudaGetLastError());
}
