// Kernel 4 focus:
//   1. Add warp specialization (load warp, MMA warp, row/softmax warps).
//   2. Keep single-stage scheduling (no ring/multi-stage buffers yet).
//   3. Keep kernel 3 math/data contracts (Q/K/V swizzled SMEM, P/O in TMEM).



#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// tcgen05 + TMA + TMEM attention with warp specialization (no multi-stage pipeline yet).
//
// Assumptions fixed for this teaching line:
//   - head_dim == 128
//   - BLOCK_M == 128 queries per CTA
//   - BLOCK_N == 128 keys per KV tile
//   - len_q % 128 == 0
//   - len_kv % 128 == 0
//   - dense, non-causal, no masking
//   - inputs/outputs are bf16
//
// Dataflow per KV tile:
//   1) TMA load K into K-major SW128 atoms.
//      TMA load V into MN-major SW128 8x64 atoms.
//   2) UMMA: S = Q @ K^T into TMEM (fp32)
//   3) softmax(S) online update (rowmax/rowsumexp) by reading S from TMEM
//      - rescale O accumulator in TMEM for KV tiles after the first tile
//      - write packed P (bf16x2 in b32 cells) into TMEM
//   4) UMMA: O += P @ V with A from TMEM and B from swizzled V_smem (MN-major) into TMEM (fp32)
//   5) after loop: write O/rowsumexp to global bf16

namespace {

constexpr int NUM_WARPS = 6;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int HEAD_DIM = 128;
constexpr int MMA_K = 16;
constexpr int BF16_BYTES = int(sizeof(nv_bfloat16));
constexpr int MMA_K_BYTES = MMA_K * BF16_BYTES;         // 32
// In this kernel's matrix-coordinate view, SW128 operand atoms are 8 rows x 64 bf16 cols.
// K-major and MN-major descriptors organize/interpret those atoms differently.
constexpr int SW128_BYTES = 128;
constexpr int SW128_ATOM_ROWS = 8;
constexpr int SW128_ATOM_COLS = SW128_BYTES / BF16_BYTES;  // 64 bf16
#include "smem_layout_swizzle128.cuh"
constexpr int TMEM_S_COLS = BLOCK_N;
constexpr int TMEM_O_COLS = HEAD_DIM;
constexpr int Q_TILE_BYTES  = BLOCK_M * HEAD_DIM * BF16_BYTES;
constexpr int K_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;
constexpr int V_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;
}  // namespace

__global__ __launch_bounds__(TB_SIZE) void attention_tcgen05_v4_kernel(
    const __grid_constant__ CUtensorMap Q_tmap, const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap, nv_bfloat16* __restrict__ O_ptr, int len_q,
    int len_kv) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  constexpr int ROW_WARP_BASE = 0;
  constexpr int ROW_WARP_COUNT = 4;
  constexpr int LOAD_WARP_ID = 4;
  constexpr int MMA_WARP_ID = 5;
  const bool is_load_warp = (warp_id == LOAD_WARP_ID);
  const bool is_mma_warp = (warp_id == MMA_WARP_ID);
  const bool is_row_warp = (warp_id >= ROW_WARP_BASE && warp_id < ROW_WARP_BASE + ROW_WARP_COUNT);
  // One CTA handles one 128-row tile of Q.
  const int q_tiles_per_batch = len_q / BLOCK_M;
  const int bid = blockIdx.x;
  const int batch_id = bid / q_tiles_per_batch;
  const int q_tile_id = bid - batch_id * q_tiles_per_batch;

  const int q_row0 = batch_id * len_q + q_tile_id * BLOCK_M;  // flattened row index

  // Shared memory.
  extern __shared__ __align__(1024) char smem_raw[];

  // 3 tiles (Q, K, V), each 128x128.

  // Integer shared addresses for TMA + UMMA descriptors.
  const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
  const int Q_smem = smem_base;
  const int K_smem = Q_smem + Q_TILE_BYTES;
  const int V_smem = K_smem + K_TILE_BYTES;

  // Stage barriers:
  //   q-ready:      load warp -> mma warp (one-time)
  //   kv-ready:     load warp -> mma warp
  //   qk-done:      mma warp  -> row warps
  //   softmax-done: row warps -> mma warp (full P ready + old O safe)
  //   pv-done:      mma warp  -> load/row warps (buffer reuse / overwrite safety)
  __shared__ uint64_t mbar_q_ready[1];
  __shared__ uint64_t mbar_kv_ready[1];
  __shared__ uint64_t mbar_qk_done[1];
  __shared__ uint64_t mbar_softmax_done[1];
  __shared__ uint64_t mbar_pv_done[1];
  __shared__ int tmem_addr[1];
  const int mbar_q_ready_addr = static_cast<int>(__cvta_generic_to_shared(mbar_q_ready));
  const int mbar_kv_ready_addr = static_cast<int>(__cvta_generic_to_shared(mbar_kv_ready));
  const int mbar_qk_done_addr = static_cast<int>(__cvta_generic_to_shared(mbar_qk_done));
  const int mbar_softmax_done_addr = static_cast<int>(__cvta_generic_to_shared(mbar_softmax_done));
  const int mbar_pv_done_addr = static_cast<int>(__cvta_generic_to_shared(mbar_pv_done));

  if (is_load_warp && elect_sync()) {
    mbarrier::init(mbar_q_ready_addr, 1);
    mbarrier::init(mbar_kv_ready_addr, 1);
    mbarrier::init(mbar_qk_done_addr, 1);
    mbarrier::init(mbar_softmax_done_addr, ROW_WARP_COUNT);
    mbarrier::init(mbar_pv_done_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (is_mma_warp) {
    // Allocate S + O. P overlays the lower half of S, and does not need extra TMEM columns.
    const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :
                 : "r"(tmem_smem_addr), "r"(TMEM_S_COLS + TMEM_O_COLS)
                 : "memory");
  }
  // setup block above should not be folded into the
  // warp specific conditionals below;
  // mbarriers must be initialized before any role waits/arrives,
  // __syncthreads() publishes that setup to all roles
  __syncthreads();

  const int taddr_base = tmem_addr[0];
  const int taddr_s = taddr_base;
  // Swizzle applies to SMEM operand descriptors (Q/K/V), not the TMEM O accumulator.
  const int taddr_o = taddr_s + TMEM_S_COLS;
  // P overlays the lower half of S in TMEM (bf16x2 packed into b32 columns).
  const int taddr_p = taddr_s;

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
  // Role-specific loops (WS-only refactor; still single-stage, no KV ring pipeline).
  // All three role loops must execute exactly kv_tiles iterations with no early exits.
  // Barrier phase toggles are coupled tile-by-tile across roles; changing one loop bound can deadlock
  if (is_load_warp && elect_sync()) {
    int phase_pv_done = 0;

    // Load Q once into SMEM (load warp only); MMA warp is the only direct Q consumer.
    swizzle128::tma_load_kmajor(Q_smem, &Q_tmap, BLOCK_M, q_row0, mbar_q_ready_addr);
    mbarrier::arrive_expect_tx(mbar_q_ready_addr, Q_TILE_BYTES);

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
      const int kv_row0 = batch_id * len_kv + kv_tile * BLOCK_N;

      // K: load directly into canonical K-major SW128 slices.
      // V: load as 8x64 swizzled MN-major slices.
      swizzle128::tma_load_kmajor(K_smem, &K_tmap, BLOCK_N, kv_row0, mbar_kv_ready_addr);
      swizzle128::tma_load_mnmajor(V_smem, &V_tmap, kv_row0, mbar_kv_ready_addr);
      mbarrier::arrive_expect_tx(mbar_kv_ready_addr, K_TILE_BYTES + V_TILE_BYTES);

      // Wait for PV completion before reusing shared K/V buffers on the next tile;
      // we have a single stage and load warps cannot proceed loading next
      // k/v untill current tile's full stage (including pv) is done
      mbarrier::wait(mbar_pv_done_addr, phase_pv_done);
      phase_pv_done ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    }
  }

  if (is_mma_warp && elect_sync()) {
    int phase_kv_ready = 0;
    int phase_softmax_done = 0;

    // q_ready is one-shot: Q is loaded once before all KV tiles, so no phase flip is needed.
    // Repeated barriers below still flip phase each time they are reused.
    mbarrier::wait(mbar_q_ready_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
      const bool first_kv = (kv_tile == 0);

      mbarrier::wait(mbar_kv_ready_addr, phase_kv_ready);
      phase_kv_ready ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      // QK^T -> S (TMEM), with K-major SW128 descriptor math (MMA warp).
      // outer loop selects SW128 64-wide block
      constexpr uint64_t qk_desc_micro_step = MMA_K_BYTES / 16;
      constexpr uint64_t qk_a_desc_panel_step = (BLOCK_M * SW128_BYTES) / 16;
      constexpr uint64_t qk_b_desc_panel_step = (BLOCK_N * SW128_BYTES) / 16;
      uint64_t a_desc_panel = swizzle128::desc_kmajor(Q_smem);
      uint64_t b_desc_panel = swizzle128::desc_kmajor(K_smem);
      for (int k1 = 0; k1 < HEAD_DIM / SW128_ATOM_COLS; ++k1) {
        uint64_t a_desc = a_desc_panel;
        uint64_t b_desc = b_desc_panel;
        // inner loop selects MMA_K=16 chunk inside that block
        for (int k2 = 0; k2 < SW128_ATOM_COLS / MMA_K; ++k2) {
          const int enable_input_d = (k1 == 0 && k2 == 0) ? 0 : 1;
          tcgen05::mma_f16(taddr_s, a_desc, b_desc, i_desc_qk, enable_input_d);
          a_desc += qk_desc_micro_step;
          b_desc += qk_desc_micro_step;
        }
        a_desc_panel += qk_a_desc_panel_step;
        b_desc_panel += qk_b_desc_panel_step;
      }

      // Track QK completion; the mbarrier arrives when S is ready in TMEM.
      tcgen05::commit_arrive(mbar_qk_done_addr);

      // PV -> O (TMEM): A=P from TMEM and B=V from swizzled SMEM (MN-major).
      mbarrier::wait(mbar_softmax_done_addr, phase_softmax_done);
      phase_softmax_done ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

      constexpr int pv_v_chunk_bytes = HEAD_DIM * MMA_K_BYTES;
      constexpr uint64_t pv_desc_chunk_step = pv_v_chunk_bytes / 16;
      uint64_t b_desc = swizzle128::desc_mnmajor(V_smem, HEAD_DIM);
      for (int k = 0; k < BLOCK_N / MMA_K; ++k) {
        // Advance by 16 V rows. In the SW128 MN-major layout this is
        // equivalent to two 8-token row-block strides in SMEM.
        // Inherited from kernel 2: each PV microstep advances 8 packed-P TMEM columns.
        const int taddr_a = taddr_p + k * 8;
        const int enable_input_d = (first_kv && k == 0) ? 0 : 1;
        tcgen05::mma_f16_a_tmem(taddr_o, taddr_a, b_desc, i_desc_pv, enable_input_d);
        b_desc += pv_desc_chunk_step;
      }

      // Track PV completion; the mbarrier arrives when O is ready in TMEM.
      tcgen05::commit_arrive(mbar_pv_done_addr);
    }
  }

  // Row warps own both softmax/P and O correction
  if (is_row_warp) {
    const int local_warp_idx = warp_id - ROW_WARP_BASE;
    const int row_base = local_warp_idx * WARP_SIZE;  // first of 32 rows owned by this warp
    const int row = row_base + lane_id;               // specific row owned by this lane
    const int trow = row_base << 16;                  // encode row_base for TMEM addressing

    int phase_qk_done = 0;
    int phase_pv_done = 0;

    // Online softmax state per row.
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
          const int taddr = taddr_s + trow + score_n8 * 8;
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

      // Skip O rescale on the first KV tile; O is initialized by the first PV MMA.
      if (!first_kv) {
        float o8[8];
#pragma unroll
        // This 8 is the tcgen05.ld/st x8 TMEM column width, not a SMEM atom/slice size.
        for (int out_n8 = 0; out_n8 < HEAD_DIM / 8; ++out_n8) {
          const int taddr = taddr_o + trow + out_n8 * 8;
          tcgen05::ld_32x32b_x8(taddr, o8);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            o8[i] *= rescale;
          }
          tcgen05::st_32x32b_x8(taddr, o8);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
      }

      // Pass 2: read S from TMEM, write packed P to TMEM (overlay), accumulate rowsum.
      // Lower-half overlay lets us scan low->high: each packed P write only overwrites
      // S columns that were already consumed by this or an earlier chunk.
      float tile_rowsum = 0.0f;
      {
        // TMEM address = row field from trow + column field from taddr_s/p.
        const int taddr_s_row = taddr_s + trow;
        const int taddr_p_row = taddr_p + trow;
        uint32_t p8_packed[8];
#pragma unroll
        for (int score_n16 = 0; score_n16 < BLOCK_N / 16; ++score_n16) {
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
      // One ordinary software arrival per row warp once this warp's P rows are ready.
      if (lane_id == 0) {
        mbarrier::arrive(mbar_softmax_done_addr);
      }

      // Wait for PV completion before reading/rescaling O for the next tile.
      mbarrier::wait(mbar_pv_done_addr, phase_pv_done);
      phase_pv_done ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    }

    // Epilogue: write O / rowsum to global bf16.
    // NOTE: this is outside of the KV loop, so executed one per output tile
    const float inv_denom = 1.0f / rowsum;
    float o8[8];
#pragma unroll
    // This 8 is the tcgen05.ld/st x8 TMEM column width, not a SMEM atom/slice size.
    for (int out_n8 = 0; out_n8 < HEAD_DIM / 8; ++out_n8) {
      const int taddr = taddr_o + trow + out_n8 * 8;
      tcgen05::ld_32x32b_x8(taddr, o8);

      nv_bfloat162 out_bf16x2[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 v = make_float2(o8[2 * i] * inv_denom, o8[2 * i + 1] * inv_denom);
        out_bf16x2[i] = __float22bfloat162_rn(v);
      }

      nv_bfloat16* out_ptr = O_ptr + (q_row0 + row) * HEAD_DIM + out_n8 * 8;
      reinterpret_cast<int4*>(out_ptr)[0] = reinterpret_cast<int4*>(out_bf16x2)[0];
    }
  }

  __syncthreads();

  // Deallocate TMEM.
  // Per tcgen05 requirements, dealloc must be executed by a single warp.
  // Use the same warp that performed allocation (the MMA warp in this kernel).
  // Softmax warps do much of the TMEM math work, but alloc/dealloc is administrative.
  // It is not tied to whichever role touches TMEM most.
  if (is_mma_warp) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(taddr_base), "r"(TMEM_S_COLS + TMEM_O_COLS)
                 : "memory");
  }
}

void attention_tcgen05_v4_launch(const nv_bfloat16* Q_ptr, const nv_bfloat16* K_ptr,
                                 const nv_bfloat16* V_ptr, nv_bfloat16* O_ptr, int bs,
                                 int len_q, int len_kv, cudaStream_t stream) {
  // Baseline restrictions.
  TORCH_CHECK(bs > 0, "bs must be positive");
  TORCH_CHECK(len_q >= BLOCK_M, "len_q must be at least 128");
  TORCH_CHECK(len_q % BLOCK_M == 0, "len_q must be a multiple of 128");
  TORCH_CHECK(len_kv >= BLOCK_N, "len_kv must be at least 128");
  TORCH_CHECK(len_kv % BLOCK_N == 0, "len_kv must be a multiple of 128");

  // Flatten batch into the height dimension for TMA.
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
                      /*shared_height=*/SW128_ATOM_ROWS,
                      /*shared_width=*/SW128_ATOM_COLS,  // 64 bf16 = 128B
                      CU_TENSOR_MAP_SWIZZLE_128B);

  const int grid = bs * (len_q / BLOCK_M);

  const size_t smem_bytes = size_t(Q_TILE_BYTES + K_TILE_BYTES + V_TILE_BYTES);

  if (smem_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(attention_tcgen05_v4_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smem_bytes)));
  }

  attention_tcgen05_v4_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
      Q_tmap, K_tmap, V_tmap, O_ptr, len_q, len_kv);
  check_cuda(cudaGetLastError());
}
