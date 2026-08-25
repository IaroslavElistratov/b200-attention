// Kernel 2 focus:
//   1. Keep QK writing S into TMEM.
//   2. Write softmax P into TMEM (bf16x2 packed b32), overlaid on S.
//   3. Run PV with A sourced from TMEM.
//   4. Keep the no-transpose V path from kernel 1 (PV B is MN-major from V_smem).



#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// Assumptions:
//   - head_dim == 128
//   - BLOCK_M == 128 queries per CTA
//   - BLOCK_N == 128 keys per KV tile
//   - len_q % 128 == 0
//   - len_kv % 128 == 0
//   - dense, non-causal, no masking
//   - inputs/outputs are bf16
//
// Dataflow per KV tile:
//   1) TMA load K into K-major no-swizzle width-8 slices.
//      Inherited from kernel 1: TMA load V directly into MN-major no-swizzle 8x8 bf16 tiles.
//   2) QK: Q @ K^T -> S in TMEM.
//   3) Softmax reads S from TMEM, updates rowmax/rowsum,
//      rescales old O in TMEM when needed, and writes packed bf16x2 P into TMEM.
//   4) PV: P_tmem @ V_smem -> O in TMEM,
//      with P as TMEM A and V as MN-major SMEM B.
//   5) Epilogue writes O / rowsum to global bf16.



// Local-to-global mapping:
//
// QK:
//   Q[block_m=Q_row,  block_k=HeadDim]
//   K[block_n=KV_row, block_k=HeadDim]    // consumed as K^T
//   -> S/P[block_m=Q_row, block_n=KV_row]
//
// PV:
//   P[block_m=Q_row,  block_k=KV_row]      // same as P above, different local axis names
//   V[block_k=KV_row, block_n=HeadDim]
//   -> O[block_m=Q_row, block_n=HeadDim]


namespace {

constexpr int NUM_WARPS = 4;
// need 4 warps to access entire TMEM
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int HEAD_DIM = 128;
constexpr int MMA_K = 16;
constexpr int BF16_BYTES = int(sizeof(nv_bfloat16));
constexpr int MMA_K_BYTES = MMA_K * BF16_BYTES;         // 32
constexpr int ATOM_ROWS = 8;
constexpr int ATOM_COLS = 8;
constexpr int TMEM_S_COLS = BLOCK_N;
constexpr int TMEM_O_COLS = HEAD_DIM;
constexpr int Q_TILE_BYTES  = BLOCK_M * HEAD_DIM * BF16_BYTES;
constexpr int K_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;
constexpr int V_TILE_BYTES = BLOCK_N * HEAD_DIM * BF16_BYTES;

#include "smem_layout_noswizzle.cuh"

// Shared is laid out as 3 tiles, each 128x128 bf16 = 32 KiB: Q, K, and V.
// P lives in TMEM in this version, so there is no extra P tile in SMEM.
// For SWIZZLE_NONE, use canonical K-major slices of width 8:
//   tile[k_block][row][col_in_0_7]
// This is the canonical no-swizzle descriptor layout used by this kernel.


}  // namespace

__global__ __launch_bounds__(TB_SIZE) void attention_tcgen05_v2_kernel(
    const __grid_constant__ CUtensorMap Q_tmap, const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap, nv_bfloat16* __restrict__ O_ptr, int len_q,
    int len_kv) {

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int row_base = warp_id * WARP_SIZE;  // first of 32 rows owned by this warp
  const int row = row_base + lane_id;        // specific row owned by this lane
  const int trow = row_base << 16;           // encode row_base for TMEM addressing

  // how many Q tiles exist inside one batch item
  const int q_tiles_per_batch = len_q / BLOCK_M;
  const int batch_id = bid / q_tiles_per_batch;
  const int q_tile_id = bid - batch_id * q_tiles_per_batch;

  // offset to the beggining of the current q tile, ie the one that the cta is responsible for
  // Used to: load this CTA's Q tile, write this CTA's O tile
  const int q_row0 = batch_id * len_q + q_tile_id * BLOCK_M;  // flattened row index

  // SMEM
  // as mentioned in the stub below, we're using dynamically allocated SMEM;
  // size is not specified here, but it is specified at launch time: smem_bytes
  extern __shared__ __align__(1024) char smem_raw[];

  // 3 tiles (Q, K, V), each 128x128.

  // Integer shared addresses for TMA + UMMA descriptors.
  const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
  const int Q_smem = smem_base;
  const int K_smem = Q_smem + Q_TILE_BYTES;
  const int V_smem = K_smem + K_TILE_BYTES;

  // mbarrier and TMEM allocation.
  __shared__ uint64_t mbar[1];
  __shared__ int tmem_addr[1];
  const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbar));
  // Parity phase for the single mbarrier.
  int phase = 0;

  if (warp_id == 0 && elect_sync()) {
    mbarrier::init(mbar_addr, 1);
    // Make the barrier init, done in the ordinary CUDA thread execution path
    // (generic proxy), visible to TMA instructions (async proxy)
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp_id == 1) {
    // Allocate S + O. P overlays the lower half of S, and does not need extra TMEM columns.
    const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :
                 : "r"(tmem_smem_addr), "r"(TMEM_S_COLS + TMEM_O_COLS)
                 : "memory");
  }
  __syncthreads();

  const int taddr_base = tmem_addr[0];
  const int taddr_s = taddr_base;
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

  // Load Q once into SMEM in 16 width-8 slices.
  if (warp_id == 0 && elect_sync()) {
    noswizzle::tma_load_kmajor(Q_smem, &Q_tmap, BLOCK_M, q_row0, mbar_addr);
    mbarrier::arrive_expect_tx(mbar_addr, Q_TILE_BYTES);
  }
  mbarrier::wait(mbar_addr, phase);
  phase ^= 1;
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

  // Online softmax state per row.
  float rowmax = -FLT_MAX;
  float rowsum = 0.0f;
  // Keep as a runtime constant (device rsqrtf) to avoid constexpr sqrt issues.
  const float softmax_scale = rsqrtf(float(HEAD_DIM));

  const int kv_tiles = len_kv / BLOCK_N;

  for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
    const int kv_row0 = batch_id * len_kv + kv_tile * BLOCK_N;
    const bool first_kv = (kv_tile == 0);

    // K: load directly into canonical K-major width-8 slices.
    // Inherited from kernel 1: V loads directly as 8x8 tiles into canonical no-swizzle MN-major atom order.
    if (warp_id == 0 && elect_sync()) {
      noswizzle::tma_load_kmajor(K_smem, &K_tmap, BLOCK_N, kv_row0, mbar_addr);
      noswizzle::tma_load_mnmajor(V_smem, &V_tmap, kv_row0, mbar_addr);
      mbarrier::arrive_expect_tx(mbar_addr, K_TILE_BYTES + V_TILE_BYTES);
    }
    mbarrier::wait(mbar_addr, phase);
    phase ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // QK^T -> S (TMEM).
    if (warp_id == 0 && elect_sync()) {

      // see "Local-to-global mapping", for QK local "block_k" is HEAD_DIM;
      // HEAD_DIM=128, MMA_K=16 -> 8 micro-steps.
      constexpr uint64_t qk_a_desc_step = (BLOCK_M * MMA_K_BYTES) / 16;
      constexpr uint64_t qk_b_desc_step = (BLOCK_N * MMA_K_BYTES) / 16;
      uint64_t a_desc = noswizzle::desc_kmajor(Q_smem, BLOCK_M);
      uint64_t b_desc = noswizzle::desc_kmajor(K_smem, BLOCK_N);
      for (int k = 0; k < HEAD_DIM / MMA_K; ++k) {
        tcgen05::mma_f16(taddr_s, a_desc, b_desc, i_desc_qk, /*enable_input_d=*/(k == 0 ? 0 : 1));
        a_desc += qk_a_desc_step;
        b_desc += qk_b_desc_step;
      }

      // Track QK completion; the mbarrier arrives when S is ready in TMEM.
      tcgen05::commit_arrive(mbar_addr);
    }
    // Make the S tile visible for tcgen05.ld.
    mbarrier::wait(mbar_addr, phase);
    phase ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // Pass 1: compute max for this KV tile's scores.
    float tile_rowmax = -FLT_MAX;
    {
#pragma unroll
      // S (block_m=Q_row, block_n=KV_row), where block_m mapped to the TMEM rows, and block_n to TMEM columns,
      // here we're iterating over columns (ie over block_n);

      // the 8 here is the width of our TMEM loads (ld.x8) instruction
      for (int score_n8 = 0; score_n8 < BLOCK_N / 8; ++score_n8) {
        float s8[8];
        const int taddr = taddr_s + trow + score_n8 * 8;
        // load 8 score values into this thread's registers
        tcgen05::ld_32x32b_x8(taddr, s8);
#pragma unroll
        // each thread computes the raw max; SDPA scaling is applied once after the scan.
        for (int i = 0; i < 8; ++i) {
          tile_rowmax = fmaxf(tile_rowmax, s8[i]);
        }
      }
    }

    // Online update: new max, rescale factor.
    // standard SDPA (keep logits normalized by the head dim)
    tile_rowmax *= softmax_scale;
    const float new_rowmax = fmaxf(rowmax, tile_rowmax);
    const float rescale = __expf(rowmax - new_rowmax);
    // correction: rescalse rowsum to the new basis
    // (rowsum is the running unnormalized softamx denominator, in the current max basis)
    rowmax = new_rowmax;
    rowsum *= rescale;

    // Skip O rescale on the first KV tile: O is initialized by the peeled
    // first PV MMA microstep (enable_input_d=0). This avoids tcgen05.ld on O
    // before any PV write. After tile 0, O is valid and can be rescaled.
    if (!first_kv) {
      float o8[8];
#pragma unroll
      // O (block_m=Q_row, block_n=HeadDim), where block_m mapped to the TMEM rows, and block_n to TMEM columns,
      // here we're iterating over columns;

      // this 8 is the tcgen05.ld/st x8 TMEM column width, not a SMEM atom/slice size
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

    // Ensure all threads finished rescaling before PV MMA reads O as input-D.
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // Pass 2: read S from TMEM, write packed P to TMEM (overlay), accumulate rowsum.
    // Lower-half overlay: each packed P write only overwrites
    // S columns that were already consumed by this or an earlier chunk
    float tile_rowsum = 0.0f;
    {
      // TMEM address as two fields:
      //   - high bits: row start
      //   - low bits: column
      // this "+" basicaly adds the high 16 bits (from trow) and low 16 bits from taddr_s/p
      const int taddr_s_row = taddr_s + trow;
      const int taddr_p_row = taddr_p + trow;
#pragma unroll
      for (int score_n16 = 0; score_n16 < BLOCK_N / 16; ++score_n16) {
        uint32_t p8_packed[8];
        // Build one 16-column logical P chunk. These are packed in 8 columns:
        // the TMEM store writes 8 b32 cells, and each b32 cell holds two packed
        // bf16 probabilities.
#pragma unroll
        // load8 loop = each 16-score chunk is two 8-score loads (ld.x8):
        //
        // this loop over half isn't strictly needed -- can instead use tcgen05_ld_x16 and load 16 elements in one go;
        // here i want to reuse the tcgen05_ld_x8 helper we already have -- thus need two iterations (each 8 elements)
        // to cover the 16 elements;
        // And we need 16 input elements because we want to store 8 u32 slots (containing 16
        // packed bf16 P values -- ie packing two P values in each TMEM memory slot)
        for (int load8 = 0; load8 < 2; ++load8) {
          float s8[8];
          // S read columns are FP32 columns
          const int col_s = score_n16 * 16 + load8 * 8;
          tcgen05::ld_32x32b_x8(taddr_s_row + col_s, s8);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float score0 = s8[2 * j] * softmax_scale;
            const float score1 = s8[2 * j + 1] * softmax_scale;
            // exponenciate S to get unnormalized probs
            const float p0 = __expf(score0 - rowmax);
            const float p1 = __expf(score1 - rowmax);
            // previsoly, we rescaled old accumuated rowsum into the new base;
            // now pass 2 computes the current tile’s rowsum in that same basis, then adds it
            tile_rowsum += p0 + p1;
            // pack these two into a single slot (to be later written in a single TMEM cell)
            p8_packed[load8 * 4 + j] = bf16::pack2_to_u32(p0, p1);
          }
        }
        // P write columns are packed b32 columns (P is bf16).
        // This is part of the TMEM representation: 16 bf16 P values occupy 8 b32 cells.
        tcgen05::st_32x32b_x8_u32(taddr_p_row + score_n16 * 8, p8_packed);
      }
      asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
    }
    rowsum += tile_rowsum;

    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // PV -> O (TMEM): A=P from TMEM and B=V from SMEM (MN-major).
    if (warp_id == 0 && elect_sync()) {
      // QK produces S/P as (block_M,block_N). PV then consumes P as its first
      // operand (block_M,block_K). So block_N in the 1st mma's frame is
      // block_K in the 2nd mma's frame. Ie that same physical axis becomes
      // the local GEMM-K reduction axis for PV
      constexpr uint64_t pv_b_desc_step = (HEAD_DIM * MMA_K_BYTES) / 16;
      uint64_t b_desc = noswizzle::desc_mnmajor(V_smem, HEAD_DIM);
      for (int k = 0; k < BLOCK_N / MMA_K; ++k) {
        // MMA_K=16 P values are packed as bf16x2, so one PV microstep
        // advances by 8 b32 TMEM columns;
        // we're under elect_sync if block so only one thread executes this code,
        // and the code selects chunks (block_m,16) of P for mma to consume --
        // this doesn't contradict erlier point point (that 4 warps needed to acces
        // all TMEM rows) because here warp doens't acces these values, it's mma acess
        // them internally
        const int taddr_a = taddr_p + k * 8;  // 8 is two bf16 per b32 TMEM cell.
        // V[block_K=KV_row, block_N=HeadDim]

        const int enable_input_d = (first_kv && k == 0) ? 0 : 1;
        tcgen05::mma_f16_a_tmem(taddr_o, taddr_a, b_desc, i_desc_pv,
                               enable_input_d);
        b_desc += pv_b_desc_step;
      }

      // Track PV completion; the mbarrier arrives when O is ready in TMEM.
      tcgen05::commit_arrive(mbar_addr);
    }
    // Prepare TMEM for tcgen05.ld in the next tile (or epilogue).
    mbarrier::wait(mbar_addr, phase);
    phase ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
  }

  // Epilogue: write O / rowsum to global bf16.
  const float inv_denom = 1.0f / rowsum;
  {
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
  // Prefer using the same warp that allocated (warp 1).
  if (warp_id == 1) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(taddr_base), "r"(TMEM_S_COLS + TMEM_O_COLS)
                 : "memory");
  }
}

void attention_tcgen05_v2_launch(const nv_bfloat16* Q_ptr, const nv_bfloat16* K_ptr,
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

  init_tmap_2d_simple(&Q_tmap, const_cast<nv_bfloat16*>(Q_ptr), /*global_height=*/Q_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/BLOCK_M, /*shared_width=*/ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_NONE);

  init_tmap_2d_simple(&K_tmap, const_cast<nv_bfloat16*>(K_ptr), /*global_height=*/KV_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/BLOCK_N, /*shared_width=*/ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_NONE);

  init_tmap_2d_simple(&V_tmap, const_cast<nv_bfloat16*>(V_ptr), /*global_height=*/KV_height,
                      /*global_width=*/HEAD_DIM,
                      /*shared_height=*/ATOM_ROWS, /*shared_width=*/ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_NONE);

  const int grid = bs * (len_q / BLOCK_M);

  // in our kernel all tile sizes are known at compile time, but we still use dynaically allocated SMEM
  // (not statically allocates SMEM), becuase cuad docs say that "Kernels relying on shared memory allocations
  // over 48 KB per block [...] must use dynamic SMEM rather than statically-sized arrays and require an explicit
  // opt-in using cudaFuncSetAttribute as follows
  const size_t smem_bytes = size_t(Q_TILE_BYTES + K_TILE_BYTES + V_TILE_BYTES);

  if (smem_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(attention_tcgen05_v2_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smem_bytes)));
  }

  attention_tcgen05_v2_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
      Q_tmap, K_tmap, V_tmap, O_ptr, len_q, len_kv);
  check_cuda(cudaGetLastError());
}
