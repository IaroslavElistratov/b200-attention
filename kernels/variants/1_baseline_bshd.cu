// Direct contiguous-BSHD I/O variant of the baseline single-Q-tile kernel.
//
// Global BSHD is viewed by TMA as a 2D [B * L, H * D] matrix. Each CTA selects
// one head by adding head_id * HEAD_DIM to the TMA x-coordinate. The resulting
// SMEM tiles and the entire attention computation remain unchanged.
//
//   1. One CTA computes one 128x128 Q tile across the KV loop.
//   2. Q and K use K-major no-swizzle SMEM layouts.
//   3. Repacked V directly into MN-major canonical order using TMA 8x8 bf16 tiles (k8, n8).
//      Table 38: 1x8 is the swizzle atom for M/N-major with no swizzle:
//      https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-swizzle-lead-dim
//   4. QK writes FP32 scores to TMEM.
//   5. Softmax writes bf16 P into SMEM by reusing K_smem after QK.
//   6. PV consumes P_smem and MN-major V_smem, accumulating O in TMEM.



#include "common.cuh"

// Helper namespaces below are just stateless groups of related helper functions
// (mbarrier, TMA, tcgen05, SMEM layout). The kernel schedule stays inline
// inside each kernel file; it is not abstracted into the helpers.

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
//   - inputs/outputs are contiguous [batch, sequence, heads, head_dim] (BSHD)
//
// Dataflow per KV tile:
//   1) TMA load K into K-major no-swizzle width-8 slices.
//      TMA load V directly into MN-major no-swizzle 8x8 bf16 tiles.
//   2) QK: Q @ K^T -> S in TMEM.
//   3) Softmax reads S from TMEM, updates rowmax/rowsum,
//      rescales O in TMEM, and writes bf16 P into K_smem.
//   4) PV: P_smem @ V_smem -> O in TMEM,
//      with P as K-major A and V as MN-major B.
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

// Element index in canonical K-major width-8 slices: [slice][row][in8].
__device__ __forceinline__ int idx_kmajor_w8(int height, int row, int col) {
  const int slice = col / ATOM_COLS;
  const int in8 = col % ATOM_COLS;
  return slice * (height * ATOM_COLS) + row * ATOM_COLS + in8;
}

// Load one head's [tile_height, HEAD_DIM] slice from a contiguous BSHD tensor.
// The tensor map views global memory as [B * L, H * D], so global_col0 selects
// the requested head while the local SMEM layout stays identical to Kernel 1.
__device__ __forceinline__ void tma_load_kmajor_bshd(
    int dst_smem_base, const CUtensorMap* tmap, int tile_height, int global_col0,
    int global_row0, int mbar_addr) {
  for (int k = 0; k < HEAD_DIM / ATOM_COLS; ++k) {
    const int off_k = k * ATOM_COLS;
    const int slice_bytes = tile_height * ATOM_COLS * BF16_BYTES;
    const int dst = dst_smem_base + k * slice_bytes;
    tma::copy_2d_gmem_to_smem(dst, tmap, global_col0 + off_k, global_row0, mbar_addr);
  }
}

// Load one head's V tile into the same canonical MN-major SMEM layout as
// Kernel 1, using the head offset only at the global-memory boundary.
__device__ __forceinline__ void tma_load_mnmajor_bshd(
    int dst_smem_base, const CUtensorMap* tmap, int global_col0, int global_row0,
    int mbar_addr) {
  const int n8_stride = ATOM_ROWS * ATOM_COLS * BF16_BYTES;
  const int k8_stride = (HEAD_DIM / ATOM_COLS) * n8_stride;

  for (int k8 = 0; k8 < BLOCK_N / ATOM_ROWS; ++k8) {
    const int off_k = k8 * ATOM_ROWS;
    for (int n8 = 0; n8 < HEAD_DIM / ATOM_COLS; ++n8) {
      const int dst =
          dst_smem_base + k8 * k8_stride + n8 * n8_stride;
      const int off_n = n8 * ATOM_COLS;
      tma::copy_2d_gmem_to_smem(dst, tmap, global_col0 + off_n,
                                global_row0 + off_k, mbar_addr);
    }
  }
}


}  // namespace

__global__ __launch_bounds__(TB_SIZE) void attention_tcgen05_v1_bshd_kernel(
    const __grid_constant__ CUtensorMap Q_tmap, const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap, nv_bfloat16* __restrict__ O_ptr, int len_q,
    int len_kv, int heads) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int row_base = warp_id * WARP_SIZE;  // first of 32 rows owned by this warp
  const int row = row_base + lane_id;        // specific row owned by this lane
  const int trow = row_base << 16;           // encode row_base for TMEM addressing
  // One CTA handles one 128-row tile of Q.
  const int q_tiles_per_head = len_q / BLOCK_M;
  const int batch_head_id = bid / q_tiles_per_head;
  const int q_tile_id = bid - batch_head_id * q_tiles_per_head;
  const int batch_id = batch_head_id / heads;
  const int head_id = batch_head_id - batch_id * heads;

  // Global BSHD is viewed as [B * L, H * D].
  const int q_row0 = batch_id * len_q + q_tile_id * BLOCK_M;
  const int head_col0 = head_id * HEAD_DIM;

  // Shared memory.
  extern __shared__ __align__(1024) char smem_raw[];
  nv_bfloat16* smem_bf16 = reinterpret_cast<nv_bfloat16*>(smem_raw);

  // 3 tiles (Q, K/P scratch, V), each 128x128.
  nv_bfloat16* Q_smem_ptr = smem_bf16;
  nv_bfloat16* K_smem_ptr = Q_smem_ptr + BLOCK_M * HEAD_DIM;
  nv_bfloat16* P_smem_ptr = K_smem_ptr;  // after QK, K is dead and this storage holds P

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
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp_id == 1) {
    // Allocate 256 columns: first 128 for S, next 128 for O.
    // P lives in SMEM by reusing the K scratch tile.
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
    tma_load_kmajor_bshd(Q_smem, &Q_tmap, BLOCK_M, head_col0, q_row0, mbar_addr);
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
    // V: load as 8x8 tiles into canonical no-swizzle MN-major atom order (16B x 8).
    // For PV, V[KV,D] is the logical B[K,N] operand. A K-major B path would first
    // transpose/reformat it into Vt[D,KV]. Instead, use MN-major B directly, which
    // matches V's natural orientation and avoids that explicit SIMT transpose
    if (warp_id == 0 && elect_sync()) {
      tma_load_kmajor_bshd(K_smem, &K_tmap, BLOCK_N, head_col0, kv_row0, mbar_addr);
      tma_load_mnmajor_bshd(V_smem, &V_tmap, head_col0, kv_row0, mbar_addr);
      mbarrier::arrive_expect_tx(mbar_addr, K_TILE_BYTES + V_TILE_BYTES);
    }
    mbarrier::wait(mbar_addr, phase);
    phase ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // QK^T -> S (TMEM).
    if (warp_id == 0 && elect_sync()) {
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

    // Skip O rescale on the first KV tile: O is initialized by the peeled
    // first PV MMA microstep (enable_input_d=0). This avoids tcgen05.ld on O
    // before any PV write. After tile 0, O is valid and can be rescaled.
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

    // Ensure all threads finished rescaling before PV MMA reads O as input-D.
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // Pass 2: compute exp(score - rowmax), write P to SMEM (reuse K buffer), accumulate rowsum.
    float tile_rowsum = 0.0f;
    {
      float s8[8];
#pragma unroll
      for (int score_n8 = 0; score_n8 < BLOCK_N / 8; ++score_n8) {
        const int taddr = taddr_s + trow + score_n8 * 8;
        tcgen05::ld_32x32b_x8(taddr, s8);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int n = score_n8 * 8 + i;
          const float s = s8[i] * softmax_scale;
          const float p = __expf(s - rowmax);
          tile_rowsum += p;

          const int p_idx = idx_kmajor_w8(BLOCK_M, /*row=*/row, /*col=*/n);
          P_smem_ptr[p_idx] = __float2bfloat16_rn(p);
        }
      }
    }
    rowsum += tile_rowsum;

    // Threads write P to SMEM with normal stores, while tcgen05.mma reads P
    // through the async proxy. Each writer runs this fence so its P stores come
    // before that async read. The following __syncthreads() separately waits for
    // every writer before the elected MMA lane starts PV.
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // PV -> O (TMEM): A=P in K_smem and B=V in MN-major.
    if (warp_id == 0 && elect_sync()) {
      constexpr uint64_t pv_a_desc_step = (BLOCK_M * MMA_K_BYTES) / 16;
      constexpr uint64_t pv_b_desc_step = (HEAD_DIM * MMA_K_BYTES) / 16;
      uint64_t a_desc = noswizzle::desc_kmajor(K_smem, BLOCK_M);
      uint64_t b_desc = noswizzle::desc_mnmajor(V_smem, HEAD_DIM);
      for (int k = 0; k < BLOCK_N / MMA_K; ++k) {
        const int enable_input_d = (first_kv && k == 0) ? 0 : 1;
        tcgen05::mma_f16(taddr_o, a_desc, b_desc, i_desc_pv, enable_input_d);
        a_desc += pv_a_desc_step;
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

      nv_bfloat16* out_ptr =
          O_ptr + (q_row0 + row) * (heads * HEAD_DIM) + head_col0 + out_n8 * 8;
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

void attention_tcgen05_v1_bshd_launch(
    const nv_bfloat16* Q_ptr, const nv_bfloat16* K_ptr,
    const nv_bfloat16* V_ptr, nv_bfloat16* O_ptr, int batch, int heads,
    int len_q, int len_kv, cudaStream_t stream) {
  // Baseline restrictions.
  TORCH_CHECK(batch > 0, "batch must be positive");
  TORCH_CHECK(heads > 0, "heads must be positive");
  TORCH_CHECK(len_q >= BLOCK_M, "len_q must be at least 128");
  TORCH_CHECK(len_q % BLOCK_M == 0, "len_q must be a multiple of 128");
  TORCH_CHECK(len_kv >= BLOCK_N, "len_kv must be at least 128");
  TORCH_CHECK(len_kv % BLOCK_N == 0, "len_kv must be a multiple of 128");

  // A contiguous [B, L, H, D] tensor is also a contiguous
  // [B * L, H * D] matrix. TMA selects one D-wide head slice from each row.
  const int Q_height = batch * len_q;
  const int KV_height = batch * len_kv;
  const int global_width = heads * HEAD_DIM;

  CUtensorMap Q_tmap{};
  CUtensorMap K_tmap{};
  CUtensorMap V_tmap{};

  init_tmap_2d_simple(&Q_tmap, const_cast<nv_bfloat16*>(Q_ptr), /*global_height=*/Q_height,
                      /*global_width=*/global_width,
                      /*shared_height=*/BLOCK_M, /*shared_width=*/ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_NONE);

  init_tmap_2d_simple(&K_tmap, const_cast<nv_bfloat16*>(K_ptr), /*global_height=*/KV_height,
                      /*global_width=*/global_width,
                      /*shared_height=*/BLOCK_N, /*shared_width=*/ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_NONE);

  init_tmap_2d_simple(&V_tmap, const_cast<nv_bfloat16*>(V_ptr), /*global_height=*/KV_height,
                      /*global_width=*/global_width,
                      /*shared_height=*/ATOM_ROWS, /*shared_width=*/ATOM_COLS,
                      CU_TENSOR_MAP_SWIZZLE_NONE);

  const int grid = batch * heads * (len_q / BLOCK_M);

  const size_t smem_bytes = size_t(Q_TILE_BYTES + K_TILE_BYTES + V_TILE_BYTES);

  if (smem_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(attention_tcgen05_v1_bshd_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smem_bytes)));
  }

  attention_tcgen05_v1_bshd_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
      Q_tmap, K_tmap, V_tmap, O_ptr, len_q, len_kv, heads);
  check_cuda(cudaGetLastError());
}
