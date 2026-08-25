namespace noswizzle {

__device__ __forceinline__ uint64_t desc_kmajor(int smem_addr_bytes, int height) {
  // Canonical no-swizzle K-major layout:
  //   SBO = 8 * 16B
  //   LBO = height * 16B
  const int LBO = height * ATOM_COLS * BF16_BYTES;
  const int SBO = ATOM_ROWS * ATOM_COLS * BF16_BYTES;
  return desc_encode(smem_addr_bytes)
         | (uint64_t(desc_encode(LBO)) << 16ULL)
         | (uint64_t(desc_encode(SBO)) << 32ULL)
         | (1ULL << 46ULL);
}

__device__ __forceinline__ uint64_t desc_mnmajor(int smem_addr_bytes, int width) {
  // Canonical no-swizzle MN-major layout:
  //   LBO = 8 rows * width bf16/row
  //   SBO = one 8x8 bf16 atom
  const int LBO = ATOM_ROWS * width * BF16_BYTES;
  const int SBO = ATOM_ROWS * ATOM_COLS * BF16_BYTES;
  return desc_encode(smem_addr_bytes)
         | (uint64_t(desc_encode(LBO)) << 16ULL)
         | (uint64_t(desc_encode(SBO)) << 32ULL)
         | (1ULL << 46ULL);
}

// TMA load helper for canonical no-swizzle K-major width-8 slices.
// Used for Q/K operands of QK, where local GEMM-K is HEAD_DIM.
__device__ __forceinline__ void tma_load_kmajor(
    int dst_smem_base, const CUtensorMap* tmap, int tile_height, int global_row0, int mbar_addr) {

  // better than hardcoding a global BLOCK_K, because what's block_k (int the traditional)
  // matmul notation changes depending on which mamtul we're in. Ie see (local-to-global mapping),
  // for QK k=HeadDim, for PV k=KV_row. Here's using HEAD_DIM because im only using this helper for QK matmul now
  //
  // loop vertically slices input tile into (block_m, 16B) slices;
  // k (slice index) is how many slices we already stepped over (to the right)
  for (int k = 0; k < HEAD_DIM / ATOM_COLS; ++k) {
    const int off_k = k * ATOM_COLS;
    // num bytes in a single slice
    const int slice_bytes = tile_height * ATOM_COLS * BF16_BYTES;
    const int dst = dst_smem_base + k * slice_bytes;
    ::tma::copy_2d_gmem_to_smem(dst, tmap, off_k, global_row0, mbar_addr);
  }
}

// Load V[K,N] directly into canonical no-swizzle MN-major order for PV.
__device__ __forceinline__ void tma_load_mnmajor(
    int dst_smem_base, const CUtensorMap* tmap, int global_row0, int mbar_addr) {

  // this whole loop -- first takes the first horisontal slice, and iterates to the right (in 8x8 steps inside of it)
  // Then goes to the second horisontal  slice, then inteartes inside of it to the right (in 8x8 steps inside of it), and so on

  // answer-now:
  // For this helper, one TMA copy is one 8x8 bf16 tile:
  //    8 rows × 8 cols × 2 bytes = 128 bytes
  const int n8_stride = ATOM_ROWS * ATOM_COLS * BF16_BYTES;
  const int k8_stride = (HEAD_DIM / ATOM_COLS) * n8_stride;

  for (int k8 = 0; k8 < BLOCK_N / ATOM_ROWS; ++k8) {
    const int off_k = k8 * ATOM_ROWS;
    for (int n8 = 0; n8 < HEAD_DIM / ATOM_COLS; ++n8) {
      // move down to the next 8-row stripe
      // because one full 8-row stripe has shape (8 rows, HEAD_DIM columns)
      // so its byte size is: 8 * HEAD_DIM * 2 (bytes per element) == HEAD_DIM * 16 bytes,
      const int k_slice_offset = k8 * k8_stride;
      // move right by n8 8×8 atoms inside the current 8-row stripe
      // ie one 8x8 atom
      const int n_tile_offset = n8 * n8_stride;
      const int dst = dst_smem_base + k_slice_offset + n_tile_offset;
      const int off_n = n8 * ATOM_COLS;
      ::tma::copy_2d_gmem_to_smem(dst, tmap, off_n, global_row0 + off_k, mbar_addr);
    }
  }
}

}  // namespace noswizzle
