
namespace swizzle128 {

// Inherited from kernel 3: Q/K use K-major SW128 SMEM descriptors.
// K-major, SWIZZLE_128B descriptor used for Q/K.
// For SW128 K-major canonical layout:
//   - LBO is not used
//   - SBO is offset from first 8 rows to next 8 rows (8 * 128B = 1024B)
__device__ __forceinline__ uint64_t desc_kmajor(int smem_addr_bytes) {
  const int SBO = SW128_ATOM_ROWS * SW128_BYTES;  // next 8-row atom group = 1024B
  return desc_encode(smem_addr_bytes)
         | (uint64_t(desc_encode(SBO)) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);  // SWIZZLE_128B
}

// Inherited from kernel 3: V uses MN-major SW128 SMEM descriptors.
__device__ __forceinline__ uint64_t desc_mnmajor(int smem_addr_bytes, int width_bf16) {
  // V tile is packed as 8-token x 64-feature atoms in SMEM.
  //   LBO: next 64-feature atom (right) -> 1024B
  //   SBO: next 8-token block (down)     -> HEAD_DIM * 16B
  const int atom_bytes = SW128_ATOM_ROWS * SW128_BYTES;  // 8 tokens x 128B = 1024B
  const int LBO = atom_bytes;
  const int SBO = (width_bf16 / SW128_ATOM_COLS) * atom_bytes;
  return desc_encode(smem_addr_bytes)
         | (uint64_t(desc_encode(LBO)) << 16ULL)
         | (uint64_t(desc_encode(SBO)) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);  // SWIZZLE_128B
}

// Inherited from kernel 3: Q/K TMA copies use K-major width-64 SW128 slices.
// TMA load helper for K-major SWIZZLE_128B width-64 slices.
// Used for Q/K tiles where global width is HEAD_DIM and SMEM is arranged as 128B swizzle atoms.
__device__ __forceinline__ void tma_load_kmajor(
    int dst_smem_base, const CUtensorMap* tmap, int tile_height, int global_row0, int mbar_addr) {
  for (int k = 0; k < HEAD_DIM / SW128_ATOM_COLS; ++k) {
    const int off_k = k * SW128_ATOM_COLS;
    // num bytes in a single width-64 slice
    const int slice_bytes = tile_height * SW128_BYTES;
    const int dst = dst_smem_base + k * slice_bytes;
    ::tma::copy_2d_gmem_to_smem(dst, tmap, off_k, global_row0, mbar_addr);
  }
}

// Inherited from kernel 3: V TMA copies use MN-major 8-token x 64-feature SW128 atoms.
// Load V[K,N] directly into SWIZZLE_128B MN-major 8x64 slices for PV.
__device__ __forceinline__ void tma_load_mnmajor(
    int dst_smem_base, const CUtensorMap* tmap, int global_row0, int mbar_addr) {
  // V is stored as 8-token x 64-feature SW128 atoms.
  // bytes in a single (8, 64) atom
  const int n64_stride = SW128_ATOM_ROWS * SW128_BYTES;             // 1024B
  // bytes in a full 8-row stripe across all N columns
  const int k8_stride = (HEAD_DIM / SW128_ATOM_COLS) * n64_stride;  // 2048B

  for (int k8 = 0; k8 < BLOCK_N / SW128_ATOM_ROWS; ++k8) {
    const int off_k = k8 * SW128_ATOM_ROWS;
    for (int n64 = 0; n64 < HEAD_DIM / SW128_ATOM_COLS; ++n64) {
      const int off_n = n64 * SW128_ATOM_COLS;
      // move down to the next 8-token / KV-row stripe
      const int k_slice_offset = k8 * k8_stride;
      // move right to the next 64-feature atom inside that stripe
      const int n_tile_offset = n64 * n64_stride;
      const int dst = dst_smem_base + k_slice_offset + n_tile_offset;
      ::tma::copy_2d_gmem_to_smem(dst, tmap, off_n, global_row0 + off_k, mbar_addr);
    }
  }
}

}  // namespace swizzle128
