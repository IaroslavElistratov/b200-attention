#include <cstdint>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>      // nv_bfloat162 / __float22bfloat162_rn for bf16::pack2_to_u32
#include <cuda_runtime.h>   // float2
#include <torch/library.h>

constexpr int WARP_SIZE = 32;

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__ inline
uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

namespace mbarrier {

__device__ inline
void init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void inval(int mbar_addr) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(mbar_addr) : "memory");
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__ inline
void wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    : "memory"
  );
}

__device__ __forceinline__ void arrive_expect_tx(int mbar_addr, int tx_bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
      :
      : "r"(mbar_addr), "r"(tx_bytes)
      : "memory");
}

// Software mbarrier helper: ordinary arrival; no TMA byte count is attached.
// introduced by kernel 4.
__device__ __forceinline__ void arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" : : "r"(mbar_addr)
               : "memory");
}

}  // namespace mbarrier

namespace tma {

__device__ inline
void copy_2d_gmem_to_smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr) : "memory");
}

}  // namespace tma

namespace tcgen05 {

__device__ __forceinline__ void commit_arrive(int mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
      :
      : "r"(mbar_addr)
      : "memory");
}

// https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cute/arch/mma_sm100_umma.hpp#L86
__device__ inline
void mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, int enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"  // predicate register enable-input-d
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
    "}"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d)
    : "memory"
  );
}

// the single how this helper differes from "tcgen05::mma_f16", is that the below helper has
// [%1] in the A operand (ie brackets around A operand) slot, it's what tells PTX this MMA reads A from TMEM
//
// The output O and A operand P are TMEM addresses;
// the B operand V is still described by an SMEM descriptor.
__device__ __forceinline__ void mma_f16_a_tmem(int taddr_d, int taddr_a, uint64_t b_desc,
                                               uint32_t i_desc, int enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      // For this tcgen05.mma form, A is read from TMEM and i_desc says A is bf16.
      // TMEM cells are b32, so each cell holds two packed bf16 P values.
      // The MMA interprets those b32 cells according to the A-from-TMEM layout and i_desc type bits.
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t"
      "}"
      :
      : "r"(taddr_d), "r"(taddr_a), "l"(b_desc), "r"(i_desc), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void ld_32x32b_x8(int taddr, float (&out)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]), "=f"(out[4]), "=f"(out[5]),
        "=f"(out[6]), "=f"(out[7])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

// todo: rm this in favor of tcgen05::st_32x32b_x8_u32?
__device__ __forceinline__ void st_32x32b_x8(int taddr, const float (&in)[8]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
      :
      : "r"(taddr), "f"(in[0]), "f"(in[1]), "f"(in[2]), "f"(in[3]), "f"(in[4]), "f"(in[5]),
        "f"(in[6]), "f"(in[7])
      : "memory");
}

// Store packed P to TMEM.
// Each uint32_t is just a 32-bit container: two bf16 probabilities packed into one b32 cell.
// Use integer "r" constraints so the bits are passed through unchanged, not as fp32 operands.
__device__ __forceinline__ void st_32x32b_x8_u32(int taddr, const uint32_t (&in)[8]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
      :
      : "r"(taddr), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]), "r"(in[4]), "r"(in[5]),
        "r"(in[6]), "r"(in[7])
      : "memory");
}

}  // namespace tcgen05

__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; };

inline
void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *msg;
  if (cuGetErrorString(err, &msg) != CUDA_SUCCESS)
    msg = "unable to get error string";
  TORCH_CHECK(false, msg);
}

inline
void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) return;
  TORCH_CHECK(false, cudaGetErrorString(err));
}

inline
void init_tmap_2d_simple(
  CUtensorMap *tmap,
  const nv_bfloat16 *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width,
  CUtensorMapSwizzle swizzle
) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank]       = {global_width, global_height};
  uint64_t globalStrides[rank-1] = {global_width * sizeof(nv_bfloat16)};  // in bytes
  uint32_t boxDim[rank]          = {shared_width, shared_height};
  uint32_t elementStrides[rank]  = {1, 1};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    swizzle,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  check_cu(err);
}





// packed f32x2 helpers for score transforms and cached-score prefix math
namespace f32x2 {

__device__ __forceinline__ float2 fma_rn_ftz(float2 x, float a, float b) {
  float2 out;
  asm volatile(
      "{\n\t"
      ".reg .b64 vx, va, vb, vo;\n\t"
      "mov.b64 vx, {%2, %3};\n\t"
      "mov.b64 va, {%4, %4};\n\t"
      "mov.b64 vb, {%5, %5};\n\t"
      "fma.rn.ftz.f32x2 vo, vx, va, vb;\n\t"
      "mov.b64 {%0, %1}, vo;\n\t"
      "}\n"
      : "=f"(out.x), "=f"(out.y)
      : "f"(x.x), "f"(x.y), "f"(a), "f"(b));
  return out;
}

__device__ __forceinline__ float2 add_rn_ftz(float2 a, float2 b) {
  float2 out;
  asm volatile(
      "{\n\t"
      ".reg .b64 va, vb, vo;\n\t"
      "mov.b64 va, {%2, %3};\n\t"
      "mov.b64 vb, {%4, %5};\n\t"
      "add.rn.ftz.f32x2 vo, va, vb;\n\t"
      "mov.b64 {%0, %1}, vo;\n\t"
      "}\n"
      : "=f"(out.x), "=f"(out.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return out;
}

}  // namespace f32x2

// Core to the P-in-TMEM path: TMEM stores are b32 cells here, while P is bf16.
// Pack two bf16 probabilities into one b32 slot so PV can later read A=P from TMEM.
namespace bf16 {

__device__ __forceinline__ uint32_t pack2_to_u32(float a, float b) {
  union {
    nv_bfloat162 v;
    uint32_t u;
  } tmp;
  tmp.v = __float22bfloat162_rn(make_float2(a, b));
  return tmp.u;
}

}  // namespace bf16
