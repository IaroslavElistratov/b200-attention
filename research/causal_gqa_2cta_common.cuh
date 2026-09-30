#pragma once
// Helpers for causal_gqa_2cta.cu: the lineage's common.cuh, smem_layout_swizzle128.cuh and approximations.cuh in their
// cluster / cta_group::2 forms, plus remote mbarrier arrives, the TMA store and a capped software exp2. A helper is
// one PTX instruction or one small piece of math; the kernel file keeps the schedule (which barrier, which MMA, when).
// Addresses are ints as in the lineage: smem = shared::cta window address, TMEM = (lane << 16) | column. A "cluster
// address" (mapa) is the same smem offset in one CTA of the pair; remote arrives and the 2-CTA TMA loads take those.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

constexpr int WARP_SIZE = 32;

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

constexpr int BF16_BYTES = int(sizeof(nv_bfloat16));

// SW128 operand panels: 128-byte rows (64 bf16), swizzled in 8-row atoms of 1024 B. One TMA box is one panel wide.
constexpr int SW128_BYTES = 128;
constexpr int SW128_ATOM_ROWS = 8;
constexpr int SW128_ATOM_COLS = SW128_BYTES / BF16_BYTES;  // 64 bf16

__device__ __forceinline__ int smem_addr_of(const void* p) {
  return static_cast<int>(__cvta_generic_to_shared(p));
}

// 16-byte smem store (one STS.128)
__device__ __forceinline__ void st_shared_v4(int addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d)
               : "memory");
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__ inline
uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred px;\n\t"
    "elect.sync _|px, 0xffffffff;\n\t"
    "@px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred)
  );
  return pred;
}

namespace mbarrier {

__device__ inline
void init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count)
               : "memory");
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
// Plain bra rather than the lineage's bra.uni: try_wait's predicate is per thread, not guaranteed warp-uniform.
// (The .aligned tcgen05 / elect instructions after a wait assume that the warp leaves it together: true in
// practice, not an ISA promise.)
__device__ inline
void wait(int mbar_addr, int phase) {
  // 10000000 = the suspend-time hint (ns), written into the PTX as an immediate (as a register operand it costs a
  // move before every wait)
  asm volatile(
    "{\n\t"
    ".reg .pred ready;\n\t"
    "RETRY:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1, 10000000;\n\t"
    "@ready bra READY;\n\t"
    "bra RETRY;\n\t"
    "READY:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase)
    : "memory"
  );
}

// the same without the suspend-time hint: the MMA warp's steady-state loop uses this one (see there for why)
__device__ inline
void wait_nohint(int mbar_addr, int phase) {
  asm volatile(
    "{\n\t"
    ".reg .pred ready;\n\t"
    "RETRY:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n\t"
    "@ready bra READY;\n\t"
    "bra RETRY;\n\t"
    "READY:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase)
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

__device__ __forceinline__ void arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" : : "r"(mbar_addr)
               : "memory");
}

// Arrive `count` times on a barrier in either CTA of the pair (a cluster address). count 32 = one message for a
// whole warp. Scope .release.cta like FA4, although the waiter may be in the other CTA: these arrives publish TMEM
// state (P after tcgen05.wait::st), not smem writes, and .release.cluster puts four fences in front of every one.
// The release scope is not what orders TMEM across threads, though: the ISA's pattern for that is
// tcgen05.fence::before_thread_sync / after_thread_sync, which this kernel omits as FA4 does (see softmax_step).
__device__ __forceinline__ void arrive_cluster(int cluster_addr, int count) {
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], %1;\n" :: "r"(cluster_addr), "r"(count)
               : "memory");
}

}  // namespace mbarrier

namespace cluster {

// shared::cta address -> shared::cluster address of the same offset in CTA `rank` of the pair. volatile so nvcc
// neither merges nor hoists these (a hoisted cluster address is one more register live across the whole loop).
__device__ __forceinline__ int map_shared_rank(int smem_addr, int rank) {
  int out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(out) : "r"(smem_addr), "r"(rank));
  return out;
}

}  // namespace cluster

namespace tma {

__device__ __forceinline__ void prefetch_descriptor(const CUtensorMap* tmap) {
  asm volatile("prefetch.tensormap [%0];\n" :: "l"(tmap) : "memory");
}

// 2-CTA versions of the lineage's copy_2d_gmem_to_smem: one box of a 4D / 5D tensor map into this CTA's smem.
// .cta_group::2 lets the bytes be counted on a barrier of EITHER CTA (`mbar_leader`, a cluster address).
__device__ inline
void copy_4d_gmem_to_smem_2cta(int dst, const CUtensorMap* tmap, int c0, int c1, int c2, int c3, int mbar_leader) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
      :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar_leader)
      : "memory");
}

__device__ inline
void copy_5d_gmem_to_smem_2cta(int dst, const CUtensorMap* tmap, int c0, int c1, int c2, int c3, int c4,
                               int mbar_leader) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
      :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(mbar_leader)
      : "memory");
}

// TMA store: one box from this CTA's smem to a 5D tensor map. Completion is tracked with bulk async-groups
// (cp.async.bulk.commit_group / wait_group.read, written inline in the store warp), not an mbarrier.
__device__ inline
void copy_5d_smem_to_gmem(const CUtensorMap* tmap, int c0, int c1, int c2, int c3, int c4, int src) {
  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];\n"
      :: "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(src)
      : "memory");
}

}  // namespace tma

__host__ __device__ constexpr uint32_t desc_encode(uint32_t bytes) { return (bytes >> 4) & 0x3FFFu; }

// Same descriptors as the lineage's smem_layout_swizzle128.cuh, split into the two 32-bit words: only the low word
// (start address, LBO) varies; the high word (SBO = one 8-row atom, 1024 B; version 1; SWIZZLE_128B) is one
// constant for every operand and goes into the MMA asm as an immediate. A k-step adds bytes >> 4 to the low word;
// the MMA applies the swizzle to the advanced address, same as TMA did when writing the tile.
namespace swizzle128 {

constexpr uint32_t ATOM_BYTES = SW128_ATOM_ROWS * SW128_BYTES;  // 1024
constexpr uint32_t DESC_HI = desc_encode(ATOM_BYTES) | (1u << (46 - 32)) | (2u << (61 - 32));
static_assert(DESC_HI == 0x40004040u, "SBO 1024, version 1, SWIZZLE_128B");

// K-major SW128 descriptor (Q, K): LBO unused (0)
__device__ __forceinline__ uint32_t desc_lo_kmajor(int smem_addr_bytes) {
  return desc_encode(smem_addr_bytes);
}

// MN-major SW128 descriptor (V: a smem row holds 64 head-dim values of one key; in O = P V the head dim is N, so
// N runs along the row). LBO = the distance between two 64-column panels along N. Each CTA's V half is one such
// panel, so this kernel's MMAs never step to a second panel along N and never read the LBO field (written as
// 1024 B, the lineage's value). The SBO (next 8-key atom, 1024 B) is the constant high word.
__device__ __forceinline__ uint32_t desc_lo_mnmajor(int smem_addr_bytes) {
  return desc_encode(smem_addr_bytes) | (desc_encode(ATOM_BYTES) << 16);
}

}  // namespace swizzle128

namespace tcgen05 {

// The 2-CTA version of the lineage's commit_arrive: once every earlier pair MMA of this warp has completed,
// arrive on the SAME barrier offset in both CTAs (multicast, CTA mask 0b11). The whole MMA warp calls it; the
// election is inside the asm so the warp stays converged (one lane commits).
__device__ __forceinline__ void commit_arrive_pair(int mbar_addr) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader;\n\t"
      ".reg .b16 mask;\n\t"
      "elect.sync _|leader, -1;\n\t"
      "mov.b16 mask, 3;\n\t"
      "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], mask;\n\t"
      "}"
      :
      : "r"(mbar_addr)
      : "memory");
}

// The 2-CTA version of the lineage's mma_f16 (tcgen05.mma.cta_group::2): the leader issues it once for both CTAs
// (see "2-CTA MMAs" in causal_gqa_2cta.cu). The whole MMA warp calls it with the election inside the asm: the warp stays
// converged, so the compiler keeps the descriptors and addresses on the uniform datapath (with a C++ `if (elected)` around
// the loop every value is per-lane and each MMA is preceded by vector -> uniform moves).
// A and B are SW128 descriptor low words; the high word is the immediate swizzle128::DESC_HI.
__device__ inline
void mma_f16_pair(int taddr_d, uint32_t a_desc_lo, uint32_t b_desc_lo, uint32_t i_desc, int enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .b64 da, db;\n\t"
    ".reg .pred p, leader;\n\t"
    "elect.sync _|leader, -1;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "mov.b64 da, {%1, %5};\n\t"
    "mov.b64 db, {%2, %5};\n\t"
    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], da, db, %3, p;\n\t"
    "}"
    :: "r"(taddr_d), "r"(a_desc_lo), "r"(b_desc_lo), "r"(i_desc), "r"(enable_input_d), "n"(swizzle128::DESC_HI)
    : "memory"
  );
}

// same with A read from TMEM (the brackets around the A operand): A = P as packed bf16 pairs
__device__ __forceinline__ void mma_f16_pair_a_tmem(int taddr_d, int taddr_a, uint32_t b_desc_lo, uint32_t i_desc,
                                                    int enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .b64 db;\n\t"
      ".reg .pred p, leader;\n\t"
      "elect.sync _|leader, -1;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "mov.b64 db, {%2, %5};\n\t"
      "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], db, %3, p;\n\t"
      "}"
      :
      : "r"(taddr_d), "r"(taddr_a), "r"(b_desc_lo), "r"(i_desc), "r"(enable_input_d), "n"(swizzle128::DESC_HI)
      : "memory");
}

// 32x32b loads: thread i of the warp gets TMEM lane (warp % 4) * 32 + i, .xN = N consecutive columns. _nowait, the
// lineage's name for the form without the wait::ld (its ld_32x32b_x8 waits): a register's first use waits for its
// load (PTX ISA 9.7.18.6.4.5), and a later tcgen05.st of the same warp is ordered after the load (9.7.18.6.2). The
// kernel puts an explicit wait::ld where the loaded columns are handed to another warp.
__device__ __forceinline__ void ld_32x32b_x32_nowait(int taddr, float* out) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]), "=f"(out[4]), "=f"(out[5]), "=f"(out[6]),
        "=f"(out[7]), "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]), "=f"(out[12]), "=f"(out[13]),
        "=f"(out[14]), "=f"(out[15]), "=f"(out[16]), "=f"(out[17]), "=f"(out[18]), "=f"(out[19]), "=f"(out[20]),
        "=f"(out[21]), "=f"(out[22]), "=f"(out[23]), "=f"(out[24]), "=f"(out[25]), "=f"(out[26]), "=f"(out[27]),
        "=f"(out[28]), "=f"(out[29]), "=f"(out[30]), "=f"(out[31])
      : "r"(taddr));
}

__device__ __forceinline__ void ld_32x32b_x16_nowait(int taddr, float* out) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]), "=f"(out[4]), "=f"(out[5]), "=f"(out[6]),
        "=f"(out[7]), "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]), "=f"(out[12]), "=f"(out[13]),
        "=f"(out[14]), "=f"(out[15])
      : "r"(taddr));
}

// Store packed P to TMEM: 16 b32 cells per lane = 32 bf16 probabilities in pairs (the lineage's st_32x32b_x8_u32,
// twice as wide). Integer "r" constraints so the bits pass through unchanged.
__device__ __forceinline__ void st_32x32b_x16_u32(int taddr, const uint32_t* in) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
      :
      : "r"(taddr), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]), "r"(in[4]), "r"(in[5]), "r"(in[6]), "r"(in[7]),
        "r"(in[8]), "r"(in[9]), "r"(in[10]), "r"(in[11]), "r"(in[12]), "r"(in[13]), "r"(in[14]), "r"(in[15])
      : "memory");
}

}  // namespace tcgen05


// Pack two bf16 probabilities into one b32 TMEM cell (lo in bits 0-15), so PV can read A = P from TMEM. The same
// word is also two neighbouring bf16 outputs in memory.
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

namespace approx {

// MUFU approximations with FTZ: 2^x, 1/x, log2(x). PTX because exp2f, __fdividef(1, x) and __log2f give the
// non-.ftz forms without -use_fast_math, and the compiler then wraps them in subnormal fix-ups.
__device__ __forceinline__ float fast_exp2(float x) {
  float out;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out;
}
__device__ __forceinline__ float fast_rcp(float x) {
  float out;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out;
}
__device__ __forceinline__ float fast_log2(float x) {
  float out;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out;
}

// Software exp2 for a pair on the FMA pipe: the lineage's e2e_exp2_pair (FA4's ex2_emulation_2) with a cap at
// +100. MUFU is the slow unit, so the softmax sends a fixed subset of its pairs here.
//   1. clamp a = max(min(x, 100), -127). Below -127 step 4 would push the exponent field below zero (x <= -127
//      gives exactly 0, a masked score included). The +100 cap is the new part: with the frozen basis a later
//      score can sit far above it; uncapped, x >= 128 overflows the exponent field into inf, NaN or a wrong finite
//      value. Capped, every x >= 100 (and NaN) returns exactly 2^100, which by itself trips the row-sum
//      certificate. (MUFU returns +inf for x >= 128, which trips as well.)
//   2. split a = n + f, n = floor(a), f in [0, 1]: add 1.5 * 2^23 with round-down, which lands in [2^23, 2^24)
//      where float spacing is 1, so the sum is exactly 1.5 * 2^23 + n; subtract it back to get n.
//   3. 2^f ~ ((c3 f + c2) f + c1) f + 1, Horner, three FMAs (max rel. error 8.8e-5, far below bf16's 2^-8).
//   4. 2^n * 2^f: add n << 23 to the bits of 2^f. (0x4B400000 + n) << 23 == n << 23 in 32 bits.
// Two spellings matter: min(x, 100) is written -max(-x, -100), as its own statement before the -127 clamp (not
// fminf, not nested); the shift is a funnel shift with a zero low word (__funnelshift_lc, the same bits as << 23),
// which lands on the integer pipe, where a plain << 23 becomes a multiply on the busy FMA pipe.
__device__ __forceinline__ void e2e_exp2_pair_capped(float x0, float x1, float& out0, float& out1) {
  // duplicate each constant so one packed instruction handles both inputs
  const float2 floor_bias = make_float2(12582912.0f, 12582912.0f);  // 1.5 * 2^23 (bits 0x4B400000)
  const float2 neg_floor_bias = make_float2(-12582912.0f, -12582912.0f);
  const float2 c3 = make_float2(0.07711909f, 0.07711909f);  // exact FP32 bits: 0x3D9DF09D
  const float2 c2 = make_float2(0.22756439f, 0.22756439f);  // exact FP32 bits: 0x3E6906A4
  const float2 c1 = make_float2(0.69514614f, 0.69514614f);  // exact FP32 bits: 0x3F31F519
  const float2 one = make_float2(1.0f, 1.0f);

  // 1. clamp to [-127, 100]
  const float m0 = -fmaxf(-x0, -100.0f);
  const float m1 = -fmaxf(-x1, -100.0f);
  const float2 clamped = make_float2(fmaxf(m0, -127.0f), fmaxf(m1, -127.0f));
  // 2. split into floor and fraction (no packed subtract: add the negation)
  const float2 biased_floor = __fadd2_rd(clamped, floor_bias);            // bits 0x4B400000 + n
  const float2 floor_x = __fadd2_rn(biased_floor, neg_floor_bias);         // n
  const float2 fraction = __fadd2_rn(clamped, make_float2(-floor_x.x, -floor_x.y));
  // 3. 2^fraction, Horner
  float2 frac_exp2 = __ffma2_rn(c3, fraction, c2);
  frac_exp2 = __ffma2_rn(frac_exp2, fraction, c1);
  frac_exp2 = __ffma2_rn(frac_exp2, fraction, one);
  // 4. 2^n * 2^fraction: (n << 23) + bits(2^fraction)
  out0 = __uint_as_float(__funnelshift_lc(0u, __float_as_uint(biased_floor.x), 23) + __float_as_uint(frac_exp2.x));
  out1 = __uint_as_float(__funnelshift_lc(0u, __float_as_uint(biased_floor.y), 23) + __float_as_uint(frac_exp2.y));
}

}  // namespace approx

namespace bits {

// x >> n with PTX semantics: shifts of 32 or more give 0 (C++ >> is undefined there; `n < 32 ? x >> n : 0` and
// __funnelshift_rc compile to more instructions)
__device__ __forceinline__ uint32_t shr_clamp(uint32_t x, int n) {
  uint32_t out;
  asm("shr.u32 %0, %1, %2;\n" : "=r"(out) : "r"(x), "r"(n));
  return out;
}

// running max of |bits| over an output row, for the O-check: max(|bits(a)|, |bits(b)|, acc) as unsigned ints.
// With the sign bit cleared, integer order = magnitude order for finite values and inf, and every NaN pattern
// sorts above inf, so NaNs are never lost (fmaxf would drop them). One 3-input max instruction.
__device__ __forceinline__ uint32_t abs_max2(float a, float b, uint32_t acc) {
  return __vimax3_u32(__float_as_uint(a) & 0x7FFFFFFFu, __float_as_uint(b) & 0x7FFFFFFFu, acc);
}

}  // namespace bits

// n / d for any 32-bit n and a divisor d fixed at launch, from host-precomputed (m, s1, s2): t = mulhi(n, m);
// q = (((n - t) >> s1) + t) >> s2 (the round-up multiply-shift division of FA4's FastDivmod). Two multiplies and
// two shifts on the integer pipe instead of the ~20-instruction float-reciprocal sequence a runtime `/` compiles
// to. Exact for every n < 2^32 and 1 <= d < 2^31.
struct FastDiv {
  uint32_t m, s1, s2;
};
__device__ __forceinline__ uint32_t fast_div(uint32_t n, const FastDiv& f) {
  const uint32_t t = __umulhi(n, f.m);
  return (((n - t) >> f.s1) + t) >> f.s2;
}
// l = ceil(log2 d), m = floor(2^32 (2^l - d) / d) + 1, s1 = min(l, 1), s2 = max(l - 1, 0)
inline FastDiv make_fast_div(uint32_t d) {
  uint32_t l = 0;
  while ((1ull << l) < d) ++l;
  FastDiv f;
  f.m = uint32_t(((1ull << 32) * ((1ull << l) - d)) / d + 1ull);
  f.s1 = l < 1 ? l : 1u;
  f.s2 = l > 0 ? l - 1 : 0u;
  return f;
}
