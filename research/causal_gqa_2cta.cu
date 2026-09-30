// Causal GQA attention with the frozen softmax basis of kernels 15-17, FA4's 2-CTA MMAs and a persistent pair
// scheduler. A research branch off kernel 18, not a lineage step.
//   1. Frozen basis, lifted by +96 so later tiles have headroom, plus a row-sum certificate: a row the basis could
//      not handle comes back as NaN and bumps a counter. Fail-closed, up to the two known limits listed below.
//   2. FA4's 2-CTA mode: a cluster of two CTAs shares every MMA (M = 256); each CTA holds half of K / V.
//   3. Causal, GQA 4:1 packed (the 4 q heads of a kv head are consecutive rows), FA4's longest-first tile order,
//      persistent clusters walking the tiles in a snake.
//   4. O goes out through smem with a TMA store; optional LSE.

#include "causal_gqa_2cta_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <stdexcept>
#include <string>

// Assumptions (checked on the host; anything else throws from the launch function):
//   - head_dim == 128, bf16 in / out, BSHD layout
//   - Hq == 4 * Hkv (GQA 4:1), Hkv > 0; any batch B > 0 (a batch is just more (b, hkv) heads for the scheduler)
//   - len_q == len_kv, a positive multiple of 128; (Sq / 128) * Hkv * B < 2^22 (tiles are counted in 32-bit ints)
//   - always causal (bottom-right aligned); there is no non-causal mode
//   - softmax_scale finite and > 0
//   - a device with at least 2 SMs (a cluster needs two) and index < 64 (the per-device caches)
//
// The frozen basis (what replaces FA4's running max):
//   A row walks its KV tiles from the diagonal tile (the only masked one) down to tile 0. The diagonal tile fixes
//   the basis once, 96 above its row max in log2 units:
//       basis = (rowmax(masked S) + 96 / softmax_scale_log2) * softmax_scale_log2
//   and every tile after that just does
//       P = exp2(S * softmax_scale_log2 - basis);   rowsum += sum(P);   O += bf16(P) V
//   No running max, no O rescale, no correction warps. The first tile's largest P is ~2^-96, which leaves ~196
//   log2 units of headroom for later scores before one term reaches 2^100 and trips the row.
//
//   Certificate, once per row after its last tile: trip if rowsum >= 2^100, rowsum is NaN, or rowsum < 2^-98 while
//   the diagonal tile's row max was not -inf. A tripped row is written as NaN (never zeros; LSE = +inf or NaN) and
//   adds 1 to `counter`, as does an untripped row whose normalized O rounds to bf16 inf or NaN. The caller zeroes
//   `counter` (one device float; nullptr = don't report; being a float it stops growing at 2^24 but stays nonzero).
//   Nonzero afterwards = some rows are garbage, recompute the call with a reference kernel. Two things a zero
//   counter does NOT cover:
//   1. P below 2^-126 flushes to zero (the emulated columns keep a subnormal down to 2^-127), i.e. keys more than
//      ~30 log2 units below the diagonal tile's max are dropped. Each carries under 2^-30 of the row's weight
//      (2^-28 of the row sum in the worst rounding of the basis), so together the dropped keys move O by at most
//      their number x 2^-30 x their largest |V - O|. Harmless for ordinary V, but neither check looks at V: at a
//      3e-2 tolerance and |O| ~ 1, one dropped key with V ~ 2^26 gives a wrong row that is not flagged, and so do
//      128k dropped keys with V ~ 2^10.
//   2. A row at query position < 128 (its diagonal tile is its only tile) whose every visible score is -inf in
//      fp32 (+-inf in Q or K, or a finite Q K^T that overflows fp32) gets O = 0, LSE = -inf, no flag. Same as FA4.
//      At a later position the same row trips.
//
// 2-CTA MMAs:
//   A cluster of 2 CTAs is a pair. The leader (rank 0) issues each MMA once for both (tcgen05.mma.cta_group::2,
//   M = 256). Each CTA supplies its own 128 rows of A (Q from its smem, P from its TMEM) and half of B (64 keys of
//   K, or 64 head-dim columns of V) at the same smem offsets, and gets its own 128 rows of D in its own TMEM.
//   Both CTAs' TMA loads complete on the leader's barriers; every MMA commit lands in both CTAs.
//
// Steady-state dataflow, per work item (128 query positions x 4 heads = 512 rows, 256 per CTA):
//   1) load warp: K / V halves into a 6-stage ring, K(diag) Q0 Q1 V(diag) K V K V ... down to tile 0
//   2) MMA warp: QK and PV for both Q stages, one KV tile apart
//   3) softmax warps: S -> P in TMEM in two halves; P[:, 0:64] is published first so PV can start early
//   4) after the last KV tile: certificate, O / rowsum, bf16 into smem, TMA store (overlaps the next item)
//
// From FA4 (flash_fwd_sm100, b29): warp layout and P in TMEM; the 2-CTA pair MMA (b29 does not enable it for
// causal hd128); the longest-first tile order and the diagonal-first KV order; MUFU exp2 + cubic emulation on the
// FMA pipe; the rowmax / rowsum reduction orders; the split P hand-off. New here: the lifted frozen basis, the +100
// cap in the emulated exp2, the certificate and the O-check, the snake, and two settings (which 24 of the 128
// columns are emulated, and the P split at 64 instead of 96).
//
// Build: nvcc -O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a; link -lcuda (the host code uses the driver
// API). No --use_fast_math: the MUFU paths are spelled in PTX already, and the rest of the softmax is meant to be
// IEEE. Tuned with the nvcc 13.0 front end + ptxas 13.4.

// Kernel parameters: the shapes and the tile schedule, computed once on the host (make_params). At global scope:
// with this type in an anonymous namespace the kernel compiles differently.
struct Params {
  int Sq, Hq, Hkv;
  float softmax_scale_log2;  // softmax_scale * log2(e)
  float* counter;            // certificate trip counter, may be nullptr
  float* lse;                // nullptr, or [B, Hq, Sq] natural-log LSE
  // the schedule: see "Persistent work assignment"
  int num_m_blocks;          // work items per head: ceil(4 Sq / 512)
  int num_heads;             // (b, hkv) pairs: Hkv * B
  int heads_per_section;     // FA4's l2_minor: heads whose K and V fit ~50 MB of L2 together, a power of 2
  int num_clusters;          // G = min(#SMs / 2, #work items)
  int rounds_per_section;    // ceil(heads_per_section * num_m_blocks / G)
  int num_rounds;            // ceil(num_heads / heads_per_section) * rounds_per_section
  // work_item's three runtime divisors, precomputed so the device never divides (FastDiv: causal_gqa_2cta_common.cuh)
  int heads_per_section_log2;  // heads_per_section is a power of 2: a full section divides by shifting
  int full_sections;           // num_heads / heads_per_section; if heads_last_section > 0, section full_sections
  int heads_last_section;      //   is the partial one with this many heads (0: every section is full)
  FastDiv div_rounds_per_section, div_heads_last_section, div_hkv;
};

namespace {

constexpr int CTA_GROUP = 2;  // CTAs per cluster: one pair
constexpr int BLOCK_M = 128;  // packed Q rows per Q stage per CTA (= TMEM lanes)
constexpr int BLOCK_N = 128;  // keys per KV tile
constexpr int HEAD_DIM = 128;
constexpr int MMA_K = 16;
constexpr int MMA_K_BYTES = MMA_K * BF16_BYTES;  // 32
constexpr int QHEADS_PER_KVHEAD = 4;

constexpr int Q_TILE_BYTES = BLOCK_M * HEAD_DIM * BF16_BYTES;  // 32 KB: one Q (or O) stage of one CTA
constexpr int PANEL_BYTES = BLOCK_M * SW128_BYTES;             // 16 KB: 128 rows x 64 cols, one Q / O box

namespace q_stage {
constexpr int STAGES = 2;                                         // softmax of one stage overlaps the other's MMAs
constexpr int POS_PER_STAGE = BLOCK_M / QHEADS_PER_KVHEAD;        // 32 query positions per (CTA, Q stage)
constexpr int ROWS_PER_WORK_ITEM = CTA_GROUP * STAGES * BLOCK_M;  // 512 packed rows per work item
}  // namespace q_stage

namespace kv_ring {
constexpr int STAGES = 6;                                         // what is left after Q and O: (224 - 128) / 16
constexpr int K_HALF_KEYS = BLOCK_N / CTA_GROUP;                  // K: 64 of the tile's 128 keys per CTA
constexpr int V_HALF_COLS = HEAD_DIM / CTA_GROUP;                 // V: 64 of the 128 head-dim columns per CTA
constexpr int STAGE_BYTES = K_HALF_KEYS * HEAD_DIM * BF16_BYTES;  // 16 KB: this CTA's half of one K or V tile
constexpr int K_PANEL_BYTES = K_HALF_KEYS * SW128_BYTES;          // 8 KB: 64 keys x 64 d, one K box
static_assert(K_HALF_KEYS * HEAD_DIM * BF16_BYTES == STAGE_BYTES && BLOCK_N * V_HALF_COLS * BF16_BYTES == STAGE_BYTES,
              "a K half and a V half each fill one ring stage exactly (one expect_tx value serves both)");
}  // namespace kv_ring

// Dynamic shared memory: sO[2], sQ[2], the KV ring. Same offsets in both CTAs, so the leader's MMA descriptors
// address the peer's Q / K / V at its own offsets. Everything is SW128, panels start on 1024-byte boundaries.
constexpr int SMEM_BYTES = 2 * q_stage::STAGES * Q_TILE_BYTES + kv_ring::STAGES * kv_ring::STAGE_BYTES;  // 224 KB
static_assert(SMEM_BYTES + 1024 <= 227 * 1024, "sO + sQ + ring plus the static barriers fit an sm_100 CTA");

// Warp roles (12 warps = 3 warpgroups, one CTA per SM):
//   0..3, 4..7  softmax for Q stage 0 / 1, one thread per row (= TMEM lane), then that stage's O epilogue
//   8           tcgen05 MMA (issued by the leader CTA only); TMEM alloc / dealloc in both CTAs
//   9           TMA store of O
//   10          TMA loads of Q, K, V
//   11          nothing. setmaxnreg works on whole warpgroups, so it rides along.
constexpr int SOFTMAX_WARPS_PER_STAGE = 4;
constexpr int SOFTMAX_WARP_COUNT = q_stage::STAGES * SOFTMAX_WARPS_PER_STAGE;
constexpr int MMA_WARP_ID = SOFTMAX_WARP_COUNT;
constexpr int STORE_WARP_ID = MMA_WARP_ID + 1;
constexpr int LOAD_WARP_ID = MMA_WARP_ID + 2;
constexpr int NUM_WARPS = 12;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
// Warps 8-11 get 64 registers (48 would also fit). 256 x 192 + 128 x 64 = 57,344 of the CTA's pool of
// 384 x 168 = 64,512 (the launch bound: 65,536 / 384 rounded down to a multiple of 8).
constexpr int SOFTMAX_REGS = 192;  // setmaxnreg.inc for warps 0-7
constexpr int OTHER_REGS = 64;     // setmaxnreg.dec for warps 8-11
static_assert(SOFTMAX_WARP_COUNT * WARP_SIZE * SOFTMAX_REGS +
                      (TB_SIZE - SOFTMAX_WARP_COUNT * WARP_SIZE) * OTHER_REGS <= TB_SIZE * 168,
              "the register split fits the CTA's launch pool (384 threads x 168)");

// TMEM (128 lanes = packed rows, 512 columns of b32): S0 [0, 128), S1 [128, 256), O0 [256, 384), O1 [384, 512).
// P_s overlays S_s at S_s + 64 (128 bf16 = 64 b32 cells).
constexpr int TMEM_S_COLS = BLOCK_N;
constexpr int TMEM_O_COLS = HEAD_DIM;
constexpr int TMEM_ALLOC_COLS = q_stage::STAGES * TMEM_S_COLS + q_stage::STAGES * TMEM_O_COLS;
constexpr int TMEM_P_COL_OFFSET = BLOCK_N / 2;
static_assert(TMEM_ALLOC_COLS == 512, "S0/S1 + O0/O1 fill the 512 TMEM columns exactly");
// P is published in two halves, [0, 64) then [64, 128), and PV is split at the same key: microsteps 0-3 need only
// the first half. (kernel 17 splits at 32 / 96 and produces the high half first.)
constexpr int P_SPLIT = 64;  // not a free knob: softmax_step publishes P[:, 0:64] after fragment 1 (p_packed[0..31])
constexpr int PV_SPLIT_MICROSTEP = P_SPLIT / MMA_K;  // 4

// Frozen basis and certificate.
constexpr float BASIS_LIFT_LOG2 = 96.0f;      // basis = rowmax * softmax_scale_log2 + 96
constexpr float ROWSUM_TRIP_HIGH = 0x1p100f;  // 2^100
constexpr float ROWSUM_TRIP_LOW = 0x1p-98f;   // 2^-98: a healthy row keeps its first max term, ~2^-96
constexpr float LN2 = 0.6931471805599453f;
// O-check: a normalized O value with |fp32 bits| >= 0x7F7F8000 rounds to bf16 inf (0x7F7F is bf16's largest
// finite, 0x8000 is half an ulp, the tie rounds up because 0x7F7F is odd). NaN patterns are larger still.
constexpr uint32_t BF16_INF_ABS_BITS = 0x7F7F8000u;

// Persistent work assignment
//
// The work item is 512 packed Q rows (128 query positions x 4 heads) of one kv head, computed by one cluster.
// FA4's static order (SingleTileLPTScheduler), for L2 reuse and load balance: the (b, hkv) heads come in SECTIONS
// of heads_per_section heads (as many as fit their K and V in ~50 MB of L2, a power of 2; the last section takes
// the leftovers); inside a section the head changes fastest and query blocks go longest first (causal: query
// block m needs m + 1 KV tiles). E.g. 3 query blocks, a section of heads h0, h1: (m2,h0) (m2,h1) (m1,h0) (m1,h1)
// (m0,h0) (m0,h1).
//
// G resident clusters deal that order out in ROUNDS of G, one item per cluster. Dealing left to right every round
// would give cluster 0 the longest item of every round, so the direction alternates (a snake): cluster c gets
// positions c, 2G-1-c, 2G+c, 4G-1-c, ... Since each section runs longest to shortest, every section is padded to
// whole rounds (rounds_per_section); a padding position is a HOLE, and the cluster sits that round out. Example:
// G = 3, one section of 7 items (item t needs 7 - t KV tiles), 3 rounds:
//               cluster 0   cluster 1   cluster 2
//   round 0         0           1           2         left to right
//   round 1         5           4           3         right to left
//   round 2         6          (7)         (8)        left to right; 7 and 8 are holes
//   -> 10 / 9 / 9 KV tiles per cluster (left to right every round would be 12 / 9 / 7).
// The walk is a pure function of (cluster, round, Params), so every role in both CTAs sees the same items and
// leaves its loop after the same round. No runtime integer division (each one would cost ~20 instructions on the
// vector pipe): the host precomputes multiply-shift divisors (FastDiv) for the three launch-time divisors, and a
// full section, whose head count is a power of 2, divides by shifting.
__device__ __forceinline__ bool work_item(const Params& p, int cluster_id, int round, int& m_block, int& hkv,
                                          int& batch_id) {
  const int section = int(fast_div(uint32_t(round), p.div_rounds_per_section));            // round / rounds_per_section
  const int col = (round % 2 == 0) ? cluster_id : p.num_clusters - 1 - cluster_id;          // the snake
  const int pos = (round - section * p.rounds_per_section) * p.num_clusters + col;         // inside the section
  int heads, block, head;  // block = pos / heads (query blocks are dealt longest first), head = pos % heads
  if (section < p.full_sections) {  // a full section: heads_per_section heads, a power of 2
    heads = p.heads_per_section;
    block = pos >> p.heads_per_section_log2;
    head = pos - (block << p.heads_per_section_log2);
  } else {                           // the last, partial section
    heads = p.heads_last_section;
    block = int(fast_div(uint32_t(pos), p.div_heads_last_section));
    head = pos - block * heads;
  }
  if (pos >= heads * p.num_m_blocks) return false;                                          // a hole
  const int bh = section * p.heads_per_section + head;                                      // flattened (b, hkv)
  batch_id = int(fast_div(uint32_t(bh), p.div_hkv));
  hkv = bh - batch_id * p.Hkv;
  m_block = p.num_m_blocks - 1 - block;                                                     // longest first
  return true;
}

// Q-stage address math. CTA r, stage q of work item m owns packed rows m * 512 + (2 q + r) * 128 + [0, 128), i.e.
// query positions from pos0 (32 of them, x 4 heads): packed row = 4 (q_pos - pos0) + h_in.
namespace q_stage {

__device__ __forceinline__ int pos0(int m_block, int cta_rank, int q) {
  return (m_block * (CTA_GROUP * STAGES) + CTA_GROUP * q + cta_rank) * POS_PER_STAGE;
}

__device__ __forceinline__ int q_smem(int Q_smem, int q) {
  return Q_smem + q * Q_TILE_BYTES;
}

__device__ __forceinline__ int o_smem(int O_smem, int q) {
  return O_smem + q * Q_TILE_BYTES;
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

// Q stage q of this CTA: 128 packed rows (32 query positions x 4 heads) x 128 head-dim columns, two 16 KB boxes.
// Q is the pair QK MMA's K-major A operand (M = 256: each CTA supplies its own rows).
__device__ __forceinline__ void load_q(int Q_smem, int q, int mbar_q_ready_base, const CUtensorMap* Q_tmap,
                                       int m_block, int hkv, int batch_id, int cta_rank) {
  const int dst = q_smem(Q_smem, q);
  const int mbar_ready = mbar_addr(mbar_q_ready_base, q);
  const int mbar_leader = cluster::map_shared_rank(mbar_ready, 0);
  const int pos = pos0(m_block, cta_rank, q);
  if (cta_rank == 0) {
    mbarrier::arrive_expect_tx(mbar_ready, CTA_GROUP * Q_TILE_BYTES);
  }
  tma::copy_5d_gmem_to_smem_2cta(dst, Q_tmap, 0, 0, pos, hkv, batch_id, mbar_leader);
  tma::copy_5d_gmem_to_smem_2cta(dst + PANEL_BYTES, Q_tmap, SW128_ATOM_COLS, 0, pos, hkv, batch_id, mbar_leader);
}

}  // namespace q_stage

// The KV ring: 6 stages of 16 KB, round-robin for K and V alike, in the order the MMA warp drains them. The loader
// and the MMA warp each keep a cursor (stage, phase); the phase flips each time the ring wraps.
namespace kv_ring {

__device__ __forceinline__ int stage_smem_addr(int KV_smem, int stage) {
  return KV_smem + stage * STAGE_BYTES;
}

__device__ __forceinline__ int stage_mbar_addr(int base, int stage) {
  return base + stage * int(sizeof(uint64_t));
}

__device__ __forceinline__ void advance(int& stage, int& phase) {
  if (++stage == STAGES) {
    stage = 0;
    phase ^= 1;
  }
}

// This CTA's half of K tile kv_tile: 64 of its 128 keys, [128 kv_tile + 64 rank, +64), x all 128 head-dim columns,
// as two 8 KB boxes (d 0-63 and 64-127). K is the pair QK MMA's K-major B operand; the MMA splits N (= keys)
// between the CTAs.
__device__ __forceinline__ void load_k_half(int KV_smem, int stage, int mbar_kv_ready_base, const CUtensorMap* K_tmap,
                                            int kv_tile, int hkv, int batch_id, int cta_rank) {
  const int dst = stage_smem_addr(KV_smem, stage);
  const int mbar_ready = stage_mbar_addr(mbar_kv_ready_base, stage);
  const int mbar_leader = cluster::map_shared_rank(mbar_ready, 0);
  const int key0 = kv_tile * BLOCK_N + K_HALF_KEYS * cta_rank;
  if (cta_rank == 0) {
    mbarrier::arrive_expect_tx(mbar_ready, CTA_GROUP * STAGE_BYTES);
  }
  tma::copy_4d_gmem_to_smem_2cta(dst, K_tmap, 0, key0, hkv, batch_id, mbar_leader);
  tma::copy_4d_gmem_to_smem_2cta(dst + K_PANEL_BYTES, K_tmap, SW128_ATOM_COLS, key0, hkv, batch_id, mbar_leader);
}

// This CTA's half of V tile kv_tile: 64 of V's 128 head-dim columns, [64 rank, +64), x all 128 keys, one 16 KB
// box. V is the pair PV MMA's MN-major B operand (a smem row = one key); the MMA splits N (= head-dim columns).
__device__ __forceinline__ void load_v_half(int KV_smem, int stage, int mbar_kv_ready_base, const CUtensorMap* V_tmap,
                                            int kv_tile, int hkv, int batch_id, int cta_rank) {
  const int dst = stage_smem_addr(KV_smem, stage);
  const int mbar_ready = stage_mbar_addr(mbar_kv_ready_base, stage);
  const int mbar_leader = cluster::map_shared_rank(mbar_ready, 0);
  const int d0 = V_HALF_COLS * cta_rank;
  const int key0 = kv_tile * BLOCK_N;
  if (cta_rank == 0) {
    mbarrier::arrive_expect_tx(mbar_ready, CTA_GROUP * STAGE_BYTES);
  }
  tma::copy_4d_gmem_to_smem_2cta(dst, V_tmap, d0, key0, hkv, batch_id, mbar_leader);
}

}  // namespace kv_ring

// MMA instruction descriptors, kernel 17's bits with M = 256 (the pair's rows):
//   dtype fp32 (bit 4), atype bf16 (bit 7), btype bf16 (bit 10), MMA_N / 8 at bit 17, MMA_M / 16 at bit 24.
//   PV's B (V) is MN-major (bit 16).
constexpr uint32_t i_desc_qk = (1U << 4) | (1U << 7) | (1U << 10) | ((BLOCK_N >> 3) << 17) |
                               (((CTA_GROUP * BLOCK_M) >> 4) << 24);
constexpr uint32_t i_desc_pv = (1U << 4) | (1U << 7) | (1U << 10) | (1U << 16) | ((HEAD_DIM >> 3) << 17) |
                               (((CTA_GROUP * BLOCK_M) >> 4) << 24);
static_assert(i_desc_qk == 0x10200490u && i_desc_pv == 0x10210490u, "the instruction descriptors this kernel was tuned with");

// S_s = Q_s K^T over d = 0..127: kernel 17's issue_qk_mma as a pair MMA. One MMA computes a 256-row tile of S,
// rows 0-127 CTA 0's, 128-255 CTA 1's; the B panel step is this CTA's 64 keys (8 KB), not 128.
__device__ __forceinline__ void issue_qk_mma(int taddr_s_stage, int Q_stage_smem, int K_stage_smem) {
  // outer loop selects the SW128 64-wide panel
  constexpr uint32_t qk_desc_micro_step = MMA_K_BYTES / 16;
  constexpr uint32_t qk_a_desc_panel_step = (BLOCK_M * SW128_BYTES) / 16;
  constexpr uint32_t qk_b_desc_panel_step = (kv_ring::K_HALF_KEYS * SW128_BYTES) / 16;
  uint32_t a_desc_panel = swizzle128::desc_lo_kmajor(Q_stage_smem);
  uint32_t b_desc_panel = swizzle128::desc_lo_kmajor(K_stage_smem);
  // opaque (an empty asm that claims to change them): otherwise the compiler keeps a result live across the
  // softmax steps, or proves the low descriptor bits zero (smem is 1024-aligned) and turns the adds into ORs
  asm volatile("" : "+r"(a_desc_panel), "+r"(b_desc_panel));
  for (int k1 = 0; k1 < HEAD_DIM / SW128_ATOM_COLS; ++k1) {
    uint32_t a_desc = a_desc_panel;
    uint32_t b_desc = b_desc_panel;
    // inner loop selects the MMA_K = 16 chunk inside that panel
    for (int k2 = 0; k2 < SW128_ATOM_COLS / MMA_K; ++k2) {
      const int enable_input_d = (k1 == 0 && k2 == 0) ? 0 : 1;  // the first MMA overwrites S_s
      tcgen05::mma_f16_pair(taddr_s_stage, a_desc, b_desc, i_desc_qk, enable_input_d);
      a_desc += qk_desc_micro_step;
      b_desc += qk_desc_micro_step;
    }
    a_desc_panel += qk_a_desc_panel_step;
    b_desc_panel += qk_b_desc_panel_step;
  }
}

// O_s (+)= P_s V over microsteps [first_microstep, end_microstep) of 16 keys: A = P_s in TMEM (8 packed columns
// = 16 bf16 per microstep), B = this CTA's V half (MN-major, +16 keys = +2 KB per microstep). Kernel 17's, with
// the 64-column V half.
__device__ __forceinline__ void issue_pv_mma_range(int taddr_o_stage, int taddr_p_stage, int V_stage_smem,
                                                   int first_microstep, int end_microstep, bool initialize_o) {
  constexpr int pv_v_chunk_bytes = kv_ring::V_HALF_COLS * MMA_K_BYTES;  // 2 KB: 16 keys x 128 B
  constexpr uint32_t pv_desc_chunk_step = pv_v_chunk_bytes / 16;
  // offset V by the range's first P microstep so P columns match the corresponding V rows
  uint32_t b_desc = swizzle128::desc_lo_mnmajor(V_stage_smem + first_microstep * pv_v_chunk_bytes);
  asm volatile("" : "+r"(taddr_o_stage), "+r"(taddr_p_stage), "+r"(b_desc));  // opaque, as in issue_qk_mma
  for (int k = first_microstep; k < end_microstep; ++k) {
    const int taddr_a = taddr_p_stage + k * 8;
    // disabling input D makes the first issued microstep initialize O
    const int enable_input_d = (initialize_o && k == first_microstep) ? 0 : 1;
    tcgen05::mma_f16_pair_a_tmem(taddr_o_stage, taddr_a, b_desc, i_desc_pv, enable_input_d);
    b_desc += pv_desc_chunk_step;
  }
}

// Causal mask of the diagonal tile: column c survives iff c < col_limit (>= 1: the diagonal key is always visible).
// Per 32-column fragment k, keep = 0xFFFFFFFF >> max(32 (k + 1) - col_limit, 0) has bit c set iff column 32 k + c
// survives; the 32 bit tests become one predicate move plus one select per column (a plain `c >= col_limit` per
// column costs 128 compares).
__device__ __forceinline__ void apply_causal_mask(float* s, int col_limit) {
#pragma unroll
  for (int k = 0; k < BLOCK_N / 32; ++k) {
    const int sh = max(32 * (k + 1) - col_limit, 0);
    const uint32_t keep = bits::shr_clamp(0xFFFFFFFFu, sh);
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      if ((keep & (1u << c)) == 0u) s[32 * k + c] = -INFINITY;
    }
  }
}
// Row max, FA4's fmax_reduce: four chains, each folding two new values per step (one FMNMX3 each)
__device__ __forceinline__ float row_max_128(const float* s) {
  float m0 = fmaxf(s[0], s[1]);
  float m1 = fmaxf(s[2], s[3]);
  float m2 = fmaxf(s[4], s[5]);
  float m3 = fmaxf(s[6], s[7]);
#pragma unroll
  for (int c = 8; c < BLOCK_N; c += 8) {
    m0 = fmaxf(fmaxf(m0, s[c]), s[c + 1]);
    m1 = fmaxf(fmaxf(m1, s[c + 2]), s[c + 3]);
    m2 = fmaxf(fmaxf(m2, s[c + 4]), s[c + 5]);
    m3 = fmaxf(fmaxf(m3, s[c + 6]), s[c + 7]);
  }
  return fmaxf(fmaxf(fmaxf(m0, m1), m2), m3);
}

// x = s * scale - basis for elements [begin, end), in place: one packed FFMA2 per pair
__device__ __forceinline__ void scale_and_subtract_basis(float* v, int begin, int end, float2 scale2,
                                                         float2 neg_basis2) {
#pragma unroll
  for (int e = begin; e < end; e += 2) {
    const float2 x = __ffma2_rn(make_float2(v[e], v[e + 1]), scale2, neg_basis2);
    v[e] = x.x;
    v[e + 1] = x.y;
  }
}

// p = exp2(x) for the whole 32-column fragments in [begin, end), in place: MUFU, or the FMA-pipe emulation for
// columns 8-11 and 20-23 of fragments 0-2 (FA4's ex2_emu 12 / 4 from fragment 0: 24 of the 128 columns). Then each
// fragment is packed into bf16 pairs, p_packed[i] = {p[2 i] (low half), p[2 i + 1] (high half)}: one 32-bit TMEM
// cell of the PV MMA's A operand.
__device__ __forceinline__ void exp2_and_pack(float* v, int begin, int end, uint32_t* p_packed) {
#pragma unroll
  for (int j = begin / 32; j < end / 32; ++j) {
#pragma unroll
    for (int k = 0; k < 32; k += 2) {
      const int e = 32 * j + k;
      const bool use_software_exp2 = (j < 3) && (k % 12 >= 8);
      if (use_software_exp2) {
        approx::e2e_exp2_pair_capped(v[e], v[e + 1], v[e], v[e + 1]);
      } else {
        v[e] = approx::fast_exp2(v[e]);
        v[e + 1] = approx::fast_exp2(v[e + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 32; k += 2) {
      const int e = 32 * j + k;
      p_packed[e / 2] = bf16::pack2_to_u32(v[e], v[e + 1]);
    }
  }
}

// Row sum, FA4's fadd_reduce: four packed accumulators; accumulator a adds elements 8 r + 2 a and 8 r + 2 a + 1
// of [begin, end)
__device__ __forceinline__ void add_to_rowsum(float2* sum, const float* p, int begin, int end) {
#pragma unroll
  for (int e = begin; e < end; e += 8) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      sum[a] = __fadd2_rn(sum[a], make_float2(p[e + 2 * a], p[e + 2 * a + 1]));
    }
  }
}

// One KV tile of one softmax row: S_q (TMEM) -> P_q (TMEM, over S_q) against the frozen basis, the row sum
// running. first_kv = the diagonal tile (visited first): apply the causal mask, take the row max, freeze the
// basis. Called once with first_kv = true and then in a loop with false: after inlining each call compiles
// its own instance, so the steady step carries neither the mask nor its branch.
// The row is transformed in separate passes (scale, exp2 + pack, row sum), in this order on purpose: the compiler
// follows source order where it has a choice, and this order publishes P[:, 0:64] early in the exp2 stream (a
// fused per-pair spelling publishes it later).
__device__ __forceinline__ void softmax_step(bool first_kv, int taddr_s_row, int taddr_p_row,
                                             int mbar_s_ready_addr, int mbar_p_lo_ready_addr,
                                             int mbar_p_full_ready_addr, bool is_leader, int col_limit,
                                             float softmax_scale_log2, float lift, int& phase_s_ready,
                                             float& frozen_neg_basis, float& rowmax_lifted, float& rowsum) {

  // 1. S_q is in TMEM: load the whole row, 128 scores in four x32 loads. No wait::ld here: each register's
  //    first use waits for it.
  mbarrier::wait(mbar_s_ready_addr, phase_s_ready);
  phase_s_ready ^= 1;
  float v[BLOCK_N];  // the row, transformed in place: scores s, then x = s * scale - basis, then p = exp2(x)
  tcgen05::ld_32x32b_x32_nowait(taddr_s_row + 0, v + 0);
  tcgen05::ld_32x32b_x32_nowait(taddr_s_row + 32, v + 32);
  tcgen05::ld_32x32b_x32_nowait(taddr_s_row + 64, v + 64);
  tcgen05::ld_32x32b_x32_nowait(taddr_s_row + 96, v + 96);

  // 2. the basis. Diagonal tile: mask, rowmax, freeze. Later tiles: reuse.
  float neg_basis;  // -basis for this tile
  if (first_kv) {
    apply_causal_mask(v, col_limit);
    const float m = row_max_128(v);
    // (m + lift) * scale in FA4's rounding order: add in score units, then scale. m = -inf (the diagonal key is
    // always visible) means +-inf in Q / K or an fp32 overflow: the frozen basis stays -inf, so on every later
    // tile p is +inf, NaN or 2^100 (the emulation's cap) and the certificate trips. This tile itself then uses
    // basis 96 (m_safe = 0): its sum is 0, so a row with no later tile gets zeros (known limit 2; FA4's
    // convention). NaN scores: fmaxf skips them, so m is the max of the others (or -inf / NaN when there are
    // none), and the NaN p trips the row either way. For finite m the two bases are the same expression.
    // Spelled with the rounding intrinsics so the compiler keeps both chains inside this step instead of
    // hoisting a negated scale out of the work-item loop (which delays the diagonal tile's P hand-off).
    const float m_safe = (m != -INFINITY) ? m : 0.0f;
    rowmax_lifted = __fadd_rn(m, lift);  // -inf stays -inf
    neg_basis = __fsub_rn(0.0f, __fmul_rn(__fadd_rn(m_safe, lift), softmax_scale_log2));
    frozen_neg_basis = __fsub_rn(0.0f, __fmul_rn(rowmax_lifted, softmax_scale_log2));
    asm volatile("" : "+f"(frozen_neg_basis));  // opaque: kept in a register, not recomputed every tile
  } else {
    neg_basis = frozen_neg_basis;
  }
  const float2 scale2 = make_float2(softmax_scale_log2, softmax_scale_log2);
  const float2 neg_basis2 = make_float2(neg_basis, neg_basis);
  uint32_t p_packed[BLOCK_N / 2];  // the row's 128 P values as bf16 pairs: 64 b32 TMEM cells

  // 3. First half, columns 0-63: x = s * scale - basis, p = exp2(x), packed. P[:, 0:64] is published first
  //    (p_lo_ready), so the MMA warp starts PV's microsteps 0-3 while the second half is still being computed
  //    (FA4's split_P_arrive, at 64 instead of 96).
  scale_and_subtract_basis(v, 0, P_SPLIT, scale2, neg_basis2);
  exp2_and_pack(v, 0, P_SPLIT, p_packed);
  // its row sum; after the diagonal tile, accumulator 0 also carries the previous tiles' sum
  float2 sum[4] = {make_float2(v[0], v[1]), make_float2(v[2], v[3]), make_float2(v[4], v[5]),
                   make_float2(v[6], v[7])};
  if (!first_kv) sum[0] = __fadd2_rn(make_float2(rowsum, 0.0f), sum[0]);
  add_to_rowsum(sum, v, 8, P_SPLIT);
  // Publish P[:, 0:64] = TMEM cells S_q + 64 + [0, 32), over S columns whose scores are not all used yet. The ISA
  // orders a warp's tcgen05.ld before its later tcgen05.st (a pipelined pair); the wait::ld is the lineage's.
  // Then wait::st (this thread's P stores are complete) and __syncwarp (so are every lane's) before the arrive.
  // Formal gap, as in FA4: the ISA's cross-thread pattern also puts tcgen05.fence::before_thread_sync before the
  // arrive and fence::after_thread_sync after the MMA warp's wait; neither is here (FA4 omits them too, and they
  // cost). In the machine code the P store, an async-proxy fence and the arrive come in that order.
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
  tcgen05::st_32x32b_x16_u32(taddr_p_row, p_packed);            // P elements 0-31
  tcgen05::st_32x32b_x16_u32(taddr_p_row + 16, p_packed + 16);  // P elements 32-63
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
  __syncwarp();
  {
    // p_lo_ready counts 256 = every row of both CTAs: the leader's threads arrive one by one, in the peer one
    // elected lane per warp arrives with count 32 (4 remote messages instead of 128). One arrive under one
    // combined condition (a separate branch for the peer's lets the compiler sink that arrive far down the step).
    const bool elected = elect_sync() != 0u;
    if (is_leader || elected) {
      mbarrier::arrive_cluster(mbar_p_lo_ready_addr, is_leader ? 1 : WARP_SIZE);
    }
  }

  // 4. Second half, columns 64-127 (fragment 3 is all MUFU). p_full_ready counts 8 = one lane per warp, both CTAs.
  scale_and_subtract_basis(v, P_SPLIT, BLOCK_N, scale2, neg_basis2);
  exp2_and_pack(v, P_SPLIT, BLOCK_N, p_packed);
  tcgen05::st_32x32b_x16_u32(taddr_p_row + 32, p_packed + 32);  // P elements 64-95
  tcgen05::st_32x32b_x16_u32(taddr_p_row + 48, p_packed + 48);  // P elements 96-127
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
  __syncwarp();
  if (elect_sync()) {
    mbarrier::arrive_cluster(mbar_p_full_ready_addr, 1);
  }
  // the rest of the row sum: each accumulator adds in the same order as one pass over 8-127, so the same bits;
  // the compiler moves most of these adds above the arrive, between the exp2s
  add_to_rowsum(sum, v, P_SPLIT, BLOCK_N);
  sum[0] = __fadd2_rn(sum[0], sum[1]);
  sum[2] = __fadd2_rn(sum[2], sum[3]);
  sum[0] = __fadd2_rn(sum[0], sum[2]);
  rowsum = __fadd_rn(sum[0].x, sum[0].y);
}

}  // namespace

__global__ void __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(TB_SIZE, 1)
attention_tcgen05_causal_gqa_2cta_kernel(const __grid_constant__ CUtensorMap Q_tmap,
                                         const __grid_constant__ CUtensorMap K_tmap,
                                         const __grid_constant__ CUtensorMap V_tmap,
                                         const __grid_constant__ CUtensorMap O_tmap, const Params prm) {
  const int tid = threadIdx.x;
  // a shuffle from lane 0 makes the warp index provably warp-uniform: every role branch below is uniform, so the
  // MMA warp's addresses and descriptors stay on the uniform datapath (with a plain tid / 32 the roles compile as
  // divergent code: a vector -> uniform move before every MMA)
  const int warp_id = __shfl_sync(0xffffffffu, tid / WARP_SIZE, 0);
  const int lane_id = tid % WARP_SIZE;
  const int cta_rank = blockIdx.x % CTA_GROUP;   // rank in the pair (== %cluster_ctarank for cluster dims (2, 1, 1))
  const int cluster_id = blockIdx.x / CTA_GROUP;  // the pair's column in the snake (same in both CTAs)
  const bool is_leader = (cta_rank == 0);

  const bool is_softmax_warp = (warp_id < SOFTMAX_WARP_COUNT);
  const bool is_mma_warp = (warp_id == MMA_WARP_ID);
  const bool is_store_warp = (warp_id == STORE_WARP_ID);
  const bool is_load_warp = (warp_id == LOAD_WARP_ID);

  extern __shared__ __align__(1024) char smem_raw[];

  const int smem_base = smem_addr_of(smem_raw);
  const int O_smem = smem_base;                                 // sO[2]: bf16 O, softmax writes, the TMA store reads
  const int Q_smem = O_smem + q_stage::STAGES * Q_TILE_BYTES;   // sQ[2]: this CTA's 128 Q rows of each stage
  const int KV_smem = Q_smem + q_stage::STAGES * Q_TILE_BYTES;  // the ring: 6 x 16 KB halves of K or V tiles

  // Every barrier has one concrete producer and consumer. Where it lives:
  //   leader = only CTA 0's copy is used (both CTAs' TMA bytes and arrives land there)
  //   pair   = each CTA's own copy; the leader's MMA commit is multicast into both
  //   local  = each CTA's own copy
  //   1+tx   = the leader's one arrive.expect_tx (for both CTAs' bytes), then the bytes
  //                                   copy    count  producer -> consumer
  //   q_ready[q]:                     leader  1+tx   Q TMA loads (both CTAs) -> MMA
  //   q_free[q]:                      pair    1      the item's last QK_q -> loader's next Q_q overwrite
  //   kv_ready[stage]:                leader  1+tx   K / V TMA loads (both CTAs) -> MMA
  //   kv_free[stage]:                 pair    1      the stage's last MMA reader -> loader's next overwrite
  //   s_ready[q]:                     pair    1      QK_q -> softmax's S loads
  //   p_lo_ready[q]:                  leader  256    every row of both CTAs wrote P_q[:, 0:64] -> PV lo
  //   p_full_ready[q]:                leader  8      one lane per warp, both CTAs: all of P_q is written -> PV hi
  //   o_ready[q]:                     pair    1      the item's last PV_q -> softmax's O readout
  //   so_ready[q]:                    local   128    softmax wrote sO_q -> store warp
  //   so_free[q]:                     local   1      the TMA store has read sO_q -> softmax's next epilogue
  //   tmem_dealloc:                   local   1      the peer's MMA warp is done with TMEM -> our dealloc
  // No tcgen05.fence::after_thread_sync after the waits below and none around the P arrives (kernel 17 has them
  // after every wait): FA4 omits them, and adding them costs 1-2 %. What orders each hand-off is the wait::ld /
  // wait::st before the arrive.
  __shared__ uint64_t mbar_q_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_q_free[q_stage::STAGES];
  __shared__ uint64_t mbar_kv_ready[kv_ring::STAGES];
  __shared__ uint64_t mbar_kv_free[kv_ring::STAGES];
  __shared__ uint64_t mbar_s_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_p_lo_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_p_full_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_o_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_so_ready[q_stage::STAGES];
  __shared__ uint64_t mbar_so_free[q_stage::STAGES];
  __shared__ uint64_t mbar_tmem_dealloc[1];
  __shared__ int tmem_addr[1];

  const int mbar_q_ready_base = smem_addr_of(mbar_q_ready);
  const int mbar_q_free_base = smem_addr_of(mbar_q_free);
  const int mbar_kv_ready_base = smem_addr_of(mbar_kv_ready);
  const int mbar_kv_free_base = smem_addr_of(mbar_kv_free);
  const int mbar_s_ready_base = smem_addr_of(mbar_s_ready);
  const int mbar_p_lo_ready_base = smem_addr_of(mbar_p_lo_ready);
  const int mbar_p_full_ready_base = smem_addr_of(mbar_p_full_ready);
  const int mbar_o_ready_base = smem_addr_of(mbar_o_ready);
  const int mbar_so_ready_base = smem_addr_of(mbar_so_ready);
  const int mbar_so_free_base = smem_addr_of(mbar_so_free);
  const int mbar_tmem_dealloc_addr = smem_addr_of(mbar_tmem_dealloc);

  if (tid == 0) {
    tma::prefetch_descriptor(&Q_tmap);
    tma::prefetch_descriptor(&K_tmap);
    tma::prefetch_descriptor(&V_tmap);
    tma::prefetch_descriptor(&O_tmap);
    // Every barrier is initialized once; barriers and their phase cursors persist across work items.
    for (int q = 0; q < q_stage::STAGES; ++q) {
      mbarrier::init(q_stage::mbar_addr(mbar_q_ready_base, q), 1);
      mbarrier::init(q_stage::mbar_addr(mbar_q_free_base, q), 1);
      mbarrier::init(q_stage::mbar_addr(mbar_s_ready_base, q), 1);
      mbarrier::init(q_stage::mbar_addr(mbar_p_lo_ready_base, q), CTA_GROUP * SOFTMAX_WARPS_PER_STAGE * WARP_SIZE);
      mbarrier::init(q_stage::mbar_addr(mbar_p_full_ready_base, q), CTA_GROUP * SOFTMAX_WARPS_PER_STAGE);
      mbarrier::init(q_stage::mbar_addr(mbar_o_ready_base, q), 1);
      mbarrier::init(q_stage::mbar_addr(mbar_so_ready_base, q), SOFTMAX_WARPS_PER_STAGE * WARP_SIZE);
      mbarrier::init(q_stage::mbar_addr(mbar_so_free_base, q), 1);
    }
    for (int st = 0; st < kv_ring::STAGES; ++st) {
      mbarrier::init(kv_ring::stage_mbar_addr(mbar_kv_ready_base, st), 1);
      mbarrier::init(kv_ring::stage_mbar_addr(mbar_kv_free_base, st), 1);
    }
    mbarrier::init(mbar_tmem_dealloc_addr, 1);
    // cluster scope: the peer's TMA bytes, commits and arrives also land on our barriers
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // Both CTAs are running and both CTAs' barriers are initialized before any remote arrive, TMA byte count,
  // multicast commit or the pair TMEM alloc (which writes into the peer's shared memory) can reach them. Relaxed is
  // enough: fence.mbarrier_init.release.cluster already published the inits. (FA4's order: cluster wait, then alloc.)
  __cluster_barrier_arrive_relaxed();
  __cluster_barrier_wait();
  if (is_mma_warp) {
    // one warp in EACH CTA allocates, collectively for the pair: each CTA gets all 512 columns of its own TMEM
    const int tmem_smem_addr = smem_addr_of(tmem_addr);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :
                 : "r"(tmem_smem_addr), "r"(TMEM_ALLOC_COLS)
                 : "memory");
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
  }
  __syncthreads();  // publishes tmem_addr[0]
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

  const int taddr_base = tmem_addr[0];
  const int taddr_s_base = taddr_base;
  const int taddr_o_base = taddr_s_base + q_stage::STAGES * TMEM_S_COLS;

  // Register split: the softmax warpgroups grow, warpgroup 2 shrinks (all four warps, the idle one included). Each
  // setmaxnreg sits at the top of its own branch so it comes before that branch's code whatever the compiler does
  // with the branches.
  if (is_softmax_warp) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(SOFTMAX_REGS) : "memory");

    // Softmax warps: one 4-warp group per Q stage, each warp 32 of its 128 rows. One thread = one row = one TMEM
    // lane of S, P and O.
    const int softmax_stage = warp_id / SOFTMAX_WARPS_PER_STAGE;
    const int local_warp_idx = warp_id % SOFTMAX_WARPS_PER_STAGE;
    const int row_base = local_warp_idx * WARP_SIZE;  // first of the 32 rows owned by this warp
    const int row = row_base + lane_id;               // the row owned by this lane
    const int trow = row_base << 16;                  // row_base in the lane field of a TMEM address
    // phase cursors persist across items (a wait returns once the barrier completed the phase with that parity):
    int phase_s_ready = 0;  // flips once per KV tile
    int phase_o_ready = 0;  // flips once per work item
    int phase_so_free = 1;  // flips once per work item. Starts at 1: a fresh barrier counts the phase before phase 0
                            // as complete, so the first item finds sO_q free without waiting.
#pragma unroll 1
    for (int round = 0; round < prm.num_rounds; ++round) {
      int m_block, hkv, batch_id;
      if (!work_item(prm, cluster_id, round, m_block, hkv, batch_id)) continue;  // a hole: sit this round out
      int kv_tiles = m_block + 1;  // the diagonal tile and everything left of it (Sq == Skv)
      asm volatile("" : "+r"(kv_tiles));  // opaque: else the tile loop is folded into a test on m_block

      // This warp's stage-local resources. Inside the item loop on purpose: hoisted above it, these values stay
      // live across every softmax step and change the register allocation.
      const int mbar_s_ready_addr = q_stage::mbar_addr(mbar_s_ready_base, softmax_stage);
      // P barriers are the LEADER's (its pair MMA reads both CTAs' P): cluster addresses
      const int mbar_p_lo_ready_addr = cluster::map_shared_rank(q_stage::mbar_addr(mbar_p_lo_ready_base, softmax_stage), 0);
      const int mbar_p_full_ready_addr = cluster::map_shared_rank(q_stage::mbar_addr(mbar_p_full_ready_base, softmax_stage), 0);
      const int taddr_s_row = q_stage::tmem_s(taddr_s_base, softmax_stage) + trow;
      const int taddr_p_row = taddr_s_row + TMEM_P_COL_OFFSET;
      const float softmax_scale_log2 = prm.softmax_scale_log2;
      // (rowmax + lift) * scale = rowmax * scale + 96; __fdiv_rn = IEEE division whatever the build flags
      const float lift = __fdiv_rn(BASIS_LIFT_LOG2, softmax_scale_log2);

      // this row's query position and head (row = 4 (q_pos - pos0) + h_in). On the diagonal tile (visited first,
      // the only masked one) key j is visible iff j <= q_pos's position inside the tile: columns [0, col_limit).
      // row = 4 (q_pos - pos0) + h_in, so the position is row >> 2 (a shift, not `/ 4`: row is an int, and the
      // compiler lowers a signed division with a sign fix-up in 16-bit registers)
      const int q_pos = q_stage::pos0(m_block, cta_rank, softmax_stage) + (row >> 2);
      // col_limit = min(Sq - 128 m_block, q_pos + 1 - 128 m_block) >= 1 (the diagonal key is always visible). The
      // first bound never binds (q_pos < Sq), but written as this min the compiler keeps col_limit as one value
      // instead of folding its terms into the mask's four shift counts.
      const int lim_seq = prm.Sq - m_block * BLOCK_N;
      const int lim_causal = q_pos + 1 - m_block * BLOCK_N;
      const int col_limit = lim_causal < lim_seq ? lim_causal : lim_seq;

      // Frozen basis per row, reset for every work item: the diagonal tile sets it, later tiles reuse it.
      // rowmax_lifted and rowsum are what the certificate reads after the last tile.
      float frozen_neg_basis = 0.0f;
      float rowmax_lifted = 0.0f;
      float rowsum = 0.0f;
      // the diagonal tile first (masked; it freezes the basis), then tiles kv_tiles - 2 ... 0 against it
      softmax_step(true, taddr_s_row, taddr_p_row, mbar_s_ready_addr, mbar_p_lo_ready_addr, mbar_p_full_ready_addr, is_leader,
                   col_limit, softmax_scale_log2, lift, phase_s_ready, frozen_neg_basis, rowmax_lifted, rowsum);
#pragma unroll 1
      for (int i = 1; i < kv_tiles; ++i) {
        softmax_step(false, taddr_s_row, taddr_p_row, mbar_s_ready_addr, mbar_p_lo_ready_addr, mbar_p_full_ready_addr, is_leader,
                   col_limit, softmax_scale_log2, lift, phase_s_ready, frozen_neg_basis, rowmax_lifted, rowsum);
      }

      // Certificate (fail closed): trip on rowsum >= 2^100 (some later score far above the diagonal tile's max),
      // NaN, or < 2^-98 for a row whose diagonal max was not -inf (its terms underflowed, or a huge |m| rounded
      // the basis away). A tripped row gets rowsum = inf (its LSE: +inf or NaN) and inv_denom = NaN, so its O is
      // written as NaN and the O-check counts it.
      float inv_denom;
      if (rowsum >= ROWSUM_TRIP_HIGH || rowsum != rowsum || (rowmax_lifted != -INFINITY && rowsum < ROWSUM_TRIP_LOW)) {
        rowsum = INFINITY;
        inv_denom = NAN;  // written as NaN, never zeros
      } else {
        inv_denom = approx::fast_rcp(rowsum == 0.0f ? 1.0f : rowsum);  // sum 0 = all -inf row: O = 0 (known limit 2)
      }

      mbarrier::wait(q_stage::mbar_addr(mbar_o_ready_base, softmax_stage), phase_o_ready);  // the item's last PV_q
      phase_o_ready ^= 1;
      mbarrier::wait(q_stage::mbar_addr(mbar_so_free_base, softmax_stage), phase_so_free);  // sO_q was stored
      phase_so_free ^= 1;

      // Epilogue: 8 x (16 O columns from TMEM, scale, O-check, bf16, two 16-byte swizzled smem stores). sO_q has
      // the layout of a Q stage: two SW128 panels of 64 columns, 16-byte chunk j of row r sits at chunk j ^ (r % 8)
      // (8 consecutive rows hit 8 different bank groups; the TMA store's SWIZZLE_128B undoes it).
      const int taddr_o_row = q_stage::tmem_o(taddr_o_base, softmax_stage) + trow;
      // Opaque copies of the row's smem address and swizzle (an empty asm that claims to change them): otherwise
      // the compiler hoists the item-invariant sO addresses out of the persistent loop and keeps them in 8
      // registers across every softmax step (and spills).
      int O_row_smem = q_stage::o_smem(O_smem, softmax_stage) + row * SW128_BYTES;
      int swz = row & 7;
      asm volatile("" : "+r"(O_row_smem), "+r"(swz));
      uint32_t maxbits = 0u;
      const float2 inv_denom2 = make_float2(inv_denom, inv_denom);
#pragma unroll
      for (int c = 0; c < HEAD_DIM / 16; ++c) {
        float o[16];
        tcgen05::ld_32x32b_x16_nowait(taddr_o_row + 16 * c, o);
        uint32_t o_packed[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float2 y = __fmul2_rn(make_float2(o[2 * i], o[2 * i + 1]), inv_denom2);  // one packed FMUL2
          maxbits = bits::abs_max2(y.x, y.y, maxbits);
          o_packed[i] = bf16::pack2_to_u32(y.x, y.y);
        }
        // columns 16 c .. 16 c + 15 = panel c / 4, chunks 2 (c % 4) and 2 (c % 4) + 1 of the row
        const int panel = O_row_smem + (c / 4) * PANEL_BYTES;
        const int j0 = 2 * (c % 4);
        st_shared_v4(panel + (((j0 + 0) ^ swz) << 4), o_packed[0], o_packed[1], o_packed[2], o_packed[3]);
        st_shared_v4(panel + (((j0 + 1) ^ swz) << 4), o_packed[4], o_packed[5], o_packed[6], o_packed[7]);
      }
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // the stores above -> visible to the TMA store
      // O-check: count the row if its output is inf or NaN (a tripped row, or an O that overflows bf16)
      if (maxbits >= BF16_INF_ABS_BITS && prm.counter != nullptr) {
        atomicAdd(prm.counter, 1.0f);
      }
      mbarrier::arrive(q_stage::mbar_addr(mbar_so_ready_base, softmax_stage));

      // LSE (optional), natural log: (basis + log2(rowsum)) * ln 2 with basis = rowmax_lifted * scale (one FMA)
      // (the position is re-derived from an opaque copy of m_block: else q_pos stays live across the softmax steps)
      int m_block_ep = m_block;
      asm volatile("" : "+r"(m_block_ep));
      const int q_pos_ep = q_stage::pos0(m_block_ep, cta_rank, softmax_stage) + (row >> 2);
      if (prm.lse != nullptr) {
        const float lse = rowsum == 0.0f ? -INFINITY
                                         : (rowmax_lifted * softmax_scale_log2 + approx::fast_log2(rowsum)) * LN2;
        const int hq = hkv * QHEADS_PER_KVHEAD + (row & (QHEADS_PER_KVHEAD - 1));  // h_in = row % 4
        prm.lse[(size_t(batch_id) * prm.Hq + hq) * prm.Sq + q_pos_ep] = lse;
      }
      // No drain barrier on O: the next item's first PV_q overwrites O_q only after the leader's p_lo_ready[q]
      // completes, and every stage-q thread of both CTAs arrives there only after its next softmax step waited
      // for all its TMEM loads (tcgen05.wait::ld), this epilogue's loads included.
    }
    // hands TMEM to the __syncthreads before the dealloc; every O value loaded above has been used
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");

  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(OTHER_REGS) : "memory");

    // MMA warp: the leader's copy issues every pair MMA. The whole warp runs the loop (waits are warp-wide, the
    // election is inside each MMA / commit asm), so its addresses and descriptors stay warp-uniform. (The peer's
    // MMA warp only takes part in the TMEM alloc / dealloc.)
    if (is_mma_warp && is_leader) {
      const int taddr_s0 = q_stage::tmem_s(taddr_s_base, 0);
      const int taddr_s1 = q_stage::tmem_s(taddr_s_base, 1);
      const int taddr_p0 = taddr_s0 + TMEM_P_COL_OFFSET;  // P_q overlays S_q
      const int taddr_p1 = taddr_s1 + TMEM_P_COL_OFFSET;
      const int taddr_o0 = q_stage::tmem_o(taddr_o_base, 0);
      const int taddr_o1 = q_stage::tmem_o(taddr_o_base, 1);
      const int Q_smem0 = q_stage::q_smem(Q_smem, 0);
      const int Q_smem1 = q_stage::q_smem(Q_smem, 1);
      // consumer phases start at 0; the ring cursor and all phases carry across work items
      int kv_stage = 0;
      int kv_phase = 0;
      int phase_p = 0;  // p_lo_ready and p_full_ready of both stages: one parity, flipped once per KV step
      int phase_q_ready = 0;
#pragma unroll 1
      for (int round = 0; round < prm.num_rounds; ++round) {
        int m_block, hkv, batch_id;
        if (!work_item(prm, cluster_id, round, m_block, hkv, batch_id)) continue;
        const int kv_tiles = m_block + 1;

        // KV tiles go diagonal first. K_i / V_i = the i-th tile visited; the load warp puts them in the ring in
        // the order K_0, V_0, K_1, V_1, ... Issue order is left to right, then top to bottom; each stage's QK runs
        // one KV tile ahead of its PV:
        //
        //                     stage 0                                stage 1
        //   prologue:                      QK_0(K_0)                             QK_1(K_0)
        //   step i:      PV_0(V_i)         QK_0(K_{i+1})       PV_1(V_i)         QK_1(K_{i+1})     (i < kv_tiles - 1)
        //   tail:        PV_0(V_last)                          PV_1(V_last)
        //
        // While softmax q turns S_q into P_q the tensor core works on the other stage. QK_q overwrites S_q (and
        // P_q with it) without an extra wait: softmax q read all of S_q before publishing P_q, and an MMA's A
        // reads are ordered before a later MMA's D writes from the same warp, so PV_q reads P_q before QK_q
        // writes. Same across work items.

        // prologue: S_0 and S_1 of the diagonal tile
        mbarrier::wait(q_stage::mbar_addr(mbar_q_ready_base, 0), phase_q_ready);
        mbarrier::wait(kv_ring::stage_mbar_addr(mbar_kv_ready_base, kv_stage), kv_phase);  // K_0
        const int K_stage_smem0 = kv_ring::stage_smem_addr(KV_smem, kv_stage);
        issue_qk_mma(taddr_s0, Q_smem0, K_stage_smem0);  // S_0 = Q_0 K_0^T
        tcgen05::commit_arrive_pair(q_stage::mbar_addr(mbar_s_ready_base, 0));
        mbarrier::wait(q_stage::mbar_addr(mbar_q_ready_base, 1), phase_q_ready);  // Q_1 lands after Q_0: wait only now
        issue_qk_mma(taddr_s1, Q_smem1, K_stage_smem0);  // S_1 = Q_1 K_0^T
        tcgen05::commit_arrive_pair(q_stage::mbar_addr(mbar_s_ready_base, 1));
        tcgen05::commit_arrive_pair(kv_ring::stage_mbar_addr(mbar_kv_free_base, kv_stage));  // both QKs read K_0
        kv_ring::advance(kv_stage, kv_phase);

        // steady state: one KV step per iteration. The six waits here carry no suspend-time hint on purpose: without
        // one there is no sleep-and-retry path, and the compiler hoists each wait's follow-up address / descriptor
        // math above it, so the warp issues the next MMA the moment it wakes. The prologue and the tail keep the hint.
#pragma unroll 1
        for (int i = 0; i < kv_tiles - 1; ++i) {
          mbarrier::wait_nohint(kv_ring::stage_mbar_addr(mbar_kv_ready_base, kv_stage), kv_phase);  // V_i
          const int v_stage = kv_stage;
          const int V_stage_smem = kv_ring::stage_smem_addr(KV_smem, v_stage);
          const bool initialize_o = (i == 0);  // step 0's PVs overwrite O_q (safe across items: "No drain barrier on O")

          // (1) PV_0(V_i): O_0 (+)= P_0 V_i. Microsteps 0-3 need only P_0[:, 0:64], 4-7 need P_0[:, 64:128].
          mbarrier::wait_nohint(q_stage::mbar_addr(mbar_p_lo_ready_base, 0), phase_p);
          issue_pv_mma_range(taddr_o0, taddr_p0, V_stage_smem, 0, PV_SPLIT_MICROSTEP, initialize_o);
          mbarrier::wait_nohint(q_stage::mbar_addr(mbar_p_full_ready_base, 0), phase_p);
          issue_pv_mma_range(taddr_o0, taddr_p0, V_stage_smem, PV_SPLIT_MICROSTEP, BLOCK_N / MMA_K, false);
          kv_ring::advance(kv_stage, kv_phase);

          // (2) QK_0(K_{i+1}): softmax 0's next scores
          mbarrier::wait_nohint(kv_ring::stage_mbar_addr(mbar_kv_ready_base, kv_stage), kv_phase);  // K_{i+1}
          const int k_stage = kv_stage;
          const int K_stage_smem = kv_ring::stage_smem_addr(KV_smem, k_stage);
          issue_qk_mma(taddr_s0, Q_smem0, K_stage_smem);
          tcgen05::commit_arrive_pair(q_stage::mbar_addr(mbar_s_ready_base, 0));

          // (3) PV_1(V_i), split the same way. PV_1 is V_i's last reader: release the stage.
          mbarrier::wait_nohint(q_stage::mbar_addr(mbar_p_lo_ready_base, 1), phase_p);
          issue_pv_mma_range(taddr_o1, taddr_p1, V_stage_smem, 0, PV_SPLIT_MICROSTEP, initialize_o);
          mbarrier::wait_nohint(q_stage::mbar_addr(mbar_p_full_ready_base, 1), phase_p);
          issue_pv_mma_range(taddr_o1, taddr_p1, V_stage_smem, PV_SPLIT_MICROSTEP, BLOCK_N / MMA_K, false);
          tcgen05::commit_arrive_pair(kv_ring::stage_mbar_addr(mbar_kv_free_base, v_stage));

          // (4) QK_1(K_{i+1}). QK_1 is K_{i+1}'s last reader: release the stage.
          issue_qk_mma(taddr_s1, Q_smem1, K_stage_smem);
          tcgen05::commit_arrive_pair(q_stage::mbar_addr(mbar_s_ready_base, 1));
          tcgen05::commit_arrive_pair(kv_ring::stage_mbar_addr(mbar_kv_free_base, k_stage));
          kv_ring::advance(kv_stage, kv_phase);
          phase_p ^= 1;
        }
        // every QK of the item is issued: Q_0 and Q_1 go back to the load warp once those QKs complete
        tcgen05::commit_arrive_pair(q_stage::mbar_addr(mbar_q_free_base, 0));
        tcgen05::commit_arrive_pair(q_stage::mbar_addr(mbar_q_free_base, 1));

        // tail: PV_0(V_last), PV_1(V_last); after each, o_ready[q] tells softmax q that O_q is final
        mbarrier::wait(kv_ring::stage_mbar_addr(mbar_kv_ready_base, kv_stage), kv_phase);  // V_last
        const int v_stage = kv_stage;
        const int V_stage_smem = kv_ring::stage_smem_addr(KV_smem, v_stage);
        const bool initialize_o = (kv_tiles == 1);  // a one-tile item has no steady step: its PV overwrites O_q
        mbarrier::wait(q_stage::mbar_addr(mbar_p_lo_ready_base, 0), phase_p);
        issue_pv_mma_range(taddr_o0, taddr_p0, V_stage_smem, 0, PV_SPLIT_MICROSTEP, initialize_o);
        mbarrier::wait(q_stage::mbar_addr(mbar_p_full_ready_base, 0), phase_p);
        issue_pv_mma_range(taddr_o0, taddr_p0, V_stage_smem, PV_SPLIT_MICROSTEP, BLOCK_N / MMA_K, false);
        tcgen05::commit_arrive_pair(q_stage::mbar_addr(mbar_o_ready_base, 0));
        mbarrier::wait(q_stage::mbar_addr(mbar_p_lo_ready_base, 1), phase_p);
        issue_pv_mma_range(taddr_o1, taddr_p1, V_stage_smem, 0, PV_SPLIT_MICROSTEP, initialize_o);
        mbarrier::wait(q_stage::mbar_addr(mbar_p_full_ready_base, 1), phase_p);
        issue_pv_mma_range(taddr_o1, taddr_p1, V_stage_smem, PV_SPLIT_MICROSTEP, BLOCK_N / MMA_K, false);
        tcgen05::commit_arrive_pair(q_stage::mbar_addr(mbar_o_ready_base, 1));
        tcgen05::commit_arrive_pair(kv_ring::stage_mbar_addr(mbar_kv_free_base, v_stage));  // PV_1 read V_last last
        kv_ring::advance(kv_stage, kv_phase);  // the ring cursor carries into the next item
        phase_p ^= 1;
        phase_q_ready ^= 1;
      }
    }

    // Store warp: sO_0, sO_1 -> O. Each CTA stores its own 128 rows per stage (there is no 2-CTA store): two
    // 16 KB boxes (d 0-63 and 64-127) closed as one bulk async-group. The whole warp waits; one lane (elected once,
    // so the same lane every time) issues the copies and owns the bulk groups: only it blocks in wait_group.read,
    // and only it arrives on so_free (count 1).
    if (is_store_warp) {
      const bool elected = elect_sync() != 0u;
      int phase_so_ready = 0;
#pragma unroll 1
      for (int round = 0; round < prm.num_rounds; ++round) {
        int m_block, hkv, batch_id;
        if (!work_item(prm, cluster_id, round, m_block, hkv, batch_id)) continue;
        for (int q = 0; q < q_stage::STAGES; ++q) {
          mbarrier::wait(q_stage::mbar_addr(mbar_so_ready_base, q), phase_so_ready);
          const int src = q_stage::o_smem(O_smem, q);
          const int pos = q_stage::pos0(m_block, cta_rank, q);
          if (elected) {
            tma::copy_5d_smem_to_gmem(&O_tmap, 0, 0, pos, hkv, batch_id, src);
            tma::copy_5d_smem_to_gmem(&O_tmap, SW128_ATOM_COLS, 0, pos, hkv, batch_id, src + PANEL_BYTES);
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
          }
        }
        asm volatile("cp.async.bulk.wait_group.read 1;\n" ::: "memory");  // sO_0 has been read
        if (elected) mbarrier::arrive(q_stage::mbar_addr(mbar_so_free_base, 0));
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");  // sO_1 has been read
        if (elected) mbarrier::arrive(q_stage::mbar_addr(mbar_so_free_base, 1));
        phase_so_ready ^= 1;
      }
      // no wait for the global writes: one item's store overlaps the next item's main loop, and the writes
      // complete with the grid
    }

    // Load warp: Q and the KV ring, filled in exactly the order the MMA warp drains it. The whole warp waits for
    // each stage; one lane (elected once) arms the barrier and issues the copy. Each CTA loads its own half of
    // every operand into its own smem; only the leader arms a ready barrier, once for both CTAs' bytes (if the
    // peer's bytes land first the tx count goes transiently negative, which mbarriers allow).
    if (is_load_warp) {
      const bool elected = elect_sync() != 0u;
      // Producer phases start at 1 (see phase_so_free). Cursor and phases carry across items, so the next item's
      // K / Q / V stream in while this item's last PVs are still running.
      int kv_stage = 0;
      int kv_phase = 1;
      int phase_q_free = 1;
#pragma unroll 1
      for (int round = 0; round < prm.num_rounds; ++round) {
        int m_block, hkv, batch_id;
        if (!work_item(prm, cluster_id, round, m_block, hkv, batch_id)) continue;
        const int kv_tiles = m_block + 1;

        // Load order (FA4's):  K(kv_tiles-1) Q_0 Q_1 V(kv_tiles-1) K(kv_tiles-2) V(kv_tiles-2) ... K(0) V(0).
        // K before V: QK reads K, and PV reads V only after softmax turned that QK's scores into P.
        // K(kv_tiles-1) before Q: its ring stage is usually free before the previous item's Q is (Q is released
        // only after its last QK). Before every copy, wait for the stage's last reader to release it.
        mbarrier::wait(kv_ring::stage_mbar_addr(mbar_kv_free_base, kv_stage), kv_phase);
        if (elected) kv_ring::load_k_half(KV_smem, kv_stage, mbar_kv_ready_base, &K_tmap, kv_tiles - 1, hkv, batch_id, cta_rank);
        kv_ring::advance(kv_stage, kv_phase);
        for (int q = 0; q < q_stage::STAGES; ++q) {
          mbarrier::wait(q_stage::mbar_addr(mbar_q_free_base, q), phase_q_free);
          if (elected) q_stage::load_q(Q_smem, q, mbar_q_ready_base, &Q_tmap, m_block, hkv, batch_id, cta_rank);
        }
        mbarrier::wait(kv_ring::stage_mbar_addr(mbar_kv_free_base, kv_stage), kv_phase);
        if (elected) kv_ring::load_v_half(KV_smem, kv_stage, mbar_kv_ready_base, &V_tmap, kv_tiles - 1, hkv, batch_id, cta_rank);
        kv_ring::advance(kv_stage, kv_phase);
#pragma unroll 1
        for (int kv_tile = kv_tiles - 2; kv_tile >= 0; --kv_tile) {
          mbarrier::wait(kv_ring::stage_mbar_addr(mbar_kv_free_base, kv_stage), kv_phase);
          if (elected) kv_ring::load_k_half(KV_smem, kv_stage, mbar_kv_ready_base, &K_tmap, kv_tile, hkv, batch_id, cta_rank);
          kv_ring::advance(kv_stage, kv_phase);
          mbarrier::wait(kv_ring::stage_mbar_addr(mbar_kv_free_base, kv_stage), kv_phase);
          if (elected) kv_ring::load_v_half(KV_smem, kv_stage, mbar_kv_ready_base, &V_tmap, kv_tile, hkv, batch_id, cta_rank);
          kv_ring::advance(kv_stage, kv_phase);
        }
        phase_q_free ^= 1;
      }
      // Producer tail: this CTA must not exit while a release (a commit the leader multicasts into both CTAs) may
      // still be on its way to one of its barriers. So wait until every ring stage and both Q stages of the last
      // item have been released. (FA4 waits for Q stage 1 only, since its commit follows stage 0's from the same
      // thread; the ISA does not order two commits' arrivals, so both are awaited here.)
      for (int st = 0; st < kv_ring::STAGES; ++st) {
        mbarrier::wait(kv_ring::stage_mbar_addr(mbar_kv_free_base, kv_stage), kv_phase);
        kv_ring::advance(kv_stage, kv_phase);
      }
      for (int q = 0; q < q_stage::STAGES; ++q) {
        mbarrier::wait(q_stage::mbar_addr(mbar_q_free_base, q), phase_q_free);
      }
    }
  }

  // Wait until every warp role leaves its persistent loop before TMEM goes away.
  __syncthreads();
  if (is_mma_warp) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    // The pair allocated together and frees together: each MMA warp tells the PEER's that its CTA is done with
    // TMEM (an arrive on the peer's dealloc barrier) and waits for the peer's word on its own (used once, phase 0).
    if (elect_sync()) {
      mbarrier::arrive_cluster(cluster::map_shared_rank(mbar_tmem_dealloc_addr, cta_rank ^ 1), 1);
    }
    mbarrier::wait(mbar_tmem_dealloc_addr, 0);
    __syncwarp();
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(taddr_base), "r"(TMEM_ALLOC_COLS)
                 : "memory");
  }
}

// Host side: refuse shapes outside the scope, describe Q, K, V and O to the TMA engine, compute the schedule, launch
// G clusters of 2 CTAs.

namespace {

constexpr const char* KERNEL_NAME = "attention_tcgen05_causal_gqa_2cta";  // prefix of every host error

inline void check_cu(CUresult err, const char* what) {
  if (err == CUDA_SUCCESS) return;
  const char* msg = nullptr;
  if (cuGetErrorString(err, &msg) != CUDA_SUCCESS || msg == nullptr) msg = "unknown CUDA driver error";
  throw std::runtime_error(std::string(KERNEL_NAME) + ": " + what + ": " + msg);
}

inline void check_cuda(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(KERNEL_NAME) + ": " + what + ": " + cudaGetErrorString(err));
}

[[noreturn]] inline void refuse(const char* why) { throw std::invalid_argument(std::string(KERNEL_NAME) + ": " + why); }

inline Params make_params(int B, int Sq, int Skv, int Hq, int Hkv, float softmax_scale, int sm_count, float* counter,
                          float* LSE) {
  Params p;
  p.Sq = Sq;
  p.Hq = Hq;
  p.Hkv = Hkv;
  // exp(s * scale) = exp2(s * scale_log2), rounded to fp32 once, like FA4 (which forms the product in double)
  p.softmax_scale_log2 = float(double(softmax_scale) * 1.4426950408889634);
  p.counter = counter;
  p.lse = LSE;
  // the schedule: see "Persistent work assignment" in the kernel
  p.num_m_blocks = cdiv(Sq * QHEADS_PER_KVHEAD, q_stage::ROWS_PER_WORK_ITEM);
  p.num_heads = Hkv * B;
  const int total_work_items = p.num_m_blocks * p.num_heads;
  const long long kv_bytes_per_head = 2LL * Skv * HEAD_DIM * BF16_BYTES;  // K and V
  p.heads_per_section = 1;
  while (2LL * p.heads_per_section * kv_bytes_per_head <= 50LL * 1024 * 1024) p.heads_per_section *= 2;
  const int sm_pairs = sm_count / CTA_GROUP;
  p.num_clusters = sm_pairs < total_work_items ? sm_pairs : total_work_items;
  p.rounds_per_section = cdiv(p.heads_per_section * p.num_m_blocks, p.num_clusters);
  p.num_rounds = cdiv(p.num_heads, p.heads_per_section) * p.rounds_per_section;
  // work_item's divisors (the device divides with multiplies and shifts only)
  p.heads_per_section_log2 = 0;
  while ((1 << p.heads_per_section_log2) < p.heads_per_section) ++p.heads_per_section_log2;
  p.full_sections = p.num_heads / p.heads_per_section;
  p.heads_last_section = p.num_heads - p.full_sections * p.heads_per_section;
  p.div_rounds_per_section = make_fast_div(uint32_t(p.rounds_per_section));
  p.div_heads_last_section = make_fast_div(uint32_t(p.heads_last_section > 0 ? p.heads_last_section : 1));  // 1: unused
  p.div_hkv = make_fast_div(uint32_t(Hkv));
  return p;
}

// TMA tensor maps. Q, K, V and O are bf16 BSHD tensors ([B, S, H, 128], head dim contiguous). A tensor map gives
// each dimension's extent (innermost first), the byte stride of every dimension but the innermost, and the box one
// copy moves. Every box is 64 head-dim columns = 128 bytes wide, the SW128 width, so a 128-column row takes two
// copies (d = 0 and d = 64). SWIZZLE_128B, L2 promotion 256 B (kernel 18).
inline void init_tmap(CUtensorMap* m, const void* ptr, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                      const cuuint32_t* box, const char* what) {
  const cuuint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  check_cu(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<void*>(ptr), dims, strides, box,
                                  elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
           what);  // e.g. a Q / K / V / O pointer that is not 16-byte aligned fails here
}

// Q and O: [B, S, Hq, 128] as 5D (d, h_in, pos, hkv, b), query head hq = 4 hkv + h_in. Splitting hq like this puts
// the 4 query heads of one kv head next to each other, so one box holds 32 positions x 4 heads = 128 packed rows
// (row = 4 pos + h_in): one CTA's Q or O stage, as two 16 KB boxes.
//   dim    extent   stride in bytes
//   d      128      -  (contiguous)
//   h_in   4        256            one query head (128 bf16)
//   pos    S        Hq x 256       one sequence position
//   hkv    Hkv      4 x 256        one group of 4 query heads
//   b      B        S x Hq x 256   one batch
//   box    (64, 4, 32, 1, 1) = 64 columns x 128 packed rows = 16 KB
inline void init_tmap_qo(CUtensorMap* m, const void* ptr, int B, int S, int Hq, int Hkv, const char* what) {
  const cuuint64_t head_bytes = cuuint64_t(HEAD_DIM) * BF16_BYTES;
  const cuuint64_t dims[5] = {cuuint64_t(HEAD_DIM), cuuint64_t(QHEADS_PER_KVHEAD), cuuint64_t(S), cuuint64_t(Hkv),
                              cuuint64_t(B)};
  const cuuint64_t strides[4] = {head_bytes, cuuint64_t(Hq) * head_bytes, cuuint64_t(QHEADS_PER_KVHEAD) * head_bytes,
                                 cuuint64_t(S) * Hq * head_bytes};
  const cuuint32_t box[5] = {cuuint32_t(SW128_ATOM_COLS), cuuint32_t(QHEADS_PER_KVHEAD),
                             cuuint32_t(q_stage::POS_PER_STAGE), 1u, 1u};
  init_tmap(m, ptr, 5, dims, strides, box, what);
}

// K and V: [B, S, Hkv, 128] as 4D (d, key, hkv, b).
//   dim    extent   stride in bytes
//   d      128      -  (contiguous)
//   key    S        Hkv x 256       one sequence position
//   hkv    Hkv      256             one kv head
//   b      B        S x Hkv x 256   one batch
//   box    (64, box_keys, 1, 1). The two CTAs of a pair split every 128-key KV tile (16 KB per CTA and stage):
//     K: box_keys = 64, an 8 KB box: this CTA's 64 keys, loaded as two boxes (d = 0 and d = 64)
//     V: box_keys = 128, a 16 KB box: this CTA's 64 head-dim columns (d = 64 rank ... 64 rank + 63) of all 128 keys
inline void init_tmap_kv(CUtensorMap* m, const void* ptr, int B, int S, int Hkv, int box_keys, const char* what) {
  const cuuint64_t head_bytes = cuuint64_t(HEAD_DIM) * BF16_BYTES;
  const cuuint64_t dims[4] = {cuuint64_t(HEAD_DIM), cuuint64_t(S), cuuint64_t(Hkv), cuuint64_t(B)};
  const cuuint64_t strides[3] = {cuuint64_t(Hkv) * head_bytes, head_bytes, cuuint64_t(S) * Hkv * head_bytes};
  const cuuint32_t box[4] = {cuuint32_t(SW128_ATOM_COLS), cuuint32_t(box_keys), 1u, 1u};
  init_tmap(m, ptr, 4, dims, strides, box, what);
}

}  // namespace

// Q, K, V, O: bf16 BSHD device tensors; LSE: nullptr or [B, Hq, Sq] fp32; counter: nullptr or one fp32 (zeroed by
// the caller) that counts the rows failing the certificate or the O-check.
void attention_tcgen05_causal_gqa_2cta_launch(const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V,
                                              nv_bfloat16* O, float* LSE, float* counter, int B, int Sq, int Skv,
                                              int Hq, int Hkv, float softmax_scale, cudaStream_t stream) {
  // 1. scope (4 * Hkv overflows int only for Hkv >= 2^29, which the tile-count limit below refuses anyway).
  if (B <= 0 || Hkv <= 0 || Hq != QHEADS_PER_KVHEAD * Hkv) refuse("needs B > 0, Hkv > 0 and Hq == 4 * Hkv");
  if (Sq != Skv || Sq <= 0 || Sq % BLOCK_N != 0) {
    refuse("scope is the square causal case: Sq == Skv, a positive multiple of 128");
  }
  // the frozen basis needs lift = 96 / scale_log2 finite: a negative scale of magnitude below ~2e-37 makes it -inf,
  // and rows 128 k + 127 would then return zeros that no check flags
  if (!(softmax_scale > 0.0f && softmax_scale < INFINITY)) refuse("softmax_scale must be finite and > 0");
  // tiles are counted in 32-bit ints: below 2^22 work items, 4 Sq + 511 and every scheduler position fit
  if ((long long)(Sq / BLOCK_N) * Hkv * B >= (1LL << 22)) refuse("needs (Sq / 128) * Hkv * B < 2^22 (32-bit tile counts)");

  // 2. TMA tensor maps. K and V differ only in the box (see init_tmap_kv).
  CUtensorMap Q_tmap, K_tmap, V_tmap, O_tmap;
  init_tmap_qo(&Q_tmap, Q, B, Sq, Hq, Hkv, "cuTensorMapEncodeTiled(Q)");
  init_tmap_kv(&K_tmap, K, B, Skv, Hkv, kv_ring::K_HALF_KEYS, "cuTensorMapEncodeTiled(K)");
  init_tmap_kv(&V_tmap, V, B, Skv, Hkv, BLOCK_N, "cuTensorMapEncodeTiled(V)");
  init_tmap_qo(&O_tmap, O, B, Sq, Hq, Hkv, "cuTensorMapEncodeTiled(O)");

  // 3. the schedule: G needs the SM count of the current device (one cluster needs 2 SMs). The SM count and the
  //    smem opt-in below belong to a device, so both are cached per device (atomics: concurrent first calls from
  //    several host threads only repeat idempotent work); a second call does no CUDA host work before the launch.
  constexpr int MAX_DEVICES = 64;
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  if (device < 0 || device >= MAX_DEVICES) refuse("supports device indices 0 ... 63 only");
  static std::atomic<int> sm_count_of[MAX_DEVICES];
  int sm_count = sm_count_of[device].load(std::memory_order_relaxed);
  if (sm_count == 0) {
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device), "cudaDeviceGetAttribute");
    sm_count_of[device].store(sm_count, std::memory_order_relaxed);
  }
  if (sm_count < CTA_GROUP) refuse("needs a device with at least 2 SMs");
  const Params prm = make_params(B, Sq, Skv, Hq, Hkv, softmax_scale, sm_count, counter, LSE);

  // 4. launch: 2 G CTAs of 384 threads, one per SM (the cluster shape (2, 1, 1) comes from the kernel's
  //    __cluster_dims__). The dynamic smem is above the 48 KB default: opt in once per device. The grid assumes
  //    that all G clusters are resident at once (74 pairs on a 148-SM B200); if fewer fit (another kernel sharing
  //    the device, MPS) the snake's rounds run in waves: slower, never a deadlock (clusters never wait on each other).
  static std::atomic<bool> smem_opt_in_done[MAX_DEVICES];
  if (!smem_opt_in_done[device].load(std::memory_order_acquire)) {
    check_cuda(cudaFuncSetAttribute(attention_tcgen05_causal_gqa_2cta_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES),
               "cudaFuncSetAttribute");
    smem_opt_in_done[device].store(true, std::memory_order_release);
  }
  const int grid = CTA_GROUP * prm.num_clusters;
  attention_tcgen05_causal_gqa_2cta_kernel<<<grid, TB_SIZE, SMEM_BYTES, stream>>>(Q_tmap, K_tmap, V_tmap, O_tmap,
                                                                                    prm);
  check_cuda(cudaGetLastError(), "launch");
}
