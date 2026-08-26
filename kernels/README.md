# Detailed walkthrough

For the detailed, **super-visual walkthrough** of the kernel lineage and every
main optimization, read
**[B200 Attention from Scratch](https://iaroslavelistratov.github.io/b200-attention/)**.
The article explains why each step is introduced, then maps its diagrams and
mental models back to the source change here.


## Kernel lineage

Read the numbered kernels in order. Each file is a complete checkpoint, and
each checkpoint is organized around one main optimization so neighboring files
remain useful to diff.

**diff adjacent kernels**

From the repository root:

```bash
git diff --no-index -- \
  kernels/1_basic/1_baseline.cu \
  kernels/1_basic/2_p_to_tmem.cu
```

Substitute any neighboring pair in the numbered lineage.


## Shared helpers

The `.cuh` files are simply for code reused by multiple kernels. If an
optimization introduced in one checkpoint is kept by later checkpoints, I put
the shared functions in a `.cuh` header instead of copying them into every later
`.cu` file. Earlier kernels do not start using helpers introduced later, so not
every kernel uses every header or every helper:

- `common.cuh` contains shared mbarrier, TMA, `tcgen05`, descriptor, and packing
  primitives.
- `approximations.cuh` contains the hardware and software exp2 helpers used by
  the approximation stages.
- `smem_layout_noswizzle.cuh` and `smem_layout_swizzle128.cuh` contain the TMA
  loaders and operand-descriptor construction for the two SMEM-layout families.


## Supported shapes

All kernels target NVIDIA B200 (`sm_100a`) and implement dense, non-causal
forward attention with BF16 inputs and outputs.

- `HEAD_DIM = 128`
- `len_kv >= 128` and `len_kv % 128 == 0`
- Kernels 1-4: `len_q >= 128` and `len_q % 128 == 0`
- Kernels 5-18: `len_q >= 256` and `len_q % 256 == 0`
- No masking, causal attention, variable-length sequences, or other head dimensions

The published performance results use three square benchmark shapes:

- **4K:** batch 8, 16 heads, Q length 4096, KV length 4096, head dim 128
- **8K:** batch 4, 16 heads, Q length 8192, KV length 8192, head dim 128
- **16K:** batch 2, 16 heads, Q length 16384, KV length 16384, head dim 128

These are fixed teaching kernels rather than a general attention API.

## Direct BSHD variants

[`variants/1_baseline_bshd.cu`](variants/1_baseline_bshd.cu)
is the smallest example of the layout change.
[`variants/18_tma_l2_promotion_bshd.cu`](variants/18_tma_l2_promotion_bshd.cu)
applies the same change to the final kernel.

Both variants read and write contiguous BSHD directly while keeping the same
SMEM/TMEM layouts and kernel schedules. All the optimizations covered in the
blog remain unchanged. These files stay outside the numbered lineage because
they change the global I/O contract rather than introduce another optimization.
See [Input layout and direct BSHD](../benchmarks/README.md#input-layout-and-direct-bshd).

## Conservative Kernel 14 variant

[`__14_persistent_conservative.cu`](4_persistent/__14_persistent_conservative.cu)
is an optional variant.
It drains all warp roles between output work items, then invalidates and rebuilds
the work-local barriers. Numbered Kernel 14 (covered in the blog) removes that global boundary.
