# Run the kernels

These scripts compile the numbered kernels on a Modal B200. Run all commands
from the repository root after installing and authenticating the Modal client:

```bash
python -m pip install modal
modal setup
```

Select a kernel by number (`1` through `18`), filename stem, or full lineage
label. The supported benchmark shapes are `paper4k`, `paper8k`, and
`paper16k`, matching the shapes listed in [`kernels/README.md`](../kernels/README.md).

## Check correctness

```bash
modal run benchmarks/check_correctness.py --kernel 18 --shape paper4k
```

The input order follows FA4's
[`benchmark_attn.py`](https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_attn.py)
for the dense, non-causal, BF16, head-dimension-128 case implemented here: seed
`0`, followed by sequential BF16 `torch.randn` Q/K/V generation. The selected
kernel is compared with PyTorch BF16 SDPA using fixed tolerances
(`rtol = atol = 0.03`).

## Benchmark

```bash
modal run benchmarks/benchmark.py \
  --kernel 18 --shape paper4k --arm-order fa4_first
```

### Input layout and direct BSHD

The numbered lineage benchmarks time each implementation on its native
contiguous layout:

```text
Stock FA4:    BSHD [B, L, H, D]
Local kernel: BHLD [B, H, L, D]
```

Layout preparation is outside the timed region for both implementations.

I also implemented a
[direct-BSHD version of Kernel 18](../kernels/variants/18_tma_l2_promotion_bshd.cu).
It receives and returns contiguous BSHD without any layout conversions.
The same modification can be straightforwardly applied to any of the other kenrles in the lineage.
Only the global-memory indexing changes:

```cpp
// Native BHLD
tmap = [B * H * L, D];
x = feature;
y = batch_head * L + row;

// Direct BSHD
tmap = [B * L, H * D];
x = head * D + feature;
y = batch * L + row;
```

TMA still writes tiles into the same SW128 SMEM layouts (as discussed in the blog).
And all optimizations we covered in the blog remain unchanged.

B200 runs per shape produced these median same-run perf:

| Shape | Native BHLD | Direct BSHD | Direct-BSHD latency cost |
|---|---:|---:|---:|
| 4K | `94.51%` of FA4 | `92.07%` of FA4 | `2.65%` |
| 8K | `94.65%` of FA4 | `92.69%` of FA4 | `2.14%` |
| 16K | `93.87%` of FA4 | `92.12%` of FA4 | `1.92%` |

The lineage plot uses native-layout timings. The direct-BSHD measurements use
the same contiguous input/output layout as FA4 and include no conversion copies.

### Getting stable numbers

One command compares one local kernel with stock FA4 on one shape.
To smooth out cloud timing noise, run each order 3-6 times:

```bash
for run in 1 2 3; do
  modal run benchmarks/benchmark.py \
    --kernel 18 --shape paper4k --arm-order fa4_first
  modal run benchmarks/benchmark.py \
    --kernel 18 --shape paper4k --arm-order local_first
done
```

Use the median of all reported `percent_stock_fa4` values.
