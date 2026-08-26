# Run the kernels

Stock FA4 is 100%. Each value below is the median of six same-run ratios;
every call timed the local kernel and stock FA4 on the same B200.
Using the same-run ratio reduces cloud timing noise.

I use same-run FA4 instead of the paper TFLOPS because, across 342 calibration runs,
stock FA4 itself reached about 97% of the paper results on the B200s I used.
See [Benchmark calibration](https://iaroslavelistratov.github.io/b200-attention/#benchmark-calibration).

| Kernel | Change | 4K | 8K | 16K |
|---:|---|---:|---:|---:|
| 1 | Baseline | 14.2% | 14.1% | 13.7% |
| 2 | P in TMEM | 15.1% | 15.0% | 15.0% |
| 3 | Swizzling | 25.9% | 25.9% | 26.2% |
| 4 | Warp specialization | 26.6% | 26.4% | 26.5% |
| 5 | Two Q tiles | 40.5% | 40.3% | 41.3% |
| 6 | Load pipeline | 48.6% | 48.6% | 48.4% |
| 7 | Compute pipeline | 58.2% | 58.5% | 58.0% |
| 8 | Hardware `exp2` | 64.1% | 64.1% | 63.8% |
| 9 | Software `exp2` | 69.3% | 70.1% | 70.6% |
| 10 | Register caching | 72.9% | 73.7% | 73.6% |
| 11 | Skip O rescale | 81.7% | 83.0% | 83.3% |
| 12 | Split correction | 84.9% | 85.8% | 86.3% |
| 13 | Early PV | 87.2% | 88.1% | 88.4% |
| 14 | Persistent CTAs | 90.7% | 90.3% | 89.8% |
| 15 | Deferred score acquisition | 92.3% | 92.5% | 92.0% |
| 16 | Split rowsum accumulation | 93.4% | 93.7% | 93.7% |
| 17 | Phase bitmasks | 93.7% | 94.0% | 94.1% |
| 18 | TMA L2 promotion | 94.4% | 94.4% | 94.1% |

The included scripts use Modal for B200 access, but any other cloud provider
with B200s works too. Run the commands below from the repository root:

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

#### Run the BSHD comparison

```bash
modal run benchmarks/benchmark_bshd.py \
  --shape paper4k --arm-order fa4_first
```

This checks correctness, then times stock FA4, direct BSHD, and native BHLD.
The available orders are `fa4_first`, `bshd_first`, and `bhld_first`.

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
