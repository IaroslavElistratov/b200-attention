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

The input order follows Dao's
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

One invocation measures exactly one local kernel and stock FlashAttention-4 on
one shape using this timing window:

```text
one-second pause before each arm
triton.testing.do_bench(warmup=5, rep=10, return_mode="mean")
```

The output reports milliseconds, TFLOPS, percentage of same-run stock FA4, and
percentage of the corresponding FA4 paper result.

A stable comparison uses six fresh invocations per kernel and shape, balanced
by order:

```bash
for run in 1 2 3; do
  modal run benchmarks/benchmark.py \
    --kernel 18 --shape paper4k --arm-order fa4_first
  modal run benchmarks/benchmark.py \
    --kernel 18 --shape paper4k --arm-order local_first
done
```

Take the median of the six full-precision `percent_stock_fa4` values. Do not
use a ratio of separately aggregated timings. Also compare the two order-specific
medians; if they differ by at least one percentage point, collect more runs or
report the order sensitivity. Do not treat one short-burst invocation as a
stable performance result.
