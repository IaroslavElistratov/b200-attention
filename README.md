# B200 Attention from Scratch

[![B200 Attention from Scratch](assets/b200-attention-social-card.png)](https://iaroslavelistratov.github.io/b200-attention/)


**14 kernels | 60 diagrams | 94.5% of stock FlashAttention-4**

A visual, diffable path from scratch to a high-performance NVIDIA B200 kernel.
Each checkpoint is organized around one main optimization, so you can
study the explanation and diff the neighboring source files.

[**Read the visual guide**](https://iaroslavelistratov.github.io/b200-attention/) |
[**Browse the kernels**](kernels/) |
[**Check and benchmark**](benchmarks/) |
[**Run the capstone**](capstone-project/)

## Read the visual guide

**[B200 Attention from Scratch](https://iaroslavelistratov.github.io/b200-attention/)**

Article introduces Blackwell-specific machinery gradually, in
**super visual** and step-by-step guide. It assumes basic
CUDA and matrix-multiplication familiarity.

**If you are familiar with basic cuda, you can read and fully understand this article.**
**No experience with Blackwell is required to follow alone.**

## Performance progression

![Performance across the B200 attention kernel lineage](assets/results_condensed_context_pct_stock_fa4.png)

> Benchmark scope: FA4 and my kernels are benchmarked on their preferred layouts.
> FA4 contiguous on BSHD, and the local kernels on contiguous BHLD.
> Layout conversion was excluded from timing for both.
>
> Also, the local kernels can be easily adapted to natively work on FA4-style BSHD
> (without calling .contiguous() on the inputs).
> I implemented a direct-BSHD version of the final kernel (same modification can be applied
> to any other kernel in the lineage), performance changes only slightly.
> See benchmarks/README.md#input-layout-and-direct-bshd.


## Kernel lineage

- [`kernels/`](kernels/) - 18 numbered checkpoints: 14 main stages followed by
  four smaller optional refinements, plus a direct-BSHD integration variant.

## Capstone: generate videos with your kernel

[![Videos generated with the B200 attention kernel](capstone-project/assets/heading.png)](capstone-project/)

I strongly recommend doing the capstone after reading the article.
It plugs the B200 attention kernel into a video generation model,
where the model calls **YOUR KERNEL (or the kernel you understand)** 1,056 times
for each generated video.

**[Run the video-generation capstone](capstone-project/)**

## Scope

The kernels implement dense, non-causal forward attention for NVIDIA B200
(`sm_100a`). They use BF16 inputs and outputs, head dimension of 128, KV
lengths that are multiples of 128 (up to 32K seq_len tested, likely will work fine for even longer),
and Q lengths compatible with each checkpoint's one- or two-tile Q staging.

This is a handwritten teaching and research lineage,
not a line-for-line port of the official FlashAttention-4
implementation.
