# B200 FlashAttention for LTX-2.3

![Video samples generated with the B200 attention kernel](assets/heading.png)

This repo integrates the
[`final kernel`](../kernels/5_minor/18_tma_l2_promotion.cu) from the article
(a dense BF16 attention kernel for NVIDIA B200) into the LTX-2.3
video-generation model.

To generate each video, the model will call our attention kernel 1,056 times.

Budget about $7 for an hour of B200 compute, depending on the provider.
That is enough time to set up the repo, download the weights, and generate multiple videos.

A 10.4-second video (249 frames) takes about 41 seconds to generate on a B200.
Runtime depends on the complete model architecture, not only on our kernel.

I highly recommend doing it; you will get a kick seeing the kernel you understand generate beautiful videos for you.

**You do not need to study the video-model integration.**
**Your time is better spent understanding the article + diagrams + kernel code.**
**This repo is just the plumbing for integrating our kernel into a video-generation model and running it.**
**The focus of my article is optimizing a dense attention kernel for B200, so I do not want to spend time covering the plumbing code.**
**The only thing worth understanding here is the kernel itself and its optimizations (`../kernels/`), which I already explained in the article.**
**Everything else in this repo is just plumbing/glue code.**

To run the capstone:

1. Rent the compute, SSH into the machine, clone this repo, and enter
   `b200-attention/capstone-project`.
2. Follow the [setup and generation instructions](GETTING_STARTED.md).
