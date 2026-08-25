# B200 FlashAttention for LTX-2.3

![Video samples generated with the B200 attention kernel](assets/heading.png)

This repo contains the integration of our final kernel from the article
(a dense BF16 attention kernel for NVIDIA B200) into the LTX-2.3
video-generation model.

To generate each video, the model will call our attention kernel 1,056 times.

Budget about $7 for an hour of B200 compute, depending on the provider.
That is enough time to set up the repo, download the weights, and generate multiple videos.

A 10.4-second video (249 frames) takes about 41 seconds to generate on a B200.
Runtime depends on the complete model architecture, not only on our kernel.

I highly recommend doing it; you will get a kick seeing the kernel you understand generate beautiful videos for you.

**Don't bother reading the video-model integration manually.**
**Your time is better spent understanding the article + diagrams + kernel code.**
**This repo is just the plumbing for integrating our kernel into a video-generation model and running it.**
**The focus of my article is optimizing a dense attention kernel for B200, so I do not want to spend time covering the plumbing code.**
**The only thing worth understanding here is the kernel itself and its optimizations (`kernel/`), which I already explained in the article.**
**Everything else in this repo is just plumbing/glue code.**

Basically:

0. Rent the compute, SSH into the machine, and clone this repo.
1. Paste your chosen article kernel into `kernel/20_deferred_wait.cu`. If its
   launcher name differs from the final kernel, update `kernel/binding.cpp` too.
2. Tell your LLM to set up this repo using the reproduction instructions (it
   will download the model weights, create the environment, and so on).
3. When it's done, run the model—and out come the videos!

---

Reproduction instructions: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
