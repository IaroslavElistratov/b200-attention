# Kernel source

`20_deferred_wait.cu` is a literal 1:1 copy of the final kernel from the
kernel lineage in my article. It was not rewritten or modified for LTX-2.3.

The complete kernel lineage and every optimization are explained in the
article. This directory simply carries a copy of that kernel and its required
headers so this repository can compile and run it inside a real video-generation
model.

The LTX-2.3 integration is plumbing around the kernel, not a different kernel.
