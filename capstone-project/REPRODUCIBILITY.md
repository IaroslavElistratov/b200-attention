# Setup and generation

This repo targets Linux x86-64 with an NVIDIA B200 and the CUDA 12.8 toolkit.
You also need `git`, `uv`, and a C++ compiler. The setup script creates a
Python 3.12 environment. The model files use about 72 GB of disk space.

## 1. Set up the environment

```bash
./scripts/setup.sh
```

This checks out the official `Lightricks/LTX-2` source at the pinned commit,
creates `.venv`, installs the dependencies, and builds the SM100a extension.

## 2. Download the models

The Gemma text encoder is gated. Accept the
[Gemma access terms](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
and authenticate once if needed:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/hf auth login
./scripts/download_models.sh
```

The script downloads the LTX-2.3 distilled checkpoint, its x2 spatial
upscaler, and Gemma from pinned Hugging Face revisions. Existing downloads are
reused, and interrupted downloads resume.

## 3. Check the setup

```bash
.venv/bin/ltx-sm100 check
```

## 4. Generate a video

```bash
.venv/bin/ltx-sm100 run \
  --prompt "A huge porcelain elephant walking through a shallow turquoise harbor, tiny sailboats circling its legs, blue floral patterns glowing in sunlight, water splashing gently, majestic low-angle tracking shot." \
  --output outputs/porcelain-elephant.mp4
```

The default output is 1536x1024, 249 frames, and 24 fps (about 10.4 seconds).
The command writes the MP4 and a same-name JSON report. It stops with an error
if the custom kernel does not handle every expected attention call or if the
final latent contains NaN or Inf values.

## 5. Test a replacement kernel

After changing `kernel/20_deferred_wait.cu`, rebuild and run the B200 tests:

```bash
.venv/bin/ltx-sm100 build
.venv/bin/pytest -m b200 tests/test_kernel.py
```

The numerical tests compare the custom BF16 kernel with PyTorch FP32 SDPA at
the attention shapes used by the video pipeline.
