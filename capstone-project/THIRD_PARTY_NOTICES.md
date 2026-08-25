# Third-party notices

The original capstone integration code is licensed under Apache-2.0. The
following components retain their own terms and attribution.

## LTX-2

The setup script obtains LTX-2 from
<https://github.com/Lightricks/LTX-2> at commit
`780984275fd47128b02bef9b5c085404276866ee`. LTX-2 and its model artifacts are
not relicensed by this project. They are governed by the LTX-2 Community
License Agreement in `LICENSES/LTX-2-COMMUNITY.txt`.

Copyright (c) Lightricks Ltd. All rights reserved.

## NVIDIA CUTLASS

`../kernels/common.cuh` contains low-level device helper code adapted from NVIDIA
CUTLASS 4.2.1 and 4.3.1. CUTLASS is distributed under the BSD 3-Clause
License in `LICENSES/CUTLASS-BSD-3-Clause.txt`.

Copyright (c) 2017–2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

## FlashAttention

`../kernels/5_minor/18_tma_l2_promotion.cu` and
`../kernels/approximations.cuh` retain comments
identifying scheduling and approximation techniques associated with
FlashAttention-4. This notice is retained conservatively for those portions.
The FlashAttention BSD 3-Clause License is
in `LICENSES/FLASH-ATTENTION-BSD-3-Clause.txt`.

Copyright (c) 2022, the respective contributors, as shown by the upstream
AUTHORS file. All rights reserved.

No LTX, Gemma, CUTLASS, or FlashAttention model weights or binary releases are
included in this repository.
