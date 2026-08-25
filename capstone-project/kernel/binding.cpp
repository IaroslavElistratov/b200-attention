// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>

using AttentionFn = void(
    const nv_bfloat16* query,
    const nv_bfloat16* key,
    const nv_bfloat16* value,
    nv_bfloat16* output,
    int batch_heads,
    int query_length,
    int key_value_length,
    cudaStream_t stream);

AttentionFn attention_tcgen05_v20_launch;

namespace {

at::Tensor attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda(),
      "ltx_sm100.attention requires CUDA query/key/value tensors");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "ltx_sm100.attention requires query/key/value on the same CUDA device");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 &&
          key.scalar_type() == at::kBFloat16 &&
          value.scalar_type() == at::kBFloat16,
      "ltx_sm100.attention requires BF16 query/key/value tensors");
  TORCH_CHECK(
      query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
      "ltx_sm100.attention expects query/key/value shaped [BH, L, 128]");
  TORCH_CHECK(
      query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
      "ltx_sm100.attention requires contiguous query/key/value tensors");

  const int64_t batch_heads = query.size(0);
  const int64_t query_length = query.size(1);
  const int64_t key_value_length = key.size(1);
  const int64_t head_dim = query.size(2);

  TORCH_CHECK(
      key.size(0) == batch_heads && value.size(0) == batch_heads,
      "ltx_sm100.attention requires matching query/key/value BH dimensions");
  TORCH_CHECK(
      key.size(2) == head_dim && value.size(2) == head_dim,
      "ltx_sm100.attention requires matching query/key/value head dimensions");
  TORCH_CHECK(
      value.size(1) == key_value_length,
      "ltx_sm100.attention requires matching key/value sequence lengths");
  TORCH_CHECK(head_dim == 128, "ltx_sm100.attention only supports head dimension 128");
  TORCH_CHECK(
      query_length % 256 == 0,
      "ltx_sm100.attention requires query length divisible by 256");
  TORCH_CHECK(
      key_value_length % 128 == 0,
      "ltx_sm100.attention requires key/value length divisible by 128");
  TORCH_CHECK(
      batch_heads > 0 && query_length > 0 && key_value_length > 0,
      "ltx_sm100.attention dimensions must be non-zero");
  TORCH_CHECK(
      batch_heads <= INT_MAX && query_length <= INT_MAX && key_value_length <= INT_MAX,
      "ltx_sm100.attention dimensions exceed launcher integer limits");

  c10::cuda::CUDAGuard guard(query.device());
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, query.get_device()));
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0,
      "ltx_sm100.attention requires an NVIDIA B200-class SM100 device; got compute capability ",
      properties.major,
      ".",
      properties.minor);

  auto output = at::empty_like(query);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();
  attention_tcgen05_v20_launch(
      reinterpret_cast<const nv_bfloat16*>(query.data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(key.data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(value.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(output.data_ptr()),
      static_cast<int>(batch_heads),
      static_cast<int>(query_length),
      static_cast<int>(key_value_length),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY(ltx_sm100, library) {
  library.def("attention(Tensor query, Tensor key, Tensor value) -> Tensor");
}

TORCH_LIBRARY_IMPL(ltx_sm100, CUDA, library) {
  library.impl("attention", &attention);
}
