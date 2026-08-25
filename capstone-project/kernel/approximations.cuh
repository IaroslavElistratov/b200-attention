#pragma once

#include <cuda_runtime.h>

namespace approx {

// hardware exp2 approximation helper introduced by kernel 8:

__device__ __forceinline__ float fast_exp2(float x) {
  float out;
  // hardware approximation of 2^x with FTZ flushing subnormal values to zero
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
  return out;
}

// software exp2 approximation helpers introduced by kernel 9:

// this is basically a CUDA/PTX copy of the official FA4/CuTe software exp2
// emulation algorithm: split x into integer/fractional parts, approximate
// the value of 2^frac with a degree-3 polynomial, then combine by editing exponent bits
// see FA4 utils.py::ex2_emulation_2 and utils.py::e2e_asm2:
// https://github.com/Dao-AILab/flash-attention/blob/fa4-v4.0.0.beta4/flash_attn/cute/utils.py#L589-L670

__device__ __forceinline__ float2 sub_rn_ftz_f32x2(float2 a, float2 b) {
  float2 out;
  asm volatile(
      "{\n\t"
      // temporary 64-bit containers, each holding two FP32 lanes
      ".reg .b64 va, vb, vo;\n\t"
      // pack a.x/a.y and b.x/b.y into the two containers
      "mov.b64 va, {%2, %3};\n\t"
      "mov.b64 vb, {%4, %5};\n\t"
      // compute {a.x - b.x, a.y - b.y}, round-nearest with FTZ
      "sub.rn.ftz.f32x2 vo, va, vb;\n\t"
      // unpack the two results into out.x and out.y
      "mov.b64 {%0, %1}, vo;\n\t"
      "}\n"
      : "=f"(out.x), "=f"(out.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return out;
}

__device__ __forceinline__ float reconstruct_exp2(float biased_floor, float frac_exp2) {
  float out;
  asm volatile(
      "{\n\t"
      ".reg .s32 biased_i, frac_i, exponent_i, result_i;\n\t"
      // reinterpret both floats as their raw 32-bit patterns
      "mov.b32 biased_i, %1;\n\t"
      "mov.b32 frac_i, %2;\n\t"
      // convert the encoded floor(x) into an FP32 exponent adjustment
      "shl.b32 exponent_i, biased_i, 23;\n\t"
      // apply 2^floor(x) by adjusting frac_exp2's exponent bits
      "add.s32 result_i, exponent_i, frac_i;\n\t"
      "mov.b32 %0, result_i;\n\t"
      "}\n"
      : "=f"(out)
      : "f"(biased_floor), "f"(frac_exp2));
  return out;
}

__device__ __forceinline__ float2 splat2(float value) {
  return make_float2(value, value);
}

// under CUDA 13.0 / sm_100a, this intrinsic spelling has the same full-kernel
// instructions, registers, operands, and schedule as the packed .b64 spelling,
// except that the packed add and FMAs omit FTZ; the add differs only for negative
// such FP32 subnormals are not expected for attention score gaps
__device__ __forceinline__ void e2e_exp2_pair(float x0, float x1, float& out0, float& out1) {
  // bias used by the FP32 floor-extraction trick
  constexpr float kFloorBias = 12582912.0f;  // floor bias equals 2^23 + 2^22

  // cubic polynomial coefficients for approximating 2^fraction
  constexpr float kC3 = 0.07711909f;  // exact FP32 bits: 0x3D9DF09D
  constexpr float kC2 = 0.22756439f;  // exact FP32 bits: 0x3E6906A4
  constexpr float kC1 = 0.69514614f;  // exact FP32 bits: 0x3F31F519

  // duplicate each constant so one packed instruction handles both inputs
  const float2 floor_bias = splat2(kFloorBias);
  const float2 c3 = splat2(kC3);
  const float2 c2 = splat2(kC2);
  const float2 c1 = splat2(kC1);
  const float2 one = splat2(1.0f);

  // keep the eventual exponent inside the supported FP32 range
  const float2 clamped = make_float2(fmaxf(x0, -127.0f), fmaxf(x1, -127.0f));

  // split x into floor(x) and fraction = x - floor(x)
  // round-down keeps the fraction in [0, 1)
  const float2 biased_floor = __fadd2_rd(clamped, floor_bias);
  const float2 floor_x = sub_rn_ftz_f32x2(biased_floor, floor_bias);
  const float2 fraction = sub_rn_ftz_f32x2(clamped, floor_x);

  // approximate 2^fraction using packed Horner-form FMAs
  // horner form is ((C3 * fraction + C2) * fraction + C1) * fraction + 1
  float2 frac_exp2 = __ffma2_rn(c3, fraction, c2);
  frac_exp2 = __ffma2_rn(frac_exp2, fraction, c1);
  frac_exp2 = __ffma2_rn(frac_exp2, fraction, one);

  // combine 2^floor(x) and the approximation of 2^fraction
  out0 = reconstruct_exp2(biased_floor.x, frac_exp2.x);
  out1 = reconstruct_exp2(biased_floor.y, frac_exp2.y);
}

}  // namespace approx
