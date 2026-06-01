# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""float8_e4m3fn <-> fp32 conversion in Triton via integer bit ops.

Ampere (sm_8x) Triton has no ``tl.float8e4nv`` type (matmul *or* convert), so
kernels that quantize into the fp8 KV cache (compressor, etc.) cannot use
``x.to(tl.float8e4nv)``. These helpers emit/consume the e4m3fn bit pattern as
``uint8`` directly so those kernels run on sm_8x. e4m3fn: 1 sign / 4 exp
(bias 7) / 3 mantissa; no inf; finite max 448; 0x7f/0xff = NaN.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def f32_to_e4m3_u8(x):
    """Round an fp32 value to an e4m3fn byte (uint8). Round-half-away; values
    are saturated to +-448. Good enough for lossy KV quant (sub-ULP vs RNE)."""
    sign = tl.where(x < 0, 1, 0).to(tl.uint8) << 7
    a = tl.abs(x)
    a = tl.minimum(a, 448.0)
    is_sub = a < 0.015625  # 2**-6, below the smallest normal
    a_safe = tl.where(a > 0, a, 1.0)
    e = tl.floor(tl.log2(a_safe))
    e = tl.maximum(tl.minimum(e, 8.0), -6.0)

    # Normal: m in [8, 16]; carry to exponent on 16.
    step = tl.exp2(e - 3.0)
    m = tl.floor(a / step + 0.5)
    carry = m >= 16.0
    e_n = tl.where(carry, e + 1.0, e)
    m_n = tl.where(carry, 8.0, m)
    bits_n = ((e_n + 7.0).to(tl.uint8) << 3) | (m_n - 8.0).to(tl.uint8)

    # Subnormal: E=0, M in [0,8]; M==8 promotes to smallest normal (E=1,M=0).
    msub = tl.floor(a * 512.0 + 0.5)  # a / 2**-9
    bits_s = tl.where(msub >= 8.0, tl.full((), 8, tl.uint8), msub.to(tl.uint8))

    bits = tl.where(is_sub, bits_s, bits_n)
    return bits | sign


@triton.jit
def u8_e4m3_to_f32(b):
    """Decode an e4m3fn byte (uint8) to fp32. Exact (no rounding). NaN bytes
    (0x7f/0xff) are not special-cased; they don't occur in the KV cache."""
    b32 = b.to(tl.int32)
    sign = (b32 >> 7) & 1
    e = (b32 >> 3) & 0xF
    m = (b32 & 0x7).to(tl.float32)
    val_norm = (1.0 + m * 0.125) * tl.exp2((e - 7).to(tl.float32))
    val_sub = m * 0.001953125  # M * 2**-9
    val = tl.where(e == 0, val_sub, val_norm)
    return tl.where(sign == 1, -val, val)
