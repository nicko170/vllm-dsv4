# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton sinkhorn for the mHC coupling matrix (Ampere fallback).

The reference (kernels/mhc/torch.py) runs a 20-iteration sinkhorn on a tiny
(T, hc_mult, hc_mult) matrix. Each iteration is two reductions (row / col
normalize) with a data dependency on the previous, so it expands to ~40
sequential elementwise/reduce kernel launches per call — ~3400/token across the
86 mHC calls and the single largest chunk of decode GPU time.

This kernel does the row-softmax + `sinkhorn_repeat` iterations entirely in
registers (one program per token), collapsing all of it into one launch.
hc_mult is small and fixed (==4 for V4-Flash), so a per-token program with the
full M×M matrix in registers is cheap and exact.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sinkhorn_kernel(
    logits_ptr,  # [T, M, M] fp32 (pre-softmax: scaled comb logits + base)
    out_ptr,  # [T, M, M] fp32
    T,
    eps,
    n_iter,
    M: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    tmask = t < T
    r = tl.arange(0, M)
    # offsets for [BLOCK_T, M, M]
    off = t[:, None, None] * (M * M) + r[None, :, None] * M + r[None, None, :]
    m = tmask[:, None, None] & tl.full((1, M, M), 1, tl.int1)
    x = tl.load(logits_ptr + off, mask=m, other=0.0).to(tl.float32)

    # row softmax over last dim (axis=2): comb = softmax(logits, -1) + eps
    xmax = tl.max(x, axis=2, keep_dims=True)
    e = tl.exp(x - xmax)
    comb = e / tl.sum(e, axis=2, keep_dims=True) + eps

    # initial column normalize (axis=1)
    comb = comb / (tl.sum(comb, axis=1, keep_dims=True) + eps)
    # sinkhorn_repeat-1 iterations of {row(axis=2), col(axis=1)}
    for _ in range(n_iter - 1):
        comb = comb / (tl.sum(comb, axis=2, keep_dims=True) + eps)
        comb = comb / (tl.sum(comb, axis=1, keep_dims=True) + eps)
    tl.store(out_ptr + off, comb, mask=m)


def sinkhorn_triton(comb_logits: torch.Tensor, eps: float, n_iter: int) -> torch.Tensor:
    """comb_logits: [T, M, M] (already scaled + base added). Returns [T, M, M]."""
    assert comb_logits.is_cuda and comb_logits.ndim == 3
    T, M, M2 = comb_logits.shape
    assert M == M2
    x = comb_logits.contiguous().to(torch.float32)
    out = torch.empty_like(x)
    BLOCK_T = 32
    grid = (triton.cdiv(T, BLOCK_T),)
    _sinkhorn_kernel[grid](x, out, T, eps, n_iter, M=M, BLOCK_T=BLOCK_T)
    return out


# Register as a torch custom op so it composes with torch.compile (opaque node,
# no graph break) and is safe to record inside a CUDA graph.
@torch.library.custom_op("vllm::mhc_sinkhorn", mutates_args=())
def mhc_sinkhorn(comb_logits: torch.Tensor, eps: float, n_iter: int) -> torch.Tensor:
    return sinkhorn_triton(comb_logits, eps, n_iter)


@mhc_sinkhorn.register_fake
def _mhc_sinkhorn_fake(
    comb_logits: torch.Tensor, eps: float, n_iter: int
) -> torch.Tensor:
    return torch.empty_like(comb_logits, dtype=torch.float32)
