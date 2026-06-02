# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashIndexer: a fused Triton lightning-indexer logits kernel for Ampere.

Computes the DeepSeek-V4 indexer MQA logits

    logit[m, n] = ( sum_h relu(q[m, h] . k[n]) * weights[m, h] ) * scale[n]

masked to cu_seqlen_ks[m] <= n < cu_seqlen_ke[m], **without** materializing the
[H, M, N] score tensor (H=64). The torch fallback materializes that (17GB at
N=128k) and is ~30x off peak; this tiles over (M, N), loops the H reduction in
registers with bf16 tensor-core dots, and fuses relu / weight / scale / mask.

q and k are passed as bf16 (the caller casts the fp8 indexer cache; the fp8
per-token scale is applied to the score, matching the reference). Validated
byte-for-value against the torch reference in
``artifacts/flash_indexer_test.py``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_NEG = -3.4028234663852886e38


@triton.jit
def _flash_mqa_logits_kernel(
    q_ptr, k_ptr, scale_ptr, w_ptr, ks_ptr, ke_ptr, out_ptr,
    M, N,
    sqm, sqh, sqd,
    skn, skd,
    swm, swh,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # k tile [BLOCK_N, D] (bf16) -> transposed [D, BLOCK_N] for tl.dot
    k = tl.load(
        k_ptr + offs_n[:, None] * skn + offs_d[None, :] * skd,
        mask=n_mask[:, None], other=0.0,
    )
    kt = tl.trans(k)  # [D, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(H):
        qh = tl.load(
            q_ptr + offs_m[:, None] * sqm + h * sqh + offs_d[None, :] * sqd,
            mask=m_mask[:, None], other=0.0,
        )  # [BLOCK_M, D] bf16
        s = tl.dot(qh, kt)                      # [BLOCK_M, BLOCK_N] f32
        s = tl.maximum(s, 0.0)                  # relu
        wh = tl.load(w_ptr + offs_m * swm + h * swh, mask=m_mask, other=0.0)
        acc += s * wh[:, None]

    scl = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc * scl[None, :]

    ks = tl.load(ks_ptr + offs_m, mask=m_mask, other=0)
    ke = tl.load(ke_ptr + offs_m, mask=m_mask, other=0)
    valid = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
    acc = tl.where(valid, acc, -3.4028234663852886e38)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc, mask=m_mask[:, None] & n_mask[None, :],
    )


def flash_mqa_logits_triton(q, k, scale, weights, cu_seqlen_ks, cu_seqlen_ke):
    """q: [M,H,D] bf16, k: [N,D] bf16, scale: [N] f32, weights: [M,H] f32.
    Returns logits [M, N] f32 (-inf outside [ks,ke))."""
    M, H, D = q.shape
    N = k.shape[0]
    scale = scale.reshape(-1).to(torch.float32)
    weights = weights.to(torch.float32)
    out = torch.empty((M, N), device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _flash_mqa_logits_kernel[grid](
        q, k, scale, weights, cu_seqlen_ks, cu_seqlen_ke, out,
        M, N,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1),
        weights.stride(0), weights.stride(1),
        H=H, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out
