# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

try:
    from vllm.model_executor.kernels.mhc.sinkhorn_triton import mhc_sinkhorn

    _HAS_SINKHORN_TRITON = True
except Exception:  # triton unavailable / import error
    _HAS_SINKHORN_TRITON = False


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn_flat.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    if (
        _HAS_SINKHORN_TRITON
        and os.environ.get("VLLM_MHC_SINKHORN", "1") == "1"
        and comb_logits.is_cuda
    ):
        # Fused Triton sinkhorn: collapses the row-softmax + `sinkhorn_repeat`
        # iterations (each a row/col normalize with a data dependency, i.e. ~40
        # sequential reduce launches) into a single per-token kernel.
        comb_mix = mhc_sinkhorn(comb_logits, hc_sinkhorn_eps, sinkhorn_repeat)
    else:
        comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
        for _ in range(sinkhorn_repeat - 1):
            comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
            comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


# On Ampere (sm_8x) the tilelang fused mHC kernels are unusable, so the model
# falls back to these eager torch ops. The 20-iteration sinkhorn loop alone
# launches ~40 tiny reduce kernels per call (~3400/token across the 86 mHC
# calls), which is the single largest chunk of decode GPU time. torch.compile
# fuses each function into a handful of kernels (the unrolled sinkhorn loop is
# kept in registers), cutting both kernel count and memory traffic. Gated by
# env for A/B testing; safe inside vLLM's FULL cudagraph because default-mode
# torch.compile does not install its own cudagraphs.
if os.environ.get("VLLM_MHC_COMPILE", "0") == "1":
    mhc_pre_torch = torch.compile(mhc_pre_torch, dynamic=True, fullgraph=True)
    mhc_post_torch = torch.compile(mhc_post_torch, dynamic=True, fullgraph=True)
