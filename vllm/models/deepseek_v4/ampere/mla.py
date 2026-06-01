# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (sm_8x) sparse-MLA attention implementation for DeepSeek-V4.

FlashMLA has no Ampere sparse build, so we reuse the ROCm sparse-MLA
implementation, which despite its name is **portable Triton + torch**:
``rocm_sparse_attn_decode`` / ``rocm_sparse_attn_prefill`` and the
``DeepseekV4ROCMAiter*`` metadata builders contain no aiter/HIP/is_rocm code
(verified) — they build on the CUDA FlashMLA backend + Triton kernels, which
run on Ampere. This is the "dense/Triton sparse-MLA fallback" of Phase 5.

If a kernel hits the sm_86 shared-memory ceiling (~100 KB) we can later tune
BLOCK_SIZE / num_stages, but the math is arch-agnostic.
"""

from __future__ import annotations

from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseImpl,
)


class DeepseekV4AmpereMLASparseImpl(DeepseekV4ROCMAiterMLASparseImpl):
    """Sparse MLA for Ampere — reuses the portable Triton ROCm impl.

    Inherits ``backend_cls`` (``DeepseekV4ROCMAiterMLASparseBackend``), its
    Triton metadata builders, ``forward_mqa`` (decode + prefill), and
    ``get_padded_num_q_heads`` (identity — no FP8-decode head constraint).
    """
