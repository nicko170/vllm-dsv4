# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (sm_8x) sparse-MLA attention implementation for DeepSeek-V4.

FlashMLA has no Ampere sparse build, so this path uses a dense / Triton
reference instead.  Phase 1 only wires up the dispatch and the backend
metadata (kv-cache shape etc., reused from the FlashMLA backend); the actual
attention kernel is filled in at Phase 5 — until then ``forward_mqa`` raises a
clear NotImplementedError so we can prove the capability gate routes here
(past the DeepGEMM/FlashMLA wall) rather than aborting in DeepGEMM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
    DeepseekV4SparseMLAAttentionImpl,
)

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention


class DeepseekV4AmpereMLASparseBackend(DeepseekV4FlashMLASparseBackend):
    """Reuse the FlashMLA backend's kv-cache spec / metadata, but report a
    distinct name and route to the Ampere impl."""

    @staticmethod
    def get_name() -> str:
        return "V4_AMPERE_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4SparseMLAAttentionImpl"]:
        return DeepseekV4AmpereMLASparseImpl


class DeepseekV4AmpereMLASparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """Dense/Triton sparse-MLA fallback for Ampere (sm_8x)."""

    backend_cls = DeepseekV4AmpereMLASparseBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # The dense reference path has no FP8-decode head-count constraint,
        # so no padding is required.
        return num_heads

    @classmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError(
            "DeepSeek-V4 Ampere sparse-MLA attention is not implemented yet "
            "(Phase 5). The capability gate correctly routed to the Ampere "
            "fallback path instead of the DeepGEMM/FlashMLA path."
        )
