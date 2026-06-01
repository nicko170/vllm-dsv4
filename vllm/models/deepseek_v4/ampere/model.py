# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 model — Ampere (sm_8x) fallback.

The Ampere fallback reuses the *portable* model structure from the ROCm
(``amd/``) path: it always uses the standard ``FusedMoE`` (never DeepGEMM
MegaMoE) and the ``CustomOp``-based mHC layers, avoiding the Hopper/Blackwell
-only DeepGEMM / FlashMLA / tilelang dependencies at module-import time.

The architecture-specific pieces that still differ from ROCm — the sparse-MLA
attention implementation and the FP8->bf16 dequant of the o-projection — are
selected at runtime via :func:`use_ampere_fallback` inside the shared
``attention.py`` (see ``ampere/mla.py``).
"""

from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4ForCausalLM as DeepseekV4ForCausalLM,
)

__all__ = ["DeepseekV4ForCausalLM"]
