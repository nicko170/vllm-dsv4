# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 MTP draft model — Ampere (sm_8x) fallback.

Reuses the portable ROCm MTP structure (standard FusedMoE + CustomOp mHC),
which avoids the Hopper/Blackwell-only kernel imports.
"""

from vllm.models.deepseek_v4.amd.mtp import (
    DeepSeekV4MTP as DeepSeekV4MTP,
)

__all__ = ["DeepSeekV4MTP"]
