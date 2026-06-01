# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability gate for the DeepSeek-V4 Ampere fallback.

Kept import-light (only ``vllm.envs`` and ``current_platform``) so it can be
imported from the package ``__init__`` and from shared layers without pulling
in model/kernel modules.
"""

from __future__ import annotations

import vllm.envs as envs
from vllm.platforms import current_platform


def use_ampere_fallback() -> bool:
    """Return True if DeepSeek-V4 should use the Ampere fallback path.

    The default ``nvidia/`` path requires DeepGEMM + FlashMLA + tilelang,
    which are only available on Hopper (sm_90) and Blackwell (sm_100).  On
    CUDA GPUs with compute capability < 9 (e.g. Ampere sm_86 / A40) none of
    those are available and the model cannot run, so we route to this
    correctness-first fallback instead.

    The env var ``VLLM_DEEPSEEK_V4_FALLBACK=1`` forces the path on for
    testing.  ROCm has its own dedicated path and is never routed here.
    """
    if not current_platform.is_cuda():
        return False
    if envs.VLLM_DEEPSEEK_V4_FALLBACK:
        return True
    cap = current_platform.get_device_capability()
    return cap is not None and cap.major < 9
