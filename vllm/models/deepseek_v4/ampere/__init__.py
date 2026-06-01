# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 Ampere (sm_8x) fallback execution path.

This subpackage provides a *functional* (correctness-first, speed-secondary)
execution path for GPUs that lack the FP8 tensor cores, DeepGEMM, FlashMLA
and tilelang kernels that the default ``nvidia/`` path depends on
(Hopper sm_90 / Blackwell sm_100 only).

The path is selected by :func:`use_ampere_fallback` and mirrors the structure
of the ROCm (``amd/``) fallback, which is likewise a non-FP8-tensor-core path.
"""
