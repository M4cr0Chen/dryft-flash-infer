"""Fused kernels, Triton where there is a GPU and torch where there is not.

The benchmark container always has Triton, so the fallback is a test harness,
not a shipping path. It is loud about itself for the same reason.
"""

import sys

from .ngram import NgramDrafter

try:
    import triton  # noqa: F401
except Exception:  # pragma: no cover - the container always has Triton
    HAVE_TRITON = False
    DecodeAttention = None
    pick_matmul = None
    fp8 = None
    cuda_fp8 = None
    cuda_gemv = None
    cuda_mlp = None
    from .reference import (
        add_rms_norm,
        add_rms_norm_partials,
        kv_norm_rope_to_cache,
        q_norm_rope,
        qkv_norm_rope_to_cache,
        qkv_planes_norm_rope_to_cache,
        rms_norm,
        swiglu,
    )

    print("kernels: Triton unavailable, using torch reference", file=sys.stderr)
else:
    HAVE_TRITON = True
    from .attention import DecodeAttention
    from . import cuda_fp8, cuda_gemv, cuda_mlp, fp8
    from .gemm import pick_matmul
    from .norm import add_rms_norm, add_rms_norm_partials, rms_norm
    from .rope import (
        kv_norm_rope_to_cache,
        q_norm_rope,
        qkv_norm_rope_to_cache,
        qkv_planes_norm_rope_to_cache,
    )
    from .swiglu import swiglu

__all__ = [
    "HAVE_TRITON",
    "NgramDrafter",
    "DecodeAttention",
    "pick_matmul",
    "fp8",
    "cuda_fp8",
    "cuda_gemv",
    "cuda_mlp",
    "add_rms_norm",
    "add_rms_norm_partials",
    "kv_norm_rope_to_cache",
    "q_norm_rope",
    "qkv_norm_rope_to_cache",
    "qkv_planes_norm_rope_to_cache",
    "rms_norm",
    "swiglu",
]
