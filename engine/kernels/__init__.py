"""Fused kernels, Triton where there is a GPU and torch where there is not.

The benchmark container always has Triton, so the fallback is a test harness,
not a shipping path. It is loud about itself for the same reason.
"""

import sys

try:
    import triton  # noqa: F401
except Exception:  # pragma: no cover - the container always has Triton
    HAVE_TRITON = False
    DecodeAttention = None
    from .reference import (
        add_rms_norm,
        kv_norm_rope_to_cache,
        q_norm_rope,
        rms_norm,
        swiglu,
    )

    print("kernels: Triton unavailable, using torch reference", file=sys.stderr)
else:
    HAVE_TRITON = True
    from .attention import DecodeAttention
    from .norm import add_rms_norm, rms_norm
    from .rope import kv_norm_rope_to_cache, q_norm_rope
    from .swiglu import swiglu

__all__ = [
    "HAVE_TRITON",
    "DecodeAttention",
    "add_rms_norm",
    "kv_norm_rope_to_cache",
    "q_norm_rope",
    "rms_norm",
    "swiglu",
]
