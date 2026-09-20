"""Triton kernels in the programmatic-dependent-launch chain, without a rewrite.

Triton's launcher cannot set the programmatic-serialization attribute. Its
compiled kernel does expose the ``CUfunction``, so after one ordinary launch
(which compiles and specialises the kernel) every later launch with the same
specialisation goes through ``cuda_jit.Kernel.launch_ex``, bit-identical to
the ordinary launch (verified) and now able to start early. The kernel body
issues ``griddepcontrol`` through inline PTX: trigger at entry, wait before
the first load. Both are no-ops for a plain launch.

Argument marshalling mirrors Triton 3.1: ``constexpr`` arguments are baked
in, and integer arguments equal to one are specialised away, so neither is
passed to the function.
"""

import ctypes
import os

import triton
import triton.language as tl

from . import cuda_jit

PDL = {"off": 0, "late": 1, "on": 2, "early": 2}[os.environ.get("DRYFT_PDL", "early")]
ENABLED = PDL != 0 and os.environ.get("DRYFT_TRITON_PDL", "off") == "on"
#: Diagnostic: relaunch through the handle but without the programmatic
#: attribute, to separate argument-marshalling errors from ordering races.
ATTRIBUTE = os.environ.get("DRYFT_TRITON_PDL_ATTR", "on") == "on"


# The asm runs once per element of its operand, and a one-element operand
# lives in one thread: the other threads would run ahead of the wait. A
# 4096-element operand puts at least one element in every thread of any
# block up to 32 warps; the redundant executions are harmless.
@triton.jit
def gdc_trigger():
    tl.inline_asm_elementwise("griddepcontrol.launch_dependents; mov.u32 $0, $1;", "=r,r",
                              [tl.zeros((4096,), tl.int32)], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def gdc_wait():
    tl.inline_asm_elementwise("griddepcontrol.wait; mov.u32 $0, $1;", "=r,r",
                              [tl.zeros((4096,), tl.int32)], dtype=tl.int32, is_pure=False, pack=1)


class Programmatic:
    """Launch a Triton ``JITFunction`` programmatically once it has compiled."""

    def __init__(self, fn):
        self.fn = fn
        self.compiled = {}

    def __call__(self, grid, *args, num_warps, **constexpr):
        if not ENABLED:
            return self.fn[grid](*args, num_warps=num_warps, **constexpr)
        key = (tuple(sorted(constexpr.items())), num_warps,
               tuple(isinstance(a, int) and not isinstance(a, bool) and a == 1 for a in args))
        entry = self.compiled.get(key)
        if entry is None:
            compiled = self.fn[grid](*args, num_warps=num_warps, **constexpr)
            cuda_jit._init()
            kernel = cuda_jit.Kernel(ctypes.c_void_p(compiled.function), self.fn.__name__)
            kernel.set_shared(compiled.metadata.shared)
            kernel.pdl = True
            # The parameters the compiled function actually takes, by position
            # in the Python signature: Triton drops constexpr and specialised
            # arguments, and this is its own record of which survived.
            # Triton 3.1 keys these by parameter name. ``constants`` holds the
            # constexpr arguments and the integers it specialised to one.
            signature = compiled.src.signature
            constants = compiled.src.constants
            keep = tuple(i for i, name in enumerate(self.fn.arg_names)
                         if signature.get(name) not in (None, "constexpr")
                         and name not in constants and i not in constants)
            self.compiled[key] = entry = (kernel, keep)
            return None
        kernel, keep = entry
        params = [args[i] for i in keep if i < len(args)]
        grid = grid if isinstance(grid, tuple) else (grid,)
        gx = grid[0]
        gy = grid[1] if len(grid) > 1 else 1
        kernel.launch_ex((gx, gy), 32 * num_warps, *params, pdl=ATTRIBUTE)
        return None
