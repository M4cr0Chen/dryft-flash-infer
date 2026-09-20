"""Compile CUDA C++ at load time, without a toolkit.

The runtime has no nvcc and no ninja, so ``cpp_extension.load_inline`` cannot
run. It does have two things that are enough: ``libnvrtc`` (a dependency of the
torch wheel, so it is wherever torch is) and the CUDA driver. NVRTC turns a
source string into PTX; the driver JITs the PTX and launches it. Both are
reached through ctypes, so nothing ships but Python source.

Kernels compiled here launch on torch's current stream and capture into a CUDA
graph like any other kernel.

NVRTC has no toolkit headers available, so kernel sources must not ``#include``
anything. bfloat16 is therefore carried as ``unsigned short`` and converted with
bit intrinsics -- see ``cuda_gemv.py`` for the two helpers that do it.
"""

import ctypes
import glob
import os
import sys

_nvrtc = None
_driver = None


def _load_nvrtc():
    """libnvrtc, from wherever the torch wheel put it."""
    import torch

    roots = [
        os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia"),
        os.path.dirname(torch.__file__),
    ]
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "libnvrtc.so*"), recursive=True):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return ctypes.CDLL("libnvrtc.so")  # last resort: the system one


def _check(result, what):
    if result != 0:
        raise RuntimeError(f"{what} failed with {result}")


def available() -> bool:
    """Can this process compile and launch CUDA at all?"""
    try:
        _init()
    except Exception:
        return False
    return True


def _init():
    global _nvrtc, _driver
    if _nvrtc is None:
        _nvrtc = _load_nvrtc()
    if _driver is None:
        _driver = ctypes.CDLL("libcuda.so.1")
        _driver.cuInit(0)


def compile_ptx(source: str, arch: str = "compute_90") -> bytes:
    """CUDA C++ to PTX. Raises with the compiler log if it does not build."""
    _init()
    program = ctypes.c_void_p()
    _check(
        _nvrtc.nvrtcCreateProgram(
            ctypes.byref(program), source.encode(), b"engine.cu", 0, None, None
        ),
        "nvrtcCreateProgram",
    )
    options = [f"--gpu-architecture={arch}".encode(), b"--use_fast_math"]
    array = (ctypes.c_char_p * len(options))(*options)
    status = _nvrtc.nvrtcCompileProgram(program, len(options), array)
    if status != 0:
        size = ctypes.c_size_t()
        _nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        _nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(f"nvrtc: {log.value.decode(errors='replace')[:2000]}")

    size = ctypes.c_size_t()
    _check(_nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "nvrtcGetPTXSize")
    buffer = ctypes.create_string_buffer(size.value)
    _check(_nvrtc.nvrtcGetPTX(program, buffer), "nvrtcGetPTX")
    _nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    return buffer.value


class Module:
    """A compiled PTX module and the kernels in it."""

    def __init__(self, source: str, arch: str = "compute_90"):
        _init()
        import torch

        # Driver calls act on the calling thread's current context; make sure
        # torch has made one before asking for it.
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        context = ctypes.c_void_p()
        _driver.cuCtxGetCurrent(ctypes.byref(context))
        if not context.value:
            raise RuntimeError("no current CUDA context")

        ptx = compile_ptx(source, arch)
        self._module = ctypes.c_void_p()
        _check(_driver.cuModuleLoadData(ctypes.byref(self._module), ptx), "cuModuleLoadData")
        self._kernels = {}

    def kernel(self, name: str):
        if name not in self._kernels:
            handle = ctypes.c_void_p()
            _check(
                _driver.cuModuleGetFunction(
                    ctypes.byref(handle), self._module, name.encode()
                ),
                f"cuModuleGetFunction({name})",
            )
            self._kernels[name] = Kernel(handle, name)
        return self._kernels[name]


class _LaunchAttribute(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("pad", ctypes.c_int), ("value", ctypes.c_char * 64)]


class _LaunchConfig(ctypes.Structure):
    _fields_ = [
        ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint), ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint), ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint), ("hStream", ctypes.c_void_p),
        ("attrs", ctypes.c_void_p), ("numAttrs", ctypes.c_uint),
    ]


class Kernel:
    """One launchable kernel. Arguments are tensors and 32-bit ints."""

    __slots__ = ("_handle", "name", "_shared", "_keep", "pdl")

    def __init__(self, handle, name):
        self._handle = handle
        self.name = name
        self._shared = 0
        self._keep = None
        self.pdl = False

    def set_shared(self, nbytes: int) -> None:
        """Opt in to more than 48 KiB of shared memory, as Hopper allows."""
        if nbytes > 48 * 1024:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
            _check(
                _driver.cuFuncSetAttribute(self._handle, 8, ctypes.c_int(nbytes)),
                "cuFuncSetAttribute",
            )
        self._shared = nbytes

    def max_blocks(self, threads: int, shared: int) -> int:
        """Blocks that can be resident at once, which bounds a cooperative grid."""
        import torch

        count = ctypes.c_int()
        _check(
            _driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(count), self._handle,
                ctypes.c_int(threads), ctypes.c_size_t(shared),
            ),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        return count.value * sms

    def _pack(self, args):
        import torch

        keep, packed = [], []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                cell = ctypes.c_void_p(arg.data_ptr())
            elif isinstance(arg, int):
                cell = ctypes.c_int(arg)
            elif isinstance(arg, float):
                cell = ctypes.c_float(arg)
            else:
                raise TypeError(f"cannot pass {type(arg)} to a kernel")
            keep.append(cell)
            packed.append(ctypes.cast(ctypes.byref(cell), ctypes.c_void_p))
        self._keep = keep
        return (ctypes.c_void_p * len(packed))(*packed)

    def cooperative(self, grid, block, *args):
        """Launch so every block is resident, which makes a grid barrier safe.

        Captures into a CUDA graph and replays correctly -- verified before any
        of this was built on.
        """
        import torch

        array = self._pack(args)
        stream = torch.cuda.current_stream().cuda_stream
        _check(
            _driver.cuLaunchCooperativeKernel(
                self._handle,
                ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
                ctypes.c_uint(block), ctypes.c_uint(1), ctypes.c_uint(1),
                ctypes.c_uint(self._shared),
                ctypes.c_void_p(stream),
                array,
            ),
            f"cuLaunchCooperativeKernel({self.name})",
        )

    # ``pdl`` (a slot, set in __init__): launch with programmatic stream
    # serialization, so the kernel may begin while the previous kernel on the
    # stream is still running; it must then execute ``griddepcontrol.wait``
    # before reading anything that kernel writes.

    def launch_2d(self, grid_x, grid_y, block, *args):
        """A two-dimensional grid, programmatic when ``pdl`` is set."""
        if self.pdl:
            return self.launch_ex((grid_x, grid_y), block, *args, pdl=True)
        return self.launch_ex((grid_x, grid_y), block, *args, pdl=False)

    def launch_ex(self, grid, block, *args, pdl: bool = True):
        """``cuLaunchKernelEx`` with the programmatic-serialization attribute.

        Captures into CUDA graphs as a programmatic edge (CUDA 12.3+).
        ``grid`` is an int or an ``(x, y)`` pair.
        """
        import torch

        if isinstance(grid, int):
            grid = (grid, 1)
        array = self._pack(args)
        stream = torch.cuda.current_stream().cuda_stream
        attrs = (_LaunchAttribute * 1)()
        attrs[0].id = 6   # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        # The union's first int is programmaticStreamSerializationAllowed. A
        # c_char field reads back as a copy, so write through the address.
        ctypes.c_int.from_address(ctypes.addressof(attrs[0]) + _LaunchAttribute.value.offset).value = 1 if pdl else 0
        config = _LaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid[0], grid[1], 1
        config.blockDimX, config.blockDimY, config.blockDimZ = block, 1, 1
        config.sharedMemBytes = self._shared
        config.hStream = ctypes.c_void_p(stream)
        config.attrs = ctypes.cast(attrs, ctypes.c_void_p)
        config.numAttrs = 1
        _check(
            _driver.cuLaunchKernelEx(ctypes.byref(config), self._handle, array, None),
            f"cuLaunchKernelEx({self.name})",
        )

    def __call__(self, grid, block, *args):
        import torch

        if self.pdl:
            return self.launch_ex(grid, block, *args, pdl=True)
        packed = []
        keep = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                cell = ctypes.c_void_p(arg.data_ptr())
            elif isinstance(arg, int):
                cell = ctypes.c_int(arg)
            elif isinstance(arg, float):
                cell = ctypes.c_float(arg)
            else:
                raise TypeError(f"cannot pass {type(arg)} to a kernel")
            keep.append(cell)
            packed.append(ctypes.cast(ctypes.byref(cell), ctypes.c_void_p))
        array = (ctypes.c_void_p * len(packed))(*packed)

        stream = torch.cuda.current_stream().cuda_stream
        _check(
            _driver.cuLaunchKernel(
                self._handle,
                ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
                ctypes.c_uint(block), ctypes.c_uint(1), ctypes.c_uint(1),
                ctypes.c_uint(self._shared),
                ctypes.c_void_p(stream),
                array, None,
            ),
            f"cuLaunchKernel({self.name})",
        )
