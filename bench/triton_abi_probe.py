"""What Triton 3.1 records about a compiled kernel's parameters, for the RoPE kernel at batch 4."""
import sys
import torch
sys.path.insert(0, "/root/engine")
from kernels import triton_pdl
from kernels.rope import qkv_planes_norm_rope_to_cache, _qkv_planes_launch

triton_pdl.ENABLED = True
batch, tokens, context, n_q, n_kv, d = 4, 1, 2048, 32, 8, 128
rows, cap = batch * tokens, context + 16
planes = torch.randn(4, rows, (n_q + 2 * n_kv) * d, dtype=torch.float32, device="cuda")
qw = torch.ones(d, dtype=torch.bfloat16, device="cuda"); kw = qw.clone()
cos = torch.randn(cap, d, dtype=torch.bfloat16, device="cuda"); sin = torch.randn(cap, d, dtype=torch.bfloat16, device="cuda")
positions = torch.full((rows,), context - 1, dtype=torch.int32, device="cuda")
k_cache = torch.zeros(batch, n_kv, cap, d, dtype=torch.bfloat16, device="cuda"); v_cache = torch.zeros_like(k_cache)
q0 = qkv_planes_norm_rope_to_cache(planes, n_q, n_kv, qw, kw, cos, sin, positions, tokens, k_cache, v_cache, 1e-6)
k0, v0 = k_cache.clone(), v_cache.clone()
(kernel, keep), = _qkv_planes_launch.compiled.values()
print("kept parameter indices:", keep, flush=True)
k_cache.zero_(); v_cache.zero_()
q1 = qkv_planes_norm_rope_to_cache(planes, n_q, n_kv, qw, kw, cos, sin, positions, tokens, k_cache, v_cache, 1e-6)
torch.cuda.synchronize()
print("q equal", torch.equal(q0, q1), "k equal", torch.equal(k0, k_cache), "v equal", torch.equal(v0, v_cache), flush=True)
if not torch.equal(k0, k_cache):
    diff = (k0.float() - k_cache.float()).abs()
    idx = (diff > 0).nonzero()
    print("k differs at batches", idx[:, 0].unique().tolist(), "heads", idx[:, 1].unique().tolist(),
          "positions", idx[:, 2].unique().tolist()[:8], flush=True)
    print("plain row0 head0 pos2047 first 4:", k0[0, 0, 2047, :4].tolist(), " handle:", k_cache[0, 0, 2047, :4].tolist())
