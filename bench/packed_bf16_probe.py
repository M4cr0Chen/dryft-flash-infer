"""Test pre-permuted BF16 activations without changing weight/activation values."""
import functools
import json
import os
import sys
os.environ['DRYFT_INT8_MMA'] = 'off'
sys.path.insert(0, '/root/engine')
import torch
import triton
from engine import Engine
from kernels import cuda_fp8 as old, cuda_jit
from kernels.timing import time_calls
from kernels.swiglu import swiglu


def pack(x):
    k = x.shape[1]
    cols = (torch.arange(k // 64, device=x.device)[:, None] * 64 + old._permutation(x.device)[None, :]).flatten()
    return x[:, cols].contiguous()


def template():
    source = old._TEMPLATE
    start = source.index('#if @STAGE@\n    // Stage')
    stop = source.index('    if (!live) return;', start)
    source = source[:start] + r'''
#if @STAGE@
    const int pieces = ngroups * 16;
    const int items = @NT@ * 8 * pieces;
    for (int base=threadIdx.x; base<items; base+=@THREADS@*8) {
        uint4 v[8];
        #pragma unroll
        for(int u=0;u<8;++u) {
            const int it=base+u*@THREADS@;
            const int n=it/pieces,piece=it%pieces;
            v[u]=make_uint4(0,0,0,0);
            if(it<items && n<B)
                v[u]=*reinterpret_cast<const uint4*>(X+(size_t)n*K+k0+piece*8);
        }
        #pragma unroll
        for(int u=0;u<8;++u) {
            const int it=base+u*@THREADS@;
            const int n=it/pieces,piece=it%pieces;
            if(it<items) *reinterpret_cast<uint4*>(xs+n*stride+piece*8)=v[u];
        }
    }
    __syncthreads();
#endif
''' + source[stop:]
    start = source.index('                    // Straight from the activation')
    stop = source.index('#endif', start)
    source = source[:start] + r'''
                    #pragma unroll
                    for(int nt=0;nt<@NT@;++nt) {
                        const int n=n0+nt*8+g;
                        const uint4* p=reinterpret_cast<const uint4*>(
                            X+(size_t)n*K+k0+local*128+j*64+t*16);
                        xb[nt][0]=n<B ? __ldg(p) : make_uint4(0,0,0,0);
                        xb[nt][1]=n<B ? __ldg(p+1) : make_uint4(0,0,0,0);
                    }
''' + source[stop:]
    return source


@functools.lru_cache(None)
def module(nt):
    pieces = [old._PRELUDE, old._REDUCE]
    body = template()
    for warps in (4, 8):
        for mode in (0, 1, 2):
            for stage in (0, 2):
                src = body
                for key, value in {'NAME': old._name(warps, nt, mode, stage, 1),
                                   'WARPS': warps, 'THREADS': warps*32,
                                   'NT': nt, 'NSHARE': 1, 'MODE': mode, 'STAGE': stage}.items():
                    src = src.replace('@'+key+'@', str(value))
                pieces.append(src)
    return cuda_jit.Module('\n'.join(pieces))


def matmul(x, w, config, swiglu_mode=False):
    b, k = x.shape
    warps, nt, splits, gps, stage, _ = old._plan(w, b, config)
    mode = 2 if swiglu_mode else (1 if splits > 1 else 0)
    if swiglu_mode and splits != 1:
        raise ValueError('SwiGLU requires a complete K reduction')
    n = w.rows // 2 if swiglu_mode else w.rows
    out = torch.empty((splits, b, n) if splits > 1 else (b, n), device=x.device,
                      dtype=torch.float32 if splits > 1 else torch.bfloat16)
    kernel = module(nt).kernel(old._name(warps, nt, mode, stage, 1))
    kernel.set_shared(old._shared_bytes(nt, gps) if stage else 0)
    rows = warps * (8 if swiglu_mode else 16)
    kernel(triton.cdiv(n, rows)*splits, warps*32, w.weight, w.scale, x, out,
           n, k, b, w.groups, gps, splits)
    if splits > 1:
        reduced = torch.empty((b,n),device=x.device,dtype=torch.bfloat16)
        module(nt).kernel('reduce_partials')(triton.cdiv(b*n,256),256,out,reduced,b*n,splits)
        return reduced
    return out


@torch.inference_mode()
def main():
    torch.manual_seed(781)
    for b in (1,3,4,12,16,24,32):
        x=torch.randn(b,2560,device='cuda',dtype=torch.bfloat16)
        w=old.prepare(old.quantize(torch.randn(128,2560,device='cuda',dtype=torch.bfloat16)))
        packed=pack(x)
        for stage in (0,2):
            for split in (1,4):
                reference=old.matmul(x,w,(4,split))
                actual=matmul(packed,w,(4,split,stage))
                assert torch.equal(reference,actual), (b,stage,split,(reference.float()-actual.float()).abs().max().item())
            reference=old.gate_up_swiglu(x,w,(4,1))
            actual=matmul(packed,w,(4,1,stage),True)
            assert torch.equal(reference,actual), (b,stage,'swiglu')
    print('prepacked BF16: 42 bit-exact projection and SwiGLU checks passed',flush=True)
    engine=Engine('/weights/qwen3-4b')
    results=[]
    for b in (1,4,16,32):
        engine._ensure(b,512,32)
        for name in ('qkv','o','gate_up','down','lm_head'):
            family=engine.quantised[name]
            x=torch.randn(b,family[0].k,device='cuda',dtype=torch.bfloat16)
            packed=pack(x)
            incumbent=time_calls(lambda w: engine.matmul[name](x,w),engine.operand[name],reps=1,trials=3)
            rows=[]
            for stage in (0,2):
                for warps,split in ((4,1),(8,1),(4,2),(8,2),(4,4),(8,4)):
                    config=(warps,split,stage)
                    ms=time_calls(lambda w: matmul(packed,w,config),family,reps=1,trials=3)
                    rows.append({'config':config,'us':ms*1000})
            best=min(rows,key=lambda r:r['us'])
            row={'batch':b,'projection':name,'incumbent_us':incumbent*1000,
                 'best':best,'speedup_ceiling':incumbent*1000/best['us'],'candidates':rows}
            if name=='gate_up':
                old_fn=(lambda w: engine.fused_mlp(x,w)) if engine.fused_mlp else (lambda w: swiglu(engine.matmul[name](x,w)))
                old_op=engine.fused_operand if engine.fused_mlp else engine.operand[name]
                ms=time_calls(old_fn,old_op,reps=1,trials=3)
                fs=[]
                for cfg in ((4,1,0),(8,1,0),(4,1,2),(8,1,2)):
                    t=time_calls(lambda w: matmul(packed,w,cfg,True),family,reps=1,trials=3)
                    fs.append({'config':cfg,'us':t*1000})
                row['swiglu']={'incumbent_us':ms*1000,'candidates':fs}
            results.append(row);print(json.dumps(row),flush=True)
    return results


if __name__=='__main__':
    print('RESULT_JSON='+json.dumps(main()))
