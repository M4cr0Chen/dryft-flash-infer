"""Remove one optimization at a time from the frozen, ranked engine.

Graphs share weights/cache but own captured intermediates. Restore host state
and module switches before each trial. Every generation overwrites its prompt
cache; all timed samples are replayed through untouched native Qwen afterward.
"""
import functools
import inspect
import json
import math
import os
import statistics
import sys
import textwrap
import time
import types
import zlib

os.environ['DRYFT_SHORT_DRAFT'] = '0'
sys.path.insert(0, '/root/ablation_engine')
sys.path.insert(1, '/root')
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import engine as mod
from engine import Engine
from harness import _prompts, _replay, _time_stream, _validate_stream, load_corpus
from kernels import cuda_fp8, DecodeAttention
from kernels.speculation import ShortVerifier


@triton.jit
def _residual_only(X,D,R,WIDTH:tl.constexpr,SPLITS:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    total=tl.full((256,),0.,tl.float32)
    for part in range(SPLITS):
        total+=tl.load(D+part*WIDTH+i,mask=i<WIDTH,other=0).to(tl.float32)
    delta=total.to(tl.bfloat16).to(tl.float32)
    x=tl.load(X+i,mask=i<WIDTH,other=0).to(tl.float32)
    tl.store(R+i,(x+delta).to(tl.bfloat16),mask=i<WIDTH)


def signature(e):
    return {'families': dict(e.families), 'partials': dict(e.partial_config),
            'integer_mlp': e.integer_mlp, 'integer_head': e.integer_head,
            'fused_mlp': e.fused_mlp is not None,
            'attention_splits': e.decode_attention.splits if e.decode_attention else None,
            'runners': {k: repr(v) for k, v in e.matmul.items()}}


def sequential_stream(self, steps):
    for step in range(steps):
        if step:
            self.graph.replay() if self.graph is not None else self._decode_step()
        self.out_host[0].copy_(self.emitted, non_blocking=True)
        self.out_event[0].record()
        self.out_event[0].synchronize()
        yield self.out_host[0].tolist()


def simple_upload(self, ids, batch, length):
    return torch.tensor(ids, dtype=torch.int64, device=self.device)


def separate_add_norm(x, delta, weight, eps):
    residual = x + delta
    return residual, mod.rms_norm(residual, weight, eps)


def separate_partial_norm(x, partials, weight, eps):
    residual=torch.empty_like(x)
    _residual_only[(triton.cdiv(x.numel(),256),)](x,partials,residual,x.numel(),partials.shape[0],num_warps=4)
    return residual,mod.rms_norm(residual,weight,eps)


def separate_decode_blocks(self,x,positions,seq_len,batch,offset,attend,matmul=None):
    if matmul is None:
        return Engine._blocks(self,x,positions,seq_len,batch,offset,attend,matmul)
    add,partials=mod.add_rms_norm,mod.add_rms_norm_partials
    mod.add_rms_norm=separate_add_norm
    mod.add_rms_norm_partials=separate_partial_norm
    try:
        return Engine._blocks(self,x,positions,seq_len,batch,offset,attend,matmul)
    finally:
        mod.add_rms_norm,mod.add_rms_norm_partials=add,partials


def remove_mlp_fusion(e):
    if e.integer_mlp:
        kwargs = dict(e.fused_mlp.keywords)
        kwargs['mode'] = 0
        e.matmul['gate_up'] = functools.partial(e.fused_mlp.func, **kwargs)
        e.operand['gate_up'] = e.fused_operand
    elif e.fused_mlp is not None and e.families.get('gate_up') == 'fp8':
        config = e.fused_mlp.__defaults__[0]
        e.matmul['gate_up'] = functools.partial(cuda_fp8.matmul, config=config)
        e.operand['gate_up'] = e.fused_operand
    e.fused_mlp = None
    e.fused_operand = None


def clear_mask_step(e):
    source = textwrap.dedent(inspect.getsource(Engine._decode_step))
    source = source.replace('    torch.le(self.arange, self.pos, out=self.mask)\n', '')
    scope = dict(vars(mod))
    exec(source, scope)
    e._decode_step = types.MethodType(scope['_decode_step'], e)


def apply_variant(e, name):
    if '+' in name:
        for part in name.split('+'): apply_variant(e,part)
        return
    if name == 'ordinary':
        return
    if name in ('shipping', 'no_spec_governor', 'spec_target120', 'spec_target125'):
        if name == 'no_spec_governor': mod.SHORT_TARGET=float('inf')
        if name == 'spec_target120': mod.SHORT_TARGET=1.20
        if name == 'spec_target125': mod.SHORT_TARGET=1.25
        e.short_verifier = getattr(e,'_shared_ablation_verifier',None) or ShortVerifier(e,max_draft=2)
        return
    if name == 'no_stream_overlap':
        e._stream = types.MethodType(sequential_stream, e)
        return
    if name == 'simple_prompt_upload':
        e._upload = types.MethodType(simple_upload, e)
        return
    if name == 'no_decode_graph':
        e.graph = None
        return
    if name == 'no_unused_mask':
        clear_mask_step(e)
    elif name == 'no_activation_pack_fusion':
        mod.INT8_NORM_FUSION = False
    elif name == 'no_w8a8':
        e.integer_mlp = e.integer_head = False
        e._choose_mlp(e.batch, e.families.get('gate_up', 'bf16'))
    elif name in ('no_mlp_fusion', 'no_mlp_or_swiglu_fusion'):
        remove_mlp_fusion(e)
        if name == 'no_mlp_or_swiglu_fusion':
            mod.swiglu = lambda x: F.silu(x[:, :x.shape[1]//2]) * x[:, x.shape[1]//2:]
    elif name == 'no_partial_consumers':
        e.partial_config = {}
    elif name == 'no_rope_fusion':
        mod.FUSE_ROPE = False
        e.partial_config.pop('qkv', None)
    elif name == 'no_residual_norm_fusion':
        # Native W8A8's combined producer must also become separate here.
        mod.INT8_NORM_FUSION = False
        mod.add_rms_norm = separate_add_norm
        mod.add_rms_norm_partials = separate_partial_norm
    elif name == 'separate_decode_norm':
        # Preserve prefill and native INT8's activation packing. Split only
        # the ordinary decode residual/reduction from its RMSNorm.
        e._blocks=types.MethodType(separate_decode_blocks,e)
    elif name == 'bf16_cublas_only':
        for projection, family in e.families.items():
            if family == 'bf16':
                e.matmul[projection] = F.linear
                e.operand[projection] = ([e.embed] if projection == 'lm_head' else
                                        [getattr(layer, projection) for layer in e.layers])
        if e.families.get('gate_up') == 'bf16':
            e.fused_mlp = None
    elif name == 'fixed_int8_configs':
        configs = {'qkv': (4,4), 'o': (4,4), 'gate_up': (8,1),
                   'down': (8,4), 'lm_head': (4,1)}
        for projection, config in configs.items():
            if e.families.get(projection) != 'fp8':
                continue
            e.matmul[projection] = functools.partial(cuda_fp8.matmul, config=config)
            e.operand[projection] = e.quantised[projection]
            if projection in ('qkv','o','down'):
                e.partial_config[projection] = config
        if not e.integer_mlp and e.families.get('gate_up') == 'fp8':
            e.fused_mlp = functools.partial(cuda_fp8.gate_up_swiglu, config=(8,1))
            e.fused_operand = e.quantised['gate_up']
    elif name == 'no_weight_quantization':
        mod.USE_FP8 = mod.INT8_MMA = False
        e._choose_matmuls(e.batch)
    elif name in ('no_ring', 'no_ring_gate'):
        previous=e.__dict__.copy()
        configs,glu=cuda_fp8.CONFIGS,cuda_fp8.SWIGLU_CONFIGS
        try:
            cuda_fp8.CONFIGS=[c for c in configs if len(c)<5 or c[4]==0]
            cuda_fp8.SWIGLU_CONFIGS=[c for c in glu if len(c)<5 or c[4]==0]
            e._choose_matmuls(e.batch)
        finally:
            cuda_fp8.CONFIGS,cuda_fp8.SWIGLU_CONFIGS=configs,glu
        if name=='no_ring_gate':
            for projection in ('qkv','o','down','lm_head'):
                e.matmul[projection]=previous['matmul'][projection]
                e.operand[projection]=previous['operand'][projection]
            e.partial_config=previous['partial_config'].copy()
            e.integer_head=previous['integer_head']
    elif name=='force_native_w8a8':
        from kernels import cuda_int8
        if not (9<=e.batch<=16 and e.families['gate_up']=='fp8'):
            raise ValueError('native INT8 MLP is validated only for this weight family and batch range')
        operands=e.integer_operands.get('gate_up')
        if operands is None:
            operands=[cuda_int8.Prepared(w) for w in e.quantised['gate_up']]
        e.fused_mlp=functools.partial(cuda_int8.matmul,config=(8,1),mode=2)
        e.fused_operand=operands
        e.integer_mlp=True
    elif name=='down_max8_planes':
        from kernels.timing import time_calls
        if e.families['down']!='fp8': raise ValueError('down projection is not quantized')
        x=torch.randn(e.batch,e.layers[0].down.shape[1],device=e.device,dtype=torch.bfloat16)
        residual=torch.randn(e.batch,e.hidden,device=e.device,dtype=torch.bfloat16)
        pairs=list(zip(e.quantised['down'],[ly.norm_in for ly in e.layers]))
        def clock(config):
            return time_calls(lambda pair: mod.add_rms_norm_partials(
                residual,cuda_fp8.matmul_partials(x,pair[0],config),pair[1],e.eps),pairs,reps=1,trials=3)
        incumbent=e.partial_config.get('down')
        if incumbent is None: incumbent=getattr(e.matmul['down'],'keywords',{}).get('config')
        best=clock(incumbent);chosen=incumbent
        candidates=[(w,k,1,1,r) for w in (4,8) for k in (4,8) for r in (0,2)]
        for cfg in candidates:
            if cuda_fp8.splits(e.quantised['down'][0],e.batch,cfg)>8: continue
            elapsed=clock(cfg)
            if elapsed<best*.98: best,chosen=elapsed,cfg
        e.matmul['down']=functools.partial(cuda_fp8.matmul,config=chosen)
        e.operand['down']=e.quantised['down']
        e.partial_config['down']=chosen
    elif name == 'no_custom_linears':
        e.matmul, e.operand, e.partial_config = {}, {}, {}
        e.fused_mlp = None
        e.integer_mlp = e.integer_head = False
    elif name == 'sdpa_decode':
        e.decode_attention = None
    elif name in ('no_attention_splits', 'half_attention_splits', 'attention_target128'):
        splits = 1 if name == 'no_attention_splits' else max(1,e.decode_attention.splits//2)
        if name == 'attention_target128':
            splits=1
            while splits<32 and e.batch*e.n_kv*splits<128 and e.capacity//(splits*2)>=128:
                splits*=2
        e.decode_attention = DecodeAttention(e.batch, e.n_kv, e.n_q//e.n_kv,
                                             e.head_dim, e.capacity, e.device,
                                             config=(splits,64,4))
    elif name == 'no_fused_qkv_projection':
        base = e.matmul['qkv']
        cuts = (0, e.q_width, e.q_width+e.kv_width, e.q_width+2*e.kv_width)
        new_operands = []
        if e.families['qkv'] == 'fp8':
            for w in e.operand['qkv']:
                parts = []
                for start, stop in zip(cuts, cuts[1:]):
                    p = object.__new__(type(w))
                    p.weight, p.scale = w.weight[start:stop], w.scale[start:stop]
                    p.rows, p.k, p.groups = stop-start, w.k, w.groups
                    if hasattr(w,'tiled'): p.tiled=w.tiled
                    parts.append(p)
                new_operands.append(parts)
            e.matmul['qkv'] = lambda x, weights: torch.cat([base(x,w) for w in weights], dim=-1)
        else:
            new_operands = [[layer.qkv[a:b] for a,b in zip(cuts,cuts[1:])] for layer in e.layers]
            e.matmul['qkv'] = lambda x, weights: torch.cat([F.linear(x,w) for w in weights], dim=-1)
        e.operand['qkv'] = new_operands
        e.partial_config.pop('qkv', None)
    elif name == 'full_batch_prefill':
        mod.PREFILL_ROW_BUDGET = e.batch * e.seq_len
        return
    elif name == 'smaller_prefill_chunks':
        mod.PREFILL_ROW_BUDGET = max(e.seq_len, e.batch*e.seq_len//2)
        return
    elif name == 'no_prefill_gqa':
        e._enable_gqa = False
        return
    else:
        raise ValueError(name)
    e._capture()


def summarize(rows, batch, output):
    seconds = [r['seconds'] for r in rows]
    median = statistics.median(seconds)
    return {'tps':batch*output/median, 'seconds':median,
            'ttft':statistics.median(r['ttft'] for r in rows),
            'tpot':statistics.median((r['seconds']-r['ttft'])/max(1,output-1) for r in rows),
            'spread':(max(seconds)-min(seconds))/median,
            'worst_gap':max(r['gap'] for r in rows),
            'passes_accuracy':all(math.isfinite(r['gap']) and r['gap']<=2 for r in rows),
            'samples':rows}


@torch.inference_mode()
def study(shape, samples=3, panel='core', choices=None, corpus='prose'):
    name,batch,length,output = shape
    if corpus == 'code':
        import transformers.models.qwen3.modeling_qwen3 as qwen
        corpus_path = qwen.__file__
    else:
        corpus_path = '/root/technical.txt' if corpus == 'technical' else '/root/corpus.txt'
    load_corpus(corpus_path,'/weights/qwen3-4b')
    seed=zlib.crc32(name.encode())%10**6
    start=time.perf_counter()
    e=Engine('/weights/qwen3-4b')
    vocab=e.embed.shape[0]
    warm=_prompts(batch,length,vocab,seed)
    list(e.generate(warm,output))
    assert e._fast and e.graph is not None
    globals_names=[k for k in vars(mod) if k.isupper()]+['swiglu','add_rms_norm','add_rms_norm_partials']
    original_globals={k:getattr(mod,k) for k in globals_names}
    original=e.__dict__.copy()

    def restore(state, switches):
        e.__dict__.clear();e.__dict__.update(state)
        for k,v in switches.items(): setattr(mod,k,v)

    variants=['ordinary']
    if batch==1: variants.append('shipping')
    if panel=='speculation':
        variants+=['no_stream_overlap']
    elif panel=='prefill':
        variants+=['full_batch_prefill','smaller_prefill_chunks','no_prefill_gqa','simple_prompt_upload']
    else:
        variants+=['no_stream_overlap','simple_prompt_upload','no_unused_mask',
                   'no_mlp_fusion','no_mlp_or_swiglu_fusion','no_partial_consumers',
                   'no_rope_fusion','no_residual_norm_fusion','fixed_int8_configs',
                   'no_weight_quantization','no_custom_linears','sdpa_decode',
                   'no_decode_graph','no_fused_qkv_projection']
        if e.decode_attention.splits>1: variants.append('no_attention_splits')
        if e.decode_attention.splits>1: variants.append('half_attention_splits')
        if 'bf16' in e.families.values(): variants.append('bf16_cublas_only')
        if e.integer_mlp: variants+=['no_w8a8','no_activation_pack_fusion']
    if choices:
        variants = list(dict.fromkeys(['ordinary'] + choices))
    if batch==1 and any(v in variants for v in ('shipping','no_spec_governor','spec_target120','spec_target125')):
        shared=ShortVerifier(e,max_draft=2)
        original=e.__dict__.copy()
        original['_shared_ablation_verifier']=shared
    states={};errors={};setup={}
    for variant in variants:
        restore(original,original_globals)
        for attr in ('matmul','operand','partial_config','fused_out','fused_scratch'):
            setattr(e,attr,dict(getattr(e,attr)))
        stamp=time.perf_counter()
        try:
            apply_variant(e,variant)
            list(e.generate(warm,output))
            assert e._fast, 'variant fell back to native'
            states[variant]=(e.__dict__.copy(),{k:getattr(mod,k) for k in globals_names})
            setup[variant]={'seconds':time.perf_counter()-stamp, 'dispatch':signature(e)}
            print('READY',name,variant,round(setup[variant]['seconds'],2),flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc()
            errors[variant]=repr(error)
    records={v:[] for v in states};outputs={}
    for sample in range(samples):
        prompt=_prompts(batch,length,vocab,seed+sample+1)
        order=list(states)
        offset=(sample*5)%len(order)
        order=order[offset:]+order[:offset]
        if sample%2: order.reverse()
        pending=[]
        for variant in order:
            restore(*states[variant])
            ttft,total,tokens=_time_stream(e.generate,prompt,output)
            _validate_stream(tokens,batch,output,vocab)
            spec = dict(e.short_verifier.stats) if e.short_verifier else None
            pending.append((variant,ttft,total,tokens,spec))
        # Replay after the timing sweep so reference forwards cannot warm the
        # following variant's weights differently within the sweep.
        for variant,ttft,total,tokens,spec in pending:
            gap,exact=_replay(e._native,e.device,prompt,tokens,output)
            row={'ttft':ttft,'seconds':total,'gap':gap,'exact':exact}
            if spec is not None: row['speculation']=spec
            records[variant].append(row)
            print('SAMPLE',name,variant,sample,json.dumps(row),flush=True)
    results={v:summarize(rows,batch,output) for v,rows in records.items()}
    baseline=results['ordinary']['seconds']
    for v,r in results.items(): r['speedup_vs_ordinary']=baseline/r['seconds']
    return {'shape':shape,'panel':panel,'corpus':corpus,'gpu':torch.cuda.get_device_name(),
            'variants':results,'errors':errors,'setup':setup,
            'study_seconds':time.perf_counter()-start,
            'peak_memory_fraction':torch.cuda.max_memory_allocated()/torch.cuda.get_device_properties(0).total_memory}


if __name__=='__main__':
    print('RESULT_JSON='+json.dumps(study(json.loads(sys.argv[1]),int(sys.argv[2]),sys.argv[3],
                                        json.loads(sys.argv[4]) if len(sys.argv)>4 else None,
                                        sys.argv[5] if len(sys.argv)>5 else 'prose')))
