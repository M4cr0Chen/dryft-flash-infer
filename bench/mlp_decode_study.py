"""Experiments only; no changes to the production dispatch or precision policy."""
import functools
import inspect
import json
import statistics
import sys
import textwrap
import time
import types
from pathlib import Path

sys.path.insert(0, "/root")
sys.path.insert(0, "/root/experiment_engine")
import torch
import torch.nn.functional as F
import engine as mod
from engine import Engine
from kernels import cuda_fp8, cuda_small
from kernels.timing import time_calls
from harness import _prompts, _time_stream, _replay, _validate_stream, load_corpus
from prefill_mlp_kernel import CONFIGS, LARGE_CONFIGS, fused, pack
from replay import FixedDecodeReplay


@torch.inference_mode()
def micro(e, layout=False):
    weights = [l.gate_up for l in e.layers[::6]]
    packed = [pack(w) for w in weights]
    results = []
    for rows in (512, 8192):
        x = torch.randn(rows,e.hidden,device=e.device,dtype=torch.bfloat16)
        reference = mod.swiglu(F.linear(x,weights[0]))
        baseline = time_calls(lambda w:mod.swiglu(F.linear(x,w)),weights,reps=1,trials=5)
        candidates = []
        candidates_to_run = [(cfg,False) for cfg in CONFIGS]
        if layout:
            candidates_to_run = [(cfg,t) for t in (False,True) for cfg in LARGE_CONFIGS]
        transposed = [w.T.contiguous() for w in packed] if layout else []
        for cfg,transpose in candidates_to_run:
            try:
                operands=transposed if transpose else packed
                got = fused(x,operands[0],cfg,transposed=transpose)
                diff = (got.float()-reference.float()).abs()
                elapsed = time_calls(lambda w:fused(x,w,cfg,transposed=transpose),operands,reps=1,trials=5)
                row = {"config":cfg,"transposed":transpose,"ms":elapsed,"speedup":baseline/elapsed,
                       "max_abs":diff.max().item(),
                       "relative_rms":(diff.square().mean().sqrt()/reference.float().square().mean().sqrt()).item()}
            except Exception as exc:
                row = {"config":cfg,"transposed":transpose,"error":repr(exc)}
            candidates.append(row)
            print("MICRO",rows,json.dumps(row),flush=True)
        results.append({"rows":rows,"baseline_ms":baseline,"candidates":candidates})
    return results


def tune_consumers(e):
    """Race unchanged weight families together with their actual consumers."""
    history = {}
    for name in ("qkv", "o", "down", "gate_up"):
        if e.families[name] != "fp8" or (name == "gate_up" and e.integer_mlp):
            continue
        weights = e.operand[name]
        x = torch.randn(e.batch, getattr(e.layers[0],name).shape[1],
                        device=e.device,dtype=torch.bfloat16)
        residual = torch.randn(e.batch,e.hidden,device=e.device,dtype=torch.bfloat16)
        norm_consumer = (cuda_small.add_rms_norm_partials_separate if e._cuda_small()
                         else mod.add_rms_norm_partials_separate)
        swiglu_consumer = cuda_small.swiglu_partials if e._cuda_small() else mod.swiglu_partials

        def run(config, index):
            operand = weights[index]
            if name == "gate_up" and config[0] == "fused":
                return cuda_fp8.gate_up_swiglu(x,operand,config[1])
            planes = cuda_fp8.matmul_partials(x,operand,config)
            if name == "qkv":
                layer=e.layers[index]
                return mod.qkv_planes_norm_rope_to_cache(
                    planes,e.n_q,e.n_kv,layer.q_norm,layer.k_norm,e.cos,e.sin,e.pos,1,
                    e.k_cache[index],e.v_cache[index],e.eps)
            if name == "gate_up":
                return swiglu_consumer(planes)
            norm = (e.layers[index].norm_post if name == "o" else
                    e.layers[index+1].norm_in if index+1<e.n_layers else e.norm_out)
            if name == "o" and e.integer_mlp and mod.INT8_NORM_FUSION:
                from kernels.cuda_int8 import add_norm_quant
                return add_norm_quant(residual,planes,norm,e.eps)
            return norm_consumer(residual,planes,norm,e.eps)

        incumbent = e.partial_config.get(name) or getattr(e.matmul[name],"keywords",{}).get("config")
        if incumbent is None:
            continue
        if name == "gate_up":
            initial = time_calls(lambda i:e.fused_mlp(x,e.fused_operand[i])
                                 if e.fused_mlp is not None else mod.swiglu(e.matmul[name](x,weights[i])),
                                 list(range(e.n_layers)),reps=1,trials=5)
        else:
            initial = time_calls(lambda i:run(incumbent,i),list(range(e.n_layers)),reps=1,trials=5)
        best = initial
        chosen = None
        trials = []
        configs = list(cuda_fp8.CONFIGS)
        configs += [(8,8,1,1,2),(4,2,1,1,2),(8,1,1,1,2)]
        if name == "gate_up":
            configs += [("fused",cfg) for cfg in cuda_fp8.SWIGLU_CONFIGS]
        for cfg in configs:
            try:
                elapsed=time_calls(lambda i:run(cfg,i),list(range(e.n_layers)),reps=1,trials=3)
                trials.append({"config":cfg,"ms":elapsed,"speedup":initial/elapsed})
                if elapsed<best*.985:
                    best,chosen=elapsed,cfg
            except Exception as exc:
                trials.append({"config":cfg,"error":repr(exc)})
        if chosen is not None:
            # Recheck the winner to limit a one-off clock/timing fluctuation.
            confirm=time_calls(lambda i:run(chosen,i),list(range(e.n_layers)),reps=1,trials=7)
            if confirm>=initial*.985:
                chosen=None
        if chosen is not None:
            if name == "gate_up":
                if chosen[0]=="fused":
                    e.fused_mlp=functools.partial(cuda_fp8.gate_up_swiglu,config=chosen[1])
                else:
                    def fused_planes(source,w,cfg=chosen,consume=swiglu_consumer):
                        return consume(cuda_fp8.matmul_partials(source,w,cfg))
                    e.fused_mlp=fused_planes
                e.fused_operand=weights
            else:
                e.matmul[name]=functools.partial(cuda_fp8.matmul,config=chosen)
                e.partial_config[name]=chosen
        history[name]={"incumbent":incumbent,"initial_ms":initial,"chosen":chosen,"trials":trials}
        print("STAGE_TUNE",e.batch,name,json.dumps(history[name]),flush=True)
    e._ordinary_matmul=dict(e.matmul)
    return history


def state(e):
    result=e.__dict__.copy()
    for n in ("matmul","operand","partial_config","fused_out","fused_scratch"):
        result[n]=result[n].copy()
    return result


def tune_graph(e, history):
    """Validate promising stage configurations in the complete decode graph."""
    selected={}
    def measure():
        fixed=FixedDecodeReplay(e)
        return time_calls(lambda _:fixed.replay(),[None],reps=20,trials=5,use_graph=False)
    for name,record in history.items():
        before=state(e)
        initial=measure()
        best=initial;best_state=None;chosen=None;trials=[]
        candidates=sorted([r for r in record['trials'] if 'ms' in r],key=lambda r:r['ms'])[:4]
        for row in candidates:
            restore(e,before)
            e.matmul=e.matmul.copy();e.partial_config=e.partial_config.copy()
            cfg=row['config']
            if name=='gate_up':
                if cfg[0]=='fused':
                    e.fused_mlp=functools.partial(cuda_fp8.gate_up_swiglu,config=cfg[1])
                else:
                    consume=cuda_small.swiglu_partials if e._cuda_small() else mod.swiglu_partials
                    e.fused_mlp=lambda x,w,c=cfg,f=consume:f(cuda_fp8.matmul_partials(x,w,c))
                e.fused_operand=e.operand[name]
            else:
                e.matmul[name]=functools.partial(cuda_fp8.matmul,config=cfg)
                e.partial_config[name]=cfg
            elapsed=measure()
            trials.append({'config':cfg,'step_ms':elapsed,'speedup':initial/elapsed})
            if elapsed<best*.99:
                best,best_state,chosen=elapsed,state(e),cfg
        restore(e,best_state or before)
        selected[name]={'initial_ms':initial,'chosen':chosen,'trials':trials}
        print('GRAPH_TUNE',e.batch,name,json.dumps(selected[name]),flush=True)
    e._ordinary_matmul=dict(e.matmul)
    return selected


def restore(e,s):
    e.__dict__.clear();e.__dict__.update(s)


def summary(rows,batch,steps):
    med=statistics.median(r["total_ms"] for r in rows)
    return {"total_ms":med,"tok_s":batch*steps*1000/med,
            "ttft_ms":statistics.median(r["ttft_ms"] for r in rows),
            "tpot_ms":statistics.median(r["tpot_ms"] for r in rows),
            "worst_gap":max(r["gap"] for r in rows),
            "passes":all(r["gap"]<=2 for r in rows),
            "spread":(max(r["total_ms"] for r in rows)-min(r["total_ms"] for r in rows))/med,
            "samples":rows}


def install_prefill_fusion(e):
    packed={ly.gate_up.data_ptr():pack(ly.gate_up).T.contiguous() for ly in e.layers}
    def run(x,w):
        cfg=(128,256,64,8,4 if x.shape[0]<=1024 else 3)
        return fused(x,packed[w.data_ptr()],cfg,transposed=True)
    source=textwrap.dedent(inspect.getsource(Engine._blocks))
    old='            inner = swiglu(project("gate_up", index, normed, layer.gate_up))'
    new='            inner = (self._prefill_fusion(normed, layer.gate_up) if matmul is None else swiglu(project("gate_up", index, normed, layer.gate_up)))'
    assert old in source
    scope=dict(vars(mod));exec(source.replace(old,new),scope)
    e._blocks=types.MethodType(scope['_blocks'],e)
    e._prefill_fusion=run
    return {"config":"128x256x64 / 8 warps / stages4 <=1024 rows else3", "weight_layout":"K x interleaved(gate,up)"}


def install_production_prefill(e):
    from kernels.prefill_mlp import PrefillMLP
    e.prefill_mlp=PrefillMLP([ly.gate_up for ly in e.layers])
    chunk=max(1,mod.PREFILL_ROW_BUDGET//e.seq_len)
    for start in range(0,e.batch,chunk):
        e.prefill_mlp.tune(min(chunk,e.batch-start)*e.seq_len)
    return {'configs':dict(e.prefill_mlp.configs)}


@torch.inference_mode()
def compare(e,shape,samples,stage):
    batch,context,steps=shape
    warm=_prompts(batch,context,e.embed.shape[0],12000)
    list(e.generate(warm,steps))
    assert e._fast
    baseline=state(e)
    e.matmul=e.matmul.copy();e.partial_config=e.partial_config.copy()
    if stage=="prefill":
        tuning=install_prefill_fusion(e)
    elif stage in ("production","combined"):
        tuning=install_production_prefill(e)
    else:
        tuning=tune_consumers(e)
    if stage=="decode_graph":
        restore(e,baseline)
        tuning={'stages':tuning,'graph':tune_graph(e,tuning)}
    e._capture()
    if stage=="combined":
        from kernels.decode_tuning import tune_output_projection
        tuning['decode_changed']=tune_output_projection(e)
        tuning['output_config']=e.partial_config.get('o')
        e._capture()
    if e.short_verifier is not None and stage not in ("prefill","production"):
        from kernels.speculation import ShortVerifier
        e.short_verifier=ShortVerifier(e,max_draft=2)
    candidate=state(e)
    rows={"baseline":[],"candidate":[]}
    fixed={};prefill={}
    for label,s in (("baseline",baseline),("candidate",candidate)):
        restore(e,s)
        e._prefill(e._upload(warm,batch,context))
        graph=FixedDecodeReplay(e)
        from kernels.timing import time_calls
        fixed[label]=time_calls(lambda _:graph.replay(),[None],reps=20,trials=7,use_graph=False)
        ids=e._upload(warm,batch,context)
        prefill[label]=time_calls(lambda _:e._prefill(ids),[None],reps=1,trials=7,use_graph=False)
    corpora=['prose','code','technical'] if stage in ("production","combined") else ['prose']
    for sample in range(samples*len(corpora)):
        corpus=corpora[sample//samples]
        if sample%samples==0:
            import transformers.models.qwen3.modeling_qwen3 as qwen
            path={'prose':'/root/corpus.txt','code':qwen.__file__,'technical':'/root/technical.txt'}[corpus]
            load_corpus(path,'/weights/qwen3-4b')
        prompt=_prompts(batch,context,e.embed.shape[0],12001+sample)
        pending=[]
        for label in (("baseline","candidate") if sample%2==0 else ("candidate","baseline")):
            restore(e,baseline if label=="baseline" else candidate)
            ttft,total,tokens=_time_stream(e.generate,prompt,steps)
            _validate_stream(tokens,batch,steps,e.embed.shape[0])
            pending.append((label,ttft,total,tokens))
        for label,ttft,total,tokens in pending:
            gap,exact=_replay(e._native,e.device,prompt,tokens,steps)
            row={"corpus":corpus,"ttft_ms":ttft*1000,"total_ms":total*1000,
                 "tpot_ms":(total-ttft)*1000/(steps-1),"gap":gap,"exact":exact}
            rows[label].append(row)
            print("PAIR",shape,label,sample,json.dumps(row),flush=True)
    summaries={label:summary(r,batch,steps) for label,r in rows.items()}
    return {"shape":shape,"stage":stage,"tuning":tuning,"fixed_decode_ms":fixed,"prefill_ms":prefill,
            "variants":summaries,"speedup":summaries["baseline"]["total_ms"]/summaries["candidate"]["total_ms"]}


if __name__ == "__main__":
    stage, shape, samples = sys.argv[1:]
    shape = json.loads(shape)
    torch.manual_seed(42)
    # This study installs each candidate explicitly after warming an otherwise
    # identical baseline, even after the production defaults enable them.
    mod.PREFILL_MLP = False
    mod.DECODE_OUT_TUNE = False
    load_corpus("/root/corpus.txt","/weights/qwen3-4b")
    e = Engine("/weights/qwen3-4b")
    if stage in ("micro","layout"):
        result = micro(e,layout=stage=="layout")
    elif stage in ("decode","decode_graph","prefill","production","combined"):
        result = compare(e,shape,int(samples),stage)
    else:
        raise ValueError(stage)
    Path("/tmp/mlp-decode-result.json").write_text(json.dumps(result))
