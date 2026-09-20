"""Does the split-K consumer fusion improve complete generation?"""
import json
import os
import sys
import time
import zlib
os.environ['DRYFT_SHORT_DRAFT']='0'
os.environ['DRYFT_PARTIAL_SWIGLU']='off'
sys.path.insert(0,'/root/engine')
sys.path.insert(1,'/root')
import torch
import engine as mod
from engine import Engine
from kernels.speculation import ShortVerifier
from harness import _prompts, _time_stream, _replay, load_corpus
from gpu_checks import check_partial_swiglu
from findings_study import summary


@torch.inference_mode()
def run(shape,samples,corpus):
    check_partial_swiglu()
    name,batch,length,steps=shape
    if corpus=='code':
        import transformers.models.qwen3.modeling_qwen3 as qwen
        path=qwen.__file__
    else:path='/root/technical.txt' if corpus=='technical' else '/root/corpus.txt'
    load_corpus(path,'/weights/qwen3-4b')
    e=Engine('/weights/qwen3-4b')
    seed=zlib.crc32(name.encode())%10**6
    vocab=e.embed.shape[0]
    warm=_prompts(batch,length,vocab,seed)
    list(e.generate(warm,steps))
    original=e.__dict__.copy()
    states={};dispatch={}
    for label,enabled in [('off',False),('on',True)]:
        e.__dict__.clear();e.__dict__.update(original)
        mod.PARTIAL_SWIGLU=enabled
        if enabled and not e.integer_mlp:
            e._choose_mlp(batch,e.families['gate_up'])
        e._capture()
        if batch==1:e.short_verifier=ShortVerifier(e,max_draft=2)
        list(e.generate(warm,steps))
        assert e._fast
        states[label]=e.__dict__.copy()
        dispatch[label]={'fused_mlp':repr(e.fused_mlp),'integer_mlp':e.integer_mlp}
        print('READY',name,corpus,label,dispatch[label],flush=True)
    records={label:[] for label in states}
    for sample in range(samples):
        prompt=_prompts(batch,length,vocab,seed+sample+1)
        pending=[]
        for label in (('off','on') if sample%2==0 else ('on','off')):
            e.__dict__.clear();e.__dict__.update(states[label])
            mod.PARTIAL_SWIGLU=label=='on'
            ttft,seconds,tokens=_time_stream(e.generate,prompt,steps)
            pending.append((label,ttft,seconds,tokens))
        for label,ttft,seconds,tokens in pending:
            gap,exact=_replay(e._native,e.device,prompt,tokens,steps)
            row={'ttft':ttft,'seconds':seconds,'gap':gap,'exact':exact}
            records[label].append(row)
            print('SAMPLE',name,corpus,label,sample,json.dumps(row),flush=True)
        if pending[0][3]!=pending[1][3]:
            print('NOTE: generated streams differ; inspect native replay gaps',flush=True)
    results={k:summary(v,batch,steps) for k,v in records.items()}
    for r in results.values():r['speedup']=results['off']['seconds']/r['seconds']
    return {'shape':shape,'corpus':corpus,'variants':results,'dispatch':dispatch}


if __name__=='__main__':
    print('RESULT_JSON='+json.dumps(run(json.loads(sys.argv[1]),int(sys.argv[2]),sys.argv[3])))
