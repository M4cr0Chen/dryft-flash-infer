"""Validate the production implementations individually and in combination."""
import json
import math
import os
import statistics
import sys
import time
import zlib

os.environ['DRYFT_SHORT_DRAFT']='0'
sys.path.insert(0,'/root/engine')
sys.path.insert(1,'/root')
import torch
import engine as mod
from engine import Engine
from kernels import attention
from kernels.speculation import ShortVerifier
from harness import _prompts, _replay, _time_stream, _validate_stream, load_corpus


def summary(rows,batch,steps):
    times=[r['seconds'] for r in rows]
    median=statistics.median(times)
    return {'tps':batch*steps/median,'seconds':median,
            'ttft':statistics.median(r['ttft'] for r in rows),
            'tpot':statistics.median((r['seconds']-r['ttft'])/(steps-1) for r in rows),
            'spread':(max(times)-min(times))/median,
            'worst_gap':max(r['gap'] for r in rows),
            'passes_accuracy':all(math.isfinite(r['gap']) and r['gap']<=2 for r in rows),
            'samples':rows}


@torch.inference_mode()
def run(shape,samples,corpus,panel):
    name,batch,length,steps=shape
    if corpus=='code':
        import transformers.models.qwen3.modeling_qwen3 as qwen
        path=qwen.__file__
    else:
        path='/root/technical.txt' if corpus=='technical' else '/root/corpus.txt'
    load_corpus(path,'/weights/qwen3-4b')
    started=time.perf_counter()
    e=Engine('/weights/qwen3-4b')
    seed=zlib.crc32(name.encode())%10**6
    vocab=e.embed.shape[0]
    warm=_prompts(batch,length,vocab,seed)
    list(e.generate(warm,steps))
    assert e._fast and e.graph is not None
    original=e.__dict__.copy()
    # Each variant keeps the ordinary decode projection choices. Speculative
    # widths retain that weight family through the normal ShortVerifier API.
    variants={'baseline':(False,132,1.15), 'all':(True,128,1.20)}
    if panel=='components':
        variants={'baseline':(False,132,1.15),'norm':(True,132,1.15),
                  'attention':(False,128,1.15),'layouts':(True,128,1.15),
                  'governor':(False,132,1.20),'all':(True,128,1.20)}
    states={};dispatch={}
    for label,(separate,target,governor) in variants.items():
        e.__dict__.clear();e.__dict__.update(original)
        mod.SEPARATE_DECODE_NORM=separate
        mod.SHORT_LONG_TARGET=governor
        attention._PROGRAM_TARGET=target
        e.short_verifier=None
        e.decode_attention=attention.DecodeAttention(batch,e.n_kv,e.n_q//e.n_kv,
                                                     e.head_dim,e.capacity,e.device)
        e._capture()
        if batch==1:
            e.short_verifier=ShortVerifier(e,max_draft=2)
        list(e.generate(warm,steps))
        assert e._fast
        states[label]=e.__dict__.copy()
        dispatch[label]={'attention_splits':e.decode_attention.splits,
                         'integer_mlp':e.integer_mlp,
                         'families':dict(e.families),
                         'short_target':mod._short_target(steps)}
        print('READY',name,corpus,label,dispatch[label],flush=True)
    records={label:[] for label in variants}
    for sample in range(samples):
        prompt=_prompts(batch,length,vocab,seed+sample+1)
        order=list(variants)
        offset=sample%len(order)
        order=order[offset:]+order[:offset]
        if sample%2:order.reverse()
        pending=[]
        for label in order:
            e.__dict__.clear();e.__dict__.update(states[label])
            separate,target,governor=variants[label]
            mod.SEPARATE_DECODE_NORM=separate
            mod.SHORT_LONG_TARGET=governor
            attention._PROGRAM_TARGET=target
            ttft,total,tokens=_time_stream(e.generate,prompt,steps)
            _validate_stream(tokens,batch,steps,vocab)
            assert e._fast
            spec=dict(e.short_verifier.stats) if e.short_verifier else None
            pending.append((label,ttft,total,tokens,spec))
        for label,ttft,total,tokens,spec in pending:
            gap,exact=_replay(e._native,e.device,prompt,tokens,steps)
            row={'ttft':ttft,'seconds':total,'gap':gap,'exact':exact,'speculation':spec}
            records[label].append(row)
            print('SAMPLE',name,corpus,label,sample,json.dumps(row),flush=True)
    result={label:summary(rows,batch,steps) for label,rows in records.items()}
    for row in result.values():row['speedup']=result['baseline']['seconds']/row['seconds']
    return {'shape':shape,'corpus':corpus,'panel':panel,'variants':result,'dispatch':dispatch,
            'gpu':torch.cuda.get_device_name(),'elapsed_seconds':time.perf_counter()-started}


if __name__=='__main__':
    print('RESULT_JSON='+json.dumps(run(json.loads(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4])))
