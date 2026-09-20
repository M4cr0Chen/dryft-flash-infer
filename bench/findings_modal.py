"""Run production feature combinations and extended shape/corpus coverage."""
import hashlib
import json
from pathlib import Path
import sys
import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from modal_bench import image,weights,_describe_gpu

image=(image.add_local_file('bench/findings_study.py','/root/findings_study.py')
       .add_local_file('bench/partial_swiglu_study.py','/root/partial_swiglu_study.py')
       .add_local_file('bench/modal_bench.py','/root/modal_bench.py'))
app=modal.App('dryft-apply-findings')


@app.function(image=image,gpu='H100',volumes={'/weights':weights},timeout=3600)
def run_studies(shapes,samples,corpora,panel):
    import subprocess
    _describe_gpu(require_h100=True)
    rows=[]
    for shape in shapes:
        for corpus in corpora:
            script='/root/partial_swiglu_study.py' if panel=='swiglu' else '/root/findings_study.py'
            process=subprocess.Popen([sys.executable,script,json.dumps(shape),
                                      str(samples),corpus,panel],stdout=subprocess.PIPE,text=True)
            for line in process.stdout:
                if line.startswith('RESULT_JSON='):rows.append(json.loads(line.removeprefix('RESULT_JSON=')))
                else:print(line,end='',flush=True)
            if process.wait():raise RuntimeError(f'findings study failed: {shape} / {corpus}')
    return rows


@app.local_entrypoint()
def main(shape_set:str='core',samples:int=5,corpus:str='all',panel:str='pair',
         output:str='bench/results/findings-coverage.json'):
    from datetime import datetime,timezone
    groups={
        'core':[('public-2',16,512,128),('coverage-long-output',1,512,128)],
        'coverage':[('coverage-odd',3,257,33),('coverage-medium',8,1024,64),
                    ('coverage-wide',32,256,32),('coverage-12',12,1024,64)],
        'long':[('coverage-long',1,4096,65),('wide-long-output',32,512,128),
                ('wide-prefill',16,4096,16)],
        'wide':[('wide-long-output',32,512,128)],
        'large-batch':[('coverage-64',64,256,32)],
        'public':[('public-0',1,512,32),('public-1',4,2048,32),('public-2',16,512,128)],
    }
    digest=hashlib.sha256()
    for p in sorted(Path('engine').rglob('*.py')):
        digest.update(p.relative_to('engine').as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    result={'started_at':datetime.now(timezone.utc).isoformat(),'engine_sha256':digest.hexdigest(),
            'samples':samples,'results':run_studies.remote(groups[shape_set],samples,
                ['prose','code','technical'] if corpus=='all' else [corpus],panel)}
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    for r in result['results']:
        print(r['shape'],r['corpus'],{k:{n:v[n] for n in ('tps','speedup','worst_gap','spread')}
                                    for k,v in r['variants'].items()})
    print('Saved',output)
