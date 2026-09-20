"""Run feature-removal studies against an immutable copy of the ranked engine."""
import hashlib
import json
import os
from pathlib import Path
import sys
import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from modal_bench import image, weights, _describe_gpu

BASELINE_COMMIT=os.environ.get('DRYFT_ABLATION_COMMIT','0bd7995')
if modal.is_local():
    import subprocess
    BASELINE_COMMIT=subprocess.check_output(['git','rev-parse',BASELINE_COMMIT],text=True).strip()
FROZEN=Path('/tmp')/('dryft-ablation-'+BASELINE_COMMIT)/'engine'
if modal.is_local() and not FROZEN.exists():
    import io
    import subprocess
    import tarfile
    FROZEN.parent.mkdir(parents=True,exist_ok=True)
    archive=subprocess.check_output(['git','archive',BASELINE_COMMIT,'engine'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        handle.extractall(FROZEN.parent,filter='data')
image=(image.add_local_dir(FROZEN,'/root/ablation_engine')
       .add_local_file('bench/ablation_study.py','/root/ablation_study.py')
       .add_local_file('bench/modal_bench.py','/root/modal_bench.py'))
app=modal.App('dryft-feature-ablation')


@app.function(image=image,gpu='H100',volumes={'/weights':weights},timeout=3600)
def run_shape(shape,samples,panel,choices=None,corpus='prose'):
    import subprocess
    _describe_gpu(require_h100=True)
    process=subprocess.Popen([sys.executable,'/root/ablation_study.py',json.dumps(shape),str(samples),panel,json.dumps(choices),corpus],
                             stdout=subprocess.PIPE,text=True)
    result=None
    for line in process.stdout:
        if line.startswith('RESULT_JSON='): result=json.loads(line.removeprefix('RESULT_JSON='))
        else: print(line,end='',flush=True)
    if process.wait() or result is None: raise RuntimeError('ablation process failed')
    return result


@app.local_entrypoint()
def main(shape: str='public-0',samples: int=3,panel: str='core',output: str='bench/results/ablation.json',
         variants: str='',corpus: str='prose'):
    from datetime import datetime,timezone
    shapes={'public-0':('public-0',1,512,32),'public-1':('public-1',4,2048,32),
            'public-2':('public-2',16,512,128),'wide':('coverage-wide',32,256,32),
            'medium':('coverage-medium',8,1024,64),'long':('coverage-long',1,4096,65),
            'long-output':('coverage-long-output',1,512,128),
            'wide-prefill':('coverage-wide-prefill',16,4096,16)}
    digest=hashlib.sha256()
    for p in sorted(FROZEN.rglob('*.py')): digest.update(p.relative_to(FROZEN).as_posix().encode()+b'\0'+p.read_bytes())
    result={'started_at':datetime.now(timezone.utc).isoformat(),'baseline_commit':BASELINE_COMMIT,
            'baseline_sha256':digest.hexdigest(),
            'study_sha256':hashlib.sha256(Path('bench/ablation_study.py').read_bytes()).hexdigest()}
    if corpus=='all':
        result['results']=[run_shape.remote(shapes[shape],samples,panel,
                                           variants.split(',') if variants else None,c)
                           for c in ('prose','code','technical')]
    else:
        result['result']=run_shape.remote(shapes[shape],samples,panel,variants.split(',') if variants else None,corpus)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    for item in result.get('results',[result.get('result')]):
        for name,r in item['variants'].items():
            print(item['corpus'],name,round(r['tps'],1),round(r['speedup_vs_ordinary'],4),r['worst_gap'],round(r['spread'],3))
    print('Saved',output)
