"""Run the next bounded experiments on the existing pinned Modal image."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modal_bench import image, weights, _describe_gpu
import modal

image = (image.add_local_file('bench/next_probe.py', '/root/next_probe.py')
         .add_local_file('bench/w8a8_kernel.py', '/root/w8a8_kernel.py')
         .add_local_file('bench/w8a8_micro.py', '/root/w8a8_micro.py')
         .add_local_file('bench/modal_bench.py', '/root/modal_bench.py'))
app = modal.App('dryft-next-experiments')


@app.function(image=image, gpu='H100', volumes={'/weights': weights}, timeout=3600)
def run_probe(stage, shapes, samples):
    import subprocess
    _describe_gpu(require_h100=True)
    results = []
    if stage == 'micro':
        shapes = [None]
    for shape in shapes:
        command = ([sys.executable, '/root/w8a8_micro.py'] if stage == 'micro' else
                   [sys.executable, '/root/next_probe.py', stage, json.dumps(shape), str(samples)])
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            if line.startswith('RESULT_JSON='):
                results.append(json.loads(line.removeprefix('RESULT_JSON=')))
            else:
                print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError(f'probe failed for {shape}')
    return results


@app.local_entrypoint()
def main(stage: str = 'graph', samples: int = 5, shape_set: str = 'public',
         output: str = 'bench/results/next-probe.json'):
    from datetime import datetime, timezone
    shapes = [('public-0', 1, 512, 32), ('public-1', 4, 2048, 32), ('public-2', 16, 512, 128)]
    if shape_set == 'quant':
        shapes = [shapes[2], ('coverage-long', 1, 4096, 65)]
    elif shape_set == 'native':
        shapes = [shapes[2], ('wide-long-output', 32, 512, 128),
                  ('coverage-12', 12, 1024, 64), ('coverage-24', 24, 257, 33)]
    elif shape_set == 'coverage':
        shapes = [('coverage-odd', 3, 257, 33), ('coverage-medium', 8, 1024, 64),
                  ('coverage-wide', 32, 256, 16), ('coverage-long', 1, 4096, 65)]
    digest = hashlib.sha256()
    for p in sorted(Path('engine').rglob('*.py')):
        digest.update(p.as_posix().encode() + b'\0' + p.read_bytes())
    result = {'started_at': datetime.now(timezone.utc).isoformat(), 'engine_sha256': digest.hexdigest(),
              'results': run_probe.remote(stage, shapes, samples)}
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    for r in result['results']:
        if isinstance(r, dict) and 'variants' in r:
            print(r['shape'], {k: {n: v[n] for n in ('tps', 'ttft', 'worst_gap', 'spread')}
                              for k, v in r['variants'].items()})
    print('Saved', output)
