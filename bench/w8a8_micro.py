"""Check native INT8 fragments, then race full projection families in graphs."""
import json
import os
import sys
os.environ['DRYFT_SHORT_DRAFT'] = '0'
sys.path.insert(0, '/root/engine')
sys.path.insert(0, '/root')
import torch
from kernels import cuda_fp8
from kernels.timing import time_calls
from kernels.swiglu import swiglu
import w8a8_kernel as new
from next_probe import integer_reference
from engine import Engine


@torch.inference_mode()
def main():
    torch.manual_seed(901)
    for m in (1, 3, 4, 16, 32):
        x = torch.randn(m, 256, device='cuda', dtype=torch.bfloat16)
        source = cuda_fp8.prepare(cuda_fp8.quantize(torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)))
        packed = new.Prepared(source)
        for components in (1, 2):
            expected = integer_reference(x, source, components)
            for config in ((4, 1), (4, 4), (8, 1)):
                actual = new.matmul(x, packed, config, components)
                error = (actual.float() - expected.float()).abs().max().item()
                assert error <= .25, (m, components, config, error)
            actual = new.matmul(x, packed, (4, 1), components, mode=2)
            error = (actual.float() - swiglu(expected).float()).abs().max().item()
            assert error <= 4, (m, components, 'swiglu', error)
    print('native integer kernel: fragment/scales/split-K/SwiGLU checks passed', flush=True)
    engine = Engine('/weights/qwen3-4b')
    operands = {name: [new.Prepared(w) for w in family] for name, family in engine.quantised.items()}
    results = []
    for m in (1, 4, 16, 32):
        engine._ensure(m, 512, 32)
        for name in ('qkv', 'o', 'gate_up', 'down', 'lm_head'):
            packed = operands[name]
            x = torch.randn(m, packed[0].k, device='cuda', dtype=torch.bfloat16)
            incumbent = time_calls(lambda w: engine.matmul[name](x, w), engine.operand[name], reps=1, trials=3)
            configs = []
            for config in ((4, 1), (8, 1), (4, 2), (8, 2), (4, 4), (8, 4)):
                expected = integer_reference(x, engine.quantised[name][0], 1)
                actual = new.matmul(x, packed[0], config)
                error = (actual.float() - expected.float()).abs().max().item()
                assert error <= max(.125, expected.float().abs().max().item() * .015), (m, name, config, error)
                elapsed = time_calls(lambda w: new.matmul(x, w, config), packed, reps=1, trials=3)
                configs.append({'config': config, 'us': elapsed * 1000})
            best = min(configs, key=lambda r: r['us'])
            row = {'batch': m, 'projection': name, 'incumbent_us': incumbent*1000,
                   'best': best, 'speedup': incumbent*1000/best['us'], 'candidates': configs}
            if name == 'gate_up':
                current = (lambda w: engine.fused_mlp(x, w)) if engine.fused_mlp else (lambda w: swiglu(engine.matmul[name](x, w)))
                old = time_calls(current, engine.fused_operand if engine.fused_mlp else engine.operand[name], reps=1, trials=3)
                fused = []
                for config in ((4, 1), (8, 1)):
                    elapsed = time_calls(lambda w: new.matmul(x, w, config, mode=2), packed, reps=1, trials=3)
                    fused.append({'config': config, 'us': elapsed*1000})
                row['swiglu'] = {'incumbent_us': old*1000, 'candidates': fused}
            print(json.dumps(row), flush=True)
            results.append(row)
    return results


if __name__ == '__main__':
    print('RESULT_JSON=' + json.dumps(main()))
