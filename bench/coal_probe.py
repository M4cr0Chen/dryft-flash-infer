"""Direct versus block-cooperative epilogue stores, per projection and batch.

Both builds compute identical values; the probe checks that bit for bit on
every race configuration, then times the best configuration of each.
"""

import sys

import torch

sys.path.insert(0, "/root/engine")
from kernels import cuda_fp8  # noqa: E402
from kernels.timing import time_calls  # noqa: E402

SHAPES = [("qkv", 6144, 2560), ("o", 2560, 4096), ("gate_up", 19456, 2560),
          ("down", 2560, 9728), ("lm_head", 151936, 2560)]


def runner(x, prepared, cfg):
    splitk = cuda_fp8.splits(prepared, x.shape[0], cfg)
    if splitk > 1:
        return lambda p, c=cfg: cuda_fp8.matmul_partials(x, p, config=c)
    return lambda p, c=cfg: cuda_fp8.matmul(x, p, config=c)


@torch.inference_mode()
def run(batches):
    torch.manual_seed(21)
    for name, n, k in SHAPES:
        count = 1 if name == "lm_head" else 12
        weights = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(count)]
        prepared = [cuda_fp8.prepare(cuda_fp8.quantize(w)) for w in weights]
        del weights
        reps = 1 if name == "lm_head" else 2
        for batch in batches:
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            best = {}
            outputs = {}
            for coal in (False, True):
                cuda_fp8.COALESCED = coal
                cuda_fp8._modules.clear()
                times = {}
                for cfg in cuda_fp8.CONFIGS:
                    if not cuda_fp8.applicable(cfg, batch):
                        continue
                    out = runner(x, prepared[0], cfg)(prepared[0])
                    outputs.setdefault(cfg, {})[coal] = out
                    times[cfg] = time_calls(runner(x, prepared[0], cfg), prepared, reps=reps, trials=3) * 1e3
                if name == "gate_up":
                    for cfg in cuda_fp8.SWIGLU_CONFIGS:
                        if not cuda_fp8.applicable(cfg, batch, swiglu=True):
                            continue
                        out = cuda_fp8.gate_up_swiglu(x, prepared[0], config=cfg)
                        outputs.setdefault(("swiglu",) + cfg, {})[coal] = out
                        times[("swiglu",) + cfg] = time_calls(
                            lambda p, c=cfg: cuda_fp8.gate_up_swiglu(x, p, config=c), prepared, reps=reps, trials=3) * 1e3
                best[coal] = times
            for cfg, pair in outputs.items():
                if not torch.equal(pair[False], pair[True]):
                    raise AssertionError(f"{name} b{batch} {cfg}: coalesced epilogue differs, "
                                         f"max {(pair[False].float() - pair[True].float()).abs().max().item()}")
            plain = {c: v for c, v in best[False].items() if c[0] != "swiglu"}
            coal = {c: v for c, v in best[True].items() if c[0] != "swiglu"}
            b0, b1 = min(plain, key=plain.get), min(coal, key=coal.get)
            line = (f"{name:8s} b{batch:<3d} direct {plain[b0]:6.1f}us {b0} | coalesced {coal[b1]:6.1f}us {b1} "
                    f"x{plain[b0]/coal[b1]:.2f}")
            if name == "gate_up":
                sp = {c: v for c, v in best[False].items() if c[0] == "swiglu"}
                sc = {c: v for c, v in best[True].items() if c[0] == "swiglu"}
                line += f" | swiglu {min(sp.values()):.1f} -> {min(sc.values()):.1f}us x{min(sp.values())/min(sc.values()):.2f}"
            print(line + f" | {len(outputs)} bit-exact", flush=True)
        del prepared
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run([int(b) for b in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "4", "16", "32", "64"])])
