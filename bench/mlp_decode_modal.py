"""Prefill fusion and consumer-aware decode experiments on a frozen engine."""
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import modal

weights = modal.Volume.from_name("dryft-qwen3-4b")
if modal.is_local():
    from bench.modal_bench import image
    snapshot = Path(tempfile.mkdtemp(prefix="dryft-mlp-decode-")) / "engine"
    shutil.copytree("engine", snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    source_hashes = {p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(snapshot.rglob("*.py"))}
    image = (image.add_local_dir(str(snapshot), "/root/experiment_engine")
             .add_local_file("bench/prefill_mlp_kernel.py", "/root/prefill_mlp_kernel.py")
             .add_local_file("bench/mlp_decode_study.py", "/root/mlp_decode_study.py"))
else:
    image = modal.Image.debian_slim()
app = modal.App("dryft-mlp-decode")


@app.function(image=image, gpu="H100", volumes={"/weights": weights}, timeout=2400)
def run_remote(stage, shapes, samples):
    import sys
    import torch
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("H100 required, retry")
    results = []
    for shape in shapes:
        subprocess.run([sys.executable, "/root/mlp_decode_study.py", stage,
                        json.dumps(shape), str(samples)], check=True)
        results.append(json.loads(Path("/tmp/mlp-decode-result.json").read_text()))
    return results


@app.local_entrypoint()
def main(stage: str = "micro", shapes: str = "public", samples: int = 5,
         output: str = "bench/results/prefill-fusion-micro-20260920.json"):
    groups = {"public": [(1,512,32),(4,2048,32),(16,512,128)],
              "micro": [(1,512,32)],
              "decode": [(1,512,32),(4,2048,32),(16,512,128),(32,256,32)],
              "coverage": [(1,4096,65),(3,257,33),(8,1024,64),(12,1024,64),(32,256,32)]}
    selected = groups[shapes] if shapes in groups else [tuple(map(int, s.split("x"))) for s in shapes.split(",")]
    result = {"started_at": datetime.now(timezone.utc).isoformat(),
              "commit": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "engine_files_sha256": source_hashes, "stage": stage,
              "results": run_remote.remote(stage, selected, samples)}
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", output)
