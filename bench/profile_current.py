"""Profile an immutable snapshot of the working engine on one H100.

Run: .venv/bin/modal run bench/profile_current.py
"""
import base64
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal
weights = modal.Volume.from_name("dryft-qwen3-4b")
if modal.is_local():
    from bench.modal_bench import image
    SNAPSHOT = Path(tempfile.mkdtemp(prefix="dryft-profile-"))
    shutil.copytree("engine", SNAPSHOT / "engine", ignore=shutil.ignore_patterns("__pycache__"))
    FILES = {p.relative_to(SNAPSHOT / "engine").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((SNAPSHOT / "engine").rglob("*.py"))}
    profile_image = (image.add_local_dir(str(SNAPSHOT / "engine"), "/root/profile_engine")
                     .add_local_file("bench/profile_worker.py", "/root/profile_worker.py"))
else:
    profile_image = modal.Image.debian_slim()
app = modal.App("dryft-current-profile")


@app.function(image=profile_image, gpu="H100", volumes={"/weights": weights}, timeout=1800)
def run_remote():
    import sys
    import torch
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("H100 required; retry rather than accepting a substitute")
    results = []
    for name, batch, context, output in [("public-0", 1, 512, 32),
                                       ("public-1", 4, 2048, 32),
                                       ("public-2", 16, 512, 128),
                                       ("wide", 32, 256, 32),
                                       ("long", 1, 4096, 65)]:
        subprocess.run([sys.executable, "/root/profile_worker.py", name,
                        str(batch), str(context), str(output)], check=True)
        row = json.loads(Path(f"/tmp/{name}.json").read_text())
        row["traces"] = {stage: base64.b64encode(Path(f"/tmp/{name}-{stage}.json.gz").read_bytes()).decode()
                         for stage in ("prefill", "decode")}
        results.append(row)
    return results


@app.local_entrypoint()
def main(output: str = "bench/results/profile-current-20260920"):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "engine_files_sha256": FILES,
                "engine_diff": subprocess.check_output(["git", "diff", "--", "engine"], text=True)}
    shutil.copytree(SNAPSHOT / "engine", destination / "engine-snapshot")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
    results = run_remote.remote()
    for row in results:
        for stage, encoded in row.pop("traces").items():
            (destination / f"{row['name']}-{stage}.json.gz").write_bytes(base64.b64decode(encoded))
    (destination / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Profile saved to {destination}")
