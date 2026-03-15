"""Benchmark Real-ESRGAN MLX inference speed and verify quality.

Outputs one line per run to results.tsv:
    timestamp\texperiment_name\ttime_512\ttime_1024\tmax_diff\tstatus

Quality gate: max pixel difference vs reference must be < 1e-4 (float32).
Reference output is generated once from the initial commit and saved.
"""

import sys
import time
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))
from model import RRDBNet, SRVGGNetCompact, pad_reflect
from upscale import load_model, upscale_image

RESULTS_FILE = Path(__file__).parent / "results.tsv"
REFERENCE_DIR = Path(__file__).parent / "reference"
BENCH_INPUTS = {
    "512": Path(__file__).parent / "bench_input.png",
}


def get_git_info():
    """Get current commit hash and message."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], 
                                       cwd=Path(__file__).parent, text=True).strip()
        msg = subprocess.check_output(["git", "log", "-1", "--format=%s"],
                                       cwd=Path(__file__).parent, text=True).strip()[:60]
        return sha, msg
    except Exception:
        return "unknown", "unknown"


def create_reference():
    """Create reference output from current code (run once at baseline)."""
    REFERENCE_DIR.mkdir(exist_ok=True)
    model, scale = load_model("x4plus", dtype=mx.float32)
    
    img = np.array(Image.open(BENCH_INPUTS["512"]).convert("RGB")).astype(np.float32) / 255.0
    output = upscale_image(model, img, scale, dtype=mx.float32)
    ref_path = REFERENCE_DIR / "ref_512_x4plus.npy"
    np.save(ref_path, output)
    print(f"Reference saved: {ref_path}")
    return ref_path


def check_quality(model, scale, input_path: Path, ref_path: Path) -> float:
    """Compare current output vs reference. Returns max absolute diff in [0,1]."""
    img = np.array(Image.open(input_path).convert("RGB")).astype(np.float32) / 255.0
    output = upscale_image(model, img, scale, dtype=mx.float32)
    
    ref = np.load(ref_path)
    cur = output
    
    max_diff = np.max(np.abs(ref - cur))
    return max_diff


def bench_speed(model, scale, input_path: Path, dtype=mx.float16, warmup=2, runs=5) -> float:
    """Benchmark inference speed. Returns median time in seconds."""
    img = np.array(Image.open(input_path).convert("RGB")).astype(np.float32) / 255.0
    
    # Warmup
    for _ in range(warmup):
        upscale_image(model, img, scale, dtype=dtype)
    
    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        upscale_image(model, img, scale, dtype=dtype)
        mx.eval(mx.zeros(1))  # sync
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    return sorted(times)[len(times) // 2]  # median


def run_benchmark(experiment_name: str = None):
    """Run full benchmark: speed + quality check."""
    sha, msg = get_git_info()
    if experiment_name is None:
        experiment_name = f"{sha}: {msg}"
    
    ref_path = REFERENCE_DIR / "ref_512_x4plus.npy"
    if not ref_path.is_file():
        print("No reference found, creating baseline...")
        create_reference()
    
    # Load model
    model_fp16, scale = load_model("x4plus", dtype=mx.float16)
    model_fp32, _ = load_model("x4plus", dtype=mx.float32)
    
    # Quality check (fp32 for fair comparison)
    max_diff = check_quality(model_fp32, scale, BENCH_INPUTS["512"], ref_path)
    quality_ok = max_diff < 1e-4
    
    # Speed benchmark (fp16)
    time_512 = bench_speed(model_fp16, scale, BENCH_INPUTS["512"])
    
    # Create 1024 input if needed
    input_1024 = Path(__file__).parent / "bench_input_1024.png"
    if not input_1024.is_file():
        img = Image.open(BENCH_INPUTS["512"]).resize((1024, 1024), Image.LANCZOS)
        img.save(input_1024)
    time_1024 = bench_speed(model_fp16, scale, input_1024)
    
    status = "KEEP" if quality_ok else "DISCARD"
    
    # Append to results.tsv
    header_needed = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a") as f:
        if header_needed:
            f.write("timestamp\texperiment\ttime_512\ttime_1024\tmax_diff\tstatus\n")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts}\t{experiment_name}\t{time_512:.4f}\t{time_1024:.4f}\t{max_diff:.6f}\t{status}\n")
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"512x512 -> 2048x2048: {time_512:.4f}s")
    print(f"1024x1024 -> 4096x4096: {time_1024:.4f}s")
    print(f"Max quality diff: {max_diff:.6f}")
    print(f"Status: {status}")
    print(f"{'='*60}")
    
    return status == "KEEP"


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_benchmark(name)
    sys.exit(0 if success else 1)
