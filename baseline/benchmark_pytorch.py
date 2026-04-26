import gc
import json
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.append(os.path.join(os.path.dirname(__file__), "ezgatr", "src"))
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel

import config as app_config


def run_benchmark():
    torch.set_num_threads(1)

    device = "cpu"
    batch_size = app_config.BATCH_SIZE
    size_channels_in = app_config.CHANNELS

    sizes = app_config.SIZES
    print(f"Loaded {len(sizes)} sizes from config: {sizes}", flush=True)

    print("Initializing ezgatr...", flush=True)
    model_config = MVOnlyGATrConfig(size_channels_in=size_channels_in)
    net = MVOnlyGATrModel(model_config).to(device)
    net.eval()

    results = []
    cpu_freq = app_config.CPU_FREQ
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "baseline")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "metrics.json")

    for N in sizes:
        print(f"\n--- Starting N = {N} ---", flush=True)
        x = torch.randn(batch_size, N, size_channels_in, 16).to(device)

        warmup_runs = 3
        runs = 10

        print(f"  Doing {warmup_runs} warmup runs...", flush=True)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = net(x)

        print(f"  Doing {runs} benchmark runs...", flush=True)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = net(x)
        end = time.perf_counter()

        avg_seconds = (end - start) / runs
        avg_cycles = avg_seconds * cpu_freq

        results.append({"N": N, "cycles": avg_cycles})
        print(
            f"  Result: {avg_cycles:.0f} cycles (Avg {avg_seconds:.4f} sec)", flush=True
        )

        out_data = {
            "user": "baseline",
            "description": "ezgatr",
            "data": results,
        }
        with open(out_file, "w") as f:
            json.dump(out_data, f, indent=2)

        del x
        gc.collect()

    print("\nBaseline complete!", flush=True)


if __name__ == "__main__":
    run_benchmark()
