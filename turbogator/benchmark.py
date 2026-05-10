import argparse
import json
import time

import numpy as np
import torch
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel

import config as app_config
from turbogator.engine import TurboGatorModel


def analyze_runs(times_ns):

    times_s = np.array(times_ns, dtype=np.float64) / 1e9

    print("raw_s:", ", ".join(f"{t:.6f}" for t in times_s))
    print("median_s:", float(np.median(times_s)))
    print("min_s:", float(np.min(times_s)))
    print("median_over_min:", float(np.median(times_s)) / float(np.min(times_s)))
    print("cv:", float(np.std(times_s) / np.mean(times_s)))

    print("rolling_median:")
    for k in range(5, len(times_s) + 1):
        rolling_median = float(np.median(times_s[:k]))
        rel_err = abs(rolling_median - float(np.median(times_s))) / float(
            np.median(times_s)
        )
        print(f"k={k} median={rolling_median:.6f}s rel_err={rel_err * 100:.2f}%")

    print("max_s:", float(np.max(times_s)))
    print("p95_s:", float(np.percentile(times_s, 95)))


def benchmark(description, T, C_in, seed=42, warmup=5, steps=10):
    torch.manual_seed(seed)

    batch_size = app_config.BATCH_SIZE
    vector_dim = app_config.VECTOR_DIM

    cfg = MVOnlyGATrConfig(size_channels_in=C_in)

    # bodge to avoid separate impl for ezgatr reference benchmark
    if description == "ezgatr":
        model = MVOnlyGATrModel(cfg).eval()
    else:
        model = TurboGatorModel(cfg).eval()

    x = torch.randn(batch_size, T, C_in, vector_dim)

    times_ns = []
    with torch.no_grad():
        for _ in range(warmup + steps):
            t0 = time.perf_counter_ns()
            model(x)
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

    analyze_runs(times_ns)

    times_ns = times_ns[warmup:]
    median_ns = float(np.median(times_ns))
    median_cycles = median_ns * app_config.CPU_FREQ / 1e9

    result = {
        "X": {"B": batch_size, "T": T, "C_in": C_in, "D": vector_dim},
        "cycles": median_cycles,
    }
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str, required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--c", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    result = benchmark(args.desc, args.t, args.c)

    with open(args.out, "w") as f:
        json.dump(result, f)
