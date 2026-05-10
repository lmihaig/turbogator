import time

import numpy as np
import torch
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig

import config as app_config
from turbogator.models.asl_gatr import TurboGatorModel


def benchmark(seed=42, warmup=10, steps=30):
    torch.manual_seed(seed)

    N = app_config.REPRESENTATIVE_N
    T, C_in = app_config.get_dimensions(N)
    batch_size = app_config.BATCH_SIZE
    vector_dim = app_config.VECTOR_DIM

    cfg = MVOnlyGATrConfig(size_channels_in=C_in)
    model = TurboGatorModel(cfg).eval()
    x = torch.randn(batch_size, T, C_in, vector_dim)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    cycles_runs = []
    with torch.no_grad():
        for _ in range(steps):
            t0 = time.perf_counter_ns()
            model(x)
            t1 = time.perf_counter_ns()
            elapsed_ns = t1 - t0
            cycles_runs.append(elapsed_ns * app_config.CPU_FREQ / 1e9)

    median_cycles = float(np.median(cycles_runs))

    result = {
        "N": N,
        "X": [batch_size, T, C_in, vector_dim],
        "cycles": median_cycles,
    }
    print(result)
    return result


if __name__ == "__main__":
    benchmark()
