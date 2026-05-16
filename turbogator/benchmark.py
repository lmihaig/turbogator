import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel
from torch.profiler import record_function

import config as app_config
from turbogator.engine import TurboGatorModel

# stupid stuff for intel advisor god
try:
    import pyitt
except ImportError:
    pyitt = None


class AdvisorRegion:
    def __init__(self, profile):
        self.profile = profile

    def __enter__(self):
        if self.profile == "advisor" and pyitt is not None:
            pyitt.resume()

    def __exit__(self, *args):
        if self.profile == "advisor" and pyitt is not None:
            pyitt.pause()


# wrapper for easier identification in traces
def GATOR_FORWARD_PASS(model, x):
    return model(x)


def analyze_runs(warmup_ns, step_ns):
    times_ns = warmup_ns + step_ns
    t = np.array(times_ns, dtype=np.float64) / 1e9
    med, mn, mx = np.median(t), np.min(t), np.max(t)

    labeled = []
    for i, x in enumerate(t):
        prefix = "W" if i < len(warmup_ns) else "S"
        labeled.append(f"{prefix}{i:02d}={x:.6f}")

    print(f"counts: warmup={len(warmup_ns)} steps={len(step_ns)}")
    print(f"raw_s: {', '.join(labeled)}")
    print(f"median_s: {med:.6f} | min_s: {mn:.6f} | max_s: {mx:.6f}")
    print(f"cv: {np.std(t) / np.mean(t):.4f} | med/min: {med / mn:.4f}")

    print("rolling_median:")
    for k in range(5, len(t) + 1):
        r_med = float(np.median(t[:k]))
        print(
            f"k={k:02d} median={r_med:.6f}s rel_err={abs(r_med - med) / med * 100:.2f}%"
        )


def run_torch_profiler(model, x, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def _on_trace_ready(prof):
        prof.export_chrome_trace(str(out_path))
        print("\n--- PyTorch Memory & CPU Summary ---")
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=_on_trace_ready,
    ) as prof:
        for step in range(2):
            step_name = "WARMUP" if step == 0 else "ACTIVE"
            with record_function(f"GATOR_FORWARD_PASS_{step_name}"):
                GATOR_FORWARD_PASS(model, x)
            prof.step()


def benchmark(desc, T, C_in, seed, warmup, steps, profile, profile_out):
    torch.manual_seed(seed)

    B, D = app_config.BATCH_SIZE, app_config.VECTOR_DIM
    cfg = MVOnlyGATrConfig(size_channels_in=C_in)

    model = (
        MVOnlyGATrModel(cfg).eval() if desc == "ezgatr" else TurboGatorModel(cfg).eval()
    )
    x = torch.randn(B, T, C_in, D)

    with torch.no_grad():
        if profile == "torch":
            return run_torch_profiler(model, x, profile_out)

        warmup_times_ns = []
        for i in range(warmup):
            t0 = time.perf_counter_ns()
            GATOR_FORWARD_PASS(model, x)
            t1 = time.perf_counter_ns()
            print(
                f"Completed warmup {i + 1}/{warmup} run in {(t1 - t0) / 1e9:.6f} seconds."
            )
            warmup_times_ns.append(t1 - t0)

        step_times_ns = []
        for i in range(steps):
            t0 = time.perf_counter_ns()
            with AdvisorRegion(profile):
                GATOR_FORWARD_PASS(model, x)
            t1 = time.perf_counter_ns()
            print(
                f"Completed step {i + 1}/{steps} run in {(t1 - t0) / 1e9:.6f} seconds."
            )
            step_times_ns.append(t1 - t0)

    analyze_runs(warmup_times_ns, step_times_ns)

    result = {
        "X": {"B": B, "T": T, "C_in": C_in, "D": D},
        "cycles": float(np.median(step_times_ns)) * app_config.CPU_FREQ / 1e9,
    }
    print(result)

    if profile == "advisor":
        time.sleep(1)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str, required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--c", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--profile", type=str, default="none")
    parser.add_argument("--profile-out", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    result = benchmark(
        args.desc,
        args.t,
        args.c,
        seed=args.seed,
        warmup=args.warmup,
        steps=args.steps,
        profile=args.profile,
        profile_out=args.profile_out,
    )

    if args.out and result is not None:
        with open(args.out, "w") as f:
            json.dump(result, f)
