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
    w_sec = np.array(warmup_ns, dtype=np.float64) / 1e9
    s_sec = np.array(step_ns, dtype=np.float64) / 1e9

    print("\n" + "=" * 50)
    if len(w_sec) > 0:
        print(f"[Warmup Phase] ({len(w_sec)} steps)")
        for i, x in enumerate(w_sec):
            print(f"  W{i + 1:02d}: {x:.6f} s")
        if len(w_sec) > 1:
            dropoff = w_sec[0] / w_sec[-1]
            print(f"Warmup Drop-off: {dropoff:.2f}x")
        print("-" * 50)

    print(f"[Active Phase] ({len(s_sec)} steps)")
    for i, x in enumerate(s_sec):
        print(f"  S{i + 1:02d}: {x:.6f}s")
    print("-" * 50)

    if len(s_sec) > 0:
        mean_s = np.mean(s_sec)
        std_s = np.std(s_sec)
        median_s = np.median(s_sec)
        min_s = np.min(s_sec)
        max_s = np.max(s_sec)

        q25, q75 = np.percentile(s_sec, [25, 75])
        iqr_s = q75 - q25

        cv_pct = (std_s / mean_s) * 100 if mean_s > 0 else 0

        print("Metrics:")
        print(f"  Mean   + StdDev : {mean_s:.6f}s + {std_s:.6f}s")
        print(f"  Median + IQR/2  : {median_s:.6f}s + {iqr_s / 2:.6f}s")
        print(f"  Min / Max       : {min_s:.6f}s / {max_s:.6f}s")
        print(f" Var   : {cv_pct:.2f}%")

    print("=" * 50 + "\n")


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
                f"[Warmup {i + 1:02d}/{warmup:02d}] {(t1 - t0) / 1e9:.6f} s", flush=True
            )
            warmup_times_ns.append(t1 - t0)

        step_times_ns = []
        for i in range(steps):
            t0 = time.perf_counter_ns()
            with AdvisorRegion(profile):
                GATOR_FORWARD_PASS(model, x)
            t1 = time.perf_counter_ns()
            print(
                f"[Active {i + 1:02d}/{steps:02d}] {(t1 - t0) / 1e9:.6f} s", flush=True
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
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
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
