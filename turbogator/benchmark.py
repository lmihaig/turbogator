import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("FORCE_COLOR", "1")

import numpy as np
import torch
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel
from torch.profiler import record_function

import config as app_config
from turbogator.engine import TurboGatorModel
from turbogator.engine.baseline_gatr import BaselineGATrModel
from collections import defaultdict
from termcolor import colored

torch.set_num_threads(1)


def _color(ok, text):
    return colored(text, "green" if ok else "red", force_color=True)


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

        # fmt: off
        std_ok = mean_s > 0 and (std_s / mean_s) < 0.01             # StdDev/Mean < 1%
        iqr_ok = median_s > 0 and (iqr_s / 2) / median_s < 0.005    # IQR/2 < 0.5% of median
        maxmin_ok = min_s > 0 and max_s <= 1.1 * min_s              # Max <= 1.1 * Min
        var_ok = cv_pct < 1.0                                       # Var ~ 1% or less

        print("Metrics:")
        print(f"  Mean   + StdDev : {mean_s:.6f}s + {_color(std_ok,    f'{std_s:.6f}s')}")
        print(f"  Median + IQR/2  : {median_s:.6f}s + {_color(iqr_ok,  f'{iqr_s / 2:.6f}s')}")
        print(f"  Min / Max       : {min_s:.6f}s / {_color(maxmin_ok,  f'{max_s:.6f}s')}")
        print(f" Var   : {_color(var_ok, f'{cv_pct:.2f}%')}")

    print("=" * 50 + "\n")
    # fmt: on


# drop zero duration events
# clip events that overshoot parent's end
def _clean_torch_trace(path, min_dur_us=1):
    with open(path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    by_track = defaultdict(list)
    kept = []

    # events like cpu_instant_event
    for ev in events:
        if ev.get("ph") == "i" and ev.get("name") == "[memory]":
            continue

        if ev.get("ph") != "X":
            kept.append(ev)
            continue

        dur = ev.get("dur") or 0
        if dur < min_dur_us:
            continue

        by_track[(ev.get("pid"), ev.get("tid"))].append(ev)

    # fix overlapping events
    fixed = []
    for evs in by_track.values():
        evs.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
        stack = []

        for ev in evs:
            ts = ev["ts"]

            while stack and stack[-1] <= ts:
                stack.pop()

            if stack:
                parent_end = stack[-1]
                if ts + ev["dur"] > parent_end:
                    new_dur = parent_end - ts
                    if new_dur < min_dur_us:
                        continue
                    ev["dur"] = new_dur

            stack.append(ts + ev["dur"])
            fixed.append(ev)

    data["traceEvents"] = kept + fixed

    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, str(path))


def run_torch_profiler(model, x, out_path, warmup, steps):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def _on_trace_ready(prof):
        prof.export_chrome_trace(str(out_path))
        _clean_torch_trace(out_path)
        print("\n--- PyTorch Memory & CPU Summary ---")
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=steps, repeat=1),
        on_trace_ready=_on_trace_ready,
    ) as prof:
        for i in range(warmup + steps):
            label = "WARMUP" if i < warmup else "ACTIVE"
            with record_function(f"GATOR_FORWARD_PASS_{label}"):
                GATOR_FORWARD_PASS(model, x)
            prof.step()


def _perf_ctl(ctl_fd, cmd):
    if ctl_fd is not None:
        os.write(ctl_fd, f"{cmd}\n".encode())


def benchmark(desc, T, C_in, seed, warmup, steps, profile, profile_out, perf_ctl_fd):
    _perf_ctl(perf_ctl_fd, "disable")
    torch.manual_seed(seed)

    B, D = app_config.BATCH_SIZE, app_config.VECTOR_DIM
    cfg = MVOnlyGATrConfig(size_channels_in=C_in)

    if desc == "ezgatr":
        model = MVOnlyGATrModel(cfg).eval()
    elif desc == "baseline":
        model = BaselineGATrModel(cfg).eval()
    else:
        model = TurboGatorModel(cfg).eval()

    x = torch.randn(B, T, C_in, D)

    with torch.no_grad():
        if profile == "torch":
            return run_torch_profiler(model, x, profile_out, warmup, steps)

        warmup_times_ns = []
        for i in range(warmup):
            t0 = time.perf_counter_ns()
            GATOR_FORWARD_PASS(model, x)
            t1 = time.perf_counter_ns()
            print(
                f"[Warmup {i + 1:02d}/{warmup:02d}] {(t1 - t0) / 1e9:.6f} s", flush=True
            )
            warmup_times_ns.append(t1 - t0)

        # Enable perf counting for the timed steps only.
        step_times_ns = []
        for i in range(steps):
            t0 = time.perf_counter_ns()
            _perf_ctl(perf_ctl_fd, "enable")
            GATOR_FORWARD_PASS(model, x)
            _perf_ctl(perf_ctl_fd, "disable")
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

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str, required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--c", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--profile", type=str, default="none")
    parser.add_argument("--profile-out", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--perf-ctl-fd", type=int, default=None)
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
        perf_ctl_fd=args.perf_ctl_fd,
    )

    if args.out and result is not None:
        with open(args.out, "w") as f:
            json.dump(result, f)
