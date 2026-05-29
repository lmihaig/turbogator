#!/usr/bin/env python3
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("FORCE_COLOR", "1")
from termcolor import colored

ENABLE_INTEL_ADVISOR = False

# perf prints these on every --control enable/disable; drop them from logs
_PERF_CTL_NOISE = re.compile(r"^Events (enabled|disabled)\s*$")

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _colored_count(n):
    return colored(str(n), "green" if n > 0 else "red", force_color=True)


def _fence(log_file, label, seconds=3):
    os.sync()
    time.sleep(seconds)


def run_cmd(cmd, job_dir, env, log_file, pass_fds=()):
    cmd_str = [str(x) for x in cmd]
    print(f"\n$ {' '.join(cmd_str)}", file=log_file, flush=True)

    proc = subprocess.Popen(
        cmd_str,
        cwd=job_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        pass_fds=pass_fds,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if _PERF_CTL_NOISE.match(line.strip()):
            continue
        log_file.write(line)
        log_file.flush()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd_str)


def build_project(job_dir, env, log_file):
    run_cmd(["uv", "sync", "--reinstall-package", "turbogator"], job_dir, env, log_file)


def validate_build(job_dir, env, log_file):
    print("\n=== Step: Validate ===", file=log_file, flush=True)

    ccdb = job_dir / "build" / "skbuild" / "compile_commands.json"
    if ccdb.exists():
        cpp = [
            e
            for e in json.loads(ccdb.read_text())
            if e.get("file", "").endswith(".cpp")
        ]
        if cpp:
            print(f"\nsample compile command:\n  {cpp[0]['command']}\n", file=log_file)

    venv = env.get("UV_PROJECT_ENVIRONMENT")
    so = next(iter(Path(venv).rglob("turbogator_ext*.so")), None) if venv else None
    if so:
        asm = subprocess.run(
            ["objdump", "-d", str(so)], capture_output=True, text=True
        ).stdout
        lines = asm.splitlines()
        ymm = sum(1 for ln in lines if "%ymm" in ln)
        xmm = sum(1 for ln in lines if "%xmm" in ln)
        fma = sum(
            1
            for ln in lines
            if any(p in ln for p in ("vfmadd", "vfmsub", "vfnmadd", "vfnmsub"))
        )
        print(
            f"{so.name}: ymm={_colored_count(ymm)} "
            f"xmm={_colored_count(xmm)} "
            f"fma={_colored_count(fma)}",
            file=log_file,
        )

    python_exe = str(Path(venv) / "bin" / "python") if venv else "python"
    try:
        run_cmd([python_exe, "turbogator/validate.py"], job_dir, env, log_file)
        return True
    except subprocess.CalledProcessError:
        print(
            colored(
                "validate.py failed - aborting before microbench/sweep",
                "red",
                force_color=True,
            ),
            file=log_file,
            flush=True,
        )
        return False


def parse_perf_csv(perf_path, events):
    results = {e: None for e in events}
    if not perf_path.exists():
        return results

    for raw_line in perf_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        raw_event = parts[2]
        if raw_event.startswith("cpu_atom/"):
            # e-core, ignore
            continue
        event = raw_event.removeprefix("cpu_core/").rstrip("/")

        if event not in results:
            continue

        value_str = parts[0]
        if not value_str or value_str == "<not counted>":
            results[event] = None
            continue
        try:
            results[event] = float(value_str)
        except (TypeError, ValueError):
            results[event] = None

    return results


def run_microbench(job_dir, env, log_file, app_config, desc):
    print("\n=== Step: Microbench (1 warmup, 1 step) ===", file=log_file, flush=True)
    pinned_core = app_config.PINNED_CPU_CORE
    n_val = app_config.REPRESENTATIVE_N
    t_val, c_val = app_config.get_dimensions(n_val)

    venv = env.get("UV_PROJECT_ENVIRONMENT")
    python_exe = str(Path(venv) / "bin" / "python") if venv else "python"
    pyspy_bin = str(Path(venv) / "bin" / "py-spy") if venv else "py-spy"

    flamegraph = job_dir / "profile.speedscope.json"
    torch_out = job_dir / "pytorch_profile.trace.json"

    ################# PY-SPY #################
    pyspy_cmd = [
        "taskset",
        "-c",
        pinned_core,
        pyspy_bin,
        "record",
        "-o",
        str(flamegraph),
        "--format",
        "speedscope",
        "--rate",
        "100",
        "--native",
        "--",
        python_exe,
        "turbogator/benchmark.py",
        "--desc",
        desc,
        "--t",
        str(t_val),
        "--c",
        str(c_val),
        "--warmup",
        "2",
        "--steps",
        "3",
        "--profile",
        "none",
    ]
    try:
        run_cmd(pyspy_cmd, job_dir, env, log_file)
    except subprocess.CalledProcessError as e:
        # py-spy can exit non-zero with ECHILD after the child returns; trust the file.
        if not flamegraph.exists():
            raise
        print(
            f"py-spy exited {e.returncode} but wrote {flamegraph.name}; continuing",
            file=log_file,
            flush=True,
        )

    ################# PYTORCH PROFILER #################
    torch_cmd = [
        "taskset",
        "-c",
        pinned_core,
        python_exe,
        "turbogator/benchmark.py",
        "--desc",
        desc,
        "--t",
        str(t_val),
        "--c",
        str(c_val),
        "--warmup",
        "2",
        "--steps",
        "3",
        "--profile",
        "torch",
        "--profile-out",
        str(torch_out),
    ]
    run_cmd(torch_cmd, job_dir, env, log_file)


def run_sweep(job_dir, env, log_file, app_config, desc, metrics):
    pinned_core = app_config.PINNED_CPU_CORE
    perf_events = [str(e) for e in getattr(app_config, "PERF_EVENTS", [])]
    dram_events = [str(e) for e in getattr(app_config, "DRAM_EVENTS", [])]
    warmup = getattr(app_config, "WARMUP", 3)
    steps = getattr(app_config, "STEPS", 5)

    for n_val in app_config.SIZES:
        t_val, c_val = app_config.get_dimensions(n_val)
        temp_out = job_dir / f"temp_{n_val}.json"
        perf_out = job_dir / f"perf_{n_val}.csv"
        dram_out = job_dir / f"dram_{n_val}.csv"

        ctl_r = ctl_w = None
        if perf_events:
            ctl_r, ctl_w = os.pipe()

        inner = ["taskset", "-c", pinned_core]

        if perf_events:
            pcore_events = ",".join(f"cpu_core/{e}/" for e in perf_events)
            inner += [
                "uv",
                "run",
                "perf",
                "stat",
                "--control",
                f"fd:{ctl_r}",
                "--no-big-num",
                "-x",
                ",",
                "-e",
                pcore_events,
                "-o",
                str(perf_out),
                "--",
            ]
        else:
            inner += ["uv", "run"]

        bench = [
            "python",
            "turbogator/benchmark.py",
            "--desc",
            desc,
            "--t",
            str(t_val),
            "--c",
            str(c_val),
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
            "--out",
            str(temp_out),
        ]
        if ctl_w is not None:
            bench += ["--perf-ctl-fd", str(ctl_w)]
        inner += bench

        if dram_events:
            full_cmd = [
                "perf",
                "stat",
                "-C",
                pinned_core,
                "--no-big-num",
                "-x",
                ",",
                "-e",
                ",".join(dram_events),
                "-o",
                str(dram_out),
                "--",
            ] + inner
        else:
            full_cmd = inner

        fds = (ctl_r, ctl_w) if ctl_r is not None else ()
        run_cmd(full_cmd, job_dir, env, log_file, pass_fds=fds)

        if ctl_r is not None:
            os.close(ctl_r)
            os.close(ctl_w)

        if temp_out.exists():
            run_data = json.loads(temp_out.read_text())
            run_data["N"] = n_val

            if perf_events:
                raw = parse_perf_csv(perf_out, perf_events)
                run_data["perf"] = {
                    k: (v / steps if v is not None else None) for k, v in raw.items()
                }

            if dram_events and dram_out.exists():
                raw_dram = parse_perf_csv(dram_out, dram_events)
                run_data["dram"] = {
                    k: (v / steps if v is not None else None)
                    for k, v in raw_dram.items()
                }

            metrics.append(run_data)
            temp_out.unlink()

        for f in (perf_out, dram_out):
            if f.exists():
                f.unlink()


def main():
    if len(sys.argv) != 4:
        print("Usage: worker.py <job_id> <user> <description>", file=sys.stderr)
        return 1

    job_id, user, desc = sys.argv[1], sys.argv[2], sys.argv[3]
    root_dir = os.environ.get("AOS_ROOT_DIR", "/opt/aos")
    workspace_dir = Path(os.environ.get("AOS_WORKSPACE_DIR", f"{root_dir}/workspaces"))
    artifact_dir = Path(os.environ.get("AOS_ARTIFACT_DIR", f"{root_dir}/artifacts"))
    job_dir = workspace_dir / job_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(job_dir / "workspace.tar.gz", "r:gz") as tf:
        tf.extractall(path=job_dir)

    sys.path.insert(0, str(job_dir))
    import config as app_config

    build_jobs = getattr(app_config, "BUILD_JOBS", 4)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
            "CMAKE_C_COMPILER_LAUNCHER": "ccache",
            "CMAKE_BUILD_PARALLEL_LEVEL": str(build_jobs),
            "MAKEFLAGS": f"-j{build_jobs}",
            "UV_PROJECT_ENVIRONMENT": f"{root_dir}/shared_venv",
            "UV_NO_BUILD_ISOLATION": "1",
        }
    )

    log_file = open(job_dir / "run.log", "w")
    metrics = []

    failure_reason = None
    try:
        build_project(job_dir, env, log_file)
        _fence(log_file, "after build")

        if not validate_build(job_dir, env, log_file):
            failure_reason = "validate_failed"
            raise RuntimeError("validate.py failed; skipping microbench and sweep")
        _fence(log_file, "after verify_build")

        run_microbench(job_dir, env, log_file, app_config, desc)
        _fence(log_file, "after microbench")

        run_sweep(job_dir, env, log_file, app_config, desc, metrics)

    except Exception as e:
        if failure_reason is None:
            failure_reason = "exception"
        print(f"\nERROR: {e}", file=log_file)
        traceback.print_exc(file=log_file)

    finally:
        final_json = {
            "server": getattr(app_config, "ACTIVE_SERVER", "mihai"),
            "user": user,
            "description": desc,
            "job_id": job_id,
        }
        if metrics:
            final_json["data"] = metrics
        if failure_reason:
            final_json["error"] = failure_reason

        (job_dir / "metrics.json").write_text(json.dumps(final_json, indent=2))

        if metrics and not failure_reason:
            with open(artifact_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(final_json) + "\n")

        log_file.close()

        artifact_dir.mkdir(parents=True, exist_ok=True)
        tmp_archive = artifact_dir / f"{job_id}.tar.gz.tmp"
        final_archive = artifact_dir / f"{job_id}.tar.gz"
        if tmp_archive.exists():
            tmp_archive.unlink()
        with tarfile.open(tmp_archive, "w:gz") as tf:
            for f in [
                "metrics.json",
                "profile.speedscope.json",
                "pytorch_profile.trace.json",
            ]:
                if (job_dir / f).exists():
                    tf.add(job_dir / f, arcname=f)

            log_path = job_dir / "run.log"
            if log_path.exists():
                clean = _ANSI_SGR.sub("", log_path.read_text()).encode()
                info = tarfile.TarInfo(name="run.log")
                info.size = len(clean)
                info.mtime = int(log_path.stat().st_mtime)
                tf.addfile(info, io.BytesIO(clean))

            advisor_dir = job_dir / "advisor_results"
            if advisor_dir.exists():
                tf.add(advisor_dir, arcname="advisor_results")

        os.replace(tmp_archive, final_archive)
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
