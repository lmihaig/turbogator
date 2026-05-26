#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

ENABLE_INTEL_ADVISOR = False


def run_cmd(cmd, job_dir, env, log_file, pass_fds=()):
    cmd_str = [str(x) for x in cmd]

    print(f"\n$ {' '.join(cmd_str)}", file=log_file, flush=True)
    subprocess.run(
        cmd_str,
        cwd=job_dir,
        env=env,
        stdout=log_file,
        stderr=log_file,
        check=True,
        pass_fds=pass_fds,
    )


def build_project(job_dir, env, log_file):
    run_cmd(["uv", "sync", "--reinstall-package", "turbogator"], job_dir, env, log_file)


def verify_build(job_dir, env, log_file):
    ccdb = job_dir / "build" / "skbuild" / "compile_commands.json"
    if ccdb.exists():
        cpp = [
            e
            for e in json.loads(ccdb.read_text())
            if e.get("file", "").endswith(".cpp")
        ]
        if cpp:
            print(f"\nsample compile command:\n  {cpp[0]['command']}", file=log_file)

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
        print(f"{so.name}: ymm={ymm} xmm={xmm} fma={fma}", file=log_file)


def parse_perf_csv(perf_path, events):
    results: dict[str, float | None] = {e: None for e in events}
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
            # e-core we dont care about
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


def run_intel_advisor(job_dir, env, log_file, app_config, desc):
    pinned_core = app_config.PINNED_CPU_CORE
    n_val = app_config.REPRESENTATIVE_N
    t_val, c_val = app_config.get_dimensions(n_val)

    advisor_bin = "/opt/intel/oneapi/advisor/latest/bin64/advisor"
    advisor_dir = job_dir / "advisor_results"

    temp_out = job_dir / "temp_advisor.json"
    venv_root = env.get("UV_PROJECT_ENVIRONMENT")
    python_exe = str(Path(venv_root) / "bin" / "python") if venv_root else "python"

    shutil.rmtree(advisor_dir, ignore_errors=True)

    cmd = [
        "taskset",
        "-c",
        pinned_core,
        advisor_bin,
        "-collect=roofline",
        "-start-paused",
        f"-project-dir={advisor_dir}",
        "--",
        python_exe,
        "turbogator/benchmark.py",
        "--desc",
        desc,
        "--t",
        str(t_val),
        "--c",
        str(c_val),
        "--profile",
        "advisor",
        "--steps",
        str(50),
        "--out",
        str(temp_out),
    ]
    run_cmd(cmd, job_dir, env, log_file)
    if temp_out.exists():
        temp_out.unlink()


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

        # wrap with outer DRAM perf
        if dram_events:
            full_cmd = [
                "perf",
                "stat",
                "-a",
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

        # Collect results
        if temp_out.exists():
            run_data = json.loads(temp_out.read_text())
            run_data["N"] = n_val

            if perf_events:
                raw = parse_perf_csv(perf_out, perf_events)
                # per step average
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

    try:
        build_project(job_dir, env, log_file)
        verify_build(job_dir, env, log_file)
        if ENABLE_INTEL_ADVISOR:
            run_intel_advisor(job_dir, env, log_file, app_config, desc)
        run_sweep(job_dir, env, log_file, app_config, desc, metrics)

    except Exception as e:
        print(f"\nERROR: {e}", file=log_file)
        traceback.print_exc(file=log_file)

    finally:
        if len(metrics) > 0:
            final_json = {
                "server": getattr(app_config, "ACTIVE_SERVER", "mihai"),
                "user": user,
                "description": desc,
                "job_id": job_id,
                "data": metrics,
            }
            (job_dir / "metrics.json").write_text(json.dumps(final_json, indent=2))

            with open(artifact_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(final_json) + "\n")

        log_file.close()

        artifact_dir.mkdir(parents=True, exist_ok=True)
        tmp_archive = artifact_dir / f"{job_id}.tar.gz.tmp"
        final_archive = artifact_dir / f"{job_id}.tar.gz"
        if tmp_archive.exists():
            tmp_archive.unlink()
        with tarfile.open(tmp_archive, "w:gz") as tf:
            for f in ["metrics.json", "run.log"]:
                if (job_dir / f).exists():
                    tf.add(job_dir / f, arcname=f)
            advisor_dir = job_dir / "advisor_results"
            if advisor_dir.exists():
                tf.add(advisor_dir, arcname="advisor_results")

        os.replace(tmp_archive, final_archive)
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
