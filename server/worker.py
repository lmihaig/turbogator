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


def run_cmd(cmd, job_dir, env, log_file):
    cmd_str = [str(x) for x in cmd]

    print(f"\n$ {' '.join(cmd_str)}", file=log_file, flush=True)
    subprocess.run(
        cmd_str, cwd=job_dir, env=env, stdout=log_file, stderr=log_file, check=True
    )


def build_project(job_dir, env, log_file):
    run_cmd(["uv", "sync", "--reinstall-package", "turbogator"], job_dir, env, log_file)
    # run_cmd(["uv", "pip", "install", "-e", "."], job_dir, env, log_file)


def parse_perf_csv(perf_path, events):
    results: dict[str, float | None] = {event: None for event in events}
    if not perf_path.exists():
        return results

    event_set = set(events)
    for raw_line in perf_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split(",")]
        value = parts[0] if len(parts) >= 1 else None
        event = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else None)

        if event is None or event not in event_set:
            continue

        if value is None:
            results[event] = None
            continue

        try:
            results[event] = float(value)
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


def run_sweep(job_dir, env, log_file, app_config, desc):
    metrics = []
    pinned_core = app_config.PINNED_CPU_CORE
    perf_events = [str(e) for e in getattr(app_config, "PERF_EVENTS", [])]

    for n_val in app_config.SIZES:
        t_val, c_val = app_config.get_dimensions(n_val)
        temp_out = job_dir / f"temp_{n_val}.json"
        perf_out = job_dir / f"perf_{n_val}.csv"

        cmd = ["taskset", "-c", pinned_core]

        if perf_events:
            cmd += [
                "uv",
                "run",
                "perf",
                "stat",
                "--no-big-num",
                "-x",
                ",",
                "-e",
                ",".join(perf_events),
                "-o",
                str(perf_out),
                "--",
            ]
        else:
            cmd += ["uv", "run"]

        cmd += [
            "python",
            "turbogator/benchmark.py",
            "--desc",
            desc,
            "--t",
            str(t_val),
            "--c",
            str(c_val),
            "--out",
            str(temp_out),
        ]

        run_cmd(cmd, job_dir, env, log_file)

        if temp_out.exists():
            run_data = json.loads(temp_out.read_text())
            run_data["N"] = n_val
            if perf_events:
                run_data["perf"] = parse_perf_csv(perf_out, perf_events)
            metrics.append(run_data)
            temp_out.unlink()
        if perf_out.exists():
            perf_out.unlink()

    return metrics


def main():
    if len(sys.argv) != 4:
        print("Usage: worker.py <job_id> <user> <description>", file=sys.stderr)
        return 1

    job_id, user, desc = sys.argv[1], sys.argv[2], sys.argv[3]
    job_dir = Path(f"/opt/aos/workspaces/{job_id}")
    artifact_dir = Path("/opt/aos/artifacts")

    with tarfile.open(job_dir / "workspace.tar.gz", "r:gz") as tf:
        tf.extractall(path=job_dir)

    sys.path.insert(0, str(job_dir))
    import config as app_config

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
            "CMAKE_C_COMPILER_LAUNCHER": "ccache",
            "UV_PROJECT_ENVIRONMENT": "/opt/aos/shared_venv",
            "UV_NO_BUILD_ISOLATION": "1",
        }
    )

    log_file = open(job_dir / "run.log", "w")
    metrics = []

    try:
        build_project(job_dir, env, log_file)
        if ENABLE_INTEL_ADVISOR:
            run_intel_advisor(job_dir, env, log_file, app_config, desc)
        metrics = run_sweep(job_dir, env, log_file, app_config, desc)

    except Exception as e:
        print(f"\nERROR: {e}", file=log_file)
        traceback.print_exc(file=log_file)

    finally:
        if len(metrics) > 0:
            final_json = {
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
