#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path


def run_cmd(cmd, job_dir, env, log_file):
    cmd_str = [str(x) for x in cmd]

    print(f"\n$ {' '.join(cmd_str)}", file=log_file, flush=True)
    subprocess.run(
        cmd_str, cwd=job_dir, env=env, stdout=log_file, stderr=log_file, check=True
    )


def build_project(job_dir, env, log_file):
    run_cmd(["uv", "sync"], job_dir, env, log_file)
    # run_cmd(["uv", "pip", "install", "-e", "."], job_dir, env, log_file)


def run_sweep(job_dir, env, log_file, app_config, desc):
    metrics = []
    pinned_core = app_config.PINNED_CPU_CORE

    for n_val in app_config.SIZES:
        t_val, c_val = app_config.get_dimensions(n_val)
        temp_out = job_dir / f"temp_{n_val}.json"

        cmd = [
            "taskset",
            "-c",
            pinned_core,
            "uv",
            "run",
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
            metrics.append(run_data)
            temp_out.unlink()

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
        with tarfile.open(artifact_dir / f"{job_id}.tar.gz", "w:gz") as tf:
            for f in ["metrics.json", "run.log"]:
                if (job_dir / f).exists():
                    tf.add(job_dir / f, arcname=f)

        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
